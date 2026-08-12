#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import json
import math
import pickle
import random
import time
from collections import namedtuple
from itertools import count
import os
import matplotlib
import matplotlib.pyplot as plt
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_quantum_maze.envs import quantum_maze_env

import sys


# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Neural Network
class DQN(nn.Module):

    def __init__(self, inputs, outputs, intermediates=None, env=None, diag_threshold=10**(-14)):
        self.inputs = inputs
        self.outputs = outputs
        self.env = env
        self.diag_threshold = diag_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(DQN, self).__init__()
        if intermediates is None:
            intermediates = [256, 128]
        if isinstance(intermediates, int):  # int to List conversion
            intermediates = [intermediates]

        intermediates.insert(0, inputs)
        intermediates.append(outputs)
        self.layers = nn.ModuleList()
        for k in range(len(intermediates) - 1):
            self.layers.append(nn.Linear(intermediates[k], intermediates[k + 1]))

    def forward(self, x):
        if self.env is not None:
            mask = torch.full((self.outputs,), -1e9, requires_grad=False, device=self.device)
            mask[0] = 0. # no action is always valid
            for n in range(self.env.quantum_system_size-1):
                if x[0][n] >= self.diag_threshold:
                    nx, ny = self.env.maze.node2xy(n)
                    if nx > 0:
                        mask[self.env.maze.xy2link(nx - 1, ny)] = 0.
                    if nx < self.env.maze.width:
                        mask[self.env.maze.xy2link(nx + 1, ny)] = 0.
                    if ny > 0:
                        mask[self.env.maze.xy2link(nx, ny - 1)] = 0.
                    if ny < self.env.maze.height:
                        mask[self.env.maze.xy2link(nx, ny + 1)] = 0.
        else:
            mask = torch.zeros(self.outputs, requires_grad=False, device=self.device)
        # Apply ReLU only to hidden layers
        for single_layer in self.layers[:-1]:
            x = F.relu(single_layer(x))
        x = self.layers[-1](x)
        return x + mask

# deep_Q_learning
def deep_Q_learning_maze(maze_filename=None, p=0.1, total_actions=4, actions_number=1,
                         num_episodes=100, changeable_links=None,  # [4, 15, 30, 84],
                         batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05,
                         eps_decay=12500, target_update=100, replay_capacity=100000,
                         save_filename=None, enable_saving=True,
                         startNode=None, sinkerNode=None, training_startNodes=None,
                         action_selector=None, state_selector=7, diag_threshold=10**(-12),
                         saving_folder='results'):
    """Function that performs the deep Q learning.

    Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    action_selector:
        None (default), i.e. pick any action
        'threshold_mask', i.e. pick any action near a node with population greater than diag_threshold
        'probability_mask', i.e. pick a node according to the population, and from that an action according to the links
                            from that node
    """
    tic = time.time()
    codeName = 'deep_Q_learning_maze'
    today = datetime.datetime.now()

    env = quantum_maze_env.QuantumMazeEnv(maze_filename=maze_filename, startNode=startNode, sinkerNode=sinkerNode,
                                          p=p, changeable_links=changeable_links,
                                          total_actions=total_actions, state_selector=state_selector,
                                          actions_number=actions_number)
    print("[ENV] Actions ({}) scheduled at walk steps: {}".format(env.actions_number, env.action_positions))
    if startNode is None:
        startNode = env.initial_maze.startNode
    if sinkerNode is None:
        sinkerNode = env.initial_maze.sinkerNode
    if training_startNodes is None or training_startNodes == []:
        training_startNodes = [startNode]

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_options = '_ST{0}_A{1}_N{2}_P{3:02.0f}'.format(state_selector, total_actions, actions_number, 10 * p)
    if save_filename is None and enable_saving:
        save_filename = ''.join((today.strftime('%Y-%m-%d_%H-%M-%S_'), codeName, config_options))


    # Number of actions (no-op + one per changeable link)
    n_actions = env.n_actions
    # Number of agent decisions per episode (one per action, spread evenly over the walk)
    num_decisions = env.num_decisions
    # Define initial reward
    env.reset()
    initial_reward = env.cumulative_reward

    # Setup Neural Network
    if action_selector == 'threshold_mask':
        policy_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
        target_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
    elif action_selector == 'probability_mask':
        policy_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
        target_net = DQN(len(env.state), n_actions, env=env, diag_threshold=diag_threshold).to(device)
    else:
        policy_net = DQN(len(env.state), n_actions).to(device)
        target_net = DQN(len(env.state), n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayMemory(replay_capacity)

    def select_action(state, eps_threshold, net=policy_net, noAction_probability=0.05):
        sample = random.random()
        if action_selector == 'threshold_mask':
            mask = set()
            for n in range(env.quantum_system_size - 1):
                if env.quantum_state.full()[n, n] >= diag_threshold:
                    nx, ny = env.maze.node2xy(n)
                    if nx > 0:
                        mask.add(env.maze.xy2link(nx - 1, ny))
                    if nx < env.maze.width:
                        mask.add(env.maze.xy2link(nx + 1, ny))
                    if ny > 0:
                        mask.add(env.maze.xy2link(nx, ny - 1))
                    if ny < env.maze.height:
                        mask.add(env.maze.xy2link(nx, ny + 1))
            if sample > eps_threshold:  # choose action from network
                with torch.no_grad():
                    masked_tensor = torch.tensor(list(mask), dtype=torch.long)
                    a_max = net(state)[0, masked_tensor].max(0)
                    return torch.tensor([[masked_tensor[a_max[1]]]], device=device, dtype=torch.long)
                    # check print(net(state)[0,masked_tensor[a_max[1]]])
            else:  # choose random action from masked ones by threshold
                return torch.tensor([[random.choice(tuple(mask))]], device=device, dtype=torch.long)
        elif action_selector == 'probability_mask':
            if sample > eps_threshold:  # choose action from network
                mask = set()
                for n in range(env.quantum_system_size - 1):
                    if env.quantum_state.full()[n, n] >= diag_threshold:
                        nx, ny = env.maze.node2xy(n)
                        if nx > 0:
                            mask.add(env.maze.xy2link(nx - 1, ny))
                        if nx < env.maze.width:
                            mask.add(env.maze.xy2link(nx + 1, ny))
                        if ny > 0:
                            mask.add(env.maze.xy2link(nx, ny - 1))
                        if ny < env.maze.height:
                            mask.add(env.maze.xy2link(nx, ny + 1))
                with torch.no_grad():
                    masked_tensor = torch.tensor(list(mask), dtype=torch.long)
                    a_max = net(state)[0, masked_tensor].max(0)
                    return torch.tensor([[masked_tensor[a_max[1]]]], device=device, dtype=torch.long)
            else:  # choose random action from masked ones by population
                # note that in this way you always define a valid action, action=0 (noAction) is never selected
                population = np.real(np.diag(env.quantum_state.full())[:env.quantum_system_size - 1])
                if population.min() < 0:
                    population = population - population.min()
                population = population/population.sum()  # normalize
                n = np.random.choice(range(env.quantum_system_size - 1), 1, p=population)
                nx, ny = env.maze.node2xy(n)
                available_links = [0]  # noAction
                if nx > 0:
                    available_links.append(1)
                if nx < env.maze.width:
                    available_links.append(3)
                if ny > 0:
                    available_links.append(2)
                if ny < env.maze.height:
                    available_links.append(4)
                selected_direction = np.random.choice(available_links, 1, p=[noAction_probability] + \
                                        [(1-noAction_probability)/(len(available_links)-1)]*(len(available_links)-1))
                if selected_direction == 1:
                    selected_link = env.maze.xy2link(nx - 1, ny)
                elif selected_direction == 3:
                    selected_link = env.maze.xy2link(nx + 1, ny)
                elif selected_direction == 2:
                    selected_link = env.maze.xy2link(nx, ny - 1)
                elif selected_direction == 4:
                    selected_link = env.maze.xy2link(nx, ny + 1)
                else:
                    selected_link = 0
                return torch.tensor([[selected_link]], device=device, dtype=torch.long)
        else:  # any actions
            if sample > eps_threshold:  # choose action from network
                with torch.no_grad():
                    return net(state).max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    episode_transfer_to_sink = []
    target_transfer_to_sink = []
    episode_sink_prob = []   # actual sink probability (undiscounted) per episode
    target_sink_prob = []    # actual sink probability evaluated by target net
    loss_history = []  # average loss per episode
    episode_losses = []  # temporary storage for losses within current episode

    def optimize_model():
        if len(memory) < batch_size:
            return None
        transitions = memory.sample(batch_size)

        batch = Transition(*zip(*transitions))

        if torch.__version__ < '1.2.0':
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.uint8)  # version on qdab server
        else:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                          dtype=torch.bool)  # current version, torch.uint8 is deprecated

        non_final_next_states_list = [s for s in batch.next_state if s is not None]
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=device)
        if non_final_next_states_list:
            non_final_next_states = torch.cat(non_final_next_states_list)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # else: every sampled transition is terminal (e.g. actions_number == 1, one
        # decision per episode), so there is nothing to bootstrap from and
        # next_state_values stays zero.
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss.item()

    def evaluate_sequence_with_target_net(sequence=None):
        """ If sequence is None, evaluates the best action with target_net
        """
        if sequence is None:
            evaluate_optimal_action = True
            sequence = [0] * num_decisions
        else:
            evaluate_optimal_action = False

        env.reset()
        state = torch.tensor(env.state, device=device, dtype=torch.float32).unsqueeze(0)
        episode_reward = initial_reward
        for t in range(num_decisions):
            # Select and perform an action
            if evaluate_optimal_action:
                eps_threshold = -1
                action = select_action(state, eps_threshold, target_net).item()
                sequence[t] = action
            else:
                action = sequence[t]

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # reward = reward.to(device='cpu') #torch.tensor([reward], device=device, dtype=torch.float32)
            episode_reward = reward + gamma * episode_reward

            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                break

        return episode_reward, sequence, env.cumulative_reward

    print("\n\n\n\n Replay Memory Generation \n\n\n\n")
    # Pre-Training loop: store in replay memory a number of episodes equal to batch_size
    for i_episode in range(batch_size):
        print("\n\nReplay step:", i_episode, "out of", batch_size, "\n\n")
        eps_threshold = 2
        for id_startNode, startNode_tmp in enumerate(training_startNodes):
            # Initialize the environment and state
            env.initial_maze.startNode = startNode_tmp
            env.reset()
            state = torch.tensor(env.state, device=device, dtype=torch.float32).unsqueeze(0)
            episode_reward = initial_reward
            for t in range(num_decisions):
                # Select and perform an action
                action = select_action(state, eps_threshold)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                episode_reward = reward + gamma * episode_reward
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)
                # Store the transition in memory
                memory.push(state, action, next_state, reward)
                # Move to the next state
                state = next_state
                if done:
                    break

    print("\n\n\n\n Start Training \n\n\n\n")
    # Training loop
    steps_done = 0
    for i_episode in range(num_episodes):
        print("\n\nTraining step:", i_episode, "out of", num_episodes, "\n\n")
        episode_reward_startNode = [0] * len(training_startNodes)
        for id_startNode, startNode_tmp in enumerate(training_startNodes):
            # Initialize the environment and state
            env.initial_maze.startNode = startNode_tmp
            env.reset()
            state = torch.tensor(env.state, device=device, dtype=torch.float32).unsqueeze(0)
            episode_reward = initial_reward
            for t in range(num_decisions):
                eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
                steps_done += 1
                # Select and perform an action
                action = select_action(state, eps_threshold)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward = torch.tensor([reward], device=device, dtype=torch.float32)
                episode_reward = reward + gamma * episode_reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization per environment step
                loss_val = optimize_model()
                if loss_val is not None:
                    episode_losses.append(loss_val)

                if done:
                    break

            episode_reward_startNode[id_startNode] = episode_reward.to(device='cpu')
            episode_transfer_to_sink.append(episode_reward_startNode[id_startNode])
            episode_sink_prob.append(float(env.cumulative_reward))

        if len(episode_losses) > 0:
            loss_history.append(sum(episode_losses) / len(episode_losses))
            episode_losses = []  # reset for next episode
        else:
            loss_history.append(0.0)

        # Update the target network, copying all weights and biases in DQN
        if i_episode % target_update == 0:
            env.initial_maze.startNode = startNode
            reward_target, _, sink_prob_target = evaluate_sequence_with_target_net(None)
            target_transfer_to_sink.extend([reward_target] * target_update)
            target_sink_prob.extend([sink_prob_target] * target_update)
            target_net.load_state_dict(policy_net.state_dict())
            # print('Completed episode', i_episode, 'of', num_episodes)

    env.initial_maze.startNode = startNode
    reward_no_actions, _, sink_prob_no_actions = evaluate_sequence_with_target_net([0] * num_decisions)
    reward_final, optimal_sequence, sink_prob_final = evaluate_sequence_with_target_net(None)

    # Save variables:
    if enable_saving:
        save_variables(os.path.join(saving_folder, save_filename),
                       episode_transfer_to_sink, env, steps_done, policy_net, maze_filename, p,
                       total_actions, actions_number,
                       num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
                       replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink,
                       saving_folder)

        # TODO: include the following into the save_variables function, but I can't find a way to create a dict without
        #  having to specify the names
        with open(os.path.join(saving_folder, save_filename) + '.json', 'w') as f:
            json.dump({"steps_done": steps_done,
                       "saving_folder": saving_folder,
                       "maze_filename": maze_filename,
                       "p": p if p < 1 else 1,  # sometimes saves it as int64
                       "total_actions": total_actions,
                       "actions_number": actions_number,
                       "action_positions": env.action_positions,
                       "num_episodes": num_episodes,
                       "changeable_links": changeable_links,
                       "batch_size": batch_size,
                       "gamma": gamma,
                       "eps_start": eps_start,
                       "eps_end": eps_end,
                       "eps_decay": eps_decay,
                       "target_update": target_update,
                       "replay_capacity": replay_capacity,
                       "reward_no_actions": reward_no_actions,
                       "reward_final": reward_final,
                       "sink_prob_no_actions": sink_prob_no_actions,
                       "sink_prob_final": sink_prob_final,
                       "optimal_sequence": optimal_sequence,
                       "target_transfer_to_sink": target_transfer_to_sink,  # at the end since it is really long
                       "target_sink_prob": target_sink_prob,
                       "episode_transfer_to_sink": [x.item() for x in episode_transfer_to_sink],  # at the end since it is really long
                       "episode_sink_prob": episode_sink_prob,
                       "loss_history": loss_history  # average training loss per episode
                       }, f, indent=4)
    toc = time.time()
    elapsed = toc - tic

    return save_filename, elapsed, reward_final, optimal_sequence


def save_variables(filename=None, *args):
    """ TODO: atm it saves only 1 net, solve to possible con with target_net, policy_net  """
    if filename is None:
        today = datetime.datetime.now()
        filename = today.strftime('%Y-%m-%d_%H-%M-%S')

    # Ensure output directory exists if a folder is included in filename
    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(filename + '.pkl', 'wb') as f:
        pickle.dump([x.to('cpu') if isinstance(x, torch.Tensor) else x for x in args if not isinstance(x, DQN)], f)

    for x in args:
        if isinstance(x, DQN):
            torch.save(x.state_dict(), filename + '_policy_net.pt')

    return filename


def plot_durations(episode_transfer_to_sink, title='Training...', constants=None, legend_labels=None, mean_window=100, filename='plot', output_dir='results'):
    if constants is None:
        constants = []
    if legend_labels is None:
        legend_labels = []
    if episode_transfer_to_sink is None or len(episode_transfer_to_sink) == 0:
        print("plot_durations: nothing to plot")
        return
    plt.figure(dpi=300)
    plt.clf()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Sink')

    training_legend_labels = []
    generate_training_labels = not len(episode_transfer_to_sink) + len(constants) == len(legend_labels)

    for (i, transfer_to_sink) in enumerate(episode_transfer_to_sink):
        transfer_to_sink_clean = [float(np.real(x.item())) if isinstance(x, torch.Tensor) else float(np.real(x)) for x in transfer_to_sink]
        y_np = np.asarray(transfer_to_sink_clean, dtype=float).ravel()
        transfer_to_sink_len = y_np.size
        plt.plot(y_np)
        if transfer_to_sink_len >= mean_window:
            means = np.convolve(y_np, np.ones(mean_window) / mean_window, mode='valid')
            means = np.concatenate([np.zeros(mean_window-1), means])
            plt.plot(means.ravel())
            if generate_training_labels:
                training_legend_labels.extend(['training' + ' ' + str(i), 'mean ' + str(mean_window) + ' ' + str(i)])
            else:
                training_legend_labels.extend([legend_labels[i], legend_labels[i] + ' mean ' + str(mean_window)])

        else:
            if generate_training_labels:
                training_legend_labels.append('training' + ' ' + str(i))
            else:
                training_legend_labels.append(legend_labels[i])

    if not constants == []:
        for k in constants:
            k_val = k.item() if isinstance(k, torch.Tensor) else k
            k_val = float(np.real(k_val))
            plt.plot([k_val] * transfer_to_sink_len)
        legend_labels = training_legend_labels + legend_labels[-len(constants):]
    else:
        legend_labels = training_legend_labels

    plt.legend(legend_labels)
    plt.ylim(0, 1)
    if os.path.isabs(filename):
        plot_path = filename + '.png'
    else:
        plot_path = os.path.join(output_dir, filename + '.png')
    print("plot_durations: saving plot to", plot_path)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    # plt.show()


# --- New helper function: plot_from_saved_run
def plot_from_saved_run(json_path, mean_window=100):
    """Load data from a saved JSON file and plot training curves.

    The JSON file is expected to be produced by deep_Q_learning_maze, i.e.
    it contains at least:
      - episode_transfer_to_sink
      - target_transfer_to_sink
      - reward_no_actions
      - reward_final
      - p
      - total_actions
    """
    json_path = os.path.abspath(json_path)
    json_dir = os.path.dirname(json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Required fields saved by deep_Q_learning_maze
    episode_transfer_to_sink = data.get("episode_transfer_to_sink")
    target_transfer_to_sink = data.get("target_transfer_to_sink")
    reward_no_actions = data.get("reward_no_actions")
    reward_final = data.get("reward_final")
    p = data.get("p")
    total_actions = data.get("total_actions")

    if episode_transfer_to_sink is None or target_transfer_to_sink is None:
        print("plot_from_saved_run: JSON file does not contain required transfer data")
        return

    # Reshape into the format expected by plot_durations: list of sequences
    episode_transfer = [episode_transfer_to_sink]
    episode_transfer.append(target_transfer_to_sink)

    constants = [reward_no_actions, reward_final]
    legend_labels = ["training", "target", "no action", "final"]

    title = f"p={p}, depth {total_actions}" if p is not None and total_actions is not None else "Training..."

    # Save as training.png in the same folder as the JSON file
    filename = os.path.join(json_dir, "training")
    print("plot_from_saved_run: plotting from", json_path)
    plot_durations(episode_transfer,
                   title=title,
                   constants=constants,
                   legend_labels=legend_labels,
                   mean_window=mean_window,
                   filename=filename,
                   output_dir=json_dir)


if __name__ == '__main__':
    print('learning_tools has started')

    # Convenience mode: if a single JSON file is passed, just plot from it and exit.
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        plot_from_saved_run(sys.argv[1])
        sys.exit(0)

    training_startNodes = []
    action_selector = 'None'  # 'threshold_mask' and 'probability_mask' do NOT work because of maze mapping
    diag_threshold = 10 ** (-4)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    p=0.0
    episodes=5000
    total_actions=30
    actions_number=2
    maze_name='resources/maze_3x3.pkl'
    if len(sys.argv) > 3:
        p=float(sys.argv[1])
        episodes=int(sys.argv[2])
        total_actions=int(sys.argv[3])
    if len(sys.argv) > 4:
        maze_name=sys.argv[4]
    if len(sys.argv) > 5:
        actions_number=int(sys.argv[5])
    saving_folder = os.path.join(os.getcwd(), 'results', f'{p}_{actions_number}_{timestamp}')
    filename, elapsed, reward_final, optimal_sequence = deep_Q_learning_maze(maze_filename=maze_name,
                                                                             num_episodes=episodes,
                                                                             p=p,
                                                                             total_actions=total_actions,
                                                                             actions_number=actions_number,
                                                                             training_startNodes=training_startNodes,
                                                                             action_selector=action_selector,
                                                                             diag_threshold=diag_threshold,
                                                                             state_selector=7,
                                                                             saving_folder=saving_folder
                                                                             )
    print('Final reward', reward_final)
    print('Variables saved in', filename + '.pkl')
    print('Trained model saved in', filename + '_policy_net.pt')
    print("Elapsed time", elapsed, "sec.\n")

    #This section prints one training
    with open(os.path.join(saving_folder, filename + '.pkl'), 'rb') as f:
        [episode_transfer_to_sink, env, steps_done, maze_filename, p, total_actions, actions_number,
         num_episodes, changeable_links, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
         replay_capacity, reward_no_actions, reward_final, optimal_sequence, target_transfer_to_sink,
         saving_folder] = pickle.load(f)

    legend_labels = []
    if len(training_startNodes) > 1:
        episode_transfer = [episode_transfer_to_sink[k::len(training_startNodes)] for k in
                            range(len(training_startNodes))]
        episode_transfer.append(
            [sum(episode_transfer_to_sink[k * len(training_startNodes):(k + 1) * len(training_startNodes)])
             / len(training_startNodes) for k in range(num_episodes)])
        legend_labels = ['tranining']*(len(training_startNodes)+1)
    else:
        episode_transfer = [episode_transfer_to_sink]
        training_startNodes = [0]

    episode_transfer.append(target_transfer_to_sink)
    legend_labels.append('target')

    constants = [reward_no_actions, reward_final]
    legend_labels.extend(['no action', 'final'])
    plot_durations(episode_transfer,
                   title='p='+str(env.p)+', depth '+str(total_actions),
                   constants=constants,
                   legend_labels=legend_labels,
                   filename=os.path.join(timestamp, 'training'),
                   output_dir='results'
                   )
