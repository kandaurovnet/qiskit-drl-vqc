import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Initialize CartPole environment
env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1")

# Configure a callback to stop training early once it hits the target performance
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./logs/",
    log_path="./logs/", 
    eval_freq=1000,
    deterministic=True, 
    render=False
)

# Initialize optimized DQN Agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=0.05,                  # Soft update for target network stability
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    exploration_fraction=0.2,  # Quick exploration decay over the first 20% of steps
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    target_update_interval=10,
    verbose=1
)

# Train the agent (500 steps max per episode means ~150,000 steps total for 300 episodes maximum)
model.learn(total_timesteps=100000, callback=eval_callback)
