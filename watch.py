#!/usr/bin/env python3
"""Render a trained policy as an animated GIF.

    python watch.py --checkpoint results/classical_zoo_policy.pt

ponytail: draws the cart and pole directly from the (x, theta) state instead of
using gymnasium's renderer, which needs pygame. The state *is* the picture.
"""

import argparse
import os

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

from cartpole_dqn import ClassicalQNetwork, normalize_obs

X_LIMIT = 2.4          # cart falls off the track past this
THETA_LIMIT = 0.2095   # 12 degrees, pole considered fallen
POLE_LEN = 1.0         # world units; gymnasium uses half-length 0.5
CART_W, CART_H = 0.4, 0.25


def rollout(policy_net, seed, max_steps=500):
    """Run one greedy episode, returning the states visited."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)
    states = [obs.copy()]
    for _ in range(max_steps):
        with torch.no_grad():
            action = int(policy_net(torch.tensor(normalize_obs(obs)).unsqueeze(0)).argmax(1).item())
        obs, _reward, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        if terminated or truncated:
            break
    env.close()
    return states


def animate(states, out_path, fps=50, stride=1):
    frames = states[::stride]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.set_xlim(-X_LIMIT - 0.6, X_LIMIT + 0.6)
    ax.set_ylim(-0.4, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.plot([-X_LIMIT - 0.6, X_LIMIT + 0.6], [0, 0], color="0.7", lw=1)
    for edge in (-X_LIMIT, X_LIMIT):  # the track limits that end an episode
        ax.plot([edge, edge], [-0.12, 0.12], color="tab:red", lw=2)

    cart = patches.Rectangle((0, 0), CART_W, CART_H, facecolor="0.25")
    ax.add_patch(cart)
    pole, = ax.plot([], [], lw=5, color="tab:orange", solid_capstyle="round")
    hub, = ax.plot([], [], "o", color="0.15", ms=5)
    label = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top",
                    fontfamily="monospace", fontsize=10)

    def draw(i):
        x, _v, theta, _w = frames[i]
        cart.set_xy((x - CART_W / 2, 0))
        top = CART_H
        pole.set_data([x, x + POLE_LEN * np.sin(theta)],
                      [top, top + POLE_LEN * np.cos(theta)])
        hub.set_data([x], [top])
        label.set_text(f"step {i * stride:3d}/{len(states) - 1}\n"
                       f"x     {x:+.2f} / {X_LIMIT}\n"
                       f"angle {np.degrees(theta):+5.1f}deg / 12.0")
        return cart, pole, hub, label

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=True,
                                   interval=1000 / fps)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="results/classical_zoo_policy.pt")
    p.add_argument("--out", default="results/cartpole_solved.gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
    args = p.parse_args()

    # Infer the hidden layer widths from the checkpoint so older runs (which used
    # a narrower net) still load after the default architecture changed.
    sd = torch.load(args.checkpoint)
    widths = [sd[k].shape[0] for k in sd if k.endswith("weight")]
    net = ClassicalQNetwork(hidden=tuple(widths[:-1]))
    net.load_state_dict(sd)
    net.eval()

    states = rollout(net, seed=args.seed)
    steps = len(states) - 1
    print(f"{args.checkpoint}: survived {steps} steps ({steps * 0.02:.1f}s simulated)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    animate(states, args.out, stride=args.stride)
    print(f"Saved {args.out}")
