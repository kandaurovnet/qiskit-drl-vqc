No. 1
Variational Quantum Circuits for Deep Reinforcement Learning in Classic Control – Damien Jian
Introduction:

Reinforcement learning (RL) plays an important role in robotics, autonomous control, game-playing, and large language model fine-tuning. Meanwhile, variational quantum circuits (VQCs), also known as parameterized quantum circuits (PQCs), have emerged as quantum counterparts to classical neural networks. By combining classical data encoding, trainable quantum gates, and quantum measurements, VQCs can approximate policies or value functions using compact hybrid quantum-classical models.

This challenge explores quantum reinforcement learning in CartPole-v1, a classic control environment with continuous observations and discrete actions.


Goal:
Fundamental: Train a VQC as a quantum deep Q-network (QDQN) that can balance the pole in CartPole-v1. Implement observation normalization, data encoding, experience replay, a target network, and epsilon-greedy exploration. Evaluate the trained agent using average episode reward.
Advanced: Explore data re-uploading, Double QDQN, quantum policy-gradient, or quantum actor-critic architectures. Compare the quantum agent with a classical neural-network baseline in terms of performance, trainable parameters, and robustness under finite-shot or noisy quantum simulation.


Reference:
Chen, S. Y.-C., et al. (2020). Variational quantum circuits for deep reinforcement learning. https://arxiv.org/abs/1907.00397
Skolik, A., Jerbi, S., & Dunjko, V. (2022). Quantum agents in the Gym: A variational quantum algorithm for deep Q-learning. https://arxiv.org/abs/2103.15084
TensorFlow Quantum. Parametrized quantum circuits for reinforcement learning. https://www.tensorflow.org/quantum/tutorials/quantum_reinforcement_learning
Pérez-Salinas, A., et al. (2020). Data re-uploading for a universal quantum classifier. https://doi.org/10.22331/q-2020-02-06-226
Towers, M., et al. (2024). Gymnasium: A standard interface for reinforcement learning environments. https://arxiv.org/abs/2407.17032