# Linking Unity to Self-made Agents

This repository is a very informal playground for me to play around with the Unity ML-agents environment, 
such that I can apply many state-of-the-art algorithm and see their performances.

As of right now, the only agent implemented is Soft Actor Critic: you can find it under
* policy_learning_algorithms/soft_actor_critic.py

## What folders are

* **agents** folder holds all agents, which are combinations of environments and certain policy learning algorithms.
* **gridworld_example_breakdown** folder holds codes and publicly available tutorials for the Unity ML-agents Grid World Environment,
  which I am using to understand how ML-agents works.
* **policy_learning_algorithms** folder holds all policy learning algorithms.
* **trainers** folder holds all "trainer" objects, which wraps around specific Unity ML-agents environment to allow interface with agents for training.
* **walker_trail_of_different_algorithms** folder will hold all attempts to apply different policy learning algorithms to the Unity Walker environment.

This is very much a work in progress!
