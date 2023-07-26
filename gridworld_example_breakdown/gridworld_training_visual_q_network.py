"""
At least now I know that this file can be run using the Unity editor, and we can  
directly see the training process, which is great!
Runs with the GridWorldColab scene.
"""

import os
import random
from typing import List

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
import matplotlib.pyplot as plt

import torch

from gridworld_example_breakdown.visual_q_network import VisualQNetwork
from gridworld_example_breakdown.gridworld_base_trainer import Buffer, GridWorldBaseTrainer

SAVE_Q_NET_AFTER_TRAINING = True

"""
ORIGINAL CODE WHICH SET UP THE ENVIRONMENT:

"""

print("About to create the environment! Press the play button to initiate training!")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
# Create the GridWorld Environment from the registry
# env = default_registry["GridWorld"].make()
print("GridWorld environment created.")

# print("Successfully created the environment!")

num_actions = 5
# The number of training steps that will be performed
NUM_TRAINING_STEPS = int(os.getenv('QLEARNING_NUM_TRAINING_STEPS', 5))
# The number of experiences to collect per training step
NUM_NEW_EXP = int(os.getenv('QLEARNING_NUM_NEW_EXP', 1000))
# The maximum size of the Buffer
BUFFER_SIZE = int(os.getenv('QLEARNING_BUFFER_SIZE', 10000))

EPSILON = 0.1

try:
    # Create a new Q-Network.
    qnet = VisualQNetwork((3, 64, 84), 126, num_actions)
    
    def wrapped_qnet(np_obs):
        return qnet(torch.from_numpy(np_obs)).detach().numpy()

    experiences: Buffer = []
    optim = torch.optim.Adam(qnet.parameters(), lr=0.001)

    cumulative_rewards: List[float] = []

    for n in range(NUM_TRAINING_STEPS):
        new_exp, _ = GridWorldBaseTrainer.generate_trajectories(
            env, wrapped_qnet, NUM_NEW_EXP, epsilon=EPSILON
        )
        random.shuffle(experiences)
        if len(experiences) > BUFFER_SIZE:
            experiences = experiences[:BUFFER_SIZE]
        experiences.extend(new_exp)
        qnet.update_q_net(optim, experiences, num_actions)

        _, rewards = GridWorldBaseTrainer.generate_trajectories(env, wrapped_qnet, 100, epsilon=0)
        cumulative_rewards.append(rewards)
        print("Training step ", n+1, "\treward ", rewards)

    if SAVE_Q_NET_AFTER_TRAINING:
        qnet.save_model(experiences[0].obs)

except KeyboardInterrupt:
    print("\nTraining interrupted, continue to next cell to save to save the model.")
finally:
    env.close()

# Show the training graph
try:
    plt.plot(range(NUM_TRAINING_STEPS), cumulative_rewards)
    plt.show()
except ValueError:
    print("\nPlot failed on interrupted training.")