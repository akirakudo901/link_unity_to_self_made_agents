"""
An integration of the Walker environment with SAC.
"""

from mlagents_envs.environment import BaseEnv
from mlagents_envs.registry import default_registry
import numpy as np
import torch

import ddqn
from ddqn import DDQN
from trainers.unityenv_base_trainer import OffPolicyBaseTrainer

SAVE_AFTER_TRAINING = True

print("About to create the environment! Press the play button to initiate training!")
# env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env = default_registry["Walker"].make()
print("Successfully created the environment!")

behavior_name = list(env.behavior_specs)[0]

learning_algorithm = DDQN(
    observation_size=env.behavior_specs[behavior_name].observation_specs[0].shape[0],
    action_size=env.behavior_specs[behavior_name].action_spec.continuous_size, 
    l_r=0.1, 
    d_r=0.95
)

trainer = OffPolicyBaseTrainer(env, behavior_name, learning_algorithm)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 70
# The number of experiences to collect per training step
NUM_NEW_EXP = 100
# The maximum size of the Buffer
BUFFER_SIZE = 1000

EPSILON = 0.1

TASK_NAME = "DDQN" + "_Walker"

ACTION_DISCRETIZED_NUMBER = ddqn.ACTION_DISCRETIZED_NUMBER

#TODO Is there a way for me to move DISCRETIZED_NUM to ddqn? 
def fixed_epsilon(discrete_actions : torch.tensor, epsilon : float, env : BaseEnv):
    discrete_actions = discrete_actions.detach().reshape(
        discrete_actions.shape[0],
        env.behavior_specs[behavior_name].action_spec.continuous_size,
        ACTION_DISCRETIZED_NUMBER
    )
    discrete_actions += epsilon * (np.random.randn(*discrete_actions.shape)).astype(np.float32)
    
    discrete_best_actions = torch.argmax(discrete_actions, dim=2, keepdim=False)
    continuous_actions = DDQN.convert_discrete_action_to_continuous(discrete_best_actions).numpy()
    return continuous_actions

l_a = trainer.train(
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP, 
    max_buffer_size=BUFFER_SIZE,
    exploration_function=fixed_epsilon,
    epsilon=EPSILON,
    save_after_training=SAVE_AFTER_TRAINING,
    task_name=TASK_NAME
    )