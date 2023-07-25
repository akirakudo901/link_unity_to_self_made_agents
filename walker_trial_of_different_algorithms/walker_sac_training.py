"""
An integration of the Walker environment with SAC.
"""

from mlagents_envs.environment import BaseEnv
from mlagents_envs.registry import default_registry
import torch

from policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from trainers.unityenv_base_trainer import OffPolicyBaseTrainer

SAVE_AFTER_TRAINING = True

print("About to create the environment! Press the play button to initiate training!")
# env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
env = default_registry["Walker"].make()
print("Successfully created the environment!")

# reset the environment to set up behavior_specs
env.reset()

behavior_name = list(env.behavior_specs)[0]

print("The environment has observation of size: ", env.behavior_specs[behavior_name].observation_specs[0].shape[0],
    " and action of size: ", env.behavior_specs[behavior_name].action_spec.continuous_size, ".")

learning_algorithm = SoftActorCritic(
    q_net_learning_rate=3e-4, 
    policy_learning_rate=1e-3, 
    discount=0.99, 
    temperature=0.9,
    observation_size=env.behavior_specs[behavior_name].observation_specs[0].shape[0],
    action_size=env.behavior_specs[behavior_name].action_spec.continuous_size, 
    update_qnet_every_N_gradient_steps=1000,
    # leave the optimizer as the default = Adam
    )

trainer = OffPolicyBaseTrainer(env, behavior_name, learning_algorithm)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 200
# The number of experiences to collect per training step
NUM_NEW_EXP = 500
# The maximum size of the Buffer
BUFFER_SIZE = 10**6

EPSILON = 0.0

TASK_NAME = "SAC" + "_Walker"

#TODO Is there a way for me to move DISCRETIZED_NUM to ddqn? 
def no_exploration(actions : torch.tensor, epsilon : float, env : BaseEnv):
    # since exploration is inherent in SAC, we don't need epsilon to do anything
    # we however need to convert torch.tensor back to numpy arrays.
    return actions.detach().numpy()

l_a = trainer.train(
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP, 
    max_buffer_size=BUFFER_SIZE,
    exploration_function=no_exploration,
    epsilon=EPSILON,
    save_after_training=SAVE_AFTER_TRAINING,
    task_name=TASK_NAME
    )