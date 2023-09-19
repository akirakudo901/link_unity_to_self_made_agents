"""
An integration of the Walker environment with SAC.
"""

from mlagents_envs.environment import BaseEnv, UnityEnvironment
from mlagents_envs.registry import default_registry
import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from models.trainers.unityenv_base_trainer import OffPolicyBaseTrainer

SAVE_AFTER_TRAINING = True

# set up a device first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Using device:', device)
print()

cpu_device = torch.device("cpu")

print("About to create the environment! Press the play button to initiate training!")
env = UnityEnvironment(file_name=None, seed=1, side_channels=[])
# env = default_registry["Walker"].make()
print("Successfully created the environment!")

# reset the environment to set up behavior_specs
env.reset()

behavior_name = list(env.behavior_specs)[0]
observation_size = env.behavior_specs[behavior_name].observation_specs[0].shape[0]
action_size = env.behavior_specs[behavior_name].action_spec.continuous_size

print("The environment has observation of size: ", observation_size,
      " and action of size: ", action_size, ".")

learning_algorithm = SoftActorCritic(
    q_net_learning_rate=3e-4, 
    policy_learning_rate=1e-3, 
    discount=0.99, 
    temperature=0.2,
    qnet_update_smoothing_coefficient=0.005,
    obs_dim_size=observation_size,
    act_dim_size=action_size, 
    act_ranges=(((-1., 1.),)*action_size),
    pol_eval_batch_size=1028,
    pol_imp_batch_size=1028,
    update_qnet_every_N_gradient_steps=1000
    # leave the optimizer as the default = Adam
    )

trainer = OffPolicyBaseTrainer(env, behavior_name)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 10000
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = 1000
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10**4

TASK_NAME = "SAC" + "_Walker"

def uniform_random_sampling(actions, env):
    # initially sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    num_agents = actions.shape[0]
    action_zero_to_one = torch.rand(size=(num_agents, learning_algorithm.act_dim_size,)).cpu()
    action_minus_one_to_one = action_zero_to_one * 2.0 - 1.0
    adjusted_actions = (action_minus_one_to_one * 
                        learning_algorithm.policy.action_multiplier.detach().cpu() + 
                        learning_algorithm.policy.action_avgs.detach().cpu())
    return adjusted_actions.numpy()

def no_exploration(actions : np.ndarray, env : BaseEnv):
    # since exploration is inherent in SAC, we don't need epsilon to do anything
    # SAC also returns an np.ndarray upon call
    return actions

l_a = trainer.train(
    learning_algorithm=learning_algorithm,
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP, 
    max_buffer_size=BUFFER_SIZE,
    num_initial_experiences=NUM_INIT_EXP,
    evaluate_every_N_steps=NUM_TRAINING_STEPS // 10,
    initial_exploration_function=uniform_random_sampling,
    training_exploration_function=no_exploration,
    save_after_training=SAVE_AFTER_TRAINING,
    task_name=TASK_NAME
    )