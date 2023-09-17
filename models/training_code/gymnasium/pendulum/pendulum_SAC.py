"""
Want to see if our implementation of SAC is able to solve pendulum,
which seems to be one of the easiest continuous environemnts available
through gym.

If we can't solve this, there is an error somewhere in the code to be fixed.
"""

import gymnasium
import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer
from models.trainers.utils.gym_observation_scaling_wrapper import GymObservationScaling

# create the environment and determine specs about it
env = gymnasium.make("Pendulum-v1")#, render_mode="human")
env = GymObservationScaling(env, obs_max=1.0, obs_min=-1.0)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_ranges = tuple([(env.action_space.low[i], env.action_space.high[i]) for i in range(action_size)])

trainer = GymOffPolicyBaseTrainer(env)
TASK_NAME = "SAC" + "_" + env.spec.id

# ++++++++++++++++++++++++++++++++++++++++++++
# SETTING HYPERPARAMETERS

parameters = {
    "online_example" : { #https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.1,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1
    },
    "play_around" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.08,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1
    }
}

param = parameters["play_around"]

learning_algorithm = SoftActorCritic(
    q_net_learning_rate=param["q_net_learning_rate"], 
    policy_learning_rate=param["policy_learning_rate"], 
    discount=param["discount"], 
    temperature=param["temperature"],
    qnet_update_smoothing_coefficient=param["qnet_update_smoothing_coefficient"],
    observation_size=observation_size,
    action_size=action_size,
    action_ranges=action_ranges,
    pol_eval_batch_size=param["pol_eval_batch_size"],
    pol_imp_batch_size=param["pol_imp_batch_size"],
    update_qnet_every_N_gradient_steps=param["update_qnet_every_N_gradient_steps"]
    # leave the optimizer as the default = Adam
    )

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 10000
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = 1000
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10**4

# SETTING HYPERPARAMETERS END
# ++++++++++++++++++++++++++++++++++++++++++++


def uniform_random_sampling(actions, env):
    # initially sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    action_zero_to_one = torch.rand(size=(learning_algorithm.act_size,)).cpu()
    action_minus_one_to_one = action_zero_to_one * 2.0 - 1.0
    adjusted_actions = (action_minus_one_to_one * 
                        learning_algorithm.policy.action_multiplier.detach().cpu() + 
                        learning_algorithm.policy.action_avgs.detach().cpu())
    return adjusted_actions.numpy()

def no_exploration(actions, env):
    # since exploration is inherent in SAC, we don't need epsilon to do anything
    if type(actions) == type(torch.tensor([0])):
        return actions.numpy()
    elif type(actions) == type(np.array([0])):
        return actions
    else:
        raise Exception("Value passed as action to no_exploration was of type ", type(actions), 
                        "but should be either a torch.tensor or np.ndarray to successfully work.") 

l_a = trainer.train(
    learning_algorithm=learning_algorithm,
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP,
    max_buffer_size=BUFFER_SIZE,
    num_initial_experiences=NUM_INIT_EXP,
    evaluate_every_N_steps=NUM_TRAINING_STEPS // 20,
    initial_exploration_function=uniform_random_sampling,
    training_exploration_function=no_exploration,
    save_after_training=True,
    task_name=TASK_NAME,
    render_evaluation=True
    )