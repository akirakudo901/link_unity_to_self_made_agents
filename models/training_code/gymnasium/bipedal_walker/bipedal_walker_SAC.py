"""
Implementation of SAC training with the bipedal walker environment with gym.
"""

import gymnasium
import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("BipedalWalker-v3")#, render_mode="human")
observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_ranges = tuple([(env.action_space.low[i], env.action_space.high[i]) for i in range(action_size)])

learning_algorithm = SoftActorCritic(
    q_net_learning_rate=1e-3, 
    policy_learning_rate=1e-3, 
    discount=0.99, 
    temperature=0.15,
    qnet_update_smoothing_coefficient=0.005,
    obs_dim_size=observation_size,
    act_dim_size=action_size, 
    act_ranges=action_ranges,
    pol_eval_batch_size=64,
    pol_imp_batch_size=64,
    update_qnet_every_N_gradient_steps=1
    # leave the optimizer as the default = Adam
    )

trainer = GymOffPolicyBaseTrainer(env)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 100000
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = 3000
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10**5

TASK_NAME = "SAC" + "_BipedalWalker"

def uniform_random_sampling(actions, env):
    # initially sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    action_zero_to_one = torch.rand(size=(learning_algorithm.act_dim_size,)).cpu()
    action_minus_one_to_one = action_zero_to_one * 2.0 - 1.0
    adjusted_actions = (action_minus_one_to_one * 
                        learning_algorithm.policy.action_multiplier.detach().cpu() + 
                        learning_algorithm.policy.action_avgs.detach().cpu())
    return adjusted_actions.numpy()

def no_exploration(actions, env):
    # since exploration is inherent in SAC, we don't need epsilon to do anything
    # we however need to convert torch.tensor back to numpy arrays.
    if type(actions) == type(torch.tensor([0])):
        return actions.detach().to(torch.device("cpu")).numpy()
    elif type(actions) == type(np.array([0])):
        return actions
    else:
        raise Exception("Value passed as action to no_exploration was of type ", type(actions), 
                        "but should be either a torch.tensor or np.ndarray to successfully work.") 

l_a = trainer.train(
    learning_algorithm=learning_algorithm,
    num_training_epochs=NUM_TRAINING_STEPS, 
    new_experience_per_epoch=NUM_NEW_EXP,
    max_buffer_size=BUFFER_SIZE,
    num_initial_experiences=NUM_INIT_EXP,
    evaluate_every_N_epochs=NUM_TRAINING_STEPS // 10,
    evaluate_N_samples=3,
    initial_exploration_function=uniform_random_sampling,
    training_exploration_function=no_exploration,
    save_after_training=True,
    task_name=TASK_NAME,
    render_evaluation=True
    )