"""
Implementation of SAC training with the continuous version of the mountain car
environment with gym. 
Initially thought to check if the implementation runs or not if in a simple environment.
Then figured out this is a quite complicated environment where the reward is shaped such 
that only a successful exploration is followed by learning to get up the hill (and exploration
helped by SAC's entropy term might not be strong enough to reach such success before the 
gas cost of outputting high speed takes over.)
"""

import gymnasium
import numpy as np
import torch

from policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from trainers.gym_base_trainer import GymOffPolicyBaseTrainer

SAVE_AFTER_TRAINING = True

# set up a device first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Using device:', device)
print()

cpu_device = torch.device("cpu")

# create the environment and determine specs about it
env = gymnasium.make("MountainCarContinuous-v0")#, render_mode="human")
observation_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
action_ranges = tuple([(env.action_space.low[i], env.action_space.high[i]) for i in range(action_size)])

print(f"The environment has observation of size: {observation_size} and \
      action of size: {action_size} with action ranges of: {action_ranges}.")

learning_algorithm = SoftActorCritic(
    q_net_learning_rate=3e-4, 
    policy_learning_rate=1e-3, 
    discount=0.99, 
    temperature=0.41,
    observation_size=observation_size,
    action_size=action_size, 
    action_ranges=action_ranges,
    pol_eval_batch_size=256, 
    pol_imp_batch_size=256, 
    update_qnet_every_N_gradient_steps=1000,
    device=device
    # leave the optimizer as the default = Adam
    )

trainer = GymOffPolicyBaseTrainer(env, learning_algorithm)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 300
# The number of experieces to be initlally collected before doing any training
NUM_INIT_EXP = 1028
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10**4

TASK_NAME = "SAC" + "_Mountain_Car_Continuous"

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
    # we however need to convert torch.tensor back to numpy arrays.
    if type(actions) == type(torch.tensor([0])):
        return actions.detach().to(cpu_device).numpy()
    elif type(actions) == type(np.array([0])):
        return actions
    else:
        raise Exception("Value passed as action to no_exploration was of type ", type(actions), 
                        "but should be either a torch.tensor or np.ndarray to successfully work.") 

l_a = trainer.train(
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP,
    max_buffer_size=BUFFER_SIZE,
    num_initial_experiences=NUM_INIT_EXP,
    evaluate_every_N_steps=NUM_TRAINING_STEPS // 5,
    initial_exploration_function=uniform_random_sampling,
    training_exploration_function=no_exploration,
    save_after_training=SAVE_AFTER_TRAINING,
    task_name=TASK_NAME,
    render_evaluation=False
    )