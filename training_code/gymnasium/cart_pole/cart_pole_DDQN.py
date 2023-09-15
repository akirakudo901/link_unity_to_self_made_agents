
"""
Implementation of DDQN training with the simple cartpole environment. 
I know that this setup should work out, since I have already trained
a model succcessfully - if this doesn't work, the trainer code has a 
problem.
"""
import random

import gymnasium
import numpy as np
import torch

from policy_learning_algorithms.double_deep_q_network import DoubleDeepQNetwork
from trainers.gym_base_trainer import GymOffPolicyBaseTrainer

SAVE_AFTER_TRAINING = True

# set up a device first
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# create the environment and determine specs about it
env = gymnasium.make("CartPole-v1")
observation_size, action_size = env.observation_space.shape[0], 1

print(f"The environment has observation of size: {observation_size} and " + 
      f"action of size: {action_size}.")

learning_algorithm = DoubleDeepQNetwork(
    observation_size=observation_size, 
    action_size=action_size, 
    device=device, 
    l_r=0.1,
    d_r=0.95
    )

trainer = GymOffPolicyBaseTrainer(env, learning_algorithm)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 2500
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = 2500 // 6
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10**4

TASK_NAME = "DDQN" + "_" + env.spec.id

def uniform_random_sampling(actions, env):
    # initially sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    a = np.array(1) if random.random() >= 0.5 else np.array(0)
    return a

class EpsilonAdjustment:
      def __init__(self, init_eps=1.0, min_eps=0.05, eps_decay=0.999):
            self.eps = init_eps
            self.min_eps = min_eps
            self.eps_decay = eps_decay
      
      def adjust_per_loop(self):
            if self.eps * self.eps_decay > self.min_eps:
                  self.eps *= self.eps_decay

eps_adjust = EpsilonAdjustment(init_eps=1.0, min_eps=0.05, eps_decay=0.999)

def epsilon_exploration(actions, env):
      threshold = random.random()
      a = actions if (threshold > eps_adjust.eps) else uniform_random_sampling(actions, env)
      eps_adjust.adjust_per_loop()
      return a

l_a = trainer.train(
    num_training_steps=NUM_TRAINING_STEPS, 
    num_new_experience=NUM_NEW_EXP,
    max_buffer_size=BUFFER_SIZE,
    num_initial_experiences=NUM_INIT_EXP,
    evaluate_every_N_steps=NUM_TRAINING_STEPS // 5,
    initial_exploration_function=uniform_random_sampling,
    training_exploration_function=epsilon_exploration,
    save_after_training=SAVE_AFTER_TRAINING,
    task_name=TASK_NAME,
    render_evaluation=False
    )