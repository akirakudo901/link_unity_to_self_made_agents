
"""
Implementation of DDQN training with the simple cartpole environment. 
I know that this setup should work out, since I have already trained
a model succcessfully - if this doesn't work, the trainer code has a 
problem.
"""
import random

import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from policy_learning_algorithms.double_deep_q_network import DoubleDeepQNetwork
from trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("CartPole-v1")
observation_size = 1 if (env.observation_space.shape == ()) else env.observation_space.shape[0]

learning_rates = [1e-1, 5e-2, 1e-2]

algorithms = {
     str(l_r) : DoubleDeepQNetwork(
           observation_dim_size=observation_size, 
           discrete_action_size=2,
           l_r=l_r,
           d_r=0.95
           ) for l_r in learning_rates
}

trainer = GymOffPolicyBaseTrainer(env)

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 20000
EVALUATE_EVERY_N_STEPS = NUM_TRAINING_STEPS // 20
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = NUM_TRAINING_STEPS // 6
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
      def __init__(self, init_eps=1.0, min_eps=0.05, eps_decay=0.9999):
            self.eps = init_eps
            self.min_eps = min_eps
            self.eps_decay = eps_decay

            self.eps_history = [init_eps]
      
      def adjust_per_loop(self):
            if self.eps * self.eps_decay > self.min_eps:
                  self.eps *= self.eps_decay
            self.eps_history.append(self.eps)
      
      def show_epsilon_history(self):
            try:
                plt.clf()
                plt.title(f"{TASK_NAME} Epsilon Over Time")
                plt.plot(range(0, len(self.eps_history)), self.eps_history)
                plt.savefig(f"{TASK_NAME}_epsilon_over_time.png")
                plt.show()
            except ValueError:
                print("\nEpsilon plotting failed.")
            
eps_adjust = EpsilonAdjustment(init_eps=1.0, min_eps=0.05, eps_decay=0.999)

def epsilon_exploration(actions, env):
      threshold = random.random()
      a = np.argmax(actions) if (threshold > eps_adjust.eps) else uniform_random_sampling(actions, env)
      eps_adjust.adjust_per_loop()
      return a

for l_r in algorithms.keys():
      
      algo = algorithms[l_r]

      l_a = trainer.train(
            learning_algorithm=algo,
            num_training_steps=NUM_TRAINING_STEPS, 
            num_new_experience=NUM_NEW_EXP,
            max_buffer_size=BUFFER_SIZE,
            num_initial_experiences=NUM_INIT_EXP,
            evaluate_every_N_steps=EVALUATE_EVERY_N_STEPS,
            initial_exploration_function=uniform_random_sampling,
            training_exploration_function=epsilon_exploration,
            save_after_training=True,
            task_name=TASK_NAME + f"_{l_r}",
            render_evaluation=False
            )

eps_adjust.show_epsilon_history()