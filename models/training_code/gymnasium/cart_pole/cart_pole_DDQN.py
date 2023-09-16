"""
Implementation of DDQN training with the simple cartpole environment. 
WORKS OUT QUITE WELL, GOT IT SOLVED! SO THE TRAINER CODE MIGHT NOT HAVE ANY ISSUES.
"""
import random

import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from models.policy_learning_algorithms.double_deep_q_network import DoubleDeepQNetwork
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("CartPole-v1")
observation_size = 1 if (env.observation_space.shape == ()) else env.observation_space.shape[0]

trainer = GymOffPolicyBaseTrainer(env)

TASK_NAME = "DDQN" + "_" + env.spec.id


# ++++++++++++++++++++++++++
# SET HYPERPARAMETERS

parameters = {
     "best_with_1e-2" : { "init_eps" : 1.0, "min_eps" : 0.05, "eps_decay" : 0.9996,
                          "l_r" : 1e-2, "d_r" : 0.95, 
                          "hard_update_every_N_updates" : None,
                           "soft_update_coefficient" : 5e-4 },
      # below is inspired from https://github.com/lsimmons2/double-dqn-cartpole-solution/blob/master/double_dqn.py
      "trial" : { "init_eps" : 0.5, "min_eps" : 0.01, "eps_decay" : 0.99,
                  "l_r" : 1e-3, "d_r" : 0.99, 
                  "hard_update_every_N_updates" : None,
                  "soft_update_coefficient" : 0.1}
}

# learning_rates = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # all
# learning_rates = [5e-1,   1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # doesn't quite work:
# learning_rates = [1e-1, 5e-2, 1e-2, 5e-3] #works well

# The number of training steps that will be performed
NUM_TRAINING_STEPS = 20000
EVALUATE_EVERY_N_STEPS = 500
# The number of experiences to be initlally collected before doing any training
NUM_INIT_EXP = 5000
# The number of experiences to collect per training step
NUM_NEW_EXP = 1
# The maximum size of the Buffer
BUFFER_SIZE = 10000

# SET HYPERPARAMETERS END
# ++++++++++++++++++++++++++


class EpsilonAdjustment:
      def __init__(self, init_eps, min_eps, eps_decay):
            self.eps = init_eps
            self.min_eps = min_eps
            self.eps_decay = eps_decay

            self.eps_history = [init_eps]
      
      def adjust_per_loop(self):
            self.eps = max(self.min_eps, self.eps * self.eps_decay)
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

def uniform_random_sampling(actions, env):
    # initially sample actions from a uniform random distribution of the right
    # range, in order to extract good reward signals
    a = np.array(1) if random.random() >= 0.5 else np.array(0)
    return a


def epsilon_exploration(actions, env, epsilon_adjustment):
      threshold = random.random()
      a = np.argmax(actions) if (threshold > epsilon_adjustment.eps) else uniform_random_sampling(actions, env)
      epsilon_adjustment.adjust_per_loop()
      return a

            
for i in range(3):
      param_name = list(parameters.keys())[0]

      print(f"Training: {param_name}")
      
      hyperparam = parameters[param_name]
      
      algo = DoubleDeepQNetwork(
           observation_dim_size=observation_size, 
           discrete_action_size=2,
           l_r=hyperparam["l_r"],
           d_r=hyperparam["d_r"],
           hard_update_every_N_updates=hyperparam["hard_update_every_N_updates"],
           soft_update_coefficient=hyperparam["soft_update_coefficient"]
           )

      eps_adjust = EpsilonAdjustment(init_eps  = hyperparam["init_eps"], 
                                     min_eps   = hyperparam["min_eps"], 
                                     eps_decay = hyperparam["eps_decay"])

      def eps_explore_fn(actions, env):
           return epsilon_exploration(actions, env, eps_adjust)

      l_a = trainer.train(
            learning_algorithm=algo,
            num_training_steps=NUM_TRAINING_STEPS, 
            num_new_experience=NUM_NEW_EXP,
            max_buffer_size=BUFFER_SIZE,
            num_initial_experiences=NUM_INIT_EXP,
            evaluate_every_N_steps=EVALUATE_EVERY_N_STEPS,
            initial_exploration_function=uniform_random_sampling,
            training_exploration_function=eps_explore_fn,
            save_after_training=True,
            task_name=TASK_NAME + f"_{hyperparam['l_r']}_trial_{i+1}",
            render_evaluation=False
            )
      
      eps_adjust.show_epsilon_history()