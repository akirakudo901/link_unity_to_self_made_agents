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

trainer = GymOffPolicyBaseTrainer(env)

TASK_NAME = "DDQN" + "_" + env.spec.id

parameters = {
     "best_with_1e2" : { 
          "init_eps" : 1.0, 
          "min_eps" : 0.05, 
          "eps_decay" : 0.9996,
          "l_r" : 1e-2, 
          "d_r" : 0.95, 
          "soft_update_coefficient" : 5e-4,
          "update_target_every_N_updates" : 1,
          "num_training_steps" : 10000,
          "num_init_exp" : 1000,
          "num_new_exp" : 1,
          "buffer_size" : 10000,
          "save_after_training" : True
          },
      # below is inspired from https://github.com/lsimmons2/double-dqn-cartpole-solution/blob/master/double_dqn.py
      "trial" : { 
           "init_eps" : 0.5, 
           "min_eps" : 0.01, 
           "eps_decay" : 0.99,
           "l_r" : 1e-3, 
           "d_r" : 0.99, 
           "soft_update_coefficient" : 0.1,
           "update_target_every_N_updates" : 1,
           "num_training_steps" : 10000,
           "num_init_exp" : 1000,
           "num_new_exp" : 1,
           "buffer_size" : 10000,
           "save_after_training" : True
           },
      # short training to see that code executes correctly
      "for_testing" : {
          "init_eps" : 1.0, 
          "min_eps" : 0.05, 
          "eps_decay" : 0.9996,
          "l_r" : 1e-2, 
          "d_r" : 0.95, 
          "soft_update_coefficient" : 5e-4,
          "update_target_every_N_updates" : 1,
          "num_training_steps" : 20,
          "num_init_exp" : 10,
          "num_new_exp" : 1,
          "buffer_size" : 10000,
          "save_after_training" : False
      }
}

# learning_rates = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # all
# learning_rates = [5e-1,   1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # doesn't quite work:
# learning_rates = [1e-1, 5e-2, 1e-2, 5e-3] #works well


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


def train_DDQN_on_cartPole(parameter_name : str):

      print(f"Training: {parameter_name}")

      param = parameters[parameter_name]

      algo = DoubleDeepQNetwork(
            l_r=param["l_r"],
            d_r=param["d_r"],
            soft_update_coefficient=param["soft_update_coefficient"],
            update_target_every_N_updates=param["update_target_every_N_updates"],
            env=env
            )

      eps_adjust = EpsilonAdjustment(init_eps  = param["init_eps"],
                                     min_eps   = param["min_eps"], 
                                     eps_decay = param["eps_decay"])

      def eps_explore_fn(actions, env):
            return epsilon_exploration(actions, env, eps_adjust)

      l_a = trainer.train(
            learning_algorithm=algo,
            num_training_steps=param["num_training_steps"], 
            num_new_experience=param["num_new_exp"],
            max_buffer_size=param["buffer_size"],
            num_initial_experiences=param["num_init_exp"],
            evaluate_every_N_steps=param["num_training_steps"] // 20,
            initial_exploration_function=uniform_random_sampling,
            training_exploration_function=eps_explore_fn,
            save_after_training=param["save_after_training"],
            task_name=TASK_NAME + f"_{param['l_r']}",
            render_evaluation=False
            )

      eps_adjust.show_epsilon_history()
      return l_a


train_DDQN_on_cartPole(parameter_name="for_testing")