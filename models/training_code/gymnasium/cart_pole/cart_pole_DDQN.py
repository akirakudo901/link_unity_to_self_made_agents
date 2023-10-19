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
          "auto_adjust" : False,
          "l_r" : 1e-2, 
          "d_r" : 0.95, 
          "soft_update_coefficient" : 5e-4,
          "update_target_every_N_updates" : 1,
          "num_training_steps" : 5000,
          "num_init_exp" : 1000,
          "num_new_exp" : 1,
          "buffer_size" : 10000,
          "save_after_training" : True,
          "training_id" : 101,
          },
      # below is inspired from https://github.com/lsimmons2/double-dqn-cartpole-solution/blob/master/double_dqn.py
      "trial" : { 
          "init_eps" : 0.5,
          "min_eps" : 0.01,
          "eps_decay" : 0.99,
          "auto_adjust" : False,
          "l_r" : 1e-3,
          "d_r" : 0.99, 
          "soft_update_coefficient" : 0.1,
          "update_target_every_N_updates" : 1,
          "num_training_steps" : 50000,
          "num_init_exp" : 500,
          "num_new_exp" : 1,
          "buffer_size" : 100000,
          "save_after_training" : False,
          "training_id" : 102,
           }
}

# learning_rates = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # all
# learning_rates = [5e-1,   1e-3, 5e-4, 1e-4, 5e-5, 1e-5] # doesn't quite work:
# learning_rates = [1e-1, 5e-2, 1e-2, 5e-3] #works well


class EpsilonAdjustment:
      def __init__(self, init_eps, min_eps, eps_decay, auto_adjust=False):
            self.eps = init_eps
            self.min_eps = min_eps
            self.eps_decay = eps_decay

            if type(auto_adjust) == type(0):
                 self.eps_decay = (min_eps / init_eps)**(1 / (auto_adjust * 0.8))

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
      a = actions if (threshold > epsilon_adjustment.eps) else uniform_random_sampling(actions, env)
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
            dqn_layer_sizes=(64, 64),
            env=env
            )

      eps_adjust = EpsilonAdjustment(
           init_eps    = param["init_eps"],
           min_eps     = param["min_eps"], 
           eps_decay   = param["eps_decay"],
           auto_adjust = param["num_training_steps"]
           )

      def eps_explore_fn(actions, env):
            return epsilon_exploration(actions, env, eps_adjust)

      l_a = trainer.train(
            learning_algorithm=algo,
            num_training_epochs=param["num_training_steps"], 
            new_experience_per_epoch=param["num_new_exp"],
            max_buffer_size=param["buffer_size"],
            num_initial_experiences=param["num_init_exp"],
            evaluate_every_N_epochs=param["num_training_steps"] // 10,
            evaluate_N_samples=10,
            initial_exploration_function=uniform_random_sampling,
            training_exploration_function=eps_explore_fn,
            training_exploration_function_name="eps_explore_fn",
            save_after_training=param["save_after_training"],
            task_name=TASK_NAME + f"_{param['l_r']}",
            training_id=param["training_id"],
            render_evaluation=False
            )

      eps_adjust.show_epsilon_history()

      # check whether the training was successful
      # defined as getting an average of over 195.0 rewards over 100 trials
      # here: [https://github.com/openai/gym/wiki/CartPole-v0]
      avg_reward = trainer.evaluate(learning_algorithm=l_a, num_samples=100)
      if avg_reward >= 195.0: 
           print(f"Cartpole-v1 solved; average reward over 100 trials: {avg_reward}!")
      else:
           print(f"Cartpole-v1 not solved... average reward over 100 trials: {avg_reward}!")

      return l_a


train_DDQN_on_cartPole(parameter_name="trial")