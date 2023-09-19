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

trainer = GymOffPolicyBaseTrainer(env)

parameters = {
    "online_example" : { #https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.1,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "num_training_steps" : 10000,
        "num_init_exp" : 1000,
        "num_new_exp" : 1,
        "buffer_size" : 10000,
        "save_after_training" : True
    },
    "play_around" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.08,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "num_training_steps" : 10000,
        "num_init_exp" : 1000,
        "num_new_exp" : 1,
        "buffer_size" : 10000,
        "save_after_training" : True
    },
    # short training to play around
    "for_testing" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.08,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "num_training_steps" : 20,
        "num_init_exp" : 10,
        "num_new_exp" : 1,
        "buffer_size" : 10000,
        "save_after_training" : False
    }
}


def train_SAC_on_pendulum(parameter_name : str):
      
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
            if type(actions) == type(torch.tensor([0])):
                return actions.numpy()
            elif type(actions) == type(np.array([0])):
                return actions
            else:
                raise Exception("Value passed as action to no_exploration was of type ", type(actions), 
                                "but should be either a torch.tensor or np.ndarray to successfully work.") 

        TASK_NAME = learning_algorithm.ALGORITHM_NAME + "_" + env.spec.id

        print(f"Training: {parameter_name}")

        param = parameters[parameter_name]

        learning_algorithm = SoftActorCritic(
            q_net_learning_rate=param["q_net_learning_rate"], 
            policy_learning_rate=param["policy_learning_rate"], 
            discount=param["discount"], 
            temperature=param["temperature"],
            qnet_update_smoothing_coefficient=param["qnet_update_smoothing_coefficient"],
            pol_eval_batch_size=param["pol_eval_batch_size"],
            pol_imp_batch_size=param["pol_imp_batch_size"],
            update_qnet_every_N_gradient_steps=param["update_qnet_every_N_gradient_steps"],
            env=env
            # leave the optimizer as the default = Adam
            )

        l_a = trainer.train(
            learning_algorithm=learning_algorithm,
            num_training_steps=param["num_training_steps"], 
            num_new_experience=param["num_new_exp"],
            max_buffer_size=param["buffer_size"],
            num_initial_experiences=param["num_init_exp"],
            evaluate_every_N_steps=param["num_training_steps"] // 20,
            initial_exploration_function=uniform_random_sampling,
            training_exploration_function=no_exploration,
            save_after_training=param["save_after_training"],
            task_name=TASK_NAME,
            render_evaluation=False
            )

        return l_a


train_SAC_on_pendulum(parameter_name="play_around")