
"""
Implementation of SAC training with the continuous version of the mountain car
environment with gym. 
Was meant to be used to check if the implementation runs or not if in a simple environment.
Turns out this environment itself is quite tricky with a need for successful exploration
for learning, which my SAC wasn't quite able to achieve.
"""

import gymnasium

from models.policy_learning_algorithms.policy_learning_algorithm import generate_parameters
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, no_exploration_wrapper, train_SAC_on_gym
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("MountainCarContinuous-v0")#, render_mode="human")
trainer = GymOffPolicyBaseTrainer(env)
MAX_EPISODE_STEPS = 1000

parameters = {
    "online_example" : { #https://github.com/zhihanyang2022/pytorch-sac/blob/main/params_pool.py
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.10,
        "qnet_update_smoothing_coefficient" : 0.005,

        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        
        "num_training_steps" : MAX_EPISODE_STEPS * 50,
        "num_init_exp" : 0,
        "num_new_exp" : 1,
        "evaluate_N_samples" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),

        "save_after_training" : True,
        "render_evaluation" : False,

        "qnet_layer_sizes" : (64, 64),
        "policy_layer_sizes" : (64, 64)
    }
}


train_SAC_on_gym(parameters=parameters["online_example"], parameter_name="q_net_0.0005_polic_0.0005_tempe_0.5",
          env=env, trainer=trainer,
          training_id="mgtfshw5")

# params_to_try = generate_parameters(default_parameters=parameters["online_example"],
#                                     default_name = "default",
#                                     q_net_learning_rate = [1e-3, 5e-4],
#                                     policy_learning_rate = [1e-3, 5e-4],
#                                     temperature = [0.1, 0.25, 0.5])

# for i, name_and_dict in enumerate(params_to_try.items()):
#     name, p = name_and_dict
#     train_SAC_on_gym(parameters=p, parameter_name=name, 
#                      env=env, trainer=trainer, training_id=None)