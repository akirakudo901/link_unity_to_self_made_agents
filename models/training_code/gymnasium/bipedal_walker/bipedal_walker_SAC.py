"""
Implementation of SAC training with the bipedal walker environment with gym.
"""

import gymnasium

from models.policy_learning_algorithms.policy_learning_algorithm import generate_name_from_parameter_dict, generate_parameters
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, uniform_random_sampling_wrapper, no_exploration_wrapper, train_SAC_on_gym
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("BipedalWalker-v3")#, render_mode="human")
trainer = GymOffPolicyBaseTrainer(env)
MAX_EPISODE_STEPS = 1600

parameters = {
    "long_run_trial" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.5,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 200,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },
    "try_256by256" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.5,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 20,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },
    "policy_learning_rate_1e-3_without_clipping" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.5,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 100,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },
    "long_trial_without_clipping" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.5,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 500,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },
    "smaller_qnet_lower_learning_rate" : {
        "q_net_learning_rate"  : 1e-4,
        "policy_learning_rate" : 1e-4,
        "discount" : 0.99,
        "temperature" : 10,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (64, 64),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 50,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },    
    "higher_learning_rate" : {
        "q_net_learning_rate"  : 1e-2,
        "policy_learning_rate" : 1e-2,
        "discount" : 0.99,
        "temperature" : 0.5,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 50,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "evaluate_N_samples" : 1,
        "render_evaluation" : False
    },

}

# JUST TRAIN ONE PARAMETER
train_SAC_on_gym(parameters=parameters["higher_learning_rate"], parameter_name="higher_learning_rate", 
                 env=env, trainer=trainer, training_id=None)


# TRAIN MANY COMBINATIONS
# params_to_try = generate_parameters(default_parameters=parameters["smaller_qnet_lower_learning_rate"],
#                                     default_name = "default",
#                                     q_net_learning_rate = [1e-3, 5e-4],
#                                     policy_learning_rate = [5e-4, 1e-4, 5e-5, 1e-5],
#                                     temperature = [5, 10],
#                                     num_training_steps = MAX_EPISODE_STEPS * 20)

# for i, name_and_dict in enumerate(params_to_try.items()):
#     name, p = name_and_dict
#     name = name[:60]
#     train_SAC_on_gym(parameters=p, parameter_name=name, 
#                      env=env, trainer=trainer, training_id=None)