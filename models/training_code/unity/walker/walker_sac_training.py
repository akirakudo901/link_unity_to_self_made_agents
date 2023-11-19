"""
An integration of the Walker environment with SAC.
"""

from mlagents_envs.environment import BaseEnv
from mlagents_envs.registry import default_registry
import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, no_exploration_wrapper
from models.trainers.unityenv_base_trainer import UnityOffPolicyBaseTrainer

parameters = {
    # "batch_size_256" : {
    #     "q_net_learning_rate"  : 1e-3,
    #     "policy_learning_rate" : 1e-3,
    #     "discount" : 0.99,
    #     "temperature" : 0.1,
    #     "qnet_update_smoothing_coefficient" : 0.005,
    #     "pol_eval_batch_size" : 256,
    #     "pol_imp_batch_size" : 256,
    #     "update_qnet_every_N_gradient_steps" : 1,
    #     "num_training_steps" : 50000,
    #     "num_init_exp" : 3000,
    #     "num_new_exp" : 1,
    #     "buffer_size" : 10000,
    #     "save_after_training" : True
    # },
    # "batch_size_1024" : { #doesn't quite work out!
    #     "q_net_learning_rate"  : 1e-3,
    #     "policy_learning_rate" : 1e-3,
    #     "discount" : 0.99,
    #     "temperature" : 0.1,
    #     "qnet_update_smoothing_coefficient" : 0.005,
    #     "pol_eval_batch_size" : 1024,
    #     "pol_imp_batch_size" : 1024,
    #     "update_qnet_every_N_gradient_steps" : 1,
    #     "num_training_steps" : 50000,
    #     "num_init_exp" : 3000,
    #     "num_new_exp" : 1,
    #     "buffer_size" : 10000,
    #     "save_after_training" : True
    # },
    "batch_64_5e3" : {
        "q_net_learning_rate"  : 5e-3,
        "policy_learning_rate" : 5e-3,
        "discount" : 0.99,
        "temperature" : 0.1,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "num_training_steps" : 1000,
        "num_init_exp" : 3000,
        "num_new_exp" : 1,
        "evaluate_N_samples" : 1,
        "evaluate_every_N_epochs" : 50000//20,
        "buffer_size" : 10000,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "save_after_training" : True,
        "render_evaluation" : False,
        "training_id" : 0
        },
    }


def train_SAC_on_walker(parameter_name : str):

    ENV_NAME = "Walker"
    env = default_registry[ENV_NAME].make()

    # reset the environment to set up behavior_specs
    env.reset()
    behavior_name = list(env.behavior_specs)[0]        
    observation_size = env.behavior_specs[behavior_name].observation_specs[0].shape[0]
    action_size = env.behavior_specs[behavior_name].action_spec.continuous_size

    trainer = UnityOffPolicyBaseTrainer(env, ENV_NAME, behavior_name)

    print(f"Training: {parameter_name}")

    param = parameters[parameter_name]

    learning_algorithm = SoftActorCritic(
        q_net_learning_rate=param["q_net_learning_rate"], 
        policy_learning_rate=param["policy_learning_rate"], 
        discount=param["discount"], 
        temperature=param["temperature"],
        obs_dim_size=observation_size,
        act_dim_size=action_size, 
        act_ranges=(((-1., 1.),)*action_size),
        qnet_update_smoothing_coefficient=param["qnet_update_smoothing_coefficient"],
        pol_eval_batch_size=param["pol_eval_batch_size"],
        pol_imp_batch_size=param["pol_imp_batch_size"],
        update_qnet_every_N_gradient_steps=param["update_qnet_every_N_gradient_steps"],
        qnet_layer_sizes=param["qnet_layer_sizes"],
        policy_layer_sizes=param["policy_layer_sizes"],
        # leave the optimizer as the default = Adam
        )

    l_a = trainer.train(
        learning_algorithm=learning_algorithm,
        num_training_epochs=param["num_training_steps"], 
        new_experience_per_epoch=param["num_new_exp"],
        max_buffer_size=param["buffer_size"],
        num_initial_experiences=param["num_init_exp"],
        evaluate_every_N_epochs=param["evaluate_every_N_epochs"],
        evaluate_N_samples=param["evaluate_N_samples"],
        initial_exploration_function=no_exploration_wrapper(learning_algorithm),
        training_exploration_function=no_exploration_wrapper(learning_algorithm),
        save_after_training=param["save_after_training"],
        task_name=f"{learning_algorithm.ALGORITHM_NAME}_{ENV_NAME}_{parameter_name}",
        training_exploration_function_name="no_exploration",
        training_id=param["training_id"]
        )

    return l_a

train_SAC_on_walker(parameter_name="batch_64_5e3")