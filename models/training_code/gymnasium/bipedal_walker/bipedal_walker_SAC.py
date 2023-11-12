"""
Implementation of SAC training with the bipedal walker environment with gym.
"""

import gymnasium

from models.policy_learning_algorithms.policy_learning_algorithm import generate_name_from_parameter_dict, generate_parameters
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, uniform_random_sampling_wrapper, no_exploration_wrapper
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("BipedalWalker-v3")#, render_mode="human")
trainer = GymOffPolicyBaseTrainer(env)
MAX_EPISODE_STEPS = 1600

parameters = {
    "play_around" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.10,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (64, 64),
        "policy_layer_sizes" : (64, 64),
        "num_training_steps" : MAX_EPISODE_STEPS * 10,
        "num_init_exp" : 0,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True
    },
    "try_exploration_from_vanilla" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.75,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS,
        "num_init_exp" : 2000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : False
    },
        "try_256by256" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.75,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (256, 256),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 10,
        "num_init_exp" : 5000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True
    },
    "try_big_policy" : {
        "q_net_learning_rate"  : 1e-3,
        "policy_learning_rate" : 1e-3,
        "discount" : 0.99,
        "temperature" : 0.50,
        "qnet_update_smoothing_coefficient" : 0.005,
        "pol_eval_batch_size" : 1024,
        "pol_imp_batch_size" : 1024,
        "update_qnet_every_N_gradient_steps" : 1,
        "qnet_layer_sizes" : (64, 64),
        "policy_layer_sizes" : (256, 256),
        "num_training_steps" : MAX_EPISODE_STEPS * 100 + 1000,
        "num_init_exp" : 10000,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True
    },
}
        
def train_SAC_on_bipedal_walker(parameter_name : str, params_dict = parameters):

    print(f"Training: {parameter_name}")

    param = params_dict[parameter_name]

    learning_algorithm = SoftActorCritic(
        q_net_learning_rate=param["q_net_learning_rate"], 
        policy_learning_rate=param["policy_learning_rate"], 
        discount=param["discount"], 
        temperature=param["temperature"],
        qnet_update_smoothing_coefficient=param["qnet_update_smoothing_coefficient"],
        pol_eval_batch_size=param["pol_eval_batch_size"],
        pol_imp_batch_size=param["pol_imp_batch_size"],
        update_qnet_every_N_gradient_steps=param["update_qnet_every_N_gradient_steps"],
        qnet_layer_sizes=param["qnet_layer_sizes"],
        policy_layer_sizes=param["policy_layer_sizes"],
        env=env
        # leave the optimizer as the default = Adam
        )
    
    TASK_NAME = learning_algorithm.ALGORITHM_NAME + "_" + env.spec.id

    l_a = trainer.train(
        learning_algorithm=learning_algorithm,
        num_training_epochs=param["num_training_steps"],
        new_experience_per_epoch=param["num_new_exp"],
        max_buffer_size=param["buffer_size"],
        num_initial_experiences=param["num_init_exp"],
        evaluate_every_N_epochs=param["evaluate_every_N_epochs"],
        evaluate_N_samples=1,
        initial_exploration_function=no_exploration_wrapper(learning_algorithm),# uniform_random_sampling_wrapper(learning_algorithm),
        training_exploration_function=no_exploration_wrapper(learning_algorithm),
        training_exploration_function_name="no_exploration",
        save_after_training=param["save_after_training"],
        task_name=TASK_NAME + parameter_name,
        training_id=param["training_id"],
        render_evaluation=False
        )

    return l_a

# TRAIN MANY COMBINATIONS
params_to_try = generate_parameters(default_parameters=parameters["try_256by256"],
                                    default_name = "default",
                                    policy_learning_rate = [1e-2, 5e-3, 5e-4, 1e-4, 5e-5],
                                    num_training_steps = MAX_EPISODE_STEPS * 20,
                                    pol_eval_batch_size = 1024,
                                    num_init_exp = 10000,
                                    temperature = 0.5,
                                    qnet_layer_sizes = (256, 256),
                                    policy_layer_sizes = (256, 256))

for i, name_and_dict in enumerate(params_to_try.items()):
    name, p = name_and_dict
    train_SAC_on_bipedal_walker(name, params_dict=params_to_try)

# JUST TRAIN ONE PARAMETER
# train_SAC_on_bipedal_walker(parameter_name="try_big_policy", params_dict=parameters)