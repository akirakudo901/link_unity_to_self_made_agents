"""
Want to see if our implementation of SAC is able to solve pendulum,
which seems to be one of the easiest continuous environemnts available
through gym.

If we can't solve this, there is an error somewhere in the code to be fixed.
"""

import gymnasium

from models.policy_learning_algorithms.policy_learning_algorithm import no_exploration
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, uniform_random_sampling_wrapper
from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer

# create the environment and determine specs about it
env = gymnasium.make("Pendulum-v1")#, render_mode="human")
trainer = GymOffPolicyBaseTrainer(env)
MAX_EPISODE_STEPS = 200

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
        "num_training_steps" : MAX_EPISODE_STEPS * 100,
        "num_init_exp" : 0,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : 200,
        "buffer_size" : int(1e6),
        "save_after_training" : False,
        "training_id" : 7
    }
}


def train_SAC_on_pendulum(parameter_name : str):

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
    
    TASK_NAME = learning_algorithm.ALGORITHM_NAME + "_" + env.spec.id

    l_a = trainer.train(
        learning_algorithm=learning_algorithm,
        num_training_epochs=param["num_training_steps"], 
        new_experience_per_epoch=param["num_new_exp"],
        max_buffer_size=param["buffer_size"],
        num_initial_experiences=param["num_init_exp"],
        evaluate_every_N_epochs=param["evaluate_every_N_epochs"],
        evaluate_N_samples=1,
        initial_exploration_function=uniform_random_sampling_wrapper(learning_algorithm),
        training_exploration_function=no_exploration,
        training_exploration_function_name="no_exploration",
        save_after_training=param["save_after_training"],
        task_name=TASK_NAME + parameter_name + str(param["temperature"]),
        training_id=param["training_id"],
        render_evaluation=False
        )

    return l_a

# temps = [r*0.05 for r in [3,5,8,2]]
# print(temps)
# for temp in temps:
#     print(f"Working on {temp}!")
#     parameters["play_around"]["temperature"] = temp
#     train_SAC_on_pendulum(parameter_name="play_around")

train_SAC_on_pendulum(parameter_name="online_example")