"""
Implementation of SAC training with the bipedal walker environment with gym.
"""

import gymnasium
from copy import deepcopy

from models.policy_learning_algorithms.policy_learning_algorithm import no_exploration
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, uniform_random_sampling_wrapper
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
        "pol_eval_batch_size" : 64,
        "pol_imp_batch_size" : 64,
        "update_qnet_every_N_gradient_steps" : 1,
        "num_training_steps" : MAX_EPISODE_STEPS * 50,
        "num_init_exp" : 0,
        "num_new_exp" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),
        "save_after_training" : True,
        "training_id" : 7
    }
}

def generate_parameters(default_parameters, **kwargs):

    def new_dict_from_old(old_name, old_dict, key, val):
        if old_name == "default":
            new_name = f"{key}_{str(val)}"
        else:
            new_name = name + f"_{key}_{str(val)}"
        old_dict[key] = val
        return new_name, old_dict

    returned = {"default" : default_parameters}
    for key, values in kwargs.items():
        if type(values) != type([]):
            for d in returned.values(): d[key] = values
        elif type(values) == type([]) and len(values) == 0:
            pass
        elif type(values) == type([]):
            new_dicts = {}
            for name, d in returned.items():
                new_name, new_dict = new_dict_from_old(old_name=name, 
                                                       old_dict=d, 
                                                       key=key, 
                                                       val=values[0])
                new_dicts[new_name] = new_dict 
                
                for v in values[1:]:
                    new_d = deepcopy(d)
                    new_name, new_dict = new_dict_from_old(old_name=name,
                                                           old_dict=new_d,
                                                           key=key,
                                                           val=v)
                    new_dicts[new_name] = new_dict
            returned = new_dicts
    return returned
        
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

params_to_try = generate_parameters(default_parameters=parameters["play_around"],
                                    tempearture = [0.05, 0.10, 0.15, 0.20, 0.25])

for i, name_and_dict in enumerate(params_to_try.items()):
    name, p = name_and_dict
    p["training_id"] = i
    train_SAC_on_bipedal_walker(name, params_dict=params_to_try)