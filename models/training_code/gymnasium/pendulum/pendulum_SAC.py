"""
Want to see if our implementation of SAC is able to solve pendulum,
which seems to be one of the easiest continuous environemnts available
through gym.

If we can't solve this, there is an error somewhere in the code to be fixed.
"""

import gymnasium

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic, no_exploration_wrapper, train_SAC_on_gym
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
        "evaluate_N_samples" : 1,
        "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
        "buffer_size" : int(1e6),

        "save_after_training" : True,
        "render_evaluation" : False,

        "qnet_layer_sizes" : (64, 64),
        "policy_layer_sizes" : (64, 64)
    }
}

train_SAC_on_gym(parameters=parameters["online_example"], parameter_name="online_example_without_clipping",
          env=env, trainer=trainer,
          training_id="7fk52q9j")