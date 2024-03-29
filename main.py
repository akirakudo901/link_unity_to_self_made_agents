# from models.training_code.unity.walker import walker_sac_training
# from models.training_code.unity.tournament_env import tournament_env_sac_training
# from models.training_code.gymnasium.bipedal_walker import bipedal_walker_SAC
# from models.training_code.gymnasium.pendulum import pendulum_SAC
from models.training_code.gymnasium.mountain_car_continuous import mountain_car_continuous_SAC
# from models.training_code.gymnasium.cart_pole import cart_pole_DDQN

# import gymnasium

# from models.trainers.gym_base_trainer import GymOffPolicyBaseTrainer
# from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic

# def test_Pendulum_on(name):
#     env = gymnasium.make("Pendulum-v1")
#     trainer = GymOffPolicyBaseTrainer(env)

#     sac_algorithm = SoftActorCritic(q_net_learning_rate=1.0, policy_learning_rate=0.1, discount=1,
#                                     temperature=2, qnet_update_smoothing_coefficient=0.5,
#                                     pol_eval_batch_size=54, pol_imp_batch_size=54,
#                                     update_qnet_every_N_gradient_steps=1, env=env)
#     sac_algorithm.load(path=f"trained_algorithms/SAC/{name}")

#     trainer.render_evaluation = True
#     reward = trainer.evaluate(sac_algorithm, 10)
#     print(f"Reward is: {reward}!")

# def test_BipedalWalker_on(path):
#     env = gymnasium.make("BipedalWalker-v3")
#     trainer = GymOffPolicyBaseTrainer(env)

#     sac_algorithm = SoftActorCritic(q_net_learning_rate=1.0, policy_learning_rate=0.1, discount=1,
#                                     temperature=2, qnet_update_smoothing_coefficient=0.5,
#                                     pol_eval_batch_size=54, pol_imp_batch_size=54,
#                                     update_qnet_every_N_gradient_steps=1, env=env,
#                                     qnet_layer_sizes=(256, 256),
#                                     policy_layer_sizes=(256, 256))
#     sac_algorithm.load(path=path)

#     trainer.render_evaluation = True
#     reward = trainer.evaluate(sac_algorithm, 10)
#     print(f"Reward is: {reward}!")

# import os

# test_BipedalWalker_on(os.path.join("trained_algorithms", "SAC", "temperature_10.0_znh5wkvu"))