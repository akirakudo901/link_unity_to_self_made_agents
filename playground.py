import torch
# from mlagents_envs.registry import default_registry
# env = default_registry["Walker"].make()
# env.reset()

# print(env.behavior_specs)
# behavior_name=list(env.behavior_specs)[0]
# print(env.behavior_specs[behavior_name].observation_specs[0].shape)
# print(type(env.behavior_specs[behavior_name].observation_specs[0].shape))
# print(env.behavior_specs[behavior_name].action_spec.continuous_size)

# env.close()
# _____________________

# t1 = torch.tensor([
#         [
#             [11, 12, 13], 
#             [21, 22, 23], 
#             [31, 32, 33]
#         ]
#     ])
# t2 = t1.argmax(dim=2, keepdim=False)
# t3 = t1.argmax(dim=2, keepdim=True)
# print(t2)
# print(t3)

# t1 = torch.tensor((0, 1, 2))
# t2 = torch.stack([t1 for _ in range(10)], dim=1)
# t3 = torch.cat([t1 for _ in range(10)])
# print(t1)
# print(t2)
# print(t3)

# t1 = torch.tensor([ [1], 
#                     [2], 
#                     [3], 
#                     [4] ])
# t2 = torch.tensor([ [5],
#                     [6], 
#                     [7], 
#                     [8] ])
# t3 = torch.tensor([ [11, 12, 13],
#                     [21, 22, 23],
#                     [31, 32, 33],
#                     [41, 42, 43]])
# t4 = torch.mean(
#     torch.minimum(
#         torch.stack([t1 + t_temp for t_temp in torch.split(t3, 1, dim=1)]),
#         torch.stack([t1 + t_temp for t_temp in torch.split(t3, 1, dim=1)])
#     ), dim=0, dtype=torch.float32
# )
# print(t4)

# from torch.distributions.normal import Normal

# myus = torch.tensor([0, 0.1, 0.2, 0.3], dtype=torch.float32)
# sigmas = torch.tensor([0.05, 0.1, 0.15, 0.20], dtype=torch.float32)

# norm = Normal(myus, sigmas)
# rs = norm.rsample(sample_shape=(10,))
# print("rs: ", rs, "\n", "shape: ", rs.shape)
# jacobian_trace = torch.sum((1 - torch.tanh(rs)**2), dim=1)
# print("jacobian_trace: ", jacobian_trace, "\n", "shape: ", jacobian_trace.shape)

# squashed = torch.tanh(rs)
# print("squashed: ", squashed, "\n", "shape: ", squashed.shape)
# derivative = (1 - squashed**2)
# print("derivative: ", derivative, "\n", "shape: ", derivative.shape)
# sum = torch.sum(derivative, dim=1)
# print("sum: ", sum, "\n", "shape: ", sum.shape)

# rsample and sample seem to be mostly equivalent

# class A:

#     def __getitem__(self, slice):
#         print(slice)
#         print(slice.start, slice.stop, slice.step)

# a = A()
# a[1:9:3]

# from typing import NamedTuple

# class B(NamedTuple):
#     attA : int
#     attB : str

# b = B(attA=1, attB="attB")
# print(list(*b))

# import numpy as np
# from timeit import default_timer as timer

# n1 = np.random.rand(1000, 1000)
# n2 = np.random.rand(1000, 800)
# n1_ = np.random.rand(1000, 1000)
# n2_ = np.random.rand(1000, 800)

# # print("initial n1, n2:", n1, "\n", n2 )

# start_index = timer()
# n1[:, :800] = n2[:,:]
# end_index = timer()

# # print("final n1, n2:", n1,"\n", n2 )

# print("index :", end_index - start_index)

# # print("initial n1_, n2_:", n1_, "\n", n2_)

# start_for = timer()
# for i in range(800):
#     n1_[:, i] = n2_[:, i]
# end_for = timer()

# # print("final n1_, n2_:", n1_, "\n", n2_)

# print("for :", end_for - start_for)


############################################
# # TRYING TO RESOLVE THE ISSUE WHERE -INF ARISES FROM LOG PROBS
# import math

# from torch.distributions.normal import Normal
# from numbers import Real

# # myus = torch.tensor([0, 0.1, 0.2, 0.3], dtype=torch.float32)
# # sigmas = torch.tensor([1, 0.1, 0.15, 0.20], dtype=torch.float32)
# # sigmas = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# myus = torch.tensor([0], dtype=torch.float32)
# sigmas = torch.tensor([0.15], dtype=torch.float32)

# def own_log_prob(value, scale, loc):
#     # if self._validate_args:
#     #     self._validate_sample(value)
#     # compute the variance
#     var = (scale ** 2)
#     print(f"var is {var}.")
#     log_scale = math.log(scale) if isinstance(scale, Real) else scale.log()
#     print(f"log_scale is {log_scale}.")
#     comp1, comp2, comp3, comp4 = (value - loc) ** 2, 2 * var, log_scale, math.log(math.sqrt(2 * math.pi))
#     print(f"comp1 is {comp1}.")
#     print(f"comp2 is {comp2}.")
#     print(f"comp3 is {comp3}.")
#     print(f"comp4 is {comp4}.")
#     return -(comp1) / (comp2) - comp3 - comp4
#     # return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

# norm = Normal(myus, sigmas)
# rs = norm.rsample(sample_shape=(3,))
# print("rs: ", rs, "\n", "shape: ", rs.shape)
# log_probs = norm.log_prob(rs)
# print("log_probs: ", log_probs, "\n", "shape: ", log_probs.shape)
# own_log_probs = own_log_prob(rs, scale=sigmas, loc=myus)
# print("own_log_probs: ", own_log_probs, "\n", "shape: ", own_log_probs.shape)


# """
# Log probabiliities - problem where log_probs goes to infinity and thus gradient to -inf, and qnet to nan
# sigmas : [0.1172, 0.0067, 0.0071, 0.0067]
# myus : [ 4.4031, -8.8166,  6.4839,  9.0240]
# action : [[ 4.3591e+00, -8.8212e+00,  6.4904e+00,  9.0213e+00]]
# pure_log_proabbilitie : [[ 1.1544,  3.8494,  3.6093,  4.0017]]
# before_correction : [12.6148]
# log_probs :  [inf]
# """
# action = torch.tensor([[[ 4.3591e+00, -8.8212e+00,  6.4904e+00,  9.0213e+00],],]) #btch_sz, smpl_sz, actn_sz
# before = torch.tensor([[12.6148],]) #btch_sz, smpl_sz

# multiplier = torch.tensor([1, 1, 1, 1])
# oneMinusTanHPow = torch.log(1 - torch.tanh(action).pow(2))
# print(f"oneMinusTanHPow is {oneMinusTanHPow}.")
# # jacobian_trace = torch.sum((multiplier * torch.log(1 - torch.tanh(action).pow(2)) ), dim=2)
# jacobian_trace = torch.sum((multiplier *  oneMinusTanHPow), dim=2)
# print(f"jacobian_trace is {jacobian_trace}.")
# after = before - jacobian_trace
# print(f"after is {after}.")

##########################################
# TRYING OUT RANDOM GENERATORS
# import numpy as np

# rng = np.random.default_rng(seed=None)
# t1 = torch.tensor(range(10))
# t2 = torch.tensor(range(10, 20))
# t3 = torch.tensor(range(20, 30))

# # print(t1, t2, t3)

# indices = rng.choice(range(len(t1)), size=3, replace=False)
# t1_choice = t1[indices]
# t2_choice = t2[indices]
# t3_choice = t3[indices]

# print(indices)

# print(t1_choice, t2_choice, t3_choice)

#############################################
# DOUBLE CHECKING HOW SQUEEZING WORKS

# t1 = torch.tensor([[[-0.0201]], [[-0.1084]]])
# single_sample = t1[:, 0, :]
# squeeze_all = torch.squeeze(single_sample)
# squeeze_zero = torch.squeeze(single_sample, dim=0)
# squeeze_one  = torch.squeeze(single_sample, dim=1)
# # squeeze_two  = torch.squeeze(single_sample, dim=2)
# print(f"single_sample: {single_sample}.")
# print(f"squeeze_all: {squeeze_all}\n squeeze_zero: {squeeze_zero}\n \
#       squeeze_one: {squeeze_one}\n squeeze_two: {squeeze_one}.")
# print(f"single_sample.shape: {single_sample.shape}")


# SHOWING THAT THE TWO IMPLEMENTATIONS OF FORWARD IN CLEAN RL AND MY VERSION RESULT IN IDENTICAL 
# VALUES
# action_multiplier, action_avgs = torch.tensor([1., 2., 1.5]), torch.tensor([0.5, -1., 0.])

# def clean_rl_imp(mean, std, samples, log_prob):
#     # normal = torch.distributions.Normal(mean, std)
#     # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#     x_t = samples
#     y_t = torch.tanh(x_t)
#     action = y_t * action_multiplier + action_avgs
#     # log_prob = normal.log_prob(x_t)
#     # Enforcing Action Bound
#     print("INSIDE CLEAR RL.")
#     print(f"log_prob: {log_prob}")
#     jacob = torch.log(action_multiplier * (1 - y_t.pow(2)) + 1e-6)
#     print(f"jacob: {jacob}")
#     print(f"jacob_trace??: {torch.sum(jacob, dim=1)}")
#     log_prob -= jacob
#     print(f"log_prob: {log_prob}")
#     log_prob = log_prob.sum(1, keepdim=True)
#     print(f"log_prob: {log_prob}")
#     mean = torch.tanh(mean) * action_multiplier + action_avgs
#     print("CLEAR RL END.")
#     return action, log_prob, mean


# def akira_imp(myus, sigmas, samples, log_prob):

#     def squashing_function(actions : torch.tensor):
#         squashed_neg_one_to_one = torch.tanh(actions)
#         t_device = squashed_neg_one_to_one.device
#         return (squashed_neg_one_to_one * 
#                 action_multiplier.to(t_device).detach() + 
#                 action_avgs.to(t_device).detach())

#     def _correct_for_squash(before : torch.tensor, actions : torch.tensor):
#         multiplier = action_multiplier.to(actions.device)
#         print(f"actions: {actions}")
#         print(f"before: {before}")
#         jacobian_trace = torch.sum(torch.log(multiplier * (1 - torch.tanh(actions).pow(2)) + 1e-6), dim=2)
#         print(f"jacobian_trace: {jacobian_trace}")
#         after = before - jacobian_trace
#         return after

#     # dist = torch.distributions.Normal(loc=myus, scale=sigmas) #MultivariateNormal with diagonal covariance
#     # actions_num_samples_first = dist.sample(sample_shape=(1, )).to(torch.float32)
#     print("INSIDE AKIRA RL.")
#     actions_num_samples_first = samples
#     actions = torch.transpose(actions_num_samples_first, dim0=0, dim1=1)
#     squashed = squashing_function(actions)

#     # pure_probabilities is log_prob for each action when sampled from each normal distribution
#     # aggregating over the entire action, it becomes their sum (product of independent events but logged)
#     # log_prob = dist.log_prob(actions_num_samples_first).to(torch.float32)
#     pure_log_probabilities = torch.transpose(log_prob, dim0=0, dim1=1)
#     print(f"pure_log_probabilities: {pure_log_probabilities}")
#     before_correction = torch.sum(pure_log_probabilities, dim=2)
#     log_probs = _correct_for_squash(
#         before_correction, actions
#         )
    
#     print("AKIRA RL END.")
    
#     return squashed, log_probs

# myu, sigma = torch.tensor([0., 1., 2.]), torch.tensor([0.2, 0.4, 0.6])
# normal = torch.distributions.Normal(myu, sigma)
# samples = normal.rsample(sample_shape=(8, 1, )).to(torch.float32)
# log_prob = normal.log_prob(samples)
# samples_without_action_dim = torch.squeeze(samples.clone(), dim=1)
# log_prob_without_action_dim = torch.squeeze(log_prob.clone(), dim=1)

# act_clean, log_clean, _ = clean_rl_imp(mean=myu, std=sigma, samples=samples_without_action_dim, log_prob=log_prob_without_action_dim)
# act_akira, log_akira = akira_imp(myus=myu, sigmas=sigma, samples=samples, log_prob=log_prob)

# print("CLEAN!")
# print(act_clean, log_clean)
# print("AKIRA!")
# print(act_akira, log_akira)

# 

# def save(task_name, second):
#     print(task_name.upper() + second.lower() + "!")

# def try_saving_except(call, *args, **kwargs):
#     try:
#         print("Saving the algorithm parameters!")
#         call(*args, **kwargs)
#         print("Saved the algorithm parameters successfully!")
#     except Exception:
#         print("Some exception occurred while saving algorithm parameters...")

# try_saving_except(save, "abc", second="Second")
# from models.policy_learning_algorithms.soft_actor_critic import uniform_random_sampling_wrapper, SoftActorCritic
# sac1 = SoftActorCritic(1.0, 1.0, 1.0, 1.0, 1.0, 1, 1, 1, 
#                        2, 3, ((0, 1),(2, 3),(-1, 2)))

# urs = uniform_random_sampling_wrapper(learning_algorithm=sac1)
# returned = urs(actions=None, env=None)
# print(returned)

# import random
# random.seed(10)
# print(random.randint(1, 100))
# print(random.randint(1, 100))

# random.seed(None)
# print(random.randint(1, 100))
# print(random.randint(1, 100))

# random.seed(10)
# print(random.randint(1, 100))
# print(random.randint(1, 100))

# from torch.distributions.independent import Independent
# from torch.distributions.normal import Normal
# from torch.distributions.multivariate_normal import MultivariateNormal

# loc = torch.zeros(3)
# scale = torch.ones(3)
# # multivariate normal
# torch.manual_seed(42)
# mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
# print(f"mvn.batch_shape: {mvn.batch_shape}, mvn.event_shape: {mvn.event_shape}")
# mvn_sample = mvn.rsample()
# print(f"mvn_sample: {mvn_sample}, mvn_sample.shape: {mvn_sample.shape}")
# mvn_log_probs = mvn.log_prob(mvn_sample)
# print(f"mvn_log_probs: {mvn_log_probs}, mvn_log_probs.shape: {mvn_log_probs.shape}")
# print("\n")

# # normal
# torch.manual_seed(42)
# normal = Normal(loc, scale)
# print(f"normal.batch_shape: {normal.batch_shape}, normal.event_shape: {normal.event_shape}")
# normal_sample = normal.rsample()
# print(f"normal_sample: {normal_sample}, normal_sample.shape: {normal_sample.shape}")
# normal_log_probs = normal.log_prob(normal_sample)
# print(f"normal_log_probs: {normal_log_probs}, normal_log_probs.shape: {normal_log_probs.shape}")
# print("\n")

# # diagonal
# torch.manual_seed(42)
# diagn = Independent(normal, 1)
# print(f"diagn.batch_shape: {diagn.batch_shape}, diagn.event_shape: {diagn.event_shape}")
# diagn_sample = diagn.rsample()
# print(f"diagn_sample: {diagn_sample}, diagn_sample.shape: {diagn_sample.shape}")
# diagn_log_probs = diagn.log_prob(diagn_sample)
# print(f"diagn_log_probs: {diagn_log_probs}, diagn_log_probs.shape: {diagn_log_probs.shape}")
# print("\n")

# multiplier = torch.tensor(3).to(torch.float32)
# actions = torch.tensor([[[1, 2, 3],],])
# print("actions: ", actions)
# print("actions.shape: ", actions.shape)

# jacobian_trace = torch.sum(
#     torch.log(multiplier) + 
#     (2. * (torch.log(torch.tensor(2.)) - 
#            actions - 
#            torch.nn.functional.softplus(-2. * actions))), 
#            dim=2)
# print("jacobian_trace: ", jacobian_trace)

# jacobian_trace2 = (torch.log(multiplier) * actions.shape[2] + 
#                    torch.sum(
#                        (2. * (torch.log(torch.tensor(2.)) - 
#                               actions - 
#                               torch.nn.functional.softplus(-2. * actions))), 
#                               dim=2))
# print("jacobian_trace2: ", jacobian_trace2)

# import gymnasium
# import numpy as np

# env1 = gymnasium.make("Pendulum-v1", render_mode="human")
# env1.reset()
# action = np.array([-0.5])

# for i in range(100):
#     env1.step(action)

# env2 = gymnasium.make("Pendulum-v1")
# env2.reset()

# for i in range(50):
#     env2.step(action)

# env2.close()

# for i in range(200):
#     env1.step(action)

# env1.close()

# import wandb
# import matplotlib.pyplot as plt

# wandb.init(
#     project='trial_playground',
#     settings=wandb.Settings(_disable_stats=True),
#     name=f'run_id={3}'
# )

# epochs, avg_reward, max_r, min_r = [1, 2, 3, 4], [10, 11, 12, 13], [18, 12, 15, 19], [9, 9, 8.5, 10]
# data = [[epoch, avg, max, min] for (epoch, avg, max, min) in zip(epochs, avg_reward, max_r, min_r)]
# table = wandb.Table(data=data, columns = ["Epochs", "Average Reward", "Max Reward", "Min Reward"])

# plt.plot(epochs, avg_reward, label="Average Reward")
# plt.plot(epochs, max_r, label="Max Reward")
# plt.plot(epochs, min_r, label="Min Reward")
# plt.xlabel("Epochs"); plt.ylabel("Rewards")
# plt.legend()

# wandb.log({"multi-trial rewards" : wandb.Image(plt)})

# import gymnasium
# from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic

# env = gymnasium.make("BipedalWalker-v3")
# sac = SoftActorCritic(1e-3, 1e-3, 0.99, 0.1, 0.005, 64, 64, 1, env=env)
# print(sac.obs_dim_size)
# print(sac.act_dim_size)
# print(sac.act_ranges)

# def generate_name_from_parameter_dict(parameter_dict):
#     """
#     Generates a name characterizing a given parameter dict.
#     The order of terms in the dictionary is the order in which 
#     parameters are listed.

#     :param Dict parameter_dict: The parameter dict for which we generate the name.
#     """
#     acc = str(list(parameter_dict.keys())[0]) + "_" + str(list(parameter_dict.values())[0])
#     [acc := acc + "_" + str(key) + "_" + str(val) for key, val in list(parameter_dict.items())[1:]]
#     return acc

# param_dict = {
#     "param1" : 1.0,
#     "param2" : "amazing",
#     "param3" : 30,
#     "param4" : (1, 2, 3)
# }
# print(generate_name_from_parameter_dict(param_dict))

# import gymnasium

# env = gymnasium.make("Pendulum-v1")
# unscaled_high, unscaled_low = env.observation_space.high, env.observation_space.low
# print(unscaled_high)
# print(unscaled_low)