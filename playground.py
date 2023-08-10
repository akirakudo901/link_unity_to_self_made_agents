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
import numpy as np

rng = np.random.default_rng(seed=123)
t1 = torch.tensor(range(10))
t2 = torch.tensor(range(10, 20))
t3 = torch.tensor(range(20, 30))

# print(t1, t2, t3)

indices = rng.choice(range(len(t1)), size=3, replace=False)
t1_choice = t1[indices]
t2_choice = t2[indices]
t3_choice = t3[indices]

print(indices)

print(t1_choice, t2_choice, t3_choice)