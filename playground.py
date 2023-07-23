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

from torch.distributions.normal import Normal

myus = torch.tensor([0, 0.1, 0.2, 0.3], dtype=torch.float32)
sigmas = torch.tensor([0.05, 0.1, 0.15, 0.20], dtype=torch.float32)

norm = Normal(myus, sigmas)
rs = norm.rsample(sample_shape=(10,))
print("rs: ", rs, "\n", "shape: ", rs.shape)
jacobian_trace = torch.sum((1 - torch.tanh(rs)**2), dim=1)
print("jacobian_trace: ", jacobian_trace, "\n", "shape: ", jacobian_trace.shape)

squashed = torch.tanh(rs)
print("squashed: ", squashed, "\n", "shape: ", squashed.shape)
derivative = (1 - squashed**2)
print("derivative: ", derivative, "\n", "shape: ", derivative.shape)
sum = torch.sum(derivative, dim=1)
print("sum: ", sum, "\n", "shape: ", sum.shape)


# rsample and sample seem to be mostly equivalent