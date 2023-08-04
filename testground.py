
import numpy as np
import torch

from policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from trainers.unityenv_base_trainer import Buffer, Experience

# sac = SoftActorCritic(learning_rate=0.5, discount=0.9, temperature=0.5, observation_size=3, action_size=1)

# exp : Buffer = []
# for i in range(10):
#     exp.append(Experience(
#         np.array([0+i, 1+i, 2+i], dtype=np.float32), #obs 
#         np.array([5+i]), #action 
#         np.array([100.0]), #reward 
#         False, #done 
#         np.array([7+i, 8+i, 9+i], dtype=np.float32) #next_obs
#         ))
    
# # for inner_exp in exp:
# #     print(inner_exp, "\n")

# sac.update(experiences=exp)



# from torch.distributions.normal import Normal
# import torch

# # Goal is to understand how torch distributions work
# myus, sigmas = torch.tensor([0.0, 100.0]), torch.tensor([1.0, 15.0])
# num_samples = 4

# dist = Normal(loc=myus, scale=sigmas)
# actions = dist.sample(sample_shape=(num_samples, ))
# squashed = torch.tanh(actions)

# log_probs = dist.log_prob(actions)


# # log_probs = correct_for_squash(
# #     dist.log_prob(actions)
# #     )

# print("actions: ", actions, "\n squashed: ", squashed, "\n log_probs: ", log_probs)

# bool_t = torch.tensor([True, False, False])
# float_t = bool_t.to(torch.float32)
# print(float_t)

# [batch, num_samples]
# t1 = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]).to(torch.float32)
# t2 = torch.mean(t1, dim=1)
# print(t2, t2.shape)

# oldSAC = SoftActorCritic(q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=0.99, temperature=0.8, observation_size=3, action_size=4)

# oldSAC.save(task_name="test")

# newSAC = SoftActorCritic(q_net_learning_rate=1e-2, policy_learning_rate=1e-2, discount=0.1, temperature=0.23, observation_size=3, action_size=4)
# newSAC.load(loaded_sac_name="trained_algorithms/SAC/test_2023_07_25_10_03")

# from timeit import default_timer as timer

# obs_t = torch.tensor([[0., 1., 2.], [3., 4., 5.]]*1024)
# print("obs_t.shape: ", obs_t.shape)
# act_t = torch.tensor([[[10.,11.,12.], [13.,14.,15.], [16.,17.,18.], [19., 20., 21.]], [[30.,31.,32.], [33.,34.,35.], [36.,37.,38.], [39.,40.,41.]]]*1024)
# print("act_t.shape: ", act_t.shape)
# # using a for loop and split
# start_for = timer()
# for_split_t = torch.stack([
#     torch.stack((obs_t, torch.squeeze(single_act_t, dim=1))) 
#     for single_act_t in torch.split(act_t, 1, dim=1) 
#     ])
# end_for = timer()
# # using index and split
# start_index = timer()
# index_split_t = torch.stack([
#     torch.stack((obs_t, torch.squeeze(act_t[:, i, :], dim=1)))
#     for i in range(act_t.shape[1])  
#     ])
# end_index = timer()

# print(f"The tensors were {'' if torch.equal(for_split_t, index_split_t) else 'not '}equal.")
# print(f"Execution time for for  : {end_for - start_for}.")
# print(f"Execution time for index: {end_index - start_index}.")

# action_ranges = ((0., 1.), (-15., 3.), (-1., 2.))
# avgs   = torch.tensor([(range[1] + range[0]) / 2 for range in action_ranges])
# ranges = torch.tensor([(range[1] - range[0]) / 2 for range in action_ranges])
# print("avgs: ", avgs, "ranges: ", ranges)

# squashed = torch.tensor([0.3, 0.6, -0.8])
# result = squashed * ranges + avgs

# print("result: ", result)

# print(tuple(((0., 1.),)*7))

# import gymnasium as gym

# env = gym.make("CartPole-v1")
# print(env.spec.id)

# print(t1 := torch.squeeze(
#                 self.learning_algorithm(
#                     torch.tensor([
#                         [0.0, -0.5, 0.3, 0.8],
#                         ])
#                 ).cpu().detach()
#                 ).numpy())
# print(t1.shape)

from torch import nn
target = torch.tensor([-0.5708, ])#, -0.5431, -0.5547, -0.5632, -0.5331, -0.5343, -0.5604, -0.5390, -0.5585, -0.5543])
# target = torch.tensor([-0.4457, -0.4419, -0.4457, -0.4404, -0.4406])
# target = torch.tensor([-194.0212, -207.6462, -197.5734, -200.3207, -214.7844])

pred = torch.tensor([-2.0165, ])#,  2.0930, -2.6004, -0.8696,  0.9794,  2.6524,  1.0425, -1.2586, 2.6342, -1.1144])

criterion = nn.KLDivLoss(reduction="batchmean")
loss = criterion(pred, target)
print(loss)
                