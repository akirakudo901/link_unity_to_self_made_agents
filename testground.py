
import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from models.trainers.unityenv_base_trainer import Buffer, Experience

import gymnasium

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

# from torch import nn
# target = torch.tensor([-0.5708, ])#, -0.5431, -0.5547, -0.5632, -0.5331, -0.5343, -0.5604, -0.5390, -0.5585, -0.5543])
# # target = torch.tensor([-0.4457, -0.4419, -0.4457, -0.4404, -0.4406])
# # target = torch.tensor([-194.0212, -207.6462, -197.5734, -200.3207, -214.7844])

# pred = torch.tensor([-2.0165, ])#,  2.0930, -2.6004, -0.8696,  0.9794,  2.6524,  1.0425, -1.2586, 2.6342, -1.1144])

# criterion = nn.KLDivLoss(reduction="batchmean")
# loss = criterion(pred, target)
# print(loss)


# import gymnasium
# env = gymnasium.make("BipedalWalker-v3", render_mode="human")
# observation_size = env.observation_space.shape[0]
# action_size = env.action_space.shape[0]
# action_ranges = tuple([(env.observation_space.low[i], env.observation_space.high[i]) for i in range(action_size)])
# print(action_ranges)

# learning_algorithm = SoftActorCritic(
#     q_net_learning_rate=3e-4, 
#     policy_learning_rate=1e-3, 
#     discount=0.99, 
#     temperature=0.8,
#     observation_size=observation_size,
#     action_size=action_size, 
#     action_ranges=action_ranges,
#     update_qnet_every_N_gradient_steps=1000,
#     device=torch.device("cpu")
#     # leave the optimizer as the default = Adam
#     )

# print("Multiplier: ", learning_algorithm.policy.action_multiplier)
# print("Averages: ", learning_algorithm.policy.action_avgs)

# def uniform_random_sampling(actions, env):
#     # initially sample actions from a uniform random distribution of the right
#     # range, in order to extract good reward signals
#     action_zero_to_one = torch.rand(size=(learning_algorithm.act_size,)).cpu()
#     action_minus_one_to_one = action_zero_to_one * 2.0 - 1.0
#     adjusted_actions = (action_minus_one_to_one * 
#                         learning_algorithm.policy.action_multiplier.detach().cpu() + 
#                         learning_algorithm.policy.action_avgs.detach().cpu())
#     return adjusted_actions.numpy()

# for _ in range(10):
#     action = uniform_random_sampling(None, None)
#     print(action, "\n")

# actions = torch.tensor([3.])
# a1 = torch.tanh(actions)**2
# a2 = torch.tanh(actions).pow(2)
# print(a1, a2)

# action_ranges = ((-1., 1.),)*4
# action_multiplier = [(range[1] - range[0]) / 2 for range in action_ranges]
# print(action_ranges, action_multiplier)

from models.policy_learning_algorithms.double_deep_q_network import DoubleDeepQNetwork
from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic

# ddqn = DoubleDeepQNetwork(obs_dim_size=3, act_num_discrete=7, l_r=0.5, d_r=0.2, 
#                    soft_update_coefficient=0.11, update_target_every_N_updates=300
#                    )

# TASK_NAME = "DDQN_Test"
# ddqn.save_training_progress(task_name=TASK_NAME, training_id=1)

# ddqn2 = DoubleDeepQNetwork(obs_dim_size=3, act_num_discrete=7, l_r=1, d_r=1, 
#                    soft_update_coefficient=1, update_target_every_N_updates=1
#                    )
# ddqn2.load_training_progress(task_name=TASK_NAME, training_id=1)

# ddqn2.delete_training_progress(task_name=TASK_NAME, training_id=1)

# sac1 = SoftActorCritic(q_net_learning_rate=0.5, policy_learning_rate=0.4, discount=0.99,
#                        temperature=0.2, qnet_update_smoothing_coefficient=0.005, 
#                        pol_eval_batch_size=124, pol_imp_batch_size=124, 
#                        update_qnet_every_N_gradient_steps=500, 
#                        obs_dim_size=2, act_dim_size=1, act_ranges=((1, 2),))
# TASK_NAME = "SAC_Test"
# sac1.save_training_progress(task_name=TASK_NAME, training_id=3)

# sac2 = SoftActorCritic(q_net_learning_rate=1, policy_learning_rate=1, discount=1,
#                        temperature=1, qnet_update_smoothing_coefficient=1, 
#                        pol_eval_batch_size=1, pol_imp_batch_size=1, 
#                        update_qnet_every_N_gradient_steps=1, 
#                        obs_dim_size=2, act_dim_size=1, act_ranges=((1, 2),))
# sac2.load_training_progress(task_name=TASK_NAME, training_id=3)

# sac2.delete_training_progress(task_name=TASK_NAME, training_id=3)
# from models.trainers.utils.buffer import NdArrayBuffer

# bf1 = NdArrayBuffer(max_size=32, obs_shape=(2,), act_shape=(3,))
# bf1.append_experience(obs=np.array([1, 2]), act=np.array([3, 11, 23]), 
#                       rew=1.0, don=True, next_obs=np.array([33, 99]))
# bf1.save(save_dir=".", file_name="Test_Buffer")
# result = bf1.sample_random_experiences(1)
# print(f"from bf1: {result}")

# bf2 = NdArrayBuffer(max_size=300, obs_shape=(1,), act_shape=(1,))
# bf2.load(path="./Test_Buffer.npz")
# result = bf2.sample_random_experiences(1)
# print(f"from bf2: {result}")

# from models.trainers.utils.buffer import NdArrayBuffer

# bf1 = NdArrayBuffer(max_size=10, obs_shape=(1,), act_shape=(1,))
# for i in range(100):
#     bf1.append_experience(obs=np.array([i,]), act=np.array([100-i, ]), 
#                           rew=i%2, don=i%2==0, next_obs=np.array([i+1, ]))

# obs, act, rew, don, next_obs = bf1.sample_random_experiences(num_samples=10, seed=None)
# print(f"obs: {np.squeeze(obs)}")
# print(f"act: {np.squeeze(act)}")
# print(f"rew: {np.squeeze(rew)}")
# print(f"don: {np.squeeze(don)}")
# print(f"next_obs: {np.squeeze(next_obs)}")
# print(f"Size of bf1: {bf1.size()}")
# print("\n")

# obs, act, rew, don, next_obs = bf1.sample_random_experiences(num_samples=10, seed=None)
# print(f"obs: {np.squeeze(obs)}")
# print(f"act: {np.squeeze(act)}")
# print(f"rew: {np.squeeze(rew)}")
# print(f"don: {np.squeeze(don)}")
# print(f"next_obs: {np.squeeze(next_obs)}")
# print(f"Size of bf1: {bf1.size()}")
# print("\n")

# obs, act, rew, don, next_obs = bf1.sample_random_experiences(num_samples=10, seed=None)
# print(f"obs: {np.squeeze(obs)}")
# print(f"act: {np.squeeze(act)}")
# print(f"rew: {np.squeeze(rew)}")
# print(f"don: {np.squeeze(don)}")
# print(f"next_obs: {np.squeeze(next_obs)}")
# print(f"Size of bf1: {bf1.size()}")
# print("\n")

# import torch.nn as nn

# class Md(nn.Module):
    
#     def __init__(self, layers):
#         super(Md, self).__init__()
        
#         # it = iter(layers)
#         # previous = next(it)

#         # layers = []

#         # for sz in it:
#         #     layers.append(nn.Linear(in_features=previous, out_features=sz))
#         #     layers.append(nn.ReLU())
#         #     previous = sz

#         l = []

#         for i, sz in enumerate(layers[:-1]):
#             l.append(nn.Linear(sz, layers[i+1]))
#             l.append(nn.ReLU())
            
#         self.md = nn.Sequential(*l)
    
#     def forward(self, input):
#         return self.md(input)

# md1 = Md([1, 2, 3])
# for param in md1.parameters():
#     print(param)
# t1 = torch.tensor([1, ])
# result = md1(t1)
# print(result)

# eps_decay = 1/2
# min_eps = 0.2
# init_eps = 1
# episodes = 5

# eps_decay = (min_eps / init_eps)**(1 / (episodes * 0.8))
# print(eps_decay)
# print(eps_decay**(episodes*0.8))

# ddqn1 = DoubleDeepQNetwork(obs_dim_size=2, act_num_discrete=3, dqn_layer_sizes=(16, 32, 64))
# for param in ddqn1.dnn_policy.parameters():
#     print(param.size())

# ddqn2 = DoubleDeepQNetwork(obs_dim_size=2, act_num_discrete=3)
# for param in ddqn2.dnn_policy.parameters():
#     print(param.size())

# from copy import deepcopy

# def generate_parameters(default_parameters, **kwargs):

#     def new_dict_from_old(old_name, old_dict, key, val):
#         if old_name == "default":
#             new_name = f"{key}_{str(val)}"
#         else:
#             new_name = name + f"_{key}_{str(val)}"
#         old_dict[key] = val
#         return new_name, old_dict

#     returned = {"default" : default_parameters}
#     for key, values in kwargs.items():
#         if type(values) != type([]):
#             for d in returned.values(): d[key] = values
#         elif type(values) == type([]) and len(values) == 0:
#             pass
#         elif type(values) == type([]):
#             new_dicts = {}
#             for name, d in returned.items():
#                 new_name, new_dict = new_dict_from_old(old_name=name, 
#                                                        old_dict=d, 
#                                                        key=key, 
#                                                        val=values[0])
#                 new_dicts[new_name] = new_dict 
                
#                 for v in values[1:]:
#                     new_d = deepcopy(d)
#                     new_name, new_dict = new_dict_from_old(old_name=name,
#                                                            old_dict=new_d,
#                                                            key=key,
#                                                            val=v)
#                     new_dicts[new_name] = new_dict
#             returned = new_dicts
#     return returned

# MAX_EPISODE_STEPS = 200

# params = generate_parameters(default_parameters={
#     "q_net_learning_rate"  : 1e-3,
#     "policy_learning_rate" : 1e-3,
#     "discount" : 0.99,
#     "temperature" : 0.10,
#     "qnet_update_smoothing_coefficient" : 0.005,
#     "pol_eval_batch_size" : 64,
#     "pol_imp_batch_size" : 64,
#     "update_qnet_every_N_gradient_steps" : 1,
#     "num_training_steps" : MAX_EPISODE_STEPS * 50,
#     "num_init_exp" : 0,
#     "num_new_exp" : 1,
#     "evaluate_every_N_epochs" : MAX_EPISODE_STEPS,
#     "buffer_size" : int(1e6),
#     "save_after_training" : False,
#     "training_id" : 7
#     },
#     temperature = [1, 2, 3, 4],
#     discount = [0, 10, 20, 30])

# for name, param in params.items():
#     print(f"{name}:")
#     print(param)
#     print("\n")

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic

def verify_correct_layer_size(layer_sizes, layers):
    for i, l in enumerate(layers):
        layersz_idx = i // 2
        if i % 2 == 0:
            assert l.in_features == layer_sizes[layersz_idx]
            assert l.out_features == layer_sizes[layersz_idx + 1]
            print(f"When i = {i}")
            print(l.in_features)
            print(layer_sizes[layersz_idx])
            print(l.out_features)
            print(layer_sizes[layersz_idx+1])

net1 = SoftActorCritic.create_net(input_size=3, output_size=5, interim_layer_sizes=(8, 32))
# verify_correct_layer_size(layer_sizes=[3, 8, 32, 5], layers=net1)
net2 = net1[:-2]
verify_correct_layer_size(layer_sizes=[3, 8, 32], layers=net2)
print(net1[:-1])