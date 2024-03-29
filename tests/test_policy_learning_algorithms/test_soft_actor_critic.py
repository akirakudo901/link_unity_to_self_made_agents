
import unittest

import numpy as np
import torch

from models.policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from models.trainers.utils.buffer import NdArrayBuffer

class TestSoftActorCritic(unittest.TestCase):
    """
    Mostly to test all functionalities that are not specific to the 
    policy or the qnets.
    """

    BUFFER_CLASS = NdArrayBuffer

    def test_get_parameter_dict_and_load_parameter_dict(self):
        # establish two different SACs
        sac1 = SoftActorCritic(q_net_learning_rate= (qlr1 := 0.01),
                               policy_learning_rate= (plr1 := 0.02),
                               discount= (disc1 := 0.03),
                               temperature= (temp1 := 0.04),
                               qnet_update_smoothing_coefficient= (qusc1 := 0.05),
                               pol_eval_batch_size= (pebs1 := 100),
                               pol_imp_batch_size= (pibs1 := 200),
                               update_qnet_every_N_gradient_steps= (Nsteps1 := 300),
                               obs_dim_size= (obs_dims_size1 := 10),
                               act_dim_size= (act_dim_size1 := 2),
                               act_ranges= (act_ranges_1 := ((1, 2),(3, 4))))
        sac1.qnet_update_counter = (qnet_update_counter1 := 1000)
        sac1.qnet1_loss_history = (qnet1_loss_history1 := [1, 2, 4])
        sac1.qnet2_loss_history = (qnet2_loss_history1 := [4, 11, 23])
        sac1.policy_loss_history = (policy_loss_history1 := [11, 12, 33])

        sac2 = SoftActorCritic(q_net_learning_rate= (qlr2 := 1.00),
                               policy_learning_rate= (plr2 := 1.00),
                               discount= (disc2 := 1.00),
                               temperature= (temp2 := 1.00),
                               qnet_update_smoothing_coefficient= (qusc2 := 1.00),
                               pol_eval_batch_size= (pebs2 := 1),
                               pol_imp_batch_size= (pibs2 := 1),
                               update_qnet_every_N_gradient_steps= (Nsteps2 := 1),
                               obs_dim_size= (obs_dims_size2 := 1),
                               act_dim_size= (act_dim_size2 := 1),
                               act_ranges= (act_ranges_2 := ((1, 2),)))
        # ensure their parameters are not equal
        self.assertNotEqual(qlr1, qlr2)
        self.assertNotEqual(plr1, plr2)
        self.assertNotEqual(disc1, disc2)
        self.assertNotEqual(temp1, temp2)
        self.assertNotEqual(qusc1, qusc2)
        self.assertNotEqual(pebs1, pebs2)
        self.assertNotEqual(pibs1, pibs2)
        self.assertNotEqual(Nsteps1, Nsteps2)
        self.assertNotEqual(obs_dims_size1, obs_dims_size2)
        self.assertNotEqual(act_dim_size1, act_dim_size2)
        self.assertNotEqual(act_ranges_1, act_ranges_2)
        # then load parameters from one to the other
        param_dict = sac1._get_parameter_dict()
        sac2._load_parameter_dict(param_dict)
        # finally compare the parameters now
        self.assertEqual(qlr1, sac2.q_net_l_r)
        self.assertEqual(plr1, sac2.pol_l_r)
        self.assertEqual(disc1, sac2.d_r)
        self.assertEqual(temp1, sac2.alpha)
        self.assertEqual(qusc1, sac2.tau)
        self.assertEqual(pebs1, sac2.pol_eval_batch_size)
        self.assertEqual(pibs1, sac2.pol_imp_batch_size)
        self.assertEqual(Nsteps1, sac2.update_qnet_every_N_gradient_steps)
        self.assertEqual(obs_dims_size1, sac2.obs_dim_size)
        self.assertEqual(act_dim_size1, sac2.act_dim_size)
        self.assertEqual(act_ranges_1, sac2.act_ranges)
        self.assertEqual(qnet_update_counter1, sac2.qnet_update_counter)
        self.assertEqual(qnet1_loss_history1, sac2.qnet1_loss_history)
        self.assertEqual(qnet2_loss_history1, sac2.qnet2_loss_history)
        self.assertEqual(policy_loss_history1, sac2.policy_loss_history)

class TestSoftActorCritic_Policy(unittest.TestCase):
    """
    Mostly non-formal test suit - only checks mostly for shapes of tensors in between,
    and prints the intermediate tensors for checking.
    """

    def test_Policy_correct_for_squash(self): 
        before1 = torch.tensor([[0.3, 0.9, 0.95], [0.5, 0.7, 0.75]]) #(2, 3)
        actions1 = torch.tensor([[[ 0.5,  0.6,  0.7,  0.8], [ 0.1,  0.2,  0.3,  0.4], [0.41, 0.52, 0.63, 0.74]],  #(2, 3, 4)
                                 [[0.33, 0.44, 0.55, 0.66], [0.12, 0.24, 0.36, 0.48], [0.60, 0.72, 0.84, 0.96]]])
        
        action_ranges = ((-5., 0.6),)*4
        action_multiplier = torch.tensor([(range[1] - range[0]) / 2 for range in action_ranges])
        # formula is: jacobianDiagonal = 2 * (log(2) + log(multiplier) - x - softplus(-2x))
        jacobian_trace_equivalent = torch.sum(torch.log(action_multiplier) + 
                                              (2.0 * (torch.log(torch.tensor(2.)) - 
                                                      actions1 -
                                                      torch.nn.functional.softplus(-2. * actions1))), dim=2)
        # ORIGINAL LESS NUMERICALLY STABLE WAY:
        # jacobian_diagonals1 = (1 - torch.tanh(actions1).pow(2))
        # jacobian_trace1 = torch.sum(torch.log(action_multiplier * jacobian_diagonals1 + 1e-6), dim=2)
        
        expected1 = before1 - jacobian_trace_equivalent
        
        pol = SoftActorCritic.Policy(observation_size=3, action_size=4, action_ranges=action_ranges)

        actual1 = pol._correct_for_squash(before=before1, actions=actions1)
        
        self.assertTrue(torch.isclose(expected1, actual1).all())

    def test_forward(self):
        # TODO ONLY TESTING IF IT RUNS
        # UNCOMMENT IT ONCE SAC IS FINALIZED
        # example_policy = SoftActorCritic.Policy(observation_size=3, action_size=7, action_ranges=tuple(((-1., 1.),)*7))
        # obs1 = torch.tensor([[0.0,1.0,2.0], [3.0,4.0,5.0]]) #[batch=2, elem=3]
        # squashed, log_probs = example_policy(obs=obs1, num_samples=5, deterministic=False)
        # self.assertEqual((2, 5, 7), tuple(squashed.shape))
        # self.assertEqual((2, 5), tuple(log_probs.shape))

        # myus = example_policy(obs=obs1, num_samples=5, deterministic=True)
        # # print("myus: ", myus)
        # self.assertEqual((2, 7), tuple(myus.shape))
        pass
    
    def test_compute_qnet_target(self):
        # TODO ONLY TESTING IF IT RUNS UNCOMMENT IT ONCE SAC IS FINALIZED
        # OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        # DISCOUNT, TEMPERATURE, TAU = 0.8, 0.6, 0.005

        # testSAC = SoftActorCritic(
        #     q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
        #     temperature=TEMPERATURE, qnet_update_smoothing_coefficient=TAU,
        #     obs_dim_size=OBSERVATION_SIZE, act_dim_size=ACTION_SIZE, 
        #     act_ranges=((-1., 1.), (-2., 3.), (0., 1.), (-10., 10.)),
        #     pol_eval_batch_size=8, pol_imp_batch_size=8,
        #     update_qnet_every_N_gradient_steps=1
        #     )

        # qnet1 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)
        # qnet2 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)

        # # assuming a batch of two experiences
        # batch_rewards = torch.tensor([0.3, -1.0])
        # batch_dones = torch.tensor([False, True])
        # batch_nextobs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3
        # batch_action_samples = torch.tensor([[[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]], # 2 action samples each
        #                                      [[0.11, 0.22, 0.33, 0.44], [0.12, 0.25, 0.37, 0.50]]]) #action_size is 4
        # batch_log_probs = torch.tensor([[0.5, 0.8], [1.0, 2.0]])

        # # first compute the clipped double Q-target
        # list_batch_single_action_sample = [torch.squeeze(sngl_smpl, dim=1) for sngl_smpl in torch.split(batch_action_samples, 1, dim=1)]
        # self.assertEqual((2, 4), tuple(list_batch_single_action_sample[0].shape))
        # # compute the Q-value relative to each q-net
        # qnet1_target = torch.cat(
        #     [qnet1(obs=batch_nextobs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
        #     dim=1
        # )
        # qnet2_target = torch.cat(
        #     [qnet2(obs=batch_nextobs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
        #     dim=1
        # )
        # self.assertEqual((2, 2), tuple(qnet1_target.shape))
        # # take the minimum of the two q-net values for each action sample
        # target_min = torch.minimum(qnet1_target, qnet2_target)
        # self.assertEqual((2, 2), tuple(target_min.shape))
        # # average this minimum over all action samples
        # target_mean = torch.mean(target_min, dim=1, dtype=torch.float32)
        # self.assertEqual((2, ), tuple(target_mean.shape))

        # # then calculate the log probabilities for the actions
        # log_probs_mean = torch.mean(batch_log_probs, dim=1)
        # self.assertEqual((2, ), tuple(log_probs_mean.shape))

        # # finally compute the target value as a whole
        # targets = (
        #     batch_rewards + 
        #     DISCOUNT * #discount rate - as set at the beggining with SAC definition
        #     (1.0 - batch_dones.to(torch.float32)) * 
        #     (
        #     target_mean - 
        #     TEMPERATURE * 
        #     log_probs_mean
        #     )
        # )
        # self.assertEqual((2, ), tuple(targets.shape))

        # # compute the exact same process using _compute_qnet_target
        # actual = testSAC._compute_qnet_target(batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs)
        # # compare that result's shape with the shape of target which we just computed
        # self.assertEqual(tuple(targets.shape), tuple(actual.shape))
        pass
    
    def test_compute_policy_val(self):
        # TODO ONLY TESTING IF IT RUNS UNCOMMENT IT ONCE SAC IS FINALIZED
        # OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        # DISCOUNT, TEMPERATURE, TAU = 0.8, 0.6, 0.005

        # testSAC = SoftActorCritic(
        #     q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
        #     temperature=TEMPERATURE, qnet_update_smoothing_coefficient=TAU,
        #     obs_dim_size=OBSERVATION_SIZE, act_dim_size=ACTION_SIZE,
        #     act_ranges=((-1., 1.), (-2., 3.), (0., 1.), (-10., 10.)),
        #     pol_eval_batch_size=8, pol_imp_batch_size=8,
        #     update_qnet_every_N_gradient_steps=1
        #     )

        # qnet1 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)
        # qnet2 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)

        # # assuming a batch of two experiences
        # batch_obs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3
        # batch_action_samples = torch.tensor([[[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]], # 2 action samples each
        #                                      [[0.11, 0.22, 0.33, 0.44], [0.12, 0.25, 0.37, 0.50]]]) #action_size is 4
        
        # # first compute the clipped double Q-target
        # list_batch_single_action_sample = [torch.squeeze(sngl_smpl, dim=1) for sngl_smpl in torch.split(batch_action_samples, 1, dim=1)]
        # self.assertEqual((2, 4), tuple(list_batch_single_action_sample[0].shape))
        # # compute the Q-value relative to each q-net
        # qnet1_val = torch.cat(
        #     [qnet1(obs=batch_obs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
        #     dim=1
        # )
        # qnet2_val = torch.cat(
        #     [qnet2(obs=batch_obs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
        #     dim=1
        # )
        # self.assertEqual((2, 2), tuple(qnet1_val.shape))
        # # take the minimum of the two q-net values for each action sample
        # val_min = torch.minimum(qnet1_val, qnet2_val)
        # self.assertEqual((2, 2), tuple(val_min.shape))
        # # average this minimum over all action samples
        # val_mean = torch.mean(val_min, dim=1, dtype=torch.float32)
        # self.assertEqual((2, ), tuple(val_mean.shape))

        # # compute the exact same process using _compute_policy_val
        # actual = testSAC._compute_policy_val(batch_obs, batch_action_samples)
        # # compare that result's shape with the shape of target which we just computed
        # self.assertEqual(tuple(val_mean.shape), tuple(actual.shape))
        pass

    def test_update(self):
        # TODO ONLY TESTING IF IT RUNS
        # UNCOMMENT IT ONCE SAC IS FINALIZED
        # OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        # DISCOUNT, TEMPERATURE, TAU = 0.8, 0.6, 0.005

        # testSAC = SoftActorCritic(
        #     q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
        #     temperature=TEMPERATURE, qnet_update_smoothing_coefficient=TAU, 
        #     obs_dim_size=OBSERVATION_SIZE, act_dim_size=ACTION_SIZE,
        #     act_ranges=((-1., 1.), (-2., 3.), (0., 1.), (-10., 10.)),
        #     pol_eval_batch_size=8, pol_imp_batch_size=8,
        #     update_qnet_every_N_gradient_steps=1
        #     )

        # # assuming a batch of two experiences
        # batch_obs = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]) #observation_size is 3
        # batch_actions = np.array([[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]]) #action_size is 4
        # batch_rewards = [0.3, -1.0]
        # batch_dones = [False, True]
        # batch_nextobs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3

        # exp_list = [Experience(obs=batch_obs[i], action=batch_actions[i], reward=batch_rewards[i], 
        #                           done=batch_dones[i], next_obs=batch_nextobs[i]) for i in range(2)]
        # experiences = ListBuffer(max_size=1000, obs_shape=(OBSERVATION_SIZE, ), act_shape=(ACTION_SIZE, ))
        # experiences.append_experience(*exp_list[0])
        # experiences.append_experience(*exp_list[1])

        # new_batch_obs, new_batch_actions, new_batch_rewards, new_batch_dones, new_batch_nextobs = SoftActorCritic._unzip_experiences(experiences)
        
        # print("new_batch_obs: ",new_batch_obs, "new_batch_actions: ", new_batch_actions, 
        #       "new_batch_rewards: ", new_batch_rewards, "new_batch_dones: ", new_batch_dones, 
        #       "new_batch_nextobs: ",new_batch_nextobs)

        # testSAC.update(experiences)
        pass
    
if __name__ == "__main__":
    unittest.main()