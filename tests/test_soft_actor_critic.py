
import unittest

import numpy as np
import torch

from policy_learning_algorithms.soft_actor_critic import SoftActorCritic
from trainers.unityenv_base_trainer import Experience

class TestSoftActorCritic(unittest.TestCase):
       
    def test_SoftActorCritic_unzip_experiences(self):
        experiences1 = [
            Experience(obs=np.array([0,1,2]), action=np.array([50,51,52]), reward=0.5, done=False, next_obs=[20,21,22]),
            Experience(obs=np.array([3,4,5]), action=np.array([53,54,55]), reward=0.7, done=False, next_obs=[23,24,25]),
            Experience(obs=np.array([6,7,8]), action=np.array([56,57,58]), reward=0.9, done=True , next_obs=[26,27,28]),
            ]
        exp_obs1     = torch.tensor([[0,1,2], [3,4,5], [6,7,8]])
        exp_acts1    = torch.tensor([[50,51,52], [53,54,55], [56,57,58]])
        exp_rews1    = torch.tensor([0.5, 0.7, 0.9])
        exp_dons1    = torch.tensor([False, False, True])
        exp_nex_obs1 = torch.tensor([[20,21,22], [23,24,25], [26,27,28]])

        act_obs1, act_acts1, act_rews1, act_dons1, act_nex_obs1 = SoftActorCritic._unzip_experiences(experiences1)

        # TODO would suggest to double check typing around here; will cast mindlessly for
        # now due to lack of knowledge, but there might be better types to set
        exp_obs1 = exp_obs1.to(act_obs1.dtype)
        exp_acts1 = exp_acts1.to(act_acts1.dtype)
        exp_rews1 = exp_rews1.to(act_rews1.dtype)
        exp_dons1 = exp_dons1.to(act_dons1.dtype)
        exp_nex_obs1 = exp_nex_obs1.to(act_nex_obs1.dtype)

        self.assertTrue(torch.equal(exp_obs1,         act_obs1))
        self.assertTrue(torch.equal(exp_acts1,       act_acts1))
        self.assertTrue(torch.allclose(exp_rews1,       act_rews1))
        self.assertTrue(torch.equal(exp_dons1,       act_dons1))
        self.assertTrue(torch.equal(exp_nex_obs1, act_nex_obs1))


class TestSoftActorCritic_Policy(unittest.TestCase):
    """
    Mostly non-formal test suit - only checks mostly for shapes of tensors in between,
    and prints the intermediate tensors for checking.
    """

    def test_Policy_correct_for_squash(self): 
        before1 = torch.tensor([[0.3, 0.9, 0.95], [0.5, 0.7, 0.75]])
        actions1 = torch.tensor([[[ 0.5,  0.6,  0.7,  0.8], [ 0.1,  0.2,  0.3,  0.4], [0.41, 0.52, 0.63, 0.74]], 
                                 [[0.33, 0.44, 0.55, 0.66], [0.12, 0.24, 0.36, 0.48], [0.60, 0.72, 0.84, 0.96]]])
        # print("before1: ", before1, "before1.shape: ", before1.shape, "\n") 
        # print("actions1: ", actions1, "actions1.shape: ", actions1.shape, "\n")
        jacobian_diagonals1 = (1 - torch.tanh(actions1)**2)
        # print("jacobian_diagonals1: ", jacobian_diagonals1, "jacobian_diagonals1.shape: ", jacobian_diagonals1.shape, "\n")
        jacobian_trace1 = torch.sum(jacobian_diagonals1, dim=2)
        # print("jacobian_trace1: ", jacobian_trace1, "jacobian_trace1.shape: ", jacobian_trace1.shape, "\n")
        expected1 = before1 - jacobian_trace1
        # print("expected1: ", expected1, "expected1.shape: ", expected1.shape, "\n")

        actual1 = SoftActorCritic.Policy._correct_for_squash(before=before1, actions=actions1)
        # print("actual1: ", actual1, "actual1s.shape: ", actual1.shape, "\n")
        
        self.assertTrue(torch.equal(expected1, actual1))

    def test_forward(self):
        example_policy = SoftActorCritic.Policy(observation_size=3, action_size=7)
        obs1 = torch.tensor([[0.0,1.0,2.0], [3.0,4.0,5.0]]) #[batch=2, elem=3]
        squashed, log_probs = example_policy(obs=obs1, num_samples=5, deterministic=False)
        # print("squashed: ", squashed, "squashed.shape: ", squashed.shape)
        # print("log_probs: ", log_probs, "log_probs.shape: ", log_probs.shape)
        self.assertEqual((2, 5, 7), tuple(squashed.shape))
        self.assertEqual((2, 5), tuple(log_probs.shape))

        myus = example_policy(obs=obs1, num_samples=5, deterministic=True)
        # print("myus: ", myus)
        self.assertEqual((2, 7), tuple(myus.shape))
    
    def test_compute_qnet_target(self):
        OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        DISCOUNT, TEMPERATURE = 0.8, 0.6

        testSAC = SoftActorCritic(
            q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
            temperature=TEMPERATURE, observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE, 
            device="cpu")

        qnet1 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)
        qnet2 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)

        # assuming a batch of two experiences
        batch_rewards = torch.tensor([0.3, -1.0])
        batch_dones = torch.tensor([False, True])
        batch_nextobs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3
        batch_action_samples = torch.tensor([[[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]], # 2 action samples each
                                             [[0.11, 0.22, 0.33, 0.44], [0.12, 0.25, 0.37, 0.50]]]) #action_size is 4
        batch_log_probs = torch.tensor([[0.5, 0.8], [1.0, 2.0]])

        # first compute the clipped double Q-target
        list_batch_single_action_sample = [torch.squeeze(sngl_smpl, dim=1) for sngl_smpl in torch.split(batch_action_samples, 1, dim=1)]
        # print("list_batch_single_action_sample[0]: ", list_batch_single_action_sample[0])
        # print("list_batch_single_action_sample[0].shape: ", list_batch_single_action_sample[0].shape)
        self.assertEqual((2, 4), tuple(list_batch_single_action_sample[0].shape))
        # compute the Q-value relative to each q-net
        qnet1_target = torch.cat(
            [qnet1(obs=batch_nextobs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
            dim=1
        )
        qnet2_target = torch.cat(
            [qnet2(obs=batch_nextobs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
            dim=1
        )
        # print("qnet1_target: ", qnet1_target, "qnet1_target.shape: ", qnet1_target.shape)
        self.assertEqual((2, 2), tuple(qnet1_target.shape))
        # take the minimum of the two q-net values for each action sample
        target_min = torch.minimum(qnet1_target, qnet2_target)
        # print("target_min: ", target_min, "target_min.shape: ", target_min.shape)
        self.assertEqual((2, 2), tuple(target_min.shape))
        # average this minimum over all action samples
        target_mean = torch.mean(target_min, dim=1, dtype=torch.float32)
        # print("target_mean: ", target_mean, "target_mean.shape: ", target_mean.shape)
        self.assertEqual((2, ), tuple(target_mean.shape))

        # then calculate the log probabilities for the actions
        log_probs_mean = torch.mean(batch_log_probs, dim=1)
        # print("log_probs_mean: ", log_probs_mean, "log_probs_mean.shape: ", log_probs_mean.shape)
        self.assertEqual((2, ), tuple(log_probs_mean.shape))

        # finally compute the target value as a whole
        targets = (
            batch_rewards + 
            DISCOUNT * #discount rate - as set at the beggining with SAC definition
            (1.0 - batch_dones.to(torch.float32)) * 
            (
            target_mean - 
            TEMPERATURE * 
            log_probs_mean
            )
        )
        # print("targets: ", targets, "targets.shape: ", targets.shape)
        self.assertEqual((2, ), tuple(targets.shape))

        # compute the exact same process using _compute_qnet_target
        actual = testSAC._compute_qnet_target(batch_rewards, batch_dones, batch_nextobs, batch_action_samples, batch_log_probs)
        # compare that result's shape with the shape of target which we just computed
        self.assertEqual(tuple(targets.shape), tuple(actual.shape))
    
    def test_compute_exponentiated_qval(self):
        OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        DISCOUNT, TEMPERATURE = 0.8, 0.6

        testSAC = SoftActorCritic(
            q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
            temperature=TEMPERATURE, observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
            device="cpu")

        qnet1 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)
        qnet2 = SoftActorCritic.QNet(observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE)

        # assuming a batch of two experiences
        batch_obs = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3
        batch_action_samples = torch.tensor([[[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]], # 2 action samples each
                                             [[0.11, 0.22, 0.33, 0.44], [0.12, 0.25, 0.37, 0.50]]]) #action_size is 4
        
        # first compute the clipped double Q-target
        list_batch_single_action_sample = [torch.squeeze(sngl_smpl, dim=1) for sngl_smpl in torch.split(batch_action_samples, 1, dim=1)]
        # print("list_batch_single_action_sample[0]: ", list_batch_single_action_sample[0])
        # print("list_batch_single_action_sample[0].shape: ", list_batch_single_action_sample[0].shape)
        self.assertEqual((2, 4), tuple(list_batch_single_action_sample[0].shape))
        # compute the Q-value relative to each q-net
        qnet1_exp_val = torch.cat(
            [qnet1(obs=batch_obs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
            dim=1
        )
        qnet2_exp_val = torch.cat(
            [qnet2(obs=batch_obs, actions=batch_single_action) for batch_single_action in list_batch_single_action_sample],
            dim=1
        )
        # print("qnet1_exp_val: ", qnet1_exp_val, "qnet1_exp_val.shape: ", qnet1_exp_val.shape)
        self.assertEqual((2, 2), tuple(qnet1_exp_val.shape))
        # take the minimum of the two q-net values for each action sample
        exp_val_min = torch.minimum(qnet1_exp_val, qnet2_exp_val)
        # print("exp_val_min: ", exp_val_min, "exp_val_min.shape: ", exp_val_min.shape)
        self.assertEqual((2, 2), tuple(exp_val_min.shape))
        # average this minimum over all action samples
        exp_val_mean = torch.mean(exp_val_min, dim=1, dtype=torch.float32)
        # print("exp_val_mean: ", exp_val_mean, "exp_val_mean.shape: ", exp_val_mean.shape)
        self.assertEqual((2, ), tuple(exp_val_mean.shape))

        
        # compute the exact same process using _compute_exponentiated_qval
        actual = testSAC._compute_exponentiated_qval(batch_obs, batch_action_samples)
        # compare that result's shape with the shape of target which we just computed
        self.assertEqual(tuple(exp_val_mean.shape), tuple(actual.shape))

    def test_update(self):
        OBSERVATION_SIZE, ACTION_SIZE = 3, 4
        DISCOUNT, TEMPERATURE = 0.8, 0.6

        testSAC = SoftActorCritic(
            q_net_learning_rate=1e-3, policy_learning_rate=1e-3, discount=DISCOUNT, 
            temperature=TEMPERATURE, observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
            device="cpu")

        # assuming a batch of two experiences
        batch_obs = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]) #observation_size is 3
        batch_actions = np.array([[0.3, 0.4, 0.5, 0.6], [0.15, 0.16, 0.17, 0.18]]) #action_size is 4
        batch_rewards = [0.3, -1.0]
        batch_dones = [False, True]
        batch_nextobs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) #observation_size is 3

        experiences = [Experience(obs=batch_obs[i], action=batch_actions[i], reward=batch_rewards[i], 
                                  done=batch_dones[i], next_obs=batch_nextobs[i]) for i in range(2)]
        
        new_batch_obs, new_batch_actions, new_batch_rewards, new_batch_dones, new_batch_nextobs = SoftActorCritic._unzip_experiences(experiences)
        
        print("new_batch_obs: ",new_batch_obs, "new_batch_actions: ", new_batch_actions, 
              "new_batch_rewards: ", new_batch_rewards, "new_batch_dones: ", new_batch_dones, 
              "new_batch_nextobs: ",new_batch_nextobs)

        testSAC.update(experiences)
        
    
if __name__ == "__main__":
    unittest.main()