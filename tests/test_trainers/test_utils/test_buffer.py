
import unittest

import numpy as np

from trainers.utils.buffer import ListBuffer, NdArrayBuffer
from trainers.utils.experience import Experience

class TestListBuffer(unittest.TestCase):

    def setUp(self) -> None:
        self.buffer_class = ListBuffer
        self.MAX_SIZE = 3
        self.OBS_SHAPE, self.ACT_SHAPE = (2, ), (3, )
        self.buffer = self.buffer_class(max_size=self.MAX_SIZE, obs_shape=self.OBS_SHAPE, act_shape=self.ACT_SHAPE)

        self.exp1 = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        self.exp2 = Experience(obs=np.array([20., 21.]), action=np.array([22., 23., 24.]), reward=20.,
                          done=False, next_obs=np.array([25., 26.]))
        self.exp3 = Experience(obs=np.array([30., 31.]), action=np.array([32., 33., 34.]), reward=30.,
                          done=True, next_obs=np.array([35., 36.]))
        self.exp4 = Experience(obs=np.array([40., 41.]), action=np.array([42., 43., 44.]), reward=40.,
                          done=True, next_obs=np.array([45., 46.]))
        
    def test_size(self):
        # initial condition
        self.assertEqual(self.buffer.size(), 0)
        # add one experience
        self.buffer.append_experience(self.exp1.obs, self.exp1.action, self.exp1.reward, self.exp1.done, self.exp1.next_obs)
        self.assertEqual(self.buffer.size(), 1)
        # add two experience
        self.buffer.append_experience(*self.exp1)
        self.assertEqual(self.buffer.size(), 2)
        # add third (and last holdable) experience
        self.buffer.append_experience(*self.exp1)
        self.assertEqual(self.buffer.size(), 3)
        # add fourth experience, which should not increase the size since it reached max
        self.buffer.append_experience(*self.exp1)
        self.assertEqual(self.buffer.size(), 3)
    
    def test_append_buffer(self):
        # initial condition
        self.assertEqual(self.buffer.size(), 0)
        # add first experience
        self.buffer.append_experience(*self.exp1)
        exp_in_buffer = self.buffer._list[0]
        self.assertEqual(self.buffer.size(), 1)
        self.assertTrue(self.exp1.is_equal(exp_in_buffer))
        # add two more to fill the buffer
        self.buffer.append_experience(*self.exp2)
        self.assertEqual(self.buffer.size(), 2)
        self.assertTrue(self.exp1.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp2.is_equal(self.buffer._list[1]))

        self.buffer.append_experience(*self.exp3)
        self.assertEqual(self.buffer.size(), 3)
        self.assertTrue(self.exp1.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp2.is_equal(self.buffer._list[1]))
        self.assertTrue(self.exp3.is_equal(self.buffer._list[2]))
        # add fourth which should be added to the end, and the first experience is removed
        self.buffer.append_experience(*self.exp4)
        self.assertEqual(self.buffer.size(), 3)
        self.assertFalse(self.exp1.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp2.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp3.is_equal(self.buffer._list[1]))
        self.assertTrue(self.exp4.is_equal(self.buffer._list[2]))
        # double check for fifth
        self.buffer.append_experience(*self.exp4)
        self.assertEqual(self.buffer.size(), 3)
        self.assertFalse(self.exp2.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp3.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp4.is_equal(self.buffer._list[1]))
        self.assertTrue(self.exp4.is_equal(self.buffer._list[2]))

    def test_extend_buffer(self):
        # initial condition
        self.assertEqual(self.buffer.size(), 0)
        # create a second buffer with one experience
        buffer2 = self.buffer_class(max_size=self.MAX_SIZE, obs_shape=self.OBS_SHAPE, act_shape=self.ACT_SHAPE)
        buffer2.append_experience(*self.exp1)
        # extend self.buffer with experiences from buffer2
        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 1)
        self.assertTrue(self.exp1.is_equal(self.buffer._list[0]))
        # extend self.buffer with buffer 2 having two experiences
        buffer2.append_experience(*self.exp2)
        self.assertEqual(buffer2.size(), 2)

        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 3)
        self.assertTrue(self.exp1.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp1.is_equal(self.buffer._list[1]))
        self.assertTrue(self.exp2.is_equal(self.buffer._list[2]))
        # extend self.buffer with buffer2 having three experiences which leads to exceeding max capacity
        buffer2.append_experience(*self.exp3)
        self.assertEqual(buffer2.size(), 3)

        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 3)
        self.assertTrue(self.exp1.is_equal(self.buffer._list[0]))
        self.assertTrue(self.exp2.is_equal(self.buffer._list[1]))
        self.assertTrue(self.exp3.is_equal(self.buffer._list[2]))

    def test_get_components(self):
        self.buffer.append_experience(*self.exp1)
        self.buffer.append_experience(*self.exp2)
        # get individual components from self.buffer
        obs, act, rew, don, next_obs = self.buffer.get_components()
        expected_obs, expected_act, expected_rew, expected_don, expected_next_obs = self.exp1
        expected_obs      = np.stack((expected_obs, self.exp2.obs), axis=0)
        expected_act      = np.stack((expected_act, self.exp2.action), axis=0)
        expected_rew      = np.stack((expected_rew, self.exp2.reward), axis=0)
        expected_don      = np.stack((expected_don, self.exp2.done), axis=0)
        expected_next_obs = np.stack((expected_next_obs, self.exp2.next_obs), axis=0)

        self.assertTrue(np.array_equal(obs, expected_obs))
        self.assertTrue(np.array_equal(act, expected_act))
        self.assertTrue(np.array_equal(rew, expected_rew))
        self.assertTrue(np.array_equal(don, expected_don))
        self.assertTrue(np.array_equal(next_obs, expected_next_obs))
    
    def test_sample_random_experiences(self):
        # append 3 experiences - which fills self.buffer
        self.buffer.append_experience(*self.exp1)
        self.buffer.append_experience(*self.exp2)
        self.buffer.append_experience(*self.exp3)
        # sample 1 random experience and see that it differs based on seeds
        obs1, act1, rew1, don1, nob1 = self.buffer.sample_random_experiences(num_samples=1, seed=123)
        obs2,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=246)
        obs3,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=369)
        obs4,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=999)
        obs5,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=989)
        obs6,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=899)
        obs7,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=226)
        obs8,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=31)
        obs9,    _,    _,    _,    _ = self.buffer.sample_random_experiences(num_samples=1, seed=341)
        self.assertEqual(obs1.shape[0], 1)
        self.assertEqual(act1.shape[0], 1)
        self.assertEqual(rew1.shape[0], 1)
        self.assertEqual(don1.shape[0], 1)
        self.assertEqual(nob1.shape[0], 1)
        self.assertTrue((not np.array_equal(obs1, obs2)) or (not np.array_equal(obs1, obs3)) or
                        (not np.array_equal(obs1, obs4)) or (not np.array_equal(obs1, obs5)) or
                        (not np.array_equal(obs1, obs6)) or (not np.array_equal(obs1, obs7)) or
                        (not np.array_equal(obs1, obs8)) or (not np.array_equal(obs1, obs9)) or 
                        (not np.array_equal(obs2, obs3)))
        # see that 2 random experiences are indeed generated
        obs4, _, _, _, _ = self.buffer.sample_random_experiences(num_samples=2, seed=123)
        self.assertEqual(obs4.shape[0], 2)
        # see that if num_samples > buffer size, we get buffer size experiences
        self.assertEqual(self.buffer.size(), 3)
        obs5, _, _, _, _ = self.buffer.sample_random_experiences(num_samples=10, seed=123)
        self.assertEqual(obs5.shape[0], 3)
        print("WARNING: SINCE THIS IS AN INCOMPLETE TEST WHICH PASSES ONLY WITH HIGH PROBABILITY, " + 
              "THERE MIGHT BE CASES WHERE THIS FAIL - RERUNNING IT AGAIN SHOULD WORK THOUGH.")
        

class TestNdArrayBuffer(TestListBuffer):

    def setUp(self) -> None:
        self.buffer_class = NdArrayBuffer
        self.MAX_SIZE = 3
        self.OBS_SHAPE, self.ACT_SHAPE = (2, ), (3, )
        self.buffer = self.buffer_class(max_size=self.MAX_SIZE, obs_shape=self.OBS_SHAPE, act_shape=self.ACT_SHAPE)

        self.exp1 = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        self.exp2 = Experience(obs=np.array([20., 21.]), action=np.array([22., 23., 24.]), reward=20.,
                          done=False, next_obs=np.array([25., 26.]))
        self.exp3 = Experience(obs=np.array([30., 31.]), action=np.array([32., 33., 34.]), reward=30.,
                          done=True, next_obs=np.array([35., 36.]))
        self.exp4 = Experience(obs=np.array([40., 41.]), action=np.array([42., 43., 44.]), reward=40.,
                          done=True, next_obs=np.array([45., 46.]))
    
    def test_append_buffer(self):
        def get_exp_at_i(i):
            return Experience(obs=self.buffer.obs[i], action=self.buffer.actions[i], 
                              reward=self.buffer.rewards[i], done=self.buffer.dones[i], 
                              next_obs=self.buffer.next_obs[i])
        
        # initial condition
        self.assertEqual(self.buffer.size(), 0)
        self.assertEqual(self.buffer._ptr, 0)
        # add first experience
        self.buffer.append_experience(*self.exp1)
        exp_in_buffer = get_exp_at_i(0)
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(self.buffer._ptr, 1)
        self.assertTrue(self.exp1.is_equal(exp_in_buffer))
        # add two more to fill the buffer
        self.buffer.append_experience(*self.exp2)
        self.assertEqual(self.buffer.size(), 2)
        self.assertEqual(self.buffer._ptr, 2)
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp2.is_equal(get_exp_at_i(1)))

        self.buffer.append_experience(*self.exp3)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 0)
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp2.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp3.is_equal(get_exp_at_i(2)))
        # add fourth which should be added to the end, and the first experience is removed
        self.buffer.append_experience(*self.exp4)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 1)
        self.assertFalse(self.exp1.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp4.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp2.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp3.is_equal(get_exp_at_i(2)))
        # double check for fifth
        self.buffer.append_experience(*self.exp4)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 2)
        self.assertTrue(self.exp4.is_equal(get_exp_at_i(0)))
        self.assertFalse(self.exp2.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp4.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp3.is_equal(get_exp_at_i(2)))

    def test_extend_buffer(self):
        def get_exp_at_i(i):
            return Experience(obs=self.buffer.obs[i], action=self.buffer.actions[i], 
                              reward=self.buffer.rewards[i], done=self.buffer.dones[i], 
                              next_obs=self.buffer.next_obs[i])
        # initial condition
        self.assertEqual(self.buffer.size(), 0)
        self.assertEqual(self.buffer._ptr, 0)
        # create a second buffer with one experience
        buffer2 = self.buffer_class(max_size=8, obs_shape=self.OBS_SHAPE, act_shape=self.ACT_SHAPE)
        buffer2.append_experience(*self.exp1)
        # extend self.buffer with experiences from buffer2
        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(self.buffer._ptr, 1)
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(0)))
        # extend self.buffer with buffer 2 having two experiences
        buffer2.append_experience(*self.exp2)
        self.assertEqual(buffer2.size(), 2)

        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 0)
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp2.is_equal(get_exp_at_i(2)))
        # extend self.buffer with buffer2 having three experiences which leads to exceeding max capacity
        buffer2.append_experience(*self.exp3)
        self.assertEqual(buffer2.size(), 3)

        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 0)
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp2.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp3.is_equal(get_exp_at_i(2)))

        # extend self.buffer with buffer2 having eight experiences which leads to multiples wrap arounds
        buffer2.append_experience(*self.exp4)
        buffer2.append_experience(*self.exp2)
        buffer2.append_experience(*self.exp3)
        buffer2.append_experience(*self.exp4)
        buffer2.append_experience(*self.exp1)
        self.assertEqual(buffer2.size(), 8)

        self.buffer.extend_buffer(buffer2)
        self.assertEqual(self.buffer.size(), 3)
        self.assertEqual(self.buffer._ptr, 2)
        self.assertTrue(self.exp4.is_equal(get_exp_at_i(0)))
        self.assertTrue(self.exp1.is_equal(get_exp_at_i(1)))
        self.assertTrue(self.exp3.is_equal(get_exp_at_i(2)))


if __name__ == "__main__":
    unittest.main()