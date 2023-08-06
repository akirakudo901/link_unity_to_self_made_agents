
import unittest

import numpy as np

from trainers.utils.experience import Experience

class TestExperience(unittest.TestCase):

    def test_is_equal(self):
        exp = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        exp_same = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        exp_obs_diff = Experience(obs=np.array([20., 21.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        exp_act_diff = Experience(obs=np.array([10., 11.]), action=np.array([22., 23., 24.]), reward=10.,
                          done=False, next_obs=np.array([15., 16.]))
        exp_rew_diff = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=20.,
                          done=False, next_obs=np.array([15., 16.]))
        exp_don_diff = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=True, next_obs=np.array([15., 16.]))
        exp_next_obs_diff = Experience(obs=np.array([10., 11.]), action=np.array([12., 13., 14.]), reward=10.,
                          done=False, next_obs=np.array([25., 26.]))
        
        exp_all_diff = Experience(obs=np.array([20., 21.]), action=np.array([22., 23., 24.]), reward=20.,
                          done=True, next_obs=np.array([25., 26.]))
        
        self.assertTrue(exp.is_equal(exp_same))
        self.assertFalse(exp.is_equal(exp_obs_diff))
        self.assertFalse(exp.is_equal(exp_act_diff))
        self.assertFalse(exp.is_equal(exp_rew_diff))
        self.assertFalse(exp.is_equal(exp_don_diff))
        self.assertFalse(exp.is_equal(exp_next_obs_diff))
        self.assertFalse(exp.is_equal(exp_all_diff))

if __name__ == "__main__":
    unittest.main()