import unittest

import train
from task import Task, REWARD_MAX

from train import *


class TestTask(unittest.TestCase):

    def test_init(self):
        """
        Test from initial position
        """

        task = Task(train.init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
        print("target_pos: %s", target_pos)

        task.sim.pose = init_pose
        reward_init = task.get_reward()
        self.assertGreaterEqual(round(reward_init), 8.)
        # Test from goal state
        task.sim.pose = target_pos + list([0, 0, 0])
        task.sim.v = np.array([0.0, 0.0, 0.0])  # no velocity
        reward_no_vol = task.get_reward()
        self.assertGreaterEqual(reward_no_vol, 9)
        task.sim.v = np.array([0.1, 0.1, 0.1])  # low velocity
        reward_low_vol = task.get_reward()
        self.assertGreater(reward_low_vol, 15)
        task.sim.v = np.array([1., 1., 1.])  # high velocity
        reward_high_vol = task.get_reward()
        self.assertGreater(reward_high_vol, 5)
        # check consistency
        self.assertGreater(reward_no_vol, reward_low_vol)
        self.assertGreater(reward_low_vol, reward_high_vol)

    def test_negs(self):
        """
        Test from initial position
        """

        target_neg = np.array([-0.4, 0.2, 10.])
        task = Task(train.init_pose, init_velocities, init_angle_velocities, runtime, target_neg)
        print("target_pos: %s", target_neg)

        task.sim.pose = init_pose
        reward_init = task.get_reward()
        self.assertGreaterEqual(round(reward_init), 8.)
        # Test from goal state
        task.sim.pose = target_neg + list([0, 0, 0])
        task.sim.v = np.array([0.0, 0.0, 0.0])  # no velocity
        reward_no_vol = task.get_reward()
        print("reward_no_vol=%d" % reward_no_vol)
        self.assertGreaterEqual(reward_no_vol, 9)
        task.sim.v = np.array([0.1, 0.1, 0.1])  # low velocity
        reward_low_vol = task.get_reward()
        print("reward_low_vol=%d" % reward_low_vol)
        self.assertGreater(reward_low_vol, 15)
        task.sim.v = np.array([1., 1., 1.])  # high velocity
        reward_high_vol = task.get_reward()
        print("reward_high_vol=%d" % reward_high_vol)
        self.assertGreater(reward_high_vol, 5)
        # check consistency
        self.assertGreater(reward_no_vol, reward_low_vol)
        self.assertGreater(reward_low_vol, reward_high_vol)

    def test_others(self):
        task = Task(train.init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
        print("target_pos: %s", target_pos)

        task.sim.pose = np.array([2., 4., 6.])
        task.sim.v = np.array([1., 2., 3.])
        reward_crazy = task.get_reward()
        print("reward_crazy = %d", reward_crazy)
        self.assertLess(reward_crazy, -15)
        self.assertGreaterEqual(reward_crazy, -REWARD_MAX)
        task.sim.pose = np.array([3., 1., 6.])
        task.sim.v = np.array([10., 0., 1.])
        reward_crazy2 = task.get_reward()
        print("reward_crazy2 = %d", reward_crazy2)
        self.assertLess(reward_crazy2, -15)
        self.assertGreaterEqual(reward_crazy2, -REWARD_MAX)


if __name__ == '__main__':
    unittest.main()
