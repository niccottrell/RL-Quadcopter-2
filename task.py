import math

import numpy as np
from physics_sim import PhysicsSim

REWARD_MAX = 30.


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None,
                 init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 100  # Never spin too slow
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """
        Uses current pose of sim to return reward.
        """
        current_pos = self.sim.pose[:3]
        pos_diff_sum = abs(current_pos - self.target_pos).sum()
        reward = REWARD_MAX * (1. / (1 + math.log(1 + pos_diff_sum)))  # as difference gets bigger this closer to zero
        # penalize angular velocity (avoid spinning)
        reward -= 3. * (abs(self.sim.angular_v)).sum()
        # reward low absolute velocity - we want to hover in position
        v__sum = 4. * (abs(self.sim.v)).sum()
        reward -= v__sum
        # clip the reward so that it's never too low
        if reward < -REWARD_MAX: reward = -REWARD_MAX
        if reward > REWARD_MAX: reward = REWARD_MAX
        return reward

    def step(self, rotor_speeds):
        """
        Uses action to obtain next state, reward, done.
        Where next_state is an ndarray.
        """
        done = None
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)  # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
