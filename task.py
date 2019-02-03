import numpy
import numpy as np
from physics_sim import PhysicsSim

REWARD_MAX = 2.

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
        self.action_low = 200  # Never spin too slow
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def computeDistance(self, a, b):
        dist = numpy.linalg.norm(a - b)
        return dist

    def get_reward(self):
        """
        Uses current pose of sim to return reward.
        """
        current_pos = self.sim.pose[:3]
        # Get the distance to the target
        distance = self.computeDistance(current_pos, self.target_pos)

        # Find to difference along the z-axis [z2 - z1]
        # create a bonus factor for current z being equal or greater than target z
        # Negative value for z_diff if the copter is above the target z.
        z_diff = self.target_pos[2] - self.sim.pose[2]
        z_factor = 1.2 if z_diff <= 0 else 1.0

        # penalize angular velocity (avoid spinning)
        angular_sum = 0.1 * (abs(self.sim.angular_v)).sum()

        # reward low absolute velocity - we want to hover in position (this would punish
        v__sum = 0.05 * (abs(self.sim.v)).sum()

        reward = (1 / (1 + distance + angular_sum + v__sum)) * z_factor

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

