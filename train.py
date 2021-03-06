import sys
import traceback

import numpy as np
import csv
import matplotlib.pyplot as plt

from agents.agent import Nic_Agent
from task import Task

num_episodes = 20000
runtime = 10.  # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0.01, 0.02, 0.01])  # initial velocities
init_angle_velocities = np.array([0.03, 0.02, 0.01])  # initial angle velocities
target_pos = np.array([0.2, 0.2, 12.])


def train_plot():
    """
    Train the actor/critic models
    """

    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos)
    agent = Nic_Agent(task)

    # Train the agent
    training = {x: [] for x in ['episode_num', 'reward']}
    with open("training.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['episode_num', 'reward'])
        for i_episode in range(1, num_episodes + 1):
            state = agent.reset_episode()  # reset the agent and task (and sim)
            while True:
                action = agent.act(state)  # Agent decides on rotor speeds
                next_state, reward, done = task.step(action)  # Move a bit based on this rotor speed
                # print("\rNext state = {},  reward = {} ".format(next_state, reward), end="")  # [debug]
                agent.step(action, reward, next_state, done)  # Save and learn
                state = next_state
                if done:
                    print("\rEpisode = {:4d},  reward = {:7.3f}, time = {:4.3f}, reason = {} "
                          .format(i_episode, reward, task.sim.time, task.sim.done_cause), end="")  # [debug]
                    writer.writerow([i_episode, reward])
                    training['episode_num'].append(i_episode)
                    training['reward'].append(reward)
                    break
            sys.stdout.flush()

    # Plot the training
    plt.plot(training['episode_num'], training['reward'], label='reward')
    plt.legend()
    # _ = plt.ylim()
    plt.savefig('training_plot.png')
    plt.show()

    # Run the simulation
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi',
              'x_velocity', 'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity', 'psi_velocity',
              'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x: [] for x in labels}

    state = agent.reset_episode()
    with open("sim.csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act(state)
            _, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(
                rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break

    # Plot the position
    plt.clf()
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    plt.savefig('sim_position.png')
    # _ = plt.ylim()
    plt.show()

    plt.clf()
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    plt.legend()
    # _ = plt.ylim()
    plt.savefig('sim_speeds.png')


if __name__ == '__main__':
    train_plot()
