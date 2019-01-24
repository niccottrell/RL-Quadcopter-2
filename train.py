import sys
import numpy as np
import pandas as pd
from agents.agent import Nic_Agent
from task import Task

num_episodes = 10000
target_pos = np.array([2., 4., 10.])  #
task = Task(target_pos=target_pos)
agent = Nic_Agent(task)

for i_episode in range(1, num_episodes + 1):
    state = agent.reset_episode()  # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(action, reward, next_state, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d},  reward = {:7.3f}".format(
                i_episode, reward), end="")  # [debug]
            break
    sys.stdout.flush()
