#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset()
plt.imshow(env.render())
plt.show()
