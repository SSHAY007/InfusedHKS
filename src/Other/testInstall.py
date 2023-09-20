#!/usr/bin/env python3
import gym
import matplotlib.pyplot as plt
env = gym.make('Humanoid-v4', render_mode="rgb_array")
env.reset()
print(env.render())
#plt.imshow(env.render())
#plt.show()
