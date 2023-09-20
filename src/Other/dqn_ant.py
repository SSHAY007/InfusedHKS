#!/usr/bin/env python3


#!/usr/bin/env python3
from __future__ import annotations
from gym.wrappers.record_video import RecordVideo
import random
from IPython.display import HTML
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
import pandas as pd
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import time
import glob
import io
import base64
import seaborn as sns
import gym
import tensorflow as tf
from tqdm import tqdm


#display = Display(visible=0, size=(1400, 900))
#display.start()

#plt.rcParams["figure.figsize"] = (10, 5)
#env = gym.make('Pendulum-v4')
env = gym.make("Ant-v2",render_mode="rgb_array")
#wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
#env = gym.make('CartPole-v1', render_mode="rgb_array")
num_actions = env.action_space.shape[0]
###env = RecordVideo(env, 'video', episode_trigger = lambda x: x == 1)
#num_features = env.observation_space.shape[0]
#num_actions = env.action_space.n

#print('Number of state features: {}'.format(num_features))
print('Number of possible actions: {}'.format(num_actions))

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()



class DQN(tf.keras.Model):
  """Dense neural network class."""
  def __init__(self,num_actions):
    super(DQN, self).__init__()
    self.dense1 = tf.keras.layers.Dense(32, activation="relu")
    self.dense2 = tf.keras.layers.Dense(32, activation="relu")
    self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32) # No activation

  def call(self, x):
    """Forward pass."""
    x = self.dense1(x)
    x = self.dense2(x)
    return self.dense3(x)

print(f"action space is {num_actions}")
main_nn = DQN(num_actions)
target_nn = DQN(num_actions)

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones

class PendulumAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_epsilon_greedy(self,state) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample() #random action from action space

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return tf.argmax(main_nn(state)[0].numpy())




    def train_step(self,states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer.
        Training your function approximation

        """
        # Calculate targets.
        next_qs = target_nn(next_states)
        max_next_qs = tf.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.discount_factor * max_next_qs
        # For autodifferentiation
        with tf.GradientTape() as tape:
            qs = main_nn(states)
            action_masks = tf.one_hot(actions, num_actions)
            masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
            loss = mse(target, masked_qs)
        #Calculate gradient
        grads = tape.gradient(loss, main_nn.trainable_variables)
        optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))

        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


#env = gym.make("InvertedPendulum-v4")
#env = gym.make("CartPole-v1")




# Hyperparameters.
num_episodes = 1000
start_epsilon = 1.0
batch_size = 32
discount = 0.99
epsilon_decay = start_epsilon / (num_episodes / 2)  # reduce the exploration over time
buffer = ReplayBuffer(100000)
cur_frame = 0
final_epsilon = 0.1
learning_rate = 0.01
# Start training. Play game once and then train with a batch.
log_rewards = []
log_loss = []

agent = PendulumAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(num_episodes+1)):
  state ,info= env.reset()
  ep_reward, done = 0, False
  while not done:
    #print(f"The state in is{state[0]}")
    state_in = tf.expand_dims(state, axis=0)#makes [x,y,z] to [[x,y,z]]
    #print(f"The state in is{state_in}")
    action = agent.get_epsilon_greedy(state_in)
    #print(f"The step is{env.step(action)}")
    next_state, reward, done,truncated, info = env.step(action)
    ep_reward += reward
    # Save to experience replay.
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    cur_frame += 1
    # Copy main_nn weights to target_nn.
    if cur_frame % 2000 == 0:
      target_nn.set_weights(main_nn.get_weights())

    # Train neural network.
    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      loss = agent.train_step(states, actions, rewards, next_states, dones)
      log_loss.append(loss.numpy())
      #print(loss.numpy())

  # if episode < 950:
  #   epsilon -= 0.001

  if len(log_rewards) == 50:
    log_rewards = log_rewards[1:]
  log_rewards.append(ep_reward)

  if episode % 100 == 0:
    print(f'Episode {episode}/{num_episodes}. '
          f'Reward in last 100 episodes: {np.mean(log_rewards):.3f}')
env.close()




# rolling_length = 500
# fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
# axs[0].set_title("Episode rewards")
# # compute and assign a rolling average of the data to provide a smoother graph
# reward_moving_average = (
#     np.convolve(
#         np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
#     )
#     / rolling_length
# )
# axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
# axs[1].set_title("Episode lengths")
# length_moving_average = (
#     np.convolve(
#         np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
#     )
#     / rolling_length
# )
# axs[1].plot(range(len(length_moving_average)), length_moving_average)
# axs[2].set_title("Training Error")
# training_error_moving_average = (
#     np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
#     / rolling_length
# )
# axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
# plt.tight_layout()
# plt.show()


x = np.arange(len(log_rewards))
rewards_to_plot =log_rewards
df1 = pd.DataFrame({"variable":x, "value":rewards_to_plot})
print(df1.head())
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)



plt.show()
