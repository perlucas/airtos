import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random


# Custom environment registration
from custom_env import register
register()

# Define the DQN agent


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=1000)  # Replay memory buffer

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Dense(24, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='sigmoid')
        ])
        model.compile(loss='mse', optimizer='adam')
        return model

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(low=0, high=1, size=(self.action_size,))
        q_values = self.model.predict(state)
        # print(f"selecting action from predict: {q_values}")
        return q_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            target_q = self.model.predict(state)
            target_q[0][action.astype(int)] = target
            self.model.fit(state, target_q, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


# Initialize the environment
env = gym.make('CustomEnv-v0')
state_shape = env.observation_space.shape
action_size = env.action_space.shape[0]

# Initialize the DQN agent
agent = DQNAgent(state_shape, action_size)

# Train the DQN agent
episodes = 1000
# batch_size = 32
batch_size = 100

for episode in range(episodes):
    state, _ = env.reset()
    # state = np.reshape(state, 10)

    state = np.reshape(state, [1, state_shape[0]])

    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        # print(f"action is: {action}")
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_shape[0]])
        if not done:
            agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Save the trained model
agent.model.save('dqn_model.h5')
