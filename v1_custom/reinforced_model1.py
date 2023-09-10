import numpy as np
import tensorflow as tf

# Define the Q-network model with 2 outputs and sigmoid activation


class QNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            64, activation='relu', input_shape=(input_size,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(
            output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# Initialize the Q-network with 2 outputs and sigmoid activation
input_size = 100
output_size = 2
q_network = QNetwork(input_size, output_size)

# Set up the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom reward function


def custom_reward(output_values, windowed_results):
    operation='B' if output_values[0] < 0.51 else 'S'

    days=15
    if output_values[1] < 0.31:
        days=7
    elif output_values[1] > 0.61:
        days=20

    amount=windowed_results[1]
    if days == 7:
        amount=windowed_results[0]
    elif days == 20:
        amount=windowed_results[2]

    return amount if operation == 'B' else -amount


# Read input data from the file
input_data=[]
with open('reinforced_model1_input.txt', 'r') as file:
    for line in file:
        numbers=line.strip().split(' ')
        input_data.append([float(num) for num in numbers])

input_data=np.array(input_data)

# Q-learning algorithm


def q_learning(state, action, reward, next_state, discount_factor=0.99):
    with tf.GradientTape() as tape:
        q_values=q_network(state)
        selected_action_values=tf.reduce_sum(q_values * action, axis=1)
        target_q_value=reward + discount_factor * \
            tf.reduce_max(q_network(next_state), axis=1)
        loss=tf.losses.mean_squared_error(
            selected_action_values, target_q_value)

    gradients=tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

# Epsilon-greedy action selection function


def choose_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform(size=(2,))
    else:
        return q_values


# Training loop
num_episodes=1000
epsilon=0.1  # Exploration parameter
for episode in range(num_episodes):
    input_pattern=input_data[np.random.randint(len(input_data))]
    state=input_pattern[:100]
    total_reward=0

    for step in range(100):
        q_values=q_network(np.expand_dims(state, axis=0))[0]
        action=choose_action(q_values, epsilon)

        reward=custom_reward(action, input_pattern[100:])
        total_reward += reward

        input_pattern=input_data[np.random.randint(len(input_data))]
        next_state=input_pattern[:100]
        q_learning(np.expand_dims(state, axis=0), np.expand_dims(
            action, axis=0), reward, np.expand_dims(next_state, axis=0))

        state=next_state

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Test the trained Q-network
test_input=input_data[np.random.randint(len(input_data))]
test_state=test_input[:100]
q_values=q_network(np.expand_dims(test_state, axis=0))
print(f"Q-Values for Test State: {q_values.numpy()}")
print(f"Reward is: {custom_reward(q_values[0], test_input[100:])}")

# q_network.save('./reinforcement_model_final')
