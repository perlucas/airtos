import numpy as np
import tensorflow as tf


# Custom stock price accumulator

class StockPriceAccumulator:
    def __init__(self):
        self.buys = 0
        self.sells = 0

    def operate(self, op):
        if op == 'S':
            if self.buys == 0:
                self.sells += 1
            else:
                self.buys -= 1
        elif op == 'B':
            if self.sells == 0:
                self.buys += 1
            else:
                self.sells -= 1

    def get_reward(self, price_variation):
        return price_variation * (self.buys - self.sells)


# Define the Q-network model with 2 outputs and sigmoid activation


class QNetwork(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_size,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_size, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


# Initialize the Q-network with 2 outputs and sigmoid activation
input_size = 1
output_size = 1
q_network = QNetwork(input_size, output_size)

# Set up the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom reward function

def q_value_to_operation(q_value):
    if q_value > 0.71:
        return 'B'
    elif q_value > 0.31:
        return 'S'
    return 'N'


# Read input data from the file
input_data=[]
with open('reinforced_model2_input.txt', 'r') as file:
    for line in file:
        numbers=line.strip().split(' ')
        input_data.append([float(num) for num in numbers])

input_data=np.array(input_data)

# Q-learning algorithm


def q_learning(state, reward, next_state, discount_factor=0.99):
    with tf.GradientTape() as tape:
        q_value=q_network(state)
        loss=tf.losses.mean_squared_error(q_value, reward)

    gradients=tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

# Epsilon-greedy action selection function


def choose_action(q_value, epsilon):
    if np.random.rand() < epsilon:
        return np.random.uniform()
    else:
        return q_value


# Training loop
# num_episodes=1000
num_episodes=len(input_data)
print(f'Episodes: {num_episodes}')
epsilon=0.3  # Exploration parameter
for episode in range(num_episodes):
    acc = StockPriceAccumulator()
    total_reward=10
    
    for step in range(99):
        state=input_data[episode][step]
        q_value=q_network(np.array([[state]]))
        action=choose_action(q_value[0][0], epsilon)
        # action=q_value[0][0]
        # print(f'step: {step}, state: {state}, q_value: {q_value[0]}, action: {action}')

        reward=acc.get_reward(state)
        acc.operate(q_value_to_operation(action))
        total_reward = total_reward + total_reward * reward

        next_state=input_data[episode][step + 1]
        # q_learning(np.array([[state]]), reward,np.array([[next_state]]))
        q_learning(np.array([[state]]), total_reward,np.array([[next_state]]))

        state=next_state

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Test the trained Q-network
# test_input=input_data[np.random.randint(len(input_data))][]
# test_state=test_input[:100]
# q_values=q_network(np.expand_dims(test_state, axis=0))
# print(f"Q-Values for Test State: {q_values.numpy()}")
# print(f"Reward is: {custom_reward(q_values[0], test_input[100:])}")

# q_network.save('./reinforcement_model_final')
