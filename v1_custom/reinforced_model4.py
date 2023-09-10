import numpy as np
import tensorflow as tf
from math import ceil

class TradingBot:
    def __init__(self, initial_stock_price, initial_cash, initial_stocks):
        self.stock_price = initial_stock_price
        self.cash = initial_cash
        self.stocks = initial_stocks

    def operate(self, operation, num_stocks):
        if operation == 'N' or num_stocks <= 0:
            return
        
        if operation == 'B':
            n = num_stocks
            while n * self.stock_price > self.cash and n > 0:
                n -= 1
            if n > 0:
                self.stocks += n
                self.cash -= self.stock_price * n
        else:
            n = num_stocks
            while n > self.stocks and n > 0:
                n -= 1
            if n > 0:
                self.stocks -= n
                self.cash += self.stock_price * n
        

    def update_stock_price(self, variations):
        for v in variations:
            self.stock_price = self.stock_price * ( 1 + v)

    def get_funds(self):
        return self.cash + self.stock_price * self.stocks

    def process_action(self, action):
        op_value = action[0]
        num_stocks = ceil(action[1] * 10)

        operation = 'N'
        if op_value > 0.71:
            operation = 'B'
        elif op_value > 0.31:
            operation = 'S'
        
        self.operate(operation, num_stocks)
        

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
input_size = 10
output_size = 2
q_network = QNetwork(input_size, output_size)

# Set up the optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Custom reward function


def custom_reward(output_values):
    return 0.2

# Read input data from the file
input_data=[]
with open('reinforced_model2_input.txt', 'r') as file:
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
num_episodes=len(input_data)
epsilon=0.05  # Exploration parameter
initial_funds = 100000
for episode in range(num_episodes):
    inputs_x_10 = input_data[episode] # contains 10 states (1 file line)

    total_reward=0
    bot = TradingBot(1000, initial_funds, 0)

    for step in range(10):
        state = inputs_x_10[10 * step : 10 * (step + 1)] # 10-sized array
        next_state = inputs_x_10[10 * (step + 1) : 10 * (step + 2)]
        # print(state)

        q_values=q_network(np.expand_dims(state, axis=0))[0]
        action=choose_action(q_values, epsilon)

        bot.update_stock_price(state)
        bot.process_action(action)

        # reward=custom_reward(action)
        reward = bot.get_funds() - initial_funds
        total_reward += reward

        if len(next_state) > 0:
            q_learning(np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), reward, np.expand_dims(next_state, axis=0))

    print(f"Episode {episode + 1} - Total Reward: {total_reward}")

# Test the trained Q-network
test_input=input_data[np.random.randint(len(input_data))]
test_state=test_input[:10]
q_values=q_network(np.expand_dims(test_state, axis=0))
print(f"Q-Values for Test State: {q_values.numpy()}")
print(f"Reward is: {custom_reward(q_values[0])}")

# q_network.save('./reinforcement_model_final')
