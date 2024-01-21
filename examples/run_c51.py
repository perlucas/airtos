from __future__ import absolute_import, division, print_function

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from envs.macd_env import MacdEnv
from envs.adx_env import AdxEnv
from envs.rsi_env import RsiEnv
from envs.moving_average_env import MovingAverageEnv
# from envs.combined_env import CombinedEnv

from utils import load_dataset

import sys


def layers_cfg(id: str):
    return {
        'v1': (100,),
        'v2': (50,),
        'v3': (100, 50),
        'v4': (100, 100),
        'v5': (100, 150),
        'v6': (100, 100, 50),
    }.get(id)


DEFINED_ENVS = ['macd', 'rsi', 'adx', 'mas']


def parse_args(args_arr):
    num_iterations = None
    learning_rate = None
    env_type = None
    agent_layers = None
    run_id = None

    learning_rate_str = ''
    agent_layers_str = ''

    for s in args_arr:
        if s.startswith('NUMIT='):
            num_iterations = int(s.removeprefix('NUMIT='))
        elif s.startswith('LRATE='):
            learning_rate = float(s.removeprefix('LRATE='))
            learning_rate_str = s.removeprefix('LRATE=').replace('.', '_')
        elif s.startswith('ENV='):
            env_type = s.removeprefix('ENV=')
        elif s.startswith('LAYERS='):
            agent_layers = layers_cfg(s.removeprefix('LAYERS='))
            agent_layers_str = s.removeprefix('LAYERS=')
        elif s.startswith('ID='):
            run_id = s.removeprefix('ID=')

    assert type(num_iterations) == int and num_iterations > 0
    assert type(learning_rate) == float and learning_rate > 0
    assert env_type in DEFINED_ENVS
    assert type(agent_layers) == tuple and len(agent_layers) > 0

    if run_id == None:
        run_id = f'{num_iterations}_{env_type}_{agent_layers_str}_{learning_rate_str}'

    return (num_iterations, learning_rate, env_type, agent_layers, run_id)


def create_env(env_type: str, df, window_size, frame_bound):
    if env_type == 'macd':
        return MacdEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'rsi':
        return RsiEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'adx':
        return AdxEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'mas':
        return MovingAverageEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    raise NotImplementedError('unknown type')


def create_training_envs(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    amzn_df = load_dataset('./resources/AMZN.csv')
    amd_df = load_dataset('./resources/AMD.csv')
    pypl_df = load_dataset('./resources/PYPL.csv')
    nflx_df = load_dataset('./resources/NFLX.csv')
    window_size = 10

    return [
        # KO training envs
        create_env(env_type, ko_df, window_size, (10, 120)),
        create_env(env_type, ko_df, window_size, (120, 230)),
        create_env(env_type, ko_df, window_size, (350, 470)),
        create_env(env_type, ko_df, window_size, (1000, 1120)),
        create_env(env_type, ko_df, window_size, (1700, 1820)),

        # AMZN training envs
        create_env(env_type, amzn_df, window_size, (10, 120)),
        create_env(env_type, amzn_df, window_size, (120, 230)),
        create_env(env_type, amzn_df, window_size, (350, 470)),
        create_env(env_type, amzn_df, window_size, (1000, 1120)),
        create_env(env_type, amzn_df, window_size, (1700, 1820)),

        # AMD training envs
        create_env(env_type, amd_df, window_size, (10, 120)),
        create_env(env_type, amd_df, window_size, (120, 230)),
        create_env(env_type, amd_df, window_size, (350, 470)),
        create_env(env_type, amd_df, window_size, (1000, 1120)),
        create_env(env_type, amd_df, window_size, (1700, 1820)),

        # PYPL training envs
        create_env(env_type, pypl_df, window_size, (10, 120)),
        create_env(env_type, pypl_df, window_size, (120, 230)),
        create_env(env_type, pypl_df, window_size, (350, 470)),
        create_env(env_type, pypl_df, window_size, (1000, 1120)),
        create_env(env_type, pypl_df, window_size, (1700, 1820)),

        # NFLX training envs
        create_env(env_type, nflx_df, window_size, (10, 120)),
        create_env(env_type, nflx_df, window_size, (120, 230)),
        create_env(env_type, nflx_df, window_size, (350, 470)),
        create_env(env_type, nflx_df, window_size, (1000, 1120)),
        create_env(env_type, nflx_df, window_size, (1700, 1820)),
    ]


def create_testing_env(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    window_size = 10
    return create_env(env_type, ko_df, window_size, (2000, 2300))


# ====================================== Parse arguments and extract params ======================================
args = sys.argv[1:]

num_iterations, learning_rate, env_type, agent_layers, run_id = parse_args(
    args)

initial_collect_steps = 1000
collect_steps_per_iteration = 10
replay_buffer_capacity = 10000
batch_size = 64

gamma = 0.99

num_atoms = 20
min_q_value = -1000
max_q_value = 1000
n_step_update = 2

log_interval = 1000
num_eval_episodes = 10
eval_interval = 200


# ====================================== Create environments ======================================

train_py_envs = create_training_envs(env_type)
train_env_sample = tf_py_environment.TFPyEnvironment(train_py_envs[0])

eval_py_env = create_testing_env(env_type)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# ====================================== Create C51 Agent and optimizer ======================================
fc_layer_params = agent_layers

categorical_q_net = categorical_q_network.CategoricalQNetwork(
    train_env_sample.observation_spec(),
    train_env_sample.action_spec(),
    num_atoms=num_atoms,
    fc_layer_params=fc_layer_params)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = categorical_dqn_agent.CategoricalDqnAgent(
    train_env_sample.time_step_spec(),
    train_env_sample.action_spec(),
    categorical_q_network=categorical_q_net,
    optimizer=optimizer,
    min_q_value=min_q_value,
    max_q_value=max_q_value,
    n_step_update=n_step_update,
    td_errors_loss_fn=common.element_wise_squared_loss,
    gamma=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

# ====================================== Helper for Avg Return ======================================


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


random_policy = random_tf_policy.RandomTFPolicy(train_env_sample.time_step_spec(),
                                                train_env_sample.action_spec())


# ====================================== Collect Data ======================================
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env_sample.batch_size,
    max_length=replay_buffer_capacity)


def collect_step(environment, policy):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    replay_buffer.add_batch(traj)


# Collect initial data
for _ in range(initial_collect_steps):
    collect_step(train_env_sample, random_policy)

# Dataset generates trajectories with shape [BxTx...] where
# T = n_step_update + 1.
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size,
    num_steps=n_step_update + 1).prefetch(3)

iterator = iter(dataset)

# ====================================== Training Loop ======================================
best_return = 400
checkpoint_stored = False

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]


def get_random_train_env():
    idx = np.random.randint(len(train_py_envs))
    return tf_py_environment.TFPyEnvironment(train_py_envs[idx])


train_env = get_random_train_env()
iterations_per_env = 10

agent_saver = policy_saver.PolicySaver(agent.policy)

for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, agent.collect_policy)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience)

    step = agent.train_step_counter.numpy()

    # Log information
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    # Evaluate agent
    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1:.2f}'.format(
            step, avg_return))
        returns.append(avg_return)

        # Store agent configuration if greater than best return
        if avg_return > best_return:
            print(
                f'===>>> Got better avg return: {avg_return}, will store checkpoint')
            best_return = avg_return
            agent_saver.save('./models/' + run_id)
            checkpoint_stored = True

    # Change training env
    if step % iterations_per_env == 0:
        train_env = get_random_train_env()

# ====================================== Render Avg Return ======================================

iterations = range(0, num_iterations + 1, eval_interval)
plt.figure(figsize=(15, 8))
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=500)
plt.savefig('./avgret/' + run_id)


# ====================================== Evaluate Agent ======================================
def render_policy_eval(policy, filename):
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
    plt.figure(figsize=(15, 8))
    eval_py_env.save_render(filename)


# Final evaluation using the best agent
if checkpoint_stored:
    loaded_policy = tf.compat.v2.saved_model.load('./models/' + run_id)
    render_policy_eval(loaded_policy, './evals/' + run_id)
