from __future__ import absolute_import, division, print_function

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import reverb
import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy, policy_saver

from envs.macd_env import MacdEnv
from envs.adx_env import AdxEnv
from envs.rsi_env import RsiEnv
from envs.moving_average_env import MovingAverageEnv
from envs.combined_env import CombinedEnv

from utils import load_dataset

import sys


def layers_cfg(id: str):
    return {
        'v1': (50,),
        'v2': (100,),
        'v3': (100, 50),
        'v4': (100, 100),
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

# num_iterations = 50000
# initial_collect_steps = 1000
collect_episodes_per_iteration = 10
replay_buffer_max_length = 2000
# batch_size = 64
# learning_rate = 0.003
log_interval = 25
num_eval_episodes = 10
eval_interval = 50


# ====================================== Create environments ======================================
train_py_envs = create_training_envs(env_type)
train_env = tf_py_environment.TFPyEnvironment(train_py_envs[0])


eval_py_env = create_testing_env(env_type)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# ====================================== Create REINFORCE Agent ======================================
# fc_layer_params = (100, 50)
fc_layer_params = agent_layers
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# ====================================== Create Optimizer ======================================
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)

agent = reinforce_agent.ReinforceAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    optimizer=optimizer,
    normalize_returns=True,
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


# ====================================== Replay Buffer ======================================
table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
    agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddEpisodeObserver(
    replay_buffer.py_client,
    table_name,
    replay_buffer_max_length
)


# ====================================== Collect Data ======================================
def collect_episode(environment, policy, num_episodes):
    driver = py_driver.PyDriver(
        environment,
        py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
        [rb_observer],
        max_episodes=num_episodes)
    initial_time_step = environment.reset()
    driver.run(initial_time_step)

# ====================================== Training Loop ======================================


# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]


def get_random_train_env():
    idx = np.random.randint(len(train_py_envs))
    return train_py_envs[idx]


# Store checkpoints for models
best_return = 400
# best_return = 100
checkpoint_stored = False

change_training_env_interval = 25

agent_saver = policy_saver.PolicySaver(agent.policy)

train_py_env = get_random_train_env()

for _ in range(num_iterations):

    # Collect a few steps and save to the replay buffer.
    collect_episode(
        train_py_env, agent.collect_policy, collect_episodes_per_iteration)

    # Sample a batch of data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)
    train_loss = agent.train(experience=trajectories)

    replay_buffer.clear()

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(
            eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

        # Store agent configuration if greater than best return
        if avg_return > best_return:
            print(
                f'===>>> Got better avg return: {avg_return}, will store checkpoint')
            best_return = avg_return
            agent_saver.save('./models/' + run_id)
            checkpoint_stored = True

    # Change environment
    if step % change_training_env_interval == 0:
        train_py_env = get_random_train_env()

# ====================================== See Avg Return ======================================

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
