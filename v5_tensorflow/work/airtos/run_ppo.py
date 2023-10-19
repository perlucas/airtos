# coding=utf-8
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Train and Eval PPO.

To run:

```bash
tensorboard --logdir $HOME/tmp/ppo/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/ppo/examples/v2/train_eval_clip_agent.py \
  --root_dir=$HOME/tmp/ppo/gym/HalfCheetah-v2/ \
  --logtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
import sys

import matplotlib
import matplotlib.pyplot as plt

from absl import app
# from absl import flags
from absl import logging
# import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
# from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.utils import common

from envs.macd_env import MacdEnv
from envs.adx_env import AdxEnv
from envs.rsi_env import RsiEnv
from envs.moving_average_env import MovingAverageEnv
from envs.combined_env import CombinedEnv

from utils import load_dataset
import numpy as np

# Utilities for env initialization and args


def layers_cfg(id: str):
    return {
        'v1': (200, 100),
        'v2': (200, 200),
        'v3': (200, 100, 50),
        'v4': (200, 200, 100),
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
    window_size = 10

    return [
        # KO training envs
        create_env(env_type, ko_df, window_size, (10, 120)),
        create_env(env_type, ko_df, window_size, (120, 230)),
        create_env(env_type, ko_df, window_size, (230, 350)),
        create_env(env_type, ko_df, window_size, (1000, 1120)),
        create_env(env_type, ko_df, window_size, (1300, 1400)),

        # AMZN training envs
        create_env(env_type, amzn_df, window_size, (10, 120)),
        create_env(env_type, amzn_df, window_size, (120, 230)),
        create_env(env_type, amzn_df, window_size, (230, 350)),
        create_env(env_type, amzn_df, window_size, (1000, 1120)),
        create_env(env_type, amzn_df, window_size, (1300, 1400)),
    ]


def create_testing_env(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    window_size = 10
    return create_env(env_type, ko_df, window_size, (2000, 2300))


def train_eval(
    root_dir,
    env_name='HalfCheetah-v2',
    run_id='run',
    get_train_py_env=None,
    get_eval_py_env=None,
    random_seed=None,
    # TODO(b/127576522): rename to policy_fc_layers.
    actor_fc_layers=(200, 100),
    value_fc_layers=(200, 100),
    use_rnns=False,
    lstm_size=(20,),
    # Params for collect
    num_environment_steps=25000000,
    collect_episodes_per_iteration=30,
    num_parallel_environments=30,
    replay_buffer_capacity=1001,  # Per-environment
    # Params for train
    num_epochs=25,
    learning_rate=1e-3,
    # Params for eval
    num_eval_episodes=30,
    eval_interval=500,
    change_env_interval=1000,
    # Params for summaries and logging
    train_checkpoint_interval=500,
    policy_checkpoint_interval=500,
    log_interval=50,
    summary_interval=50,
    summaries_flush_secs=1,
    use_tf_functions=True,
    debug_summaries=False,
    summarize_grads_and_vars=False,
):
    """A simple train and eval for PPO."""
    if root_dir is None:
        raise AttributeError('train_eval requires a root_dir.')

    root_dir = os.path.expanduser(root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')
    # saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000
    )
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000
    )
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
    ]

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with tf.compat.v2.summary.record_if(
        lambda: tf.math.equal(global_step % summary_interval, 0)
    ):
        if random_seed is not None:
            tf.compat.v1.set_random_seed(random_seed)

        eval_py_env = get_eval_py_env(env_name)
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)
        tf_env = tf_py_environment.TFPyEnvironment(get_train_py_env(env_name))
        # tf_env = tf_py_environment.TFPyEnvironment(
        #     parallel_py_environment.ParallelPyEnvironment(
        #         [lambda: get_train_py_env(env_name)] *
        #         num_parallel_environments
        #     )
        # )
        # optimizer = tf.compat.v1.train.AdamOptimizer(
        #     learning_rate=learning_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if use_rnns:
            actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                input_fc_layer_params=actor_fc_layers,
                output_fc_layer_params=None,
                lstm_size=lstm_size,
            )
            value_net = value_rnn_network.ValueRnnNetwork(
                tf_env.observation_spec(),
                input_fc_layer_params=value_fc_layers,
                output_fc_layer_params=None,
            )
        else:
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                tf_env.observation_spec(),
                tf_env.action_spec(),
                fc_layer_params=actor_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )
            value_net = value_network.ValueNetwork(
                tf_env.observation_spec(),
                fc_layer_params=value_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )

        tf_agent = ppo_clip_agent.PPOClipAgent(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            optimizer,
            actor_net=actor_net,
            value_net=value_net,
            entropy_regularization=0.0,
            importance_ratio_clipping=0.2,
            normalize_observations=False,
            normalize_rewards=False,
            use_gae=True,
            num_epochs=num_epochs,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step,
        )
        tf_agent.initialize()

        environment_steps_metric = tf_metrics.EnvironmentSteps()
        step_metrics = [
            tf_metrics.NumberOfEpisodes(),
            environment_steps_metric,
        ]

        train_metrics = step_metrics + [
            # tf_metrics.AverageReturnMetric(
            #     batch_size=num_parallel_environments),
            # tf_metrics.AverageEpisodeLengthMetric(
            #     batch_size=num_parallel_environments
            # ),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=1,
            # batch_size=num_parallel_environments,
            max_length=replay_buffer_capacity,
        )

        # train_checkpointer = common.Checkpointer(
        #     ckpt_dir=train_dir,
        #     agent=tf_agent,
        #     global_step=global_step,
        #     metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
        # )
        # policy_checkpointer = common.Checkpointer(
        #     ckpt_dir=os.path.join(train_dir, 'policy'),
        #     policy=eval_policy,
        #     global_step=global_step,
        # )
        # saved_model = policy_saver.PolicySaver(
        #     eval_policy, train_step=global_step)

        # train_checkpointer.initialize_or_restore()

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            tf_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_episodes=collect_episodes_per_iteration,
        )

        def train_step():
            trajectories = replay_buffer.gather_all()
            return tf_agent.train(experience=trajectories)

        if use_tf_functions:
            # TODO(b/123828980): Enable once the cause for slowdown was identified.
            collect_driver.run = common.function(
                collect_driver.run, autograph=False)
            tf_agent.train = common.function(tf_agent.train, autograph=False)
            train_step = common.function(train_step)

        collect_time = 0
        train_time = 0
        timed_at_step = global_step.numpy()

        avg_returns = []
        while environment_steps_metric.result() < num_environment_steps:
            global_step_val = global_step.numpy()
            if global_step_val % eval_interval == 0:
                eval_result = metric_utils.eager_compute(
                    eval_metrics,
                    eval_tf_env,
                    eval_policy,
                    num_episodes=num_eval_episodes,
                    train_step=global_step,
                    summary_writer=eval_summary_writer,
                    summary_prefix='Metrics',
                )
                logging.info('avg_ret: %f', eval_result['AverageReturn'])
                print('avg_ret: %f', eval_result['AverageReturn'])
                avg_returns.append(eval_result['AverageReturn'])

            start_time = time.time()
            collect_driver.run()
            collect_time += time.time() - start_time

            start_time = time.time()
            total_loss, _ = train_step()
            replay_buffer.clear()
            train_time += time.time() - start_time

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                    train_step=global_step, step_metrics=step_metrics
                )

            if global_step_val % log_interval == 0:
                logging.info('step = %d, loss = %f',
                             global_step_val, total_loss)
                print('step = %d, loss = %f', global_step_val, total_loss)
                logging.info('real_step = %d',
                             environment_steps_metric.result())
                print('real_step = %d', environment_steps_metric.result())
                steps_per_sec = (global_step_val - timed_at_step) / (
                    collect_time + train_time
                )
                logging.info('%.3f steps/sec', steps_per_sec)
                logging.info('collect_time = %.3f, train_time = %.3f',
                             collect_time, train_time)
                with tf.compat.v2.summary.record_if(True):
                    tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step
                    )

                # if global_step_val % train_checkpoint_interval == 0:
                #     train_checkpointer.save(global_step=global_step_val)

                # if global_step_val % policy_checkpoint_interval == 0:
                #     policy_checkpointer.save(global_step=global_step_val)
                #     saved_model_path = os.path.join(
                #         saved_model_dir, 'policy_' +
                #         ('%d' % global_step_val).zfill(9)
                #     )
                #     saved_model.save(saved_model_path)

                timed_at_step = global_step_val
                collect_time = 0
                train_time = 0

            if global_step_val % change_env_interval == 0:
                collect_driver.env = tf_py_environment.TFPyEnvironment(
                    get_train_py_env(env_name))

        # One final eval before exiting.
        metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )

        # ====================================== See Avg Return ======================================
        iterations = range(0, global_step_val + 1, eval_interval)
        plt.figure(figsize=(15, 8))
        plt.plot(iterations, avg_returns)
        plt.ylabel('Average Return')
        plt.xlabel('Iterations')
        plt.ylim(top=500)
        plt.savefig(root_dir + '/avgret/' + run_id)

        # ====================================== Evaluate Agent ======================================

        def render_policy_eval(policy, filename):
            time_step = eval_tf_env.reset()
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_tf_env.step(action_step.action)
            plt.figure(figsize=(15, 8))
            eval_py_env.save_render(filename)

        render_policy_eval(tf_agent.policy, root_dir + '/evals/' + run_id)


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()

    # Parse arguments to command
    args = sys.argv[1:]
    num_iterations, learning_rate, env_type, agent_layers, run_id = parse_args(
        args)

    # Defined parameters
    root_dir = './ppo_test'
    # num_environment_steps = 25000000
    num_environment_steps = num_iterations
    collect_episodes_per_iteration = 30
    num_parallel_environments = 1
    replay_buffer_capacity = 1001
    num_epochs = 25
    num_eval_episodes = 10

    # Environments
    def _get_eval_env(_env_name):
        return create_testing_env(env_type)

    training_envs = create_training_envs(env_type)

    def _get_train_env(_env_name):
        return training_envs[np.random.randint(len(training_envs))]

    train_eval(
        root_dir,
        run_id=run_id,
        env_name='AirtosTrading',
        get_train_py_env=_get_train_env,
        get_eval_py_env=_get_eval_env,
        use_rnns=False,
        num_environment_steps=num_environment_steps,
        collect_episodes_per_iteration=collect_episodes_per_iteration,
        num_parallel_environments=num_parallel_environments,
        replay_buffer_capacity=replay_buffer_capacity,
        num_epochs=num_epochs,
        num_eval_episodes=num_eval_episodes,
        learning_rate=learning_rate,
        actor_fc_layers=agent_layers,
        value_fc_layers=agent_layers
    )


if __name__ == '__main__':
    # multiprocessing.handle_main(functools.partial(app.run, main))
    app.run(main)
