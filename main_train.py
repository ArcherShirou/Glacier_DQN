##################################
# Import Required Packages
import torch
import time
import random
import numpy as np
import argparse
from collections import deque
from agent import Agent

from env import AgentEnv
from tensorboardX import SummaryWriter
import os
import yaml

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting   `
    parser.add_argument('--num_episodes', type=int, default=2000)
    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.2)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--save_cycle', type=int, default=10)
    parser.add_argument('--epsilon_limit', type=int, default=100)
    parser.add_argument('--dqn_type', type=str, default='DQN')
    parser.add_argument('--replay_memory_size', type=int, default=5e3)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--target_tau', type=float, default=2e-3)

    parser.add_argument('--update_rate', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--config', type=str, default='./model_config')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = {}
    config['url'] = 'http://121.89.205.93:8030'
    config['get_npc_id_url'] = '/api/train/robot_id'
    config['reset_url'] = '/api/train/start'
    config['step_url'] = '/api/train/step'

    env = AgentEnv(config)
    args = get_common_args()
    args.state_size = env.npc_state_size
    args.action_size = env.npc_action_size
    args.agent_id = env.npc_id

    if not os.path.exists(args.config):
        os.makedirs(args.config)

    filename = args.config + '/config.yaml'
    with open(filename, 'w') as file_obj:
        yaml.dump(vars(args), file_obj, allow_unicode=True)

    writer = SummaryWriter(args.result_dir)
    agent = {}
    memory_size = args.replay_memory_size
    batch_size = args.batch_size

    for i in args.agent_id:
        agent[i] = Agent(agent_id=i,
                         state_size=args.state_size[i],
                         action_size=args.action_size[i],
                         dqn_type=args.dqn_type,
                         replay_memory_size=args.replay_memory_size,
                         batch_size=args.batch_size,
                         gamma=args.gamma,
                         learning_rate=args.learning_rate,
                         target_tau=args.target_tau,
                         update_rate=args.update_rate,
                         model_dir=args.model_dir,
                         load_model=args.load_model)


    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay
    # loop from num_episodes
    for i_episode in range(1, args.num_episodes + 1):

        # reset the environment at the beginning of each episode
        env.reset()

        # get initial state of the environment
        state = {}
        next_state = {}
        done = False
        step = 0



        for i in env.npc_id:
            state[i] = env.get_obs(i)

        # set the initial episode score to zero.
        score = {i: 0 for i in env.npc_id}

        while not done and step < args.epsilon_limit:
            # determine epsilon-greedy action from current sate
            for i in env.npc_id:
                npc_id = i

                action_v = agent[npc_id].act(state[i], epsilon)

                # send the action to the environment and receive resultant environment information
                env_info = env.step(i, action_v)

                reward = env_info  # get the reward
                next_state[i] = env.get_obs(i)  # get the next state
                # done = True if next_state[npc_id][3] < 1 else False

                # Send (S, A, R, S') info to the DQN agent for a neural network update
                agent[npc_id].step(state[i], action_v, reward, next_state[i], done)

                # set new state to current state for determining next action
                state[i] = next_state[i]

                # Update episode score
                score[npc_id] += reward

                print('{} ---- agent: {} train i_episode {} - {} current score {} '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                                                           i, i_episode, step, score[npc_id]))

            step += 1

        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        for i in args.agent_id:
            writer.add_scalar('agent_{}_reward'.format(i), score[i], global_step=i_episode)
            print('{} ---- agent: {} train epoch {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                            i, i_episode))

        if i_episode % args.save_cycle == 0:
            for i in args.agent_id:
                agent[i].save_model(episode=i_episode)
