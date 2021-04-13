# Import Required Packages
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import random
from collections import namedtuple, deque
from RND import RND

from network import QNetwork
from replay_memory import ReplayBuffer
import os








# Determine if CPU or GPU computation should be used
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
"""
##################################################
Agent Class
Defines DQN Agent Methods
Agent interacts with and learns from an environment.
"""


class Agent():
    """
    Initialize Agent, inclduing:
        DQN Hyperparameters
        Local and Targat State-Action Policy Networks
        Replay Memory Buffer from Replay Buffer Class (define below)
    """

    def __init__(self, agent_id, state_size, action_size,
                 dqn_type='DQN', replay_memory_size=1e5, batch_size=64,
                 gamma=0.99, learning_rate=1e-3, target_tau=2e-3,
                 update_rate=4, model_dir='./model', load_model=False, seed=0):

        """
        DQN Agent Parameters
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            dqn_type (string): can be either 'DQN' for vanillia dqn learning (default) or 'DDQN' for double-DQN.
            replay_memory size (int): size of the replay memory buffer (typically 5e4 to 5e6)
            batch_size (int): size of the memory batch used for model updates (typically 32, 64 or 128)
            gamma (float): paramete for setting the discoun ted value of future rewards (typically .95 to .995)
            learning_rate (float): specifies the rate of model learing (typically 1e-4 to 1e-3))
            seed (int): random seed for initializing training point.
        """

        self.agent_id = agent_id
        self.dqn_type = dqn_type
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = int(replay_memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.learn_rate = learning_rate
        self.tau = target_tau
        self.update_rate = update_rate
        self.seed = random.seed(seed)
        self.model_dir = model_dir + "/{}/agent_{}".format(dqn_type, agent_id)

        """
        # DQN Agent Q-Network
        # For DQN training, two nerual network models are employed;
        # (a) A network that is updated every (step % update_rate == 0)
        # (b) A target network, with weights updated to equal the network at a slower (target_tau) rate.
        # The slower modulation of the target network weights operates to stablize learning.
        """

        self.num_simulations = 32
        self.rnd = RND(state_size, action_size, 256)




        self.network = QNetwork(state_size, action_size, seed).to(device)
        self.target_network = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learn_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        if load_model:
            if os.path.exists(os.path.join(self.model_dir, 'dqn_params.pkl')):
                path = os.path.join(self.model_dir, 'dqn_params.pkl')
                map_location = 'cpu'
                self.network.load_state_dict(torch.load(path, map_location=map_location))
                self.target_network.load_state_dict(torch.load(path, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path, path))
            else:
                raise Exception("No model!")


    # STEP() method
    #
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state = torch.tensor(state)
        reward_i = self.rnd.get_reward(state).detach().clamp(-1.0, 1.0).item()
        combined_reward = reward + reward_i
        self.memory.add(state, action, combined_reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_rate
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    ########################################################
    # ACT() method
    #
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.network.eval()
        with torch.no_grad():
            action_values = self.network(state)
        self.network.train()

        if random.random() > eps:
            action_values = action_values.cpu().numpy().tolist()[0]
        else:
            action_values = np.random.rand(self.action_size).tolist()

        # Epsilon-greedy action selection
        return action_values

    ########################################################
    # LEARN() method
    # Update value parameters using given batch of experience tuples.
    def learn(self, experiences, gamma, DQN=True):

        """
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        Ri = self.rnd.get_reward(states)
        self.rnd.update(Ri)




        # Get Q values from current observations (s, a) using model nextwork
        Qsa = self.network(states).gather(1, actions)

        # if (self.dqn_type == 'DDQN'):
        #     # Double DQN
        #     # ************************
        #     Qsa_prime_actions = self.network(next_states).detach().view(self.batch_size, 1)
        #     Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].view(self.batch_size, 1)
        #
        # else:
        #     # Regular (Vanilla) DQN
        #     # ************************
        #     # Get max Q values for (s',a') from target model
        #     Qsa_prime_target_values = self.target_network(next_states).detach()
        #     Qsa_prime_targets = Qsa_prime_target_values.max(1)[0].view(self.batch_size, 1)

            # Compute Q targets for current states

       # Qsa_targets = rewards + (gamma * Qsa_prime_targets * (1 - dones))
        Qsa_targets = rewards.squeeze() + gamma * self.target_network(next_states).max(dim=1)[0].detach() * (1 - dones)

        # Compute loss (error)
        #loss = F.msm(Qsa, Qsa_targets)
        loss = F.mse_loss(Qsa.squeeze(), Qsa_targets.squeeze())

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()


        # ------------------- update target network ------------------- #
        self.soft_update(self.network, self.target_network, self.tau)

    ########################################################
    """
    Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    """

    def soft_update(self, local_model, target_model, tau):
        """
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def save_model(self, episode=0):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.network.state_dict(), self.model_dir + '/' + str(episode) + '_dqn_params.pkl')


