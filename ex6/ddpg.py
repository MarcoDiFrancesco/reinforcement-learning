import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import helper as h
from common.buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state):
        return self.max_action * torch.tanh(self.actor(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return self.value(x)  # output shape [batch, 1]


class DDPG(object):
    def __init__(
        self,
        state_shape,
        action_dim,
        max_action,
        lr,
        gamma,
        tau,
        batch_size,
        buffer_size=1e6,
    ):
        state_dim = state_shape[0]
        self.action_dim = action_dim
        self.max_action = max_action
        self.pi = Policy(state_dim, action_dim, max_action).to(device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr)

        self.q = Critic(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.q)
        self.q_optim = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(state_shape, action_dim, max_size=int(buffer_size))

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # used to count number of transitions in a trajectory
        self.buffer_ptr = 0
        self.buffer_head = 0
        self.random_transition = 5000  # collect 5k random data for better exploration

    def update(self):
        """After collecting one trajectory, update the pi and q for #transition times:"""
        info = {}
        update_iter = (
            self.buffer_ptr - self.buffer_head
        )  # update the network once per transiton

        if self.buffer_ptr > self.random_transition:  # update once have enough data
            for _ in range(update_iter):
                info = self._update()

        # update the buffer_head:
        self.buffer_head = self.buffer_ptr
        return info

    def _update(self):
        batch = self.buffer.sample(self.batch_size, device=device)

        # Task 2
        ########## Your code starts here. ##########
        state = batch.state
        action = batch.action
        next_state = batch.next_state
        not_done = batch.not_done
        reward = batch.reward

        # Hints: 1. compute the Q target with the q_target and pi_target networks
        next_action_pi = self.pi(next_state)
        q_func_out = self.q_target(next_state, next_action_pi)
        q_target = reward + self.gamma * q_func_out * not_done

        #        2. compute the critic loss and update the q's parameters
        self.q_optim.zero_grad()
        value_loss = F.mse_loss(q_target.detach(), self.q(state, action))
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q.parameters()), 1)
        self.q_optim.step()

        #        3. compute actor loss and update the pi's parameters
        self.pi_optim.zero_grad()
        pi_target = -self.q(batch.state, self.pi(batch.state))
        pi_loss = pi_target.mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.pi.parameters()), 1)
        self.pi_optim.step()

        #        4. update the target q and pi using h.soft_update_params() (See the DQN code)
        h.soft_update_params(self.q, self.q_target, self.tau)
        h.soft_update_params(self.pi, self.pi_target, self.tau)
        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}

    @torch.no_grad()
    def get_action(self, observation, evaluation=False):
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        if (
            self.buffer_ptr < self.random_transition and not evaluation
        ):  # collect random trajectories for better exploration.
            action = torch.rand(self.action_dim)
        else:
            expl_noise = (
                0.1 * self.max_action
            )  # the stddev of the expl_noise if not evaluation

            # Task 2
            ########## Your code starts here. ##########
            # Use the policy to calculate the action to execute
            # if evaluation equals False, add normal noise to the action, where the std of the noise is expl_noise
            # Hint: Make sure the returned action's shape is correct.
            # torch.Size([1, 17]) - tensor([[ 0.0471, -0.0923, -0.0743,  0.0241, ...]]]
            action = self.pi(x)
            if not evaluation:
                # Add epsilon noise
                dist = torch.distributions.Normal(0, expl_noise)
                action += dist.sample([self.action_dim]).to(device)
            ########## Your code ends here. ##########

        return action, {}  # just return a positional value

    def record(self, state, action, next_state, reward, done):
        """Save transitions to the buffer."""
        self.buffer_ptr += 1
        self.buffer.add(state, action, next_state, reward, done)

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        d = torch.load(filepath)
        self.pi.load_state_dict(d["pi"])
        self.q.load_state_dict(d["q"])
        self.pi_target.load_state_dict(d["pi_target"])
        self.q_target.load_state_dict(d["q_target"])
        print("Successfully loaded model from {}".format(filepath))

    def save(self, filepath):
        torch.save(
            {
                "pi": self.pi.state_dict(),
                "pi_target": self.pi_target.state_dict(),
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
            },
            filepath,
        )
        print("Successfully saved model to {}".format(filepath))
