import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from common import helper as h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-critic agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # Task 1: Implement actor_logstd as a learnable parameter
        # Use log of std to make sure std doesn't become negative during training
        self.actor_logstd = torch.Tensor([[0.0]]).to(device)
        self.actor_logstd = torch.nn.Parameter(self.actor_logstd)

    def forward(self, state):
        # Get mean of a Normal distribution (the output of the neural network)
        action_mean = self.actor_mean(state)

        # Make sure action_logstd matches dimension of action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)

        # Exponentiate the log std to get actual std
        action_std = torch.exp(action_logstd)

        # Task 1: Create a Normal distribution with mean of 'action_mean' and standard deviation of 'action_logstd', and return the distribution
        probs = Normal(action_mean, action_std)

        return probs


class Value(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.value = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1)),
        )

    def forward(self, x):
        return self.value(x).squeeze(1)  # output shape [batch,]


class PG(object):
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.policy = Policy(state_dim, action_dim).to(device)
        self.value = Value(state_dim).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr,
        )

        self.gamma = gamma

        # a simple buffer
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def update(self):
        action_probs = torch.stack(self.action_probs, dim=0).to(device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(device).squeeze(-1)
        dones = torch.stack(self.dones, dim=0).to(device).squeeze(-1)
        # clear buffer
        self.states, self.action_probs, self.rewards, self.dones, self.next_states = (
            [],
            [],
            [],
            [],
            [],
        )

        # Task 1
        ########## Your code starts here. ##########
        # action_probs: tensor([[-4.9391], [-0.9320], [-0.9878]])
        # rewards: tensor([1., 1., 1.])
        # states: Size([3, 4]) - tensor([[-0.0171,  0.0374, -0.9457,  2.1420], ...)
        # next_states: Size([3, 4]) - tensor([[-0.0171,  0.0374, -0.9457,  2.1420], ...)
        # dones: tensor([0., 0., 1.])

        # Hints: 1. calculate the TD target as well as the MSE loss between the predicted value and the TD target
        # Actor
        with torch.no_grad():
            not_dones = 1 - dones
            td_target = rewards + self.gamma * self.value(next_states) * not_dones
        value_loss = F.mse_loss(td_target.detach(), self.value(states))

        # Critic
        #        2. calculate the policy loss (similar to ex5) with advantage calculated from the value function. Normalise
        #           the advantage to zero mean and unit variance.
        with torch.no_grad():
            # Advantage
            adv = td_target - self.value(states)
            # adv_norm: Size([24]) or Size([19]), tensor([-1.0160,  0.7055, -0.0833, ...]
            adv_norm = (adv - adv.mean()) / adv.std()
        policy_loss = torch.mean(-action_probs.squeeze() * adv_norm.detach())

        # Loss: Actor + Critic
        loss = value_loss + policy_loss

        #        3. update parameters of the policy and the value function jointly
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        ########## Your code ends here. ##########

        # if you want to log something in wandb, you can put them inside the {}, otherwise, just leave it empty.
        return {}

    def get_action(self, observation, evaluation=False):
        """Return action (np.ndarray) and logprob (torch.Tensor) of this action."""
        if observation.ndim == 1:
            observation = observation[None]  # add the batch dimension
        x = torch.from_numpy(observation).float().to(device)

        # Task 1
        ########## Your code starts here. ##########
        # Hints: 1. the self.policy returns a normal distribution, check the PyTorch document to see
        #           how to calculate the log_prob of an action and how to sample.

        # x: tensor([[ 0.0020, -0.0057, -0.0058, -0.0096]])
        # dist: tensor([[-2.8355]]) - Normal distribution of the model
        dist = self.policy(x)

        #        2. if evaluation, return mean, otherwise, return a sample
        if evaluation:
            action = dist.mean()
        else:
            action = dist.sample()

        # action: tensor([[-2.8355]])
        # act_logprob: tensor([[-4.9391]]) - Probability that action was taken
        act_logprob = dist.log_prob(action)

        #        3. the returned action and the act_logprob should be the torch.Tensors.
        #            Please always make sure the shape of variables is as you expected.
        assert type(action) is torch.Tensor
        assert type(act_logprob) is torch.Tensor
        ########## Your code ends here. ###########

        return action, act_logprob

    def record(self, observation, action_prob, reward, done, next_observation):
        self.states.append(torch.tensor(observation, dtype=torch.float32))
        self.action_probs.append(action_prob)
        self.rewards.append(torch.tensor([reward], dtype=torch.float32))
        self.dones.append(torch.tensor([done], dtype=torch.float32))
        self.next_states.append(torch.tensor(next_observation, dtype=torch.float32))

    # You can implement these if needed, following the previous exercises.
    def load(self, filepath):
        pass

    def save(self, filepath):
        pass
