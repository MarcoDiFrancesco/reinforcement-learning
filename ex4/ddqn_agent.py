import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import copy
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import helper as h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_dim, mlp_dims: List[int], out_dim, act_fn=nn.ReLU, out_act=nn.Identity):
    """Returns an MLP."""
    if isinstance(mlp_dims, int):
        raise ValueError("mlp dimensions should be list, but got int.")

    layers = [nn.Linear(in_dim, mlp_dims[0]), act_fn()]
    for i in range(len(mlp_dims) - 1):
        layers += [nn.Linear(mlp_dims[i], mlp_dims[i + 1]), act_fn()]
    # the output layer
    layers += [nn.Linear(mlp_dims[-1], out_dim), out_act()]
    return nn.Sequential(*layers)


class DDQNAgent(object):
    def __init__(
        self,
        state_shape,
        n_actions,
        batch_size=32,
        hidden_dims=[12],
        gamma=0.98,
        lr=1e-3,
        grad_clip_norm=1000,
        tau=0.001,
    ):
        self.n_actions = n_actions
        self.state_dim = state_shape[0]
        self.policy_net = mlp(self.state_dim, hidden_dims, n_actions).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.grad_clip_norm = grad_clip_norm
        self.tau = tau

        self.counter = 0

    def update(self, buffer):
        """One gradient step, update the policy net."""
        self.counter += 1
        # Do one step gradient update
        batch = buffer.sample(self.batch_size, device=device)

        # TODO: Task 3: Finish the DQN implementation.
        ########## You code starts here #########
        # Hints: 1. You can use torch.gather() to gather values along an axis specified by dim.
        #        2. torch.max returns a namedtuple (values, indices) where values is the maximum
        #           value of each row of the input tensor in the given dimension dim.
        #           And indices is the index location of each maximum value found (argmax).
        #        3.  batch is a namedtuple, which has state, action, next_state, not_done, reward
        #           you can access the value be batch.<name>, e.g, batch.state
        #        4. check torch.nn.utils.clip_grad_norm_() to know how to clip grad norm
        #        5. You can go throught the PyTorch Tutorial given on MyCourses if you are not familiar with it.

        # Double DQN
        # Tensor with best action for all next states of the batch, size = [512]
        action = torch.Tensor.argmax(self.policy_net(batch.next_state), 1)
        # Get evaluation of greedy policy but from target network
        # Tensor with Q for all next_states in batch and with action from before
        q_max = (
            self.target_net(batch.next_state)
            .gather(1, action.view(-1, 1))
            .reshape(batch.not_done.size())
        )
        q_tar = batch.reward + self.gamma * q_max * batch.not_done
        q_tar = q_tar.detach()

        # calculate the q(s,a)
        qs = torch.gather(
            self.policy_net(batch.state), 1, batch.action.type(torch.int64)
        )

        # DQN
        """
        # calculate the q(s,a)
        qs = self.policy_net(batch.state)
        qs = torch.gather(qs, dim=1, index=batch.action.type(torch.int64))

        target_next = self.target_net(batch.next_state)
        target_max_val, target_max_idx = target_next.max(dim=1)
        target_max_val = target_max_val.unsqueeze(1)


        # calculate q target (check q-learning)
        q_tar = batch.reward + batch.not_done * (self.gamma * target_max_val)

        # Detach q-target
        q_tar = q_tar.detach()
        """

        # calculate the loss
        loss = ((qs - q_tar) ** 2).sum() * 0.5

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.grad_clip_norm
        )
        # clip grad norm and perform the optimization step
        self.optimizer.step()

        ########## You code ends here #########

        # update the target network
        h.soft_update_params(self.policy_net, self.target_net, self.tau)

        return {
            "loss": loss.item(),
            "q_mean": qs.mean().item(),
            "num_update": self.counter,
        }

    @torch.no_grad()
    def get_action(self, state, epsilon=0.05):
        # Task 3: implement epsilon-greedy action selection
        ########## You code starts here #########
        if random.random() > epsilon:
            state = torch.Tensor(state).to(device)
            res = self.policy_net(state)
            action = res.argmax()
        else:
            action = np.random.choice(np.arange(self.n_actions))
        action = action.item()
        return action
        ########## You code ends here #########

    def save(self, fp):
        path = fp / "dqn.pt"
        torch.save(
            {
                "policy": self.policy_net.state_dict(),
                "policy_target": self.target_net.state_dict(),
            },
            path,
        )

    def load(self, fp):
        path = fp / "dqn.pt"
        d = torch.load(path)
        self.policy_net.load_state_dict(d["policy"])
        self.target_net.load_state_dict(d["policy_target"])
