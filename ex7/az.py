import os
import sys

sys.path.append("..")

import bsuite
import gym
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.buffer import ReplayBuffer
from common.simulator import SimulatorWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### Neural Networks #####
class Actor(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(np.prod(state_shape), 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, num_actions)

    def forward(self, state):
        state_flatten = torch.flatten(state, start_dim=1)
        a = F.elu(self.l1(state_flatten))
        a = F.elu(self.l2(a))
        return self.l3(a)


class Critic(nn.Module):
    def __init__(self, state_shape):
        super(Critic, self).__init__()
        state_dim = np.prod(state_shape)

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        state_flatten = torch.flatten(state, start_dim=1)
        q = F.elu(self.l1(state_flatten))
        q = F.elu(self.l2(q))
        q = self.l3(q)
        return q


##### MCTS #####
class Node(object):
    """A MCTS Node."""

    def __init__(self, prior=1.0):
        self.reward: float = 0.0
        # Visit counts N
        self.visit_count: int = 0
        self.terminal: bool = False
        # P(s, a)
        self.prior: float = prior  # action prior
        self.total_value: float = 0.0  # cumulative value
        self.children: dict = {}  # children nodes, index is the action

    def expand(self, prior: torch.Tensor):
        """Expands this node by adding cild nodes."""
        assert prior.ndim == 1  # Prior should be a flat vector.
        for a, p in enumerate(prior):
            self.children[a] = Node(prior=p)

    # Mean value Q(s, a)
    @property
    def value(self):
        """Returns the value of this node."""
        if self.visit_count:
            return self.total_value / self.visit_count
        return 0.0

    # Sum over N
    @property
    def children_visits(self) -> np.ndarray:
        """Return array of visit counts of visited children."""
        return np.array([c.visit_count for c in self.children.values()])

    @property
    def children_values(self) -> np.ndarray:
        """Return array of values of visited children."""
        return np.array([c.value for c in self.children.values()])


class MCTS(object):
    def __init__(
        self,
        env,
        eval_fn,
        num_simulations,
        num_actions,
        discount,
        dirichlet_alpha,
        exploration_fraction,
    ):
        self.env = env
        self.eval_fn = eval_fn
        self.num_simulations = num_simulations
        self.num_actions = num_actions
        self.discount = discount
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction

    def mcts(self, state) -> Node:
        """Does Monte Carlo tree search (MCTS), AlphaZero style."""

        # Evaluate the prior policy for this state.
        prior, value = self.eval_fn(state)
        assert prior.shape == (self.num_actions,)

        # Add exploration noise to the prior.
        noise = np.random.dirichlet(alpha=[self.dirichlet_alpha] * self.num_actions)
        prior = (
            prior * (1 - self.exploration_fraction) + noise * self.exploration_fraction
        )

        # Tree search.
        root = Node()
        root.expand(prior)

        # Save the model state so that we can reset it for each simulation.
        self.env.save_checkpoint()

        for _ in range(self.num_simulations):
            # Start a new simulation from the top.
            trajectory = [root]
            node = root

            # Generate a trajectory.
            obs = None
            while node.children:
                # 1. select action according to the search policy (puct: you should implement)
                action = self.puct(node)
                # Task 2.4. Greedy
                action = node.children_values.argmax()

                # 2. consider the children node according to the action
                node = node.children[action]

                # 3. execute the action and update information for new node ()
                # Make step from Slack
                obs, reward, terminal, _ = self.env.step(action)
                # "Update node.reward and node.terminal" from Slack
                node.reward = reward
                node.terminal = terminal

                # 4. append new node to the trajectory
                trajectory.append(node)

            if obs is None:
                raise ValueError("Generated an empty rollout; this should not happen.")

            # Calculate the bootstrap for leaf nodes.
            if node.terminal:
                # If terminal, there is no bootstrap value.
                value = 0.0
            else:
                # Otherwise, bootstrap from this node with our value function.
                prior, value = self.eval_fn(obs)
                # We also want to expand this node for next time.
                node.expand(prior)

            # Load the saved model state.
            self.env.load_checkpoint()

            # Monte Carlo back-up with bootstrap from value function.
            ret = value
            while trajectory:
                # Pop off the latest node in the trajectory.
                node = trajectory.pop()

                # 1 compute discounted return the node
                ret = node.reward + ret * self.discount

                # 2.update node (total_value, visit_count)
                node.total_value += ret
                node.visit_count += 1

        return root

    def bfs(self, node: Node):
        """Breadth-first search policy."""
        visit_counts = np.array([c.visit_count for c in node.children.values()])
        return self._argmax(-visit_counts)

    def puct(self, node: Node, ucb_scaling: float = 1.0) -> int:
        # Implement PUCT search policy policy search
        # Hint: compute value, prior (probs), and visit ratio for each child node
        # change these values!
        value_scores = np.zeros(self.num_actions)
        priors = np.zeros(self.num_actions)
        visit_ratios = np.zeros(self.num_actions)

        for idx, child in node.children.items():
            value_scores[idx] = child.value
            priors[idx] = child.prior
            visit_ratios[idx] = np.sqrt(node.children_visits.sum()) / (
                1 + child.visit_count
            )

        # Combine.
        puct_scores = value_scores + ucb_scaling * priors * visit_ratios
        # Greedy policy
        puct_scores = value_scores
        return self._argmax(puct_scores)

    def _argmax(self, values: np.ndarray):
        """Argmax with random tie-breaking."""
        max_value = np.max(values)
        return np.int32(np.random.choice(np.flatnonzero(values == max_value)))


##### The AlphaZero Agent #####
class AZAgent(object):
    def __init__(self, state_shape, num_actions, env, lr, gamma, num_simulations):
        self.policy = Actor(state_shape, num_actions).to(device)
        self.value = Critic(state_shape).to(device)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()), lr=lr
        )
        self.env = env
        self.planner = MCTS(
            env,
            self._eval_fn,
            num_simulations,
            num_actions,
            gamma,
            dirichlet_alpha=1.0,
            exploration_fraction=0.0,
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        self.num_actions = num_actions
        self.num_simulations = num_simulations

    def _eval_fn(self, observation):
        with torch.no_grad():
            observation = torch.FloatTensor(np.expand_dims(observation, axis=0)).to(
                device
            )
            logits, value = self.policy(observation), self.value(observation)

        logits = logits.cpu().numpy().squeeze(axis=0)
        value = value.item()
        probs = scipy.special.softmax(logits)

        return probs, value

    def visit_count_policy(self, root: Node, temperature: float = 1.0):
        """Probability weighted by visit^{1/temp} of children nodes."""
        visits = root.children_visits
        if np.sum(visits) == 0:  # uniform policy for zero visits
            visits += 1
        rescaled_visits = visits ** (1 / temperature)
        probs = rescaled_visits / np.sum(rescaled_visits)
        return probs

    def act(self, observation):
        """Compute the agent's policy via MCTS."""
        if self.env.needs_reset:
            self.env.reset()

        # compute a fresh MCTS plan.
        root = self.planner.mcts(observation)

        probs = self.visit_count_policy(root)
        action = np.int32(np.random.choice(self.num_actions, p=probs))

        return action, probs.astype(np.float32)

    def update(self, data):
        """Do a gradient update step on the loss."""
        # Update the actor and critic
        # 1. Use TD learning to update value
        # Hint: state=data.state, action=data.action, ..., probs= data.extra['pi'],
        # See buffer
        state, action, next_state, reward, not_done, extra = data
        # Probs is the action distribution you want the neural network to mimic
        probs = extra["pi"]

        # Critic - Value function Q
        with torch.no_grad():
            td_target = reward + self.gamma * self.value(next_state) * not_done

        value_loss = F.mse_loss(td_target, self.value(state))

        # Actor - Policy function π
        # 2. Use self.criterion as loss function for policy network (actor)
        policy_loss = torch.mean(self.criterion(self.policy(state), probs))

        # Loss: Actor + Critic
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {"loss": loss.item()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-use_wandb", type=int, default=1)
    args = parser.parse_args()
    if args.use_wandb:
        import uuid

        import wandb

        run_id = str(uuid.uuid4())
        wandb.init(
            project="rl_aalto",
            entity="marcodifrancesco",
            name=f"ex7-DeepSea-{run_id}",
            group=f"ex7-DeepSea",
        )
    # A wrapper to align the api of the environment, you can ignore it.
    # We will include this wrapper in the /common next year.
    class BsuiteToGymWrapper(gym.Env):
        def __init__(self, env):
            obs_shp = env.observation_spec().shape
            n_action = env.action_spec().num_values

            self.observation_space = gym.spaces.Box(
                low=np.full(obs_shp, -np.inf),
                high=np.full(obs_shp, np.inf),
                shape=obs_shp,
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Discrete(n_action)
            self.env = env
            self.t = 0

        def reset(self):
            self.t = 0
            return self.env.reset().observation

        def step(self, action):
            self.t += 1
            time_step = self.env.step(action)
            return time_step.observation, time_step.reward, time_step.last(), {}

    env = bsuite.load_from_id("deep_sea/0")
    env = BsuiteToGymWrapper(env)
    env = SimulatorWrapper(env)

    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    buffer = ReplayBuffer(state_shape, n_actions, 10000)

    agent = AZAgent(
        state_shape,
        n_actions,
        env,
        lr=1e-3,
        gamma=0.99,
        num_simulations=50,
    )

    for i in range(500):  # num_episodes
        ep_reward, ep_len = 0, 0
        state, done = env.reset(), False

        while not done:
            action, probs = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            ep_reward += reward
            ep_len += 1

            buffer.add(state, action, next_state, reward, done, {"pi": probs})

            if i > 1:
                update_info = agent.update(buffer.sample(batch_size=16, device=device))

            state = next_state

        if args.use_wandb:
            wandb.log({"Episode": i, "Episode reward": ep_reward})
        print(f"Episode: {i}, Episode length: {ep_len}, Episode reward: {ep_reward}")
