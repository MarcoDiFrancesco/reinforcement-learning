import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from common import helper as h
from common.buffer import Batch


class RBFAgent(object):
    def __init__(self, num_actions, gamma=0.98, batch_size=32):
        self.scaler = None
        self.featurizer = None
        self.q_functions = None
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self._initialize_model()

    def _initialize_model(self):
        # Draw some samples from the observation range and initialize the scaler (used to normalize data)
        obs_limit = np.array([4.8, 5, 0.5, 5])
        samples = np.random.uniform(-obs_limit, obs_limit, (1000, obs_limit.shape[0]))

        # calculate the mean and var of samples, used later to normalize training data
        self.scaler = StandardScaler().fit(samples)

        # Initialize the RBF featurizer
        self.featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=80)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(samples))

        # Create a value approximator for each action dimension
        self.q_functions = [
            SGDRegressor(learning_rate="constant", max_iter=500, tol=1e-3)
            for _ in range(self.num_actions)
        ]

        # Initialize it to whatever values; implementation detail
        for q_a in self.q_functions:
            q_a.partial_fit(self.featurize(samples), np.zeros((samples.shape[0],)))

    def featurize(self, state):
        """Map state to a higher dimension for better representation."""
        # e.g. (1000, 4)
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Manual features, Task 1a
        # e.g. (1000, 8)
        # return np.concatenate((state, np.abs(state)), axis=1)

        # RBF features, Task 1b
        # map a state to a higher dimension (100+80+50)
        # e.g. (1000, 230)
        return self.featurizer.transform(self.scaler.transform(state))

    def get_action(self, state: np.ndarray, epsilon=0.0):
        ########## Your code starts here ##########
        import random

        # Hints:
        # 1. self.q_functions is a list which defines a q function for each action dimension
        # 2. for each q function, use predict(feature) to obtain the q value

        state = self.featurize(state)

        actions = np.zeros(self.num_actions) + np.inf
        for idx, regressor in enumerate(self.q_functions):
            # e.g. regressor: SGDRegressor(learning_rate='constant', max_iter=500)
            # e.g. regressor.predict: Real val, e.g. 0.923 or 1.125 or 1.573 or 2.045
            actions[idx] = regressor.predict(state)

        if random.random() > epsilon:
            a = actions.argmax()
        else:
            # Get random idx
            a = np.random.choice(np.arange(actions.size))
        return a
        ########## Your code ends here #########

    def _to_squeezed_np(self, batch: Batch) -> Batch:
        """A simple helper function squeeze data dimension and cover data format from tensor to np.ndarray."""
        _ret = {}
        for name, value in batch._asdict().items():
            if isinstance(value, dict):  # this is for extra, which is a dict
                for k, v in value.items():
                    value[k] = v.squeeze().numpy()
                _ret[name] = value
            else:
                _ret[name] = value.squeeze().numpy()
        return Batch(**_ret)

    def update(self, buffer):
        # batch is a namedtuple, which has state, action, next_state, not_done, reward
        # you can access the value be batch.<name>, e.g, batch.state
        batch = buffer.sample(self.batch_size)

        # the returned batch is a namedtuple, where the data is torch.Tensor
        # we first squeeze dim and then covert it to Numpy array.
        batch = self._to_squeezed_np(batch)

        ########## You code starts here #########
        # Hints:
        # 1. featurize the state and next_state
        # 2. calculate q_target (check q-learning)
        # 3. self.q_functions is a list which defines a q function for each action dimension
        #    for each q function, use q.predict(featurized_state) to obtain the q value
        # 4. remember to use not_done to mask out the q values at terminal states (treat them as 0)

        f_state = self.featurize(batch.state)
        f_next_state = self.featurize(batch.next_state)

        # q target
        q_tar = np.zeros(len(batch.state)) - np.inf
        # For each step in the batch, e.g. 14 steps
        for idx, (state, action, next_state, not_done, reward) in enumerate(
            zip(
                batch.state,
                batch.action,
                batch.next_state,
                batch.not_done,
                batch.reward,
            )
        ):
            # assert action == int(
            #     action
            # ), "Use 'action = math.ceil(action)' instead to get 1.8 -> 2"
            # # e.g. 1.8 -> 1
            # action = int(action)

            # TODO states vs next_states: undestand why from formula is s+1, from intuition we want
            # to lean the q-value of the current state (taken from f_state[idx])
            # from slack: ... just use the reward, next_state, not_done from the batch
            f_next_state_for = self.featurize(next_state)
            # Q-values of state s+1
            q_value_new = [
                regressor.predict(f_next_state_for).item()
                for regressor in self.q_functions
            ]

            if not_done:
                # Batch target
                b_target = reward + (self.gamma * max(q_value_new))
                # one-item list to float
                b_target = b_target.item()
            else:
                b_target = reward

            q_tar[idx] = b_target

            # print(
            #     f"""BATCH,
            #     STATE: {state}
            #     ACTION: {action}
            #     NEXT_STATE: {next_state}
            #     NOT_DONE: {not_done}
            #     REWARD: {reward}
            #     TARGET: {b_target}
            #     """
            # )

        ########## You code ends here #########
        # Get new weights for each action separately
        for a in range(self.num_actions):
            # Find states where `a` was taken
            idx = batch.action == a

            # If a not present in the batch, skip and move to the next action
            if np.any(idx):
                act_states = f_state[idx]
                act_targets = q_tar[idx]

                # Perform a single SGD step on the Q-function params to update the q_function corresponding to action a
                self.q_functions[a].partial_fit(act_states, act_targets)

        # if you want to log something in wandb, you can put it inside the {}, otherwise, just leave it empty.
        return {}

    def save(self, fp):
        path = fp / "rbf.pkl"
        h.save_object({"q": self.q_functions, "featurizer": self.featurizer}, path)

    def load(self, fp):
        path = fp / "rbf.pkl"
        d = h.load_object(path)

        self.q_functions = d["q"]
        self.featurizer = d["featurizer"]
