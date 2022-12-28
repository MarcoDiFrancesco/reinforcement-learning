import time
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
import tqdm
import wandb
from ddqn_agent import DDQNAgent
from dqn_agent import DQNAgent
from matplotlib import pyplot as plt
from rbf_agent import RBFAgent

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer


@hydra.main(config_path="cfg", config_name="lunarlander_continuous_easy")
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    # create folders if needed
    work_dir = Path().cwd() / "results" / f"{cfg.env_name}_{cfg.agent_name}_{cfg.seed}"
    model_path = work_dir / "model"

    # create env
    env_kwargs = cfg.env_parameters
    env_kwargs = dict()

    env = gym.make(
        cfg.env_name,
        render_mode="rgb_array" if cfg.save_video else None,
        max_episode_steps=cfg.max_episode_steps,
        **env_kwargs,
    )

    env.seed(cfg.seed)
    env.reset(seed=cfg.seed)

    if cfg.save_video:
        env = gym.wrappers.RecordVideo(
            env,
            work_dir / "video" / "test",
            episode_trigger=lambda x: True,
            name_prefix=cfg.exp_name,
        )
    # get number of actions and state dimensions
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    # init agent
    if cfg.agent_name == "dqn":
        print("Using DQN")
        agent = DQNAgent(
            state_shape,
            n_actions,
            batch_size=cfg.batch_size,
            hidden_dims=cfg.hidden_dims,
            gamma=cfg.gamma,
            lr=cfg.lr,
            tau=cfg.tau,
        )
    elif cfg.agent_name == "ddqn":
        print("Using DDQN")
        agent = DDQNAgent(
            state_shape,
            n_actions,
            batch_size=cfg.batch_size,
            hidden_dims=cfg.hidden_dims,
            gamma=cfg.gamma,
            lr=cfg.lr,
            tau=cfg.tau,
        )
    elif cfg.agent_name == "rbf":
        agent = RBFAgent(
            n_actions,
            gamma=cfg.gamma,
            batch_size=cfg.batch_size,
        )
    else:
        raise ValueError(f"No {cfg.agent_name} agent implemented")

    # Load policy / q functions
    agent.load(model_path)

    total_test_reward = 0
    for ep in range(cfg.test_episodes):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        rewards = []

        # collecting data and fed into replay buffer
        while not done:
            # Select and perform an action
            action = agent.get_action(state, epsilon=0.0)
            if isinstance(action, np.ndarray):
                action = action.item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)

        total_test_reward += ep_reward

        info = {"episode": ep, "ep_reward": ep_reward}
        if not cfg.silent:
            print(info)
    print("Average test reward:", total_test_reward / cfg.test_episodes)


if __name__ == "__main__":
    main()
