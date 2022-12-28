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

from project.make_env import create_env

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer


@hydra.main(config_path="cfg", config_name="lunarlander_continuous_easy")
def main(cfg):
    # set random seed
    h.set_seed(cfg.seed)

    cfg.run_id = int(time.time())
    # create folders if needed
    # work_dir = Path().cwd() / "results" / cfg.env_name
    work_dir = Path().cwd() / "results" / f"{cfg.env_name}_{cfg.agent_name}_{cfg.seed}"

    if cfg.save_logging:
        logging_path = work_dir / "logging"
        h.make_dir(logging_path)
        L = logger.Logger()  # create a simple logger to record stats
    if cfg.save_model:
        model_path = work_dir / "model"
        h.make_dir(model_path)

    # use wandb to store stats
    if cfg.use_wandb:
        wandb.init(
            project="rl_aalto",
            entity="marcodifrancesco",
            name=f"project-{cfg.env_name}-{cfg.agent_name}-{cfg.seed}-{cfg.run_id}",
            group=f"project-{cfg.env_name}-{cfg.agent_name}",
            config=cfg,
        )

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
            work_dir / "video" / "train",
            episode_trigger=lambda x: x % 100 == 0,
            name_prefix=cfg.exp_name,
        )  # save video for every 100 episodes
    # get number of actions and state dimensions
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape

    print("ACTION", n_actions, state_shape)

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

    #  init buffer
    buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(cfg.buffer_size))

    for ep in range(cfg.train_episodes):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        eps = max(cfg.glie_b / (cfg.glie_b + ep), 0.05)

        # collecting data and fed into replay buffer
        while not done:
            env_step += 1
            if (
                ep < cfg.random_episodes
            ):  # in the first #random_episodes, collect random trajectories
                # print("RANDOM TRAJ", env_step, env.action_space, env.action_space.sample())
                action = env.action_space.sample()
            else:
                # Select and perform an action
                action = agent.get_action(state, eps)
                if isinstance(action, np.ndarray):
                    action = action.item()
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            # Store the transition in replay buffer
            buffer.add(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one update_per_episode step of the optimization
            if ep >= cfg.random_episodes:
                update_info = agent.update(buffer)
            else:
                update_info = {}

        info = {"episode": ep, "epsilon": eps, "ep_reward": ep_reward}
        info.update(update_info)

        if cfg.use_wandb:
            wandb.log(info)
        if cfg.save_logging:
            L.log(**info)
        if (not cfg.silent) and (ep % 50 == 0):
            print(info)
        if ep % 100 == 0:
            print(f"Saving model in {model_path}")
            agent.save(model_path)

    # save model and logging
    if cfg.save_model:
        agent.save(model_path)
    if cfg.save_logging:
        L.save(logging_path / "logging.pkl")

    print("------ Training Finished ------")


if __name__ == "__main__":
    main()
