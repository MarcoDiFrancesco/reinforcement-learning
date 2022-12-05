import os
import sys

sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl"  # for mujoco rendering
import argparse
import time
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
import wandb
from gym import Env

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.ddpg import DDPG
from agents.dqn_agent import DQNAgent
from agents.pg import PG
from make_env import create_env

from common import helper as h
from common import logger as logger
from common.buffer import ReplayBuffer


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


# Policy training function
def train(agent, env):
    timesteps, reward_sum = np.inf, np.inf

    # Store action's outcome (so that the agent can improve its policy)
    if isinstance(agent, PG):
        timesteps, reward_sum = train_pg(agent, env, 1000)
    elif isinstance(agent, DDPG):
        timesteps, reward_sum = train_ddpg(agent, env, 1000)
    else:
        raise ValueError

    assert timesteps != np.inf and reward_sum != np.inf

    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update(
        {
            "timesteps": timesteps,
            "ep_reward": reward_sum,
        }
    )
    return info


def train_pg(agent: PG, env: Env):
    raise NotImplementedError()


def train_ddpg(agent: DDPG, env: Env, max_episode_steps: int):
    # Run actual training
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1

        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        # ignore the time truncated terminal signal
        done_bool = float(done) if episode_timesteps < max_episode_steps else 0
        agent.record(obs, action, next_obs, reward, done_bool)

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    return timesteps, reward_sum


@torch.no_grad()
def test(agent, env: Env, num_episode: int):
    total_test_reward = 0
    for ep in range(num_episode):
        obs, done = env.reset(), False
        test_reward = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))

            test_reward += reward

        total_test_reward += test_reward
        print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward / num_episode)


def main():
    args = _get_args()

    cfg = create_env(args.config_file_name, args.seed)

    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = (
        Path().cwd() / "results" / f"{cfg.env_name}_{args.agent_name}_{args.seed}"
    )
    MODEL_PATH = work_dir / "model"
    h.make_dir(MODEL_PATH)
    h.make_dir(work_dir / "logging")
    L = logger.Logger()  # create a simple logger to record stats

    # Old names examples: "ex1", "ex2", ...
    cfg.exp_name = "project"
    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    # if cfg.use_wandb and not cfg.testing:
    wandb.init(
        project="rl_aalto_project",
        entity="marcodifrancesco",
        name=f"{cfg.exp_name}-{cfg.env_name}-{args.agent_name}-{str(cfg.seed)}-{str(cfg.run_id)}",
        group=f"{cfg.exp_name}-{cfg.env_name}-{args.agent_name}",
        config=cfg,
    )
    wandb.config.update(args)

    # create a env
    env = gym.make(cfg.env_name)
    env.seed(cfg.seed)

    if args.save_video:
        print("Saving videos")
        # During testing, save every episode
        if args.testing:
            ep_trigger = 1
            video_path = work_dir / "video" / "test"
        # During training, save every 50th episode
        else:
            ep_trigger = 200
            video_path = work_dir / "video" / "train"
        env = gym.wrappers.RecordVideo(
            env,
            video_path,
            episode_trigger=lambda x: x % ep_trigger == 0,
            name_prefix=cfg.exp_name,
        )

    agent = _get_agent(args, env)

    if not args.testing:
        for ep in range(args.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            wandb.log(train_info)
            L.log(**train_info)
            if ep % 100 == 0:
                print({"ep": ep, **train_info})

        agent.save(MODEL_PATH)
    else:
        # Testing
        print("Loading model from", MODEL_PATH, "...")
        # load model
        agent.load(MODEL_PATH)
        print("Testing ...")
        test(agent, env, 1000)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_name", help="e.g. mountaincarcontinuous_easy")
    parser.add_argument("--agent_name", help="e.g. ddpg", required=True)

    parser.add_argument("--train_episodes", required=True, type=int)
    parser.add_argument("--seed", help="e.g. 4", type=str, required=True)
    parser.add_argument("--save_video", help="<True|False>")
    parser.add_argument("--testing", help="<True|False>")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--actor_lr", type=float)
    parser.add_argument("--critic_lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    # Parse variables
    args.seed = int(args.seed)
    args.save_video = True if args.save_video == "True" else False
    args.testing = True if args.testing == "True" else False

    return args


def _get_agent(args, env):
    if args.agent_name == "dqn":
        # ex4/train.py:69
        n_actions = env.action_space.n
        state_shape = env.observation_space.shape
        agent = DQNAgent(
            state_shape,
            n_actions,
            # Config file have: cartpole_dqn.yaml: 256, lunarlander_dqn.yaml: 512
            batch_size=args.batch_size,
            hidden_dims=args.hidden_dims,
            gamma=args.gamma,
            lr=args.lr,
            tau=args.tau,
        )
    elif args.agent_name == "pg":
        # ex5/train.py:160
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = PG(
            state_dim,
            action_dim,
            lr=args.lr,
            gamma=args.gamma,
        )
    elif args.agent_name == "ddpg":
        # ex6/train.py:156
        state_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]

        assert type(args.actor_lr) is float
        assert type(args.critic_lr) is float
        assert type(args.gamma) is float
        assert type(args.tau) is float
        assert type(args.batch_size) is int

        agent = DDPG(
            state_shape,
            action_dim,
            max_action,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
            batch_size=args.batch_size,
            buffer_size=1e6,
        )
    else:
        raise KeyError("Algorithm not known")

    return agent


# Entry point of the script
if __name__ == "__main__":
    main()
