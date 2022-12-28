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

from agents import helper as h
from agents.ddpg import DDPG
from agents.dqn_agent import DQNAgent
from agents.pg import PG as PG_BASE

# from agents.pg_ac import PG as PG_AC
from agents.pg_ac_taversion import PG as PG_AC
from make_env import create_env

from common import logger as logger
from common.buffer import ReplayBuffer


def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()


# Policy training function
def train(agent, env, max_episode_steps=1000):
    # Run actual training
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1

        if isinstance(agent, PG_AC):
            # Sample action from policy
            action, (act_logprob, action_ent) = agent.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _ = env.step(to_numpy(action))

            done_bool = done
            # observation, action_prob, action_ent, reward, done, next_observation
            agent.record(obs, action, action_ent, reward, done_bool, next_obs)
        elif isinstance(agent, DDPG):
            # Sample action from policy
            action, act_logprob = agent.get_action(obs)

            # Perform the action on the environment, get new state and reward
            next_obs, reward, done, _ = env.step(to_numpy(action))

            # Store action's outcome (so that the agent can improve its policy)
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0
            # agent.record(obs, action, next_obs, reward, done_bool)
            agent.record(obs, action, next_obs, reward, done_bool)
        else:
            raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

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


@torch.no_grad()
def test(agent, env: Env):
    total_test_reward = 0
    num_episodes = 50
    for _ in range(num_episodes):
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

    print("Average test reward:", total_test_reward / num_episodes)


def main():
    args = _get_args()
    h.set_seed(args.seed)
    env = create_env(args.config_file_name, args.seed)

    env.run_id = int(time.time())

    # create folders if needed
    work_dir = (
        Path().cwd() / "results" / f"{env.env_name}_{args.agent_name}_{args.seed}"
    )
    MODEL_PATH = work_dir / "model"
    h.make_dir(MODEL_PATH)
    MODEL_PATH_BEST = MODEL_PATH / "best.pth"
    MODEL_PATH_LAST = MODEL_PATH / "last.pth"
    h.make_dir(work_dir / "logging")
    L = logger.Logger()  # create a simple logger to record stats

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    wandb.init(
        project="rl_aalto_project",
        entity="marcodifrancesco",
        name=f"project-{env.env_name}-{args.agent_name}-{env.seed}-{env.run_id}",
        group=f"project-{env.env_name}-{args.agent_name}",
        config=env,
    )
    wandb.config.update(args)

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
            name_prefix=env.exp_name,
        )

    agent = _get_agent(args, env)

    if not args.testing:
        best_reward = -1000
        for ep in range(args.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            wandb.log(train_info)
            L.log(**train_info)
            if ep % 100 == 0:
                print({"ep": ep, **train_info})
            if ep % 1000 == 0:
                agent.save(MODEL_PATH_LAST)
            if best_reward < train_info["ep_reward"]:
                print("Best reward", train_info["ep_reward"])
                best_reward = train_info["ep_reward"]
                agent.save(MODEL_PATH_BEST)

        agent.save(MODEL_PATH_LAST)
    else:
        loading_dir = MODEL_PATH_LAST
        # loading_dir = MODEL_PATH_BEST
        # Testing
        print("Loading model from", loading_dir, "...")
        # load model
        agent.load(loading_dir)
        print("Testing ...")
        test(agent, env)


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_name", help="e.g. mountaincarcontinuous_easy")
    parser.add_argument("--agent_name", help="e.g. ddpg", required=True)

    parser.add_argument("--train_episodes", required=True, type=int)
    parser.add_argument("--seed", help="e.g. 4", type=str, required=True)
    parser.add_argument("--save_video", help="<True|False>")
    parser.add_argument("--testing", help="<True|False>")

    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--random_transition", type=int)
    parser.add_argument("--buffer_size", type=int)
    args = parser.parse_args()

    # Parse variables
    args.seed = int(args.seed)
    args.save_video = True if args.save_video == "True" else False
    args.testing = True if args.testing == "True" else False

    return args


def _get_agent(args, env):
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    if args.agent_name == "pg":
        # ex5/train.py:160
        assert type(args.lr) is float
        assert type(args.gamma) is float

        agent = PG_AC(
            state_shape[0],
            action_dim,
            lr=args.lr,
            gamma=args.gamma,
            ent_coeff=0.01,  # https://github.com/openai/baselines/issues/371
        )
    elif args.agent_name == "ddpg":
        # ex6/train.py:156
        assert type(args.lr) is float
        assert type(args.gamma) is float
        assert type(args.tau) is float
        assert type(args.batch_size) is int
        assert type(args.random_transition) is int
        assert type(args.buffer_size) is int

        agent = DDPG(
            state_shape=state_shape,
            action_dim=action_dim,
            max_action=max_action,
            actor_lr=0.0003,  # args.lr
            critic_lr=0.0003,  # args.lr
            gamma=0.99,  # args.gamma
            tau=0.005,  # args.tau
            batch_size=256,  # args.batch_size
            random_transition=5000,  # args.random_transition
            use_ou=True,
            # normalize=True,
            buffer_size=1e6,  # args.buffer_size
        )
    else:
        raise KeyError("Algorithm not known")

    return agent


# Entry point of the script
if __name__ == "__main__":
    main()
