import os
import sys

sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl"  # for pygame rendering
import time
import warnings
from pathlib import Path

import gym
import hydra
import numpy as np
import torch
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agent import PG

from common import helper as h
from common import logger as logger


def to_numpy(tensor):
    return tensor.squeeze(0).cpu().numpy()


def train(agent: PG, env):
    # Policy training function

    # Reset the environment and observe the initial state
    reward_sum, timesteps, done = 0, 0, False
    obs = env.reset()

    while not done:
        ########### Your code starts here ###########
        #  1. Call agent.get_action to get action and log prob of the action
        #  2. Call env.step with the action (note: you need to convert the action into a numpy array -- use the function to_numpy for this)
        #  (Steps 1. and 2. you can also find from the 'test' function below)
        action, act_logprob = agent.get_action(obs)
        # 5. Use the observation you receive from env.step to call agent.get_action for the next timestep
        obs, reward, done, info = env.step(action)
        # obs, reward, done, info = env.step(action)

        # 3. Store the log prob of action and reward by calling agent.record
        agent.record(act_logprob, reward)

        # 4. Update reward_sum by adding the reward received from env.step, and increase timesteps by one
        reward_sum += reward
        timesteps += 1
        ########## Your codes ends here. ##########

    # Update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update(
        {
            "timesteps": timesteps,
            "ep_reward": reward_sum,
        }
    )
    return info


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episodes=50):
    total_test_reward = 0
    for ep in range(num_episodes):
        obs, done = env.reset(), False
        test_reward = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            # obs, reward, done, info = env.step(to_numpy(action))
            obs, reward, done, info = env.step(action)

            test_reward += reward

        total_test_reward += test_reward

        print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward / num_episodes)


# The main function
@hydra.main(config_path="cfg", config_name="ex5_cfg")
def main(cfg):

    # Set seed for random number generators
    h.set_seed(cfg.seed)

    # Define a run id based on current time
    cfg.run_id = int(time.time())

    # Create folders if needed
    work_dir = Path().cwd() / "results"
    if cfg.save_model:
        h.make_dir(work_dir / "model")
    if cfg.save_logging:
        h.make_dir(work_dir / "logging")
        L = logger.Logger()  # create a simple logger to record stats

    # Model filename
    if cfg.model_path == "default":
        cfg.model_path = Path().cwd() / "results" / f"{cfg.env_name}_{cfg.seed}"
        # cfg.model_path = work_dir / "model" / f"{cfg.env_name}_params.pt"

    # Use wandb to store stats
    if cfg.use_wandb and not cfg.testing:
        wandb.init(
            project="rl_aalto",
            entity="marcodifrancesco",
            name=f"{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}",
            group=f"{cfg.exp_name}-{cfg.env_name}",
            config=cfg,
        )

    # Create the gym env
    env = gym.make(cfg.env_name, render_mode="rgb_array" if cfg.save_video else None)

    # Set env random seed
    env.seed(cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir / "video" / cfg.env_name / "test"
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir / "video" / cfg.env_name / "train"
        env = gym.wrappers.RecordVideo(
            env,
            video_path,
            episode_trigger=lambda x: x % ep_trigger == 0,
            name_prefix=cfg.exp_name,
        )

    # Get state and action dimensionality
    if cfg.env_name == "InvertedPendulum-v4":
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    elif cfg.env_name == "LunarLander-v2":
        state_dim = 8  # (8,)
        action_dim = 4
        # action_dim = env.action_space.n
        # state_dim = env.observation_space.shape
    else:
        raise KeyError
    print("action_dim", action_dim)

    # Initialise the policy gradient agent
    print(
        "Initializing with parameters state_dim:",
        state_dim,
        "action_dim",
        action_dim,
        "cfg.lr",
        cfg.lr,
        "cfg.gamma",
        cfg.gamma,
    )
    agent = PG(
        state_dim,
        action_dim,
        cfg.lr,
        cfg.gamma,
    )

    if not cfg.testing:  # training
        for ep in range(cfg.train_episodes):
            # collect data and update the policy
            train_info = train(agent, env)
            train_info.update({"episodes": ep})

            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})
            if ep % 100 == 0:
                print(f"Saving model in {cfg.model_path}")
                agent.save(cfg.model_path)

        if cfg.save_model:
            agent.save(cfg.model_path)

    else:  # testing
        print("Loading model from", cfg.model_path, "...")
        # load model
        agent.load(cfg.model_path)
        print("Testing ...")
        test(agent, env, num_episodes=10)


# Entry point of the script
if __name__ == "__main__":
    main()
