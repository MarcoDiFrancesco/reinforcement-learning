import os
import sys

sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl"  # for mujoco rendering
import time
import warnings
from pathlib import Path

import gym
import hydra
import torch
import wandb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from ddpg import DDPG
from pg_ac import PG

from common import helper as h
from common import logger as logger


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

        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        print("ACTION", action, to_numpy(action))

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, PG):
            done_bool = done
            agent.record(obs, act_logprob, reward, done_bool, next_obs)
        elif isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0
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


# Function to test a trained policy
@torch.no_grad()
def test(agent, env, num_episode=10):
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


# The main function
@hydra.main(config_path="cfg", config_name="project_cfg")
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    # work_dir = Path().cwd() / "results" / f"{cfg.env_name}"
    work_dir = Path().cwd() / "results" / f"{cfg.env_name}_{cfg.agent_name}_{cfg.seed}"
    # if cfg.save_model:
    #     h.make_dir(work_dir / "model")
    # if cfg.save_logging:
    #     h.make_dir(work_dir / "logging")
    #     L = logger.Logger()  # create a simple logger to record stats

    # Model filename
    # if cfg.model_path == "default":
    #     cfg.model_path = work_dir / "model" / f"{cfg.env_name}_params.pt"

    MODEL_PATH = work_dir / "model"
    h.make_dir(MODEL_PATH)
    MODEL_PATH_BEST = MODEL_PATH / "best.pth"
    MODEL_PATH_LAST = MODEL_PATH / "last.pth"
    h.make_dir(work_dir / "logging")
    L = logger.Logger()  # create a simple logger to record stats

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(
            project="rl_aalto",
            entity="marcodifrancesco",
            # name=f"{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}",
            name=f"project-{cfg.env_name}-{cfg.agent_name}-{cfg.seed}-{cfg.run_id}",
            # group=f"{cfg.exp_name}-{cfg.env_name}",
            group=f"project-{cfg.env_name}-{cfg.agent_name}",
            config=cfg,
        )

    # create a env
    env = gym.make(cfg.env_name)
    env.seed(cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir / "video" / "test"
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir / "video" / "train"
        env = gym.wrappers.RecordVideo(
            env,
            video_path,
            episode_trigger=lambda x: x % ep_trigger == 0,
            name_prefix=cfg.exp_name,
        )  # save video every 50 episode

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # init agent
    if cfg.agent_name == "pg_ac":
        agent = PG(state_shape[0], action_dim, cfg.lr, cfg.gamma)
    else:  # ddpg
        agent = DDPG(
            state_shape,
            action_dim,
            max_action,
            cfg.lr,
            cfg.gamma,
            cfg.tau,
            cfg.batch_size,
            cfg.buffer_size,
        )

    if not cfg.testing:  # training
        best_reward = -1000
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})

            if ep % 1000 == 0:
                agent.save(MODEL_PATH_LAST)
            if best_reward < train_info["ep_reward"]:
                print("Best reward", train_info["ep_reward"])
                best_reward = train_info["ep_reward"]
                agent.save(MODEL_PATH_BEST)

        # if cfg.save_model:
        #     agent.save(cfg.model_path)
        agent.save(MODEL_PATH_LAST)

    else:  # testing
        # if cfg.model_path == "default":
        #     cfg.model_path = work_dir / "model" / f"{cfg.env_name}_params.pt"

        # loading_dir = MODEL_PATH_LAST
        loading_dir = MODEL_PATH_BEST

        # print("Loading model from", cfg.model_path, "...")
        print("Loading model from", loading_dir, "...")

        # load model
        # agent.load(cfg.model_path)
        agent.load(loading_dir)

        print("Testing ...")
        test(agent, env, num_episode=50)


# Entry point of the script
if __name__ == "__main__":
    main()
