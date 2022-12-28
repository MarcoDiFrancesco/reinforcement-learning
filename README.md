# Reinforcement Learning repository

<br />

<p align="center">
    <img src="assets/aalto-logo-long.png" width="400">
</p>

<br />

Place where I do cool stuff with RL.

## First exercise highlights

<p align="center">
    <img src="assets/ex1-task4-best_action.png" width="400">
    <img src="assets/ex1-task4-rewards.png" width="400">
</p>

<p align="center">
    <img src="assets/ex1-cartpole.gif" width="500" border=0>
</p>


<p align="center">
    <img src="assets/ex1-reacher.gif" width="400" border=0>
</p>


## Second exercise highlights

<p align="center">
    <img src="assets/ex2.png" width="400">
</p>



<p align="center">
    <img src="assets/ex2-env.gif" width="400" border=0>
</p>

<p align="center">
    <img src="assets/ex2-value.gif" width="700" border=0>
</p>


## Setup

```
pip install -U pip wheel
pip install -r requirements.txt
```


## Run

```sh
WANDB_MODE=disabled \
python train.py \
--config_file_name bipedalwalker_easy \
--agent_name ddpg \
--seed 4 \
--lr 1e-3 \
--gamma 0.98 \
--tau 0.005 \
--batch_size 256 \
--train_episodes 500000 \
--random_transition 10000 \
--buffer_size 200000
```
