# WANDB_MODE=disabled \
python train.py \
--config_file_name=hopper_easy \
--agent_name=ddpg \
--seed=4 \
--save_video=True \
--actor_lr 3e-4 \
--critic_lr 3e-4 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--train_episodes 500000 \
# --testing=True \

# Options:
# mountaincarcontinuous_easy.yaml
#
# lunarlander_continuous_easy.yaml
# lunarlander_continuous_medium.yaml
# lunarlander_discrete_easy.yaml
# lunarlander_discrete_medium.yaml
#
# bipedalwalker_easy.yaml
# bipedalwalker_medium.yaml
#
# hopper_easy.yaml
# hopper_medium.yaml
# hopper_hard.yaml
