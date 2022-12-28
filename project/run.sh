WANDB_MODE=disabled \
python train.py \
--config_file_name hopper_easy \
--agent_name ddpg \
--seed=2 \
--lr 0.0003 \
--gamma 0.99 \
--tau 0.005 \
--batch_size 256 \
--train_episodes 500000 \
--random_transition 10000 \
--buffer_size 200000 \




# --testing=True \
# --save_video=True \

# Options:
# mountaincarcontinuous_easy.yaml
#
# lunarlander_continuous_easy.yaml (DDPG=continuous)
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
