WANDB_MODE=disabled python train.py \
--config_file_name=bipedalwalker_easy \
--agent_name=ddpg \
--seed=2 \
--save_video=True \
--actor_lr 0.3 \
--critic_lr 0.3 \
--gamma 0.3 \
--tau 0.3 \
--batch_size 10 \
--train_episodes 1000 \
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
