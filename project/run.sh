source ../.venv/bin/activate
# export MUJOCO_GL=osmesa
# export MJLIB_PATH=$HOME/.mujoco/mujoco200/bin/libmujoco200.so
# export MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
# export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
# export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco200/
# export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco200/mjkey.txt
export MUJOCOGL_GL=glfw
python train.py
