MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL='egl' python env_test.py

# cd into examples/rl/stablebaselines3
MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL='egl' python test_sb3_ppo.py
MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_GL='egl' python train_sb3_ppo.py