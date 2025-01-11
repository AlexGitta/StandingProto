import os

# Constants
VISUALISE = True
TARGET_HEIGHT = 1.43
SAVE_AT_EPOCH = 999
HEADLESS_EPOCHS = 1000
PRINT_EPOCHS=10
PLOT_STEPS = 500

# Hyperparameters
TRAIN_INTERVAL = 1024
UPH_COST_WEIGHT = 0.5
CTRL_COST_WEIGHT = 0.5
HEALTH_COST_WEIGHT = 0.1

# PPO Hyperparameters
CLIP_PARAM = 0.2 # increase if stuck in local minima
PPO_EPOCHS = 4 # increase if training too slowly
LOSS_COEF = 0.9 # higher = emphasis on value function, lower = emphasis on policy improvement
ENTROPY_COEF = 0.001 # increase for more exploration

MAX_EPISODE_STEPS = 1024
EARLY_TERMINATION_HEIGHT = 0.9
NO_REWARD_HEIGHT = 1.0