import os

# Constants
VISUALISE = False # change to choose simulation or headless
TARGET_HEIGHT = 1.43
SAVE_AT_EPOCH = 2000
HEADLESS_EPOCHS = 10000
PRINT_EPOCHS=10
PLOT_STEPS = 500
BUFFER_CAP = 100000

# Hyperparameters
TRAIN_INTERVAL = 1024
UPH_COST_WEIGHT = 0.5
CTRL_COST_WEIGHT = 0.5
HEALTH_COST_WEIGHT = 0.1
LEARNING_RATE = 0.0005

# PPO Hyperparameters
CLIP_PARAM = 0.2 # increase if stuck in local minima
PPO_EPOCHS = 4 # increase if training too slowly
LOSS_COEF = 0.9 # higher = emphasis on value function, lower = emphasis on policy improvement
ENTROPY_COEF = 0.001 # increase for more exploration

MAX_EPISODE_STEPS = 1024
EARLY_TERMINATION_HEIGHT = 0.9
