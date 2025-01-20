import os
import torch
import numpy as np
import mujoco
from datetime import datetime
from config.config import *
from utils.state_utils import get_state

class StandupTask:
    def __init__(self):
        with open('./resources/humanoidold.xml', 'r') as f:
            humanoid = f.read()
            self.model = mujoco.MjModel.from_xml_string(humanoid)
            self.data = mujoco.MjData(self.model)

        self.initial_pose = self.data.qpos.copy()
        self.last_height = self.data.xpos[self.model.body('head').id][2]
        self.total_steps = 0
        self.total_reward = 0
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    
    def save_checkpoint(self, actor_critic, steps, reward):
        """Save model checkpoint with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.checkpoint_dir}/model_steps{steps}_{timestamp}.pt"
        
        checkpoint = {
            'model_state_dict': actor_critic.state_dict(),
            'steps': steps,
            'reward': reward,
            'timestamp': timestamp
        }
        
        try:
            torch.save(checkpoint, filename)
            print(f"Checkpoint saved at step {steps}: {filename}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, actor_critic, filename):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(filename)
            actor_critic.load_state_dict(checkpoint['model_state_dict'])
            reward = checkpoint['reward']
            print(f"Loaded checkpoint from step {checkpoint['steps']}")
            return reward
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, 0
    
    def calculate_reward(self, data):
        head_height = data.xpos[self.model.body('head').id][2]

        time_bonus = 0.01 * self.episode_steps

        # Calculate feet on ground reward
        feet_on_ground = FEET_COST_WEIGHT * (np.exp(-5.0 * data.xpos[self.model.body('foot_left').id][2]) * np.exp(-5.0 * data.xpos[self.model.body('foot_right').id][2]))

        # Calculate survival bonus
        height_diff = abs(head_height - TARGET_HEIGHT)
    # Height reward
        height_diff = abs(head_height - TARGET_HEIGHT)
        height_reward = max(0, 1 - (height_diff / TARGET_HEIGHT))  # 1 when perfect, 0 when far off

        # Control cost
        quad_ctrl_cost = np.sum(np.square(data.ctrl))
        control_cost = CTRL_COST_WEIGHT * quad_ctrl_cost

        # Balance reward
        torso_orientation = data.xmat[self.model.body('torso').id].reshape(3, 3)
        balance_reward = max(0, 1 - abs(torso_orientation[2, 2] - 1.0))  # 1 when upright, 0 when not

        # Final reward
        reward = (height_reward + balance_reward  + feet_on_ground + time_bonus) - control_cost
        if self.episode_steps % 1 == 0 and VISUALISE:  # print off current states for debugging
            print(f"Height: {head_height}")
            print(f"Time bonus: {time_bonus}")
            print(f"Height Reward: {height_reward:.2f}")
            print(f"Control Cost: {control_cost:.2f}")
            print(f"Balance Reward: {balance_reward:.2f}")
            print(f"Feet on Ground: {feet_on_ground:.2f}")
            print(f"Total Reward: {reward:.2f}")

            self.last_height = head_height

        return reward, head_height
    
    def reset(self, data):         
            mujoco.mj_resetData(self.model, data)
            data.qvel[:] = 0
            mujoco.mj_forward(self.model, data)   
            self.last_height = self.data.xpos[self.model.body('head').id][2]
            self.episode_steps = 0
            return get_state(data)
