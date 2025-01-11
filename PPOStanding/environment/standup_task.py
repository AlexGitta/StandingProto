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

        height_diff = abs(head_height - TARGET_HEIGHT)
        height_ratio = max(-1, 1 - (height_diff / TARGET_HEIGHT))  # 1 when perfect, -1 when far off
        survival_bonus = HEALTH_COST_WEIGHT * self.episode_steps * height_ratio

        # Control cost (unchanged)
        quad_ctrl_cost =  np.sum(np.square(data.ctrl))
        quad_ctrl_cost = CTRL_COST_WEIGHT * np.clip(quad_ctrl_cost, 0, 1)

        # Final reward
        reward =  survival_bonus - quad_ctrl_cost

        if(self.episode_steps % 10 == 0 and VISUALISE):
            print(f"Height: {head_height:.2f}")
          #  print(f"Height Ratio: {height_ratio:.2f}")
            print(f"Height Ratio: {height_ratio:.2f}")
            print(f"Survival Bonus: {survival_bonus:.2f}")
            print(f"Control Cost: {quad_ctrl_cost:.2f}")
            print(f"Total Reward: {reward:.2f}")

        return reward, head_height
    
    def reset(self, data):
            # Use the supine pose 
            #starting_pose = model.key_qpos[5]  
            #data.qpos[:] = starting_pose
            mujoco.mj_resetData(self.model, data)
            data.qvel[:] = 0
            mujoco.mj_forward(self.model, data)   
            self.episode_steps = 0
            self.total_reward = 0
            return get_state(data)