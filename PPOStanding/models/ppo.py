import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
from .actor_critic import ActorCritic
from config.config import *

class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.0005)
        
        self.clip_param = CLIP_PARAM
        self.ppo_epochs = PPO_EPOCHS
        self.value_loss_coef = LOSS_COEF
        self.entropy_coef = ENTROPY_COEF
        
        # Running state normalization - convert to device immediately
        self.state_mean = torch.zeros(state_dim, device=self.device)
        self.state_std = torch.ones(state_dim, device=self.device)
    
    def get_action(self, state):
        with torch.no_grad():
            # Add batch dimension if needed
            state_tensor = torch.FloatTensor(state).to(self.device)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            normalized_state = self.normalize_state(state_tensor)
            action_mean, action_std, value = self.actor_critic(normalized_state)
            
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Remove batch dimension for output
            return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy()
    
    def normalize_state(self, state_tensor):
        if isinstance(state_tensor, np.ndarray):
            state_tensor = torch.FloatTensor(state_tensor).to(self.device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        return (state_tensor - self.state_mean) / (self.state_std + 1e-8)

    def update_state_stats(self, state):
        # Convert state to tensor and update running stats
        state_tensor = torch.FloatTensor(state).to(self.device)
        self.state_mean = 0.99 * self.state_mean + 0.01 * state_tensor.mean(dim=0)
        self.state_std = 0.99 * self.state_std + 0.01 * state_tensor.std(dim=0)

    def train(self, memory):
        # Convert all memory items to tensors on device
        states = torch.FloatTensor(np.array(memory.states)).to(self.device)
        actions = torch.FloatTensor(np.array(memory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(memory.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(memory.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(memory.values)).to(self.device)
        
        # Normalize states before training
        states = self.normalize_state(states)
        
        # Calculate advantages
        advantages = rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get current policy distributions
            action_mean, action_std, current_value = self.actor_critic(states)
            dist = Normal(action_mean, action_std)
            current_log_probs = dist.log_prob(actions).sum(-1)
            
            # Calculate ratio and surrogate loss
            ratio = (current_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(current_value.squeeze(), rewards)
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()