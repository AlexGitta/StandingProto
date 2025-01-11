import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        # Shared features
        self.features = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor_mean = nn.Linear(256, num_outputs)
        self.actor_log_std = nn.Parameter(torch.zeros(num_outputs)) 
        
        # Critic (value) head
        self.critic = nn.Linear(256, 1)
        
        # Initialize weights
        for layer in self.features:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1)

    def forward(self, x):
        # Ensure input has batch dimension
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        features = self.features(x)
        action_mean = self.actor_mean(features)
        
        # Properly handle action std for batched input
        action_std = self.actor_log_std.exp().expand(*action_mean.size())
        value = self.critic(features)
        
        return action_mean, action_std, value
