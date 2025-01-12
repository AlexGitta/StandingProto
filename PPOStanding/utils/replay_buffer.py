import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        """
        Initialize a fixed-size replay buffer to store transitions.
        
        Args:
            capacity (int): Maximum number of transitions to store
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Preallocate memory for all arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros(capacity, dtype=np.float32)  # For tracking episode endings

    def add(self, state, action, reward, next_state, value, log_prob, done):
        """Add a new transition to the buffer"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.values[self.position] = value
        self.log_probs[self.position] = log_prob
        self.masks[self.position] = 0.0 if done else 1.0
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.values[indices]),
            torch.FloatTensor(self.log_probs[indices]),
            torch.FloatTensor(self.masks[indices])
        )

    def clear(self):
        """Clear the replay buffer"""
        self.position = 0
        self.size = 0