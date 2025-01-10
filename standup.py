import mujoco
import glfw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import os
from datetime import datetime
import imgui
from imgui.integrations.glfw import GlfwRenderer
import glob
import matplotlib.pyplot as plt


# Initialize MuJoCo simulation
with open('./humanoidold.xml', 'r') as f:
    humanoid = f.read()
    model = mujoco.MjModel.from_xml_string(humanoid)
    data = mujoco.MjData(model)



# Constants
VISUALISE = True
TARGET_HEIGHT = 1.8  # Target standing height
SAVE_INTERVAL = 100000
TRAIN_INTERVAL = 512
MAX_STEPS = 1000000
UPH_COST_WEIGHT = 1.5
CTRL_COST_WEIGHT = 0.00
IMPACT_COST_WEIGHT = 1e-7
IMPACT_COST_RANGE = 10.0

class StandupTask:
    def __init__(self):
        self.total_steps = 0
        self.total_reward = 0
        self.checkpoint_dir = "checkpoints"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.height_maintained_steps = 0  # Add counter for successful maintenance
        self.height_threshold = 0.1  # How close to target height is considered successful
        self.survival_bonus_scale = 0.001  # Scale factor for survival bonus
    
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
        head_heightall = data.xpos[model.body('head').id]
        head_height1 = data.xpos[model.body('head').id][2]
        head_height2 = data.xpos[model.body('head').id][1]

        # Height reward 
        uph_cost = (UPH_COST_WEIGHT * (head_height1 - TARGET_HEIGHT)) + 1
            

        # Control cost (-1 to 0)
        quad_ctrl_cost = CTRL_COST_WEIGHT * np.sum(np.square(data.ctrl))
        quad_ctrl_cost = np.clip(quad_ctrl_cost, 0, 1)
        
        # Impact cost (-0.5 to 0)
      #  impact_forces = np.sum(np.square(data.cfrc_ext))
     #   quad_impact_cost = IMPACT_COST_WEIGHT * min(impact_forces, IMPACT_COST_RANGE)
      #  quad_impact_cost = np.clip(quad_impact_cost, 0, 0.5)
        
        # Final reward (-1.5 to 2)
        reward = uph_cost - quad_ctrl_cost # - quad_impact_cost
        
        #if self.episode_steps % 100 == 0:
         #   print(f"Height: {head_height:.2f}, Reward Components: Height={uph_cost:.2f}, Control={-quad_ctrl_cost:.2f}, Impact={-quad_impact_cost:.2f}")
        
        
        if(self.total_steps % 1000 == 0 and VISUALISE):
            print(f"Distance from target: {head_height1 - TARGET_HEIGHT:.2f}")   
            print(f"Control Cost: {quad_ctrl_cost}")
            print(f"Height1: {head_height1})")
            print(f"UPH Cost: {uph_cost}")
            print(f"Reward: {reward:.2f}")
            

        return reward, head_height1
    
    def reset(self, data):
            # Use the supine pose 
            starting_pose = model.key_qpos[5]  
            data.qpos[:] = starting_pose
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            
           
            self.episode_steps = 0
            self.total_reward = 0
            self.height_maintained_steps = 0
            return get_state(data)

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
        self.actor_log_std = nn.Parameter(torch.zeros(num_outputs))  # Changed to 1D
        
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

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
    
    def add(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.0005)
        
        self.clip_param = 0.2 # increase if stuck in local minima
        self.ppo_epochs = 10  # increase if training too slowly
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.1 # increase for more exploration
        
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

def get_state(data):
    # Get relevant state information
        return np.concatenate([
        data.qpos.flat[2:],  # Skip root x/y coordinates
        data.qvel.flat,
        data.cinert.flat,
        data.cvel.flat,
        data.qfrc_actuator.flat,
        data.cfrc_ext.flat,
    ])

def train_headless():
    task = StandupTask()
    state_dim = len(get_state(data))
    action_dim = model.nu
    agent = PPO(state_dim, action_dim)
    memory = Memory()
    
    state = task.reset(data)
    cumulative_reward = 0

    rewards_history = []
    heights_history = []
    
    while task.total_steps < MAX_STEPS:
        # Get action and value
        action, log_prob = agent.get_action(state)
        _, _, value = agent.actor_critic(torch.FloatTensor(agent.normalize_state(state)).to(agent.device))
        
        # Execute action
        data.ctrl = np.clip(action, -1, 1)
       # data.ctrl = action
        mujoco.mj_step(model, data)
        next_state = get_state(data)
        reward, height = task.calculate_reward(data)

               # Update stats
        
        task.total_steps += 1
        
        # Store transition
        memory.add(state, action, reward, value.cpu().item(), log_prob)
        agent.update_state_stats(next_state)
        state = next_state
        
        # Train periodically
        if len(memory.states) >= TRAIN_INTERVAL:
            agent.train(memory)
            memory.clear()
            
        # Save checkpoint periodically
        if task.total_steps % SAVE_INTERVAL == 0:
            
            task.save_checkpoint(agent.actor_critic, task.total_steps, reward)
           
            
        # Print progress
        if task.total_steps % 1000 == 0:
            print(f"Step {task.total_steps}, Recent Reward: {reward:.3f}, Recent Height: {height:.3f}")
            rewards_history.append(reward)
            heights_history.append(height)

    # After training, create and save plots
    window = 1000  # Moving average window
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Create plots at end of training
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(rewards_history, 'b-')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    
    ax2.plot(heights_history, 'r-')
    ax2.axhline(y=TARGET_HEIGHT, color='g', linestyle='--', label='Target')
    ax2.set_title('Head Height')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Height')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()


def center_camera_on_humanoid(camera, data, model):
    # Get torso position
    torso_pos = data.xpos[model.body('torso').id]
    
    # Update camera lookat to track torso
    camera.lookat[0] = torso_pos[0]  # x
    camera.lookat[1] = torso_pos[1]  # y
    camera.lookat[2] = torso_pos[2]  # z


def main():
    if VISUALISE:

        # Initialize GLFW and create window
        count = 0
        glfw.init()
        window = glfw.create_window(1200, 900, "Standup Task", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # Setup scene with original camera setup
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=100000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Set default camera and options with adjusted angle
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 6.0
        camera.azimuth = 180 
        camera.elevation = -20
        mujoco.mjv_defaultOption(option)

        # Initialize task and agent
        task = StandupTask()
        state_dim = len(get_state(data))
        action_dim = model.nu
        agent = PPO(state_dim, action_dim)
        memory = Memory()

        # Initialize state
        state = task.reset(data)
        paused = False

        # Remove unused mouse variables
        lastx = 0
        lasty = 0
        button_left = False
        button_right = False

        def keyboard(window, key, scancode, act, mods):
            nonlocal paused
            if act == glfw.PRESS or act == glfw.REPEAT:  # Handle key repeats
                if key == glfw.KEY_ESCAPE:
                    glfw.set_window_should_close(window, True)
                elif key == glfw.KEY_SPACE:
                    paused = not paused
                elif key == glfw.KEY_S:
                    task.save_checkpoint(agent.actor_critic, count, task.total_reward)
                # Add camera rotation controls
                elif key == glfw.KEY_LEFT:
                    camera.azimuth = (camera.azimuth + 5) % 360
                elif key == glfw.KEY_RIGHT:
                    camera.azimuth = (camera.azimuth - 5) % 360

        glfw.set_key_callback(window, keyboard)

        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window)
        
        # Add checkpoint list state
        checkpoint_files = []
        selected_checkpoint = -1
        
        def update_checkpoint_list():
            nonlocal checkpoint_files
            checkpoint_files = glob.glob(os.path.join("checkpoints", "*.pt"))
            checkpoint_files.sort(key=os.path.getctime, reverse=True)  # Sort by creation time

        update_checkpoint_list()

        while not glfw.window_should_close(window):
            if not paused:
                # Get action from policy
                action, log_prob = agent.get_action(state)
                _, _, value = agent.actor_critic(torch.FloatTensor(agent.normalize_state(state)).to(agent.device))
                
                # Execute action
                data.ctrl = np.clip(action, -1, 1)
               # data.ctrl = action
                mujoco.mj_step(model, data)
                next_state = get_state(data)
                reward, height = task.calculate_reward(data)
                
                task.total_steps += 1
                task.total_reward += reward
                
                # Store transition
                memory.add(state, action, reward, value.cpu().item(), log_prob)
                agent.update_state_stats(next_state)
                state = next_state

                # Train periodically
                if len(memory.states) >= TRAIN_INTERVAL:
                    agent.train(memory)
                    memory.clear()
                
                # Save checkpoint periodically
                if task.total_steps % SAVE_INTERVAL == 0:
                    task.save_checkpoint(agent.actor_critic, task.total_steps, task.total_reward/SAVE_INTERVAL)

            center_camera_on_humanoid(camera,data,model)
            # Start ImGui frame
            impl.process_inputs()
            imgui.new_frame()

            imgui.set_next_window_position(10, 10, imgui.ONCE)
            imgui.begin("Simulation Stats", True)
            imgui.set_window_size(200, 100, imgui.FIRST_USE_EVER)
            imgui.text(f"Steps: {task.total_steps}")
            imgui.text(f"Current Reward: {reward:.3f}")
            imgui.end()

            # Create checkpoint selector window
            imgui.set_next_window_position(10, 100, imgui.ONCE)
            imgui.begin("Checkpoint Selector", True)
            imgui.set_window_size(300, 200, imgui.FIRST_USE_EVER)
            
            if imgui.button("Refresh List"):
                update_checkpoint_list()
            
            imgui.separator()
            
            for i, checkpoint_path in enumerate(checkpoint_files):
                filename = os.path.basename(checkpoint_path)
                if imgui.selectable(filename, selected_checkpoint == i)[0]:
                    selected_checkpoint = i
                    reward = task.load_checkpoint(agent.actor_critic, checkpoint_path)
            
            imgui.end()

            # Render ImGui
            imgui.render()
            
            # Original rendering code
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            
            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # Render ImGui over MuJoCo
            impl.render(imgui.get_draw_data())
            
            glfw.swap_buffers(window)
            glfw.poll_events()

        # Cleanup
        impl.shutdown()
        glfw.terminate()
    
    else:
        train_headless()

if __name__ == "__main__":
    main()
