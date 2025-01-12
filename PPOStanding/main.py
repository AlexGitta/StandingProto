import mujoco
import glfw
import torch
import numpy as np
import imgui
from imgui.integrations.glfw import GlfwRenderer
import matplotlib.pyplot as plt
import glob
from config.config import *
from models.ppo import PPO
from environment.standup_task import StandupTask
from environment.camera import InputState, center_camera_on_humanoid
from utils.memory import Memory
from utils.state_utils import get_state
from utils.replay_buffer import ReplayBuffer



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

def train_headless(episodes = HEADLESS_EPOCHS, max_steps = MAX_EPISODE_STEPS, print_epochs = 10):
    task = StandupTask()
    state_dim = len(get_state(task.data))
    action_dim = task.model.nu
    agent = PPO(state_dim, action_dim)
    memory = Memory()

    # Initialize replay buffer
    buffer_capacity = BUFFER_CAP  # Adjust size as needed
    replay_buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
    
    rewards_history = []
    heights_history = []
    timealive_history = []
    
    for episode in range(episodes):
        state = task.reset(task.data)
        episode_reward = 0
        episode_steps = 0
        all_height = 0

        for step in range(max_steps):
            # Get action and value
            action, log_prob = agent.get_action(state)
            _, _, value = agent.actor_critic(torch.FloatTensor(agent.normalize_state(state)).to(agent.device))
            
            # Execute action
            task.data.ctrl = np.clip(action, -1, 1)
            mujoco.mj_step(task.model, task.data)
            next_state = get_state(task.data)
            reward, height = task.calculate_reward(task.data)

            episode_reward += reward
            episode_steps += 1

            all_height += height
            
            # Check if episode is done
            torso_height = task.data.xpos[task.model.body('torso').id][2]
            done = torso_height < EARLY_TERMINATION_HEIGHT or step >= MAX_EPISODE_STEPS
            
            # Store in replay buffer
            replay_buffer.add(state, action, reward, next_state, value.cpu().item(), log_prob, done)
            
            # Store in regular memory for PPO update
            memory.add(state, action, reward, value.cpu().item(), log_prob)
            
            episode_reward += reward
            episode_steps += 1
            all_height += height
            
            agent.update_state_stats(next_state)
            state = next_state
            
            # Train periodically
            if len(memory.states) >= TRAIN_INTERVAL:
               # First train on current experiences
                agent.train(memory)
                memory.clear()
                
                # Then train on a batch of replay experiences
                if replay_buffer.size > 1000:  # Wait until buffer has enough samples
                    replay_batch_size = 1024  # Adjust as needed
                    replay_states, replay_actions, replay_rewards, replay_next_states, \
                    replay_values, replay_log_probs, replay_masks = replay_buffer.sample(replay_batch_size)
                    
                    # Create temporary memory with sampled experiences
                    replay_memory = Memory()
                    for i in range(replay_batch_size):
                        replay_memory.add(
                            replay_states[i].numpy(),
                            replay_actions[i].numpy(),
                            replay_rewards[i].item(),
                            replay_values[i].item(),
                            replay_log_probs[i].item()
                        )
                    agent.train(replay_memory)
                

            
            if done:
                break
        
        if episode % print_epochs == 0:
            print(f"Episode {episode}, Average Reward: {episode_reward/episode_steps:.3f} , Lasted {episode_steps} steps")

        if episode % SAVE_AT_EPOCH == 0 and episode != 0:
            task.save_checkpoint(agent.actor_critic, task.total_steps, reward)
            print(f"Checkpoint saved at episode {episode}")
        
        rewards_history.append(episode_reward/episode_steps)
        heights_history.append(all_height/episode_steps)
        timealive_history.append(episode_steps)

   
    # Create plots at end of training
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    ax1.plot(rewards_history, 'b-')
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Reward')
    
    ax2.plot(heights_history, 'r-')
    ax2.axhline(y=TARGET_HEIGHT, color='g', linestyle='--', label='Target')
    ax2.set_title('Average Head Height')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Height')
    ax2.legend()
    
    ax3.plot(timealive_history, 'g-')
    ax3.set_title('Time Alive per Episode')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Steps')

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()


def main():
    task = StandupTask()
    if VISUALISE:
        
        input_state = InputState()
        # Initialize GLFW and create window
        episode = 1
        glfw.init()
        window = glfw.create_window(1200, 900, "Standup Task", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # Setup scene with original camera setup
        camera = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(task.model, maxgeom=100000)
        context = mujoco.MjrContext(task.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Set default camera and options with adjusted angle
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 6.0
        camera.azimuth = 180 
        camera.elevation = -20
        mujoco.mjv_defaultOption(option)

        # Initialize agent
        state_dim = len(get_state(task.data))
        action_dim = task.model.nu
        agent = PPO(state_dim, action_dim)
        memory = Memory()

        # Initialize state
        state = task.reset(task.data)
        paused = False

        def handle_input(window, input_state, camera):
            def mouse_button_callback(window, button, action, mods):
                if button == glfw.MOUSE_BUTTON_LEFT:
                    input_state.left_down = action == glfw.PRESS
                    if input_state.left_down:
                        input_state.last_x, input_state.last_y = glfw.get_cursor_pos(window)
                elif button == glfw.MOUSE_BUTTON_RIGHT:
                    input_state.right_down = action == glfw.PRESS

            def mouse_move_callback(window, xpos, ypos):
                input_state.mouse_x = xpos
                input_state.mouse_y = ypos
                
                if input_state.left_down and input_state.camera_mode:
                    dx = xpos - input_state.last_x
                    dy = ypos - input_state.last_y
                    camera.azimuth += dx * 0.5
                    camera.elevation = np.clip(camera.elevation - dy * 0.5, -90, 90)
                    input_state.last_x = xpos
                    input_state.last_y = ypos

            def keyboard_callback(window, key, scancode, action, mods):
                nonlocal paused
                if action == glfw.PRESS:
                    if key == glfw.KEY_ESCAPE:
                        glfw.set_window_should_close(window, True)
                    elif key == glfw.KEY_SPACE:
                        paused = not paused
                    elif key == glfw.KEY_S:
                        task.save_checkpoint(agent.actor_critic, episode, task.total_reward)
            
                # Track Alt key for camera mode
                input_state.camera_mode = (mods & glfw.MOD_ALT)

                # Camera azimuth control
                if key in [glfw.KEY_LEFT, glfw.KEY_RIGHT]:
                    if action == glfw.PRESS or action == glfw.REPEAT:
                        if key == glfw.KEY_LEFT:
                            camera.azimuth = (camera.azimuth + 2) % 360
                        elif key == glfw.KEY_RIGHT: 
                            camera.azimuth = (camera.azimuth - 2) % 360

            glfw.set_mouse_button_callback(window, mouse_button_callback)
            glfw.set_cursor_pos_callback(window, mouse_move_callback) 
            glfw.set_key_callback(window, keyboard_callback)

        # Initialize ImGui
        imgui.create_context()
        impl = GlfwRenderer(window)
        handle_input(window, input_state, camera)
        
        # Add checkpoint list state
        checkpoint_files = []
        selected_checkpoint = -1
        
        def update_checkpoint_list():
            nonlocal checkpoint_files
            checkpoint_files = glob.glob(os.path.join("checkpoints", "*.pt"))
            checkpoint_files.sort(key=os.path.getctime, reverse=True)  # Sort by creation time

        update_checkpoint_list()

                
        
        reward_buffer = np.zeros(PLOT_STEPS, dtype=np.float32)
        reward_index = 0
        reward_min = float('inf')
        reward_max = float('-inf')

        while not glfw.window_should_close(window):
            if not paused:
                # Get action from policy
                action, log_prob = agent.get_action(state)
                _, _, value = agent.actor_critic(torch.FloatTensor(agent.normalize_state(state)).to(agent.device))
                
                # Execute action
                task.data.ctrl = np.clip(action, -1, 1)
               # data.ctrl = action
                mujoco.mj_step(task.model, task.data)
                next_state = get_state(task.data)
                reward, height = task.calculate_reward(task.data)
                reward_buffer[reward_index] = reward
                reward_index = (reward_index + 1) % PLOT_STEPS
                reward_min = min(reward_min, np.min(reward_buffer))
                reward_max = max(reward_max, np.max(reward_buffer))


                task.total_steps += 1
                task.episode_steps += 1
                task.total_reward += reward

                # Train and clear at interval
                if len(memory.states) >= TRAIN_INTERVAL:
                    agent.train(memory)
                    memory.clear()

                # Store transition
                memory.add(state, action, reward, value.cpu().item(), log_prob)
                agent.update_state_stats(next_state)
                state = next_state

                # Check episode end
                torso_height = task.data.xpos[task.model.body('torso').id][2]
                if task.episode_steps >= MAX_EPISODE_STEPS or torso_height < EARLY_TERMINATION_HEIGHT:
                    print(f"Episode " +str(episode)+ " ended after ", task.episode_steps, "steps")
                    print(f"Average episode reward: {task.total_reward / task.episode_steps:.3f}")
                    if episode % SAVE_AT_EPOCH == 0 and episode != 0:
                        task.save_checkpoint(agent.actor_critic, task.total_steps, reward)
                        print(f"Checkpoint saved at episode {episode}")
                    task.episode_steps = 0
                    episode += 1
                   
                    state = task.reset(task.data)

            center_camera_on_humanoid(camera,task.data,task.model)
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

            imgui.set_next_window_position(10, 300, imgui.ONCE)
            imgui.begin('Controls', True)
            imgui.text("Hold Alt + Left Mouse to rotate camera")
            imgui.text("Press Space key to pause/unpause")
            imgui.text("Press S key to save checkpoint")
            imgui.end()

            imgui.set_next_window_position(10, 500, imgui.ONCE)
            imgui.begin("Reward History", True)
            imgui.set_window_size(300, 200, imgui.FIRST_USE_EVER)

           
            # Plot the rewards
            if len(reward_buffer) > 1:
                imgui.plot_lines("##rewards", 
                                reward_buffer,
                                graph_size=(285, 150),
                                scale_min=reward_min,
                                scale_max=reward_max)                                    
            imgui.text(f"Min: {reward_min:.2f} Max: {reward_max:.2f}")
            imgui.end()


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
            
            # Mujoco rendering code
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            
            mujoco.mjv_updateScene(
                task.model,
                task.data,
                option,
                None,
                camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene)
            mujoco.mjr_render(viewport, scene, context)
            
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
