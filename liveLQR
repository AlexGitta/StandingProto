import mujoco
import glfw
import mediapy as media
import numpy as np
from typing import Callable, Optional, Union, List
import scipy.linalg
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

with open('./humanoid.xml', 'r') as f:
  humanoid = f.read()
  model = mujoco.MjModel.from_xml_string(humanoid)
  data = mujoco.MjData(model)

glfw.init()

# Add global pause state
paused = False

def keyboard(window, key, scancode, act, mods):
    global paused
    if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mujoco.mj_resetData(model, data)
        data.time = 0.0
    elif act == glfw.PRESS and key == glfw.KEY_SPACE:
        paused = not paused
    
    # Camera movement controls
    if key == glfw.KEY_W:
        camera.lookat[2] -= 0.1  # Move forward
    elif key == glfw.KEY_S:
        camera.lookat[2] += 0.1  # Move backward
    elif key == glfw.KEY_A:
        camera.lookat[0] -= 0.1  # Move left
    elif key == glfw.KEY_D:
        camera.lookat[0] += 0.1  # Move right
    elif key == glfw.KEY_Q:
        camera.lookat[1] -= 0.1  # Move down
    elif key == glfw.KEY_E:
        camera.lookat[1] += 0.1  # Move up

lastx = 0
lasty = 0
button_left = False
button_right = False

def mouse_button(window, button, act, mods):
    global button_left, button_right
    
    if button == glfw.MOUSE_BUTTON_LEFT:
        button_left = True if act == glfw.PRESS else False
    if button == glfw.MOUSE_BUTTON_RIGHT:
        button_right = True if act == glfw.PRESS else False

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_right
    
    if button_right:
        dx = xpos - lastx
        dy = ypos - lasty
        camera.azimuth += dx * 0.1
        camera.elevation += dy * 0.1
    
    lastx = xpos
    lasty = ypos

window = glfw.create_window(1200, 900, "Live Model Viewer", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Set up callback functions
glfw.set_key_callback(window, keyboard)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_cursor_pos_callback(window, mouse_move)

camera = mujoco.MjvCamera()
option = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
scene = mujoco.MjvScene(model, maxgeom=100000)
perturb = mujoco.MjvPerturb()

mujoco.mjv_defaultCamera(camera)
mujoco.mjv_defaultPerturb(perturb)
mujoco.mjv_defaultOption(option)

simstart = 0.0
dt = 0.001

def findOffset(squatorbalance):
    height_offsets = np.linspace(-0.001, 0.001, 2001)
    vertical_forces = []
    for offset in height_offsets:
        mujoco.mj_resetDataKeyframe(model, data, squatorbalance)
        mujoco.mj_forward(model, data)
        data.qacc = 0
        data.qpos[2] += offset
        mujoco.mj_inverse(model, data)
        vertical_forces.append(data.qfrc_inverse[2])
    idx = np.argmin(np.abs(vertical_forces))
    return height_offsets[idx]

def calculateForces(squatorbalance):
    mujoco.mj_resetDataKeyframe(model, data, squatorbalance)
    mujoco.mj_forward(model, data)
    data.qacc = 0
    data.qpos[2] += findOffset(squatorbalance)
    qpos0 = data.qpos.copy()
    mujoco.mj_inverse(model, data)
    qfrc0 = data.qfrc_inverse.copy()
    return qfrc0, qpos0

def calculateActuatorValues(squatorbalance):
    qfrc0, qpos0 = calculateForces(squatorbalance)
    actuator_moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        actuator_moment,
        data.actuator_moment.reshape(-1),
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind.reshape(-1),
    )
    ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
    return ctrl0.flatten(), qpos0

def balancingCost(squatorbalance):
    ctrl0, qpos0 = calculateActuatorValues(squatorbalance)
    nv = model.nv
    
    mujoco.mj_resetData(model, data)
    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    
    jac_com = np.zeros((3, nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body('torso').id)
    
    jac_foot = np.zeros((3, nv))
    mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body('foot_left').id)
    
    jac_diff = jac_com - jac_foot
    return jac_diff.T @ jac_diff

def poseCost(squatorbalance):
    nv = model.nv  # shortcut for the degrees of freedom
    # Get all joint names
    joint_names = [model.joint(i).name for i in range(model.njnt)]

    # Get indices into relevant sets of joints
    root_dofs = range(6)
    body_dofs = range(6, nv)
    abdomen_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'abdomen' in name
        and not 'z' in name
    ]
    left_leg_dofs = [
        model.joint(name).dofadr[0]
        for name in joint_names
        if 'left' in name
        and ('hip' in name or 'knee' in name or 'ankle' in name)
        and not 'z' in name
    ]
    if squatorbalance == 0:
        right_leg_dofs = [
            model.joint(name).dofadr[0]
            for name in joint_names
            if 'right' in name
            and ('hip' in name or 'knee' in name or 'ankle' in name)
            and not 'z' in name
        ]
        balance_dofs = abdomen_dofs + left_leg_dofs + right_leg_dofs
    else:
        balance_dofs = abdomen_dofs + left_leg_dofs

    other_dofs = np.setdiff1d(body_dofs, balance_dofs)
    return balance_dofs, other_dofs, root_dofs

def constructQ(squatorbalance, balance_cost, important_joint_cost, other_joint_cost):
    nv = model.nv  # shortcut for the degrees of freedom
    # get indices into relevant sets of joints
    balance_dofs, other_dofs, root_dofs = poseCost(squatorbalance)

    Qjoint = np.eye(nv)
    # don't penalize root joint
    Qjoint[root_dofs, root_dofs] *= 0
    # multiply key balancing joints deviation by cost
    Qjoint[balance_dofs, balance_dofs] *= important_joint_cost
    # multiply other joints deviation by correlating cost
    Qjoint[other_dofs, other_dofs] *= other_joint_cost

    # calculate full cost
    Qpos = balance_cost * balancingCost(squatorbalance) + Qjoint

    # not penalising speed of joint movements, only deviation from pose
    Q = np.block([[Qpos, np.zeros((nv, nv))],
                  [np.zeros((nv, 2*nv))]])
    return Q

def constructK(squatorbalance):
    nu = model.nu
    nv = model.nv
    ctrl0, qpos0 = calculateActuatorValues(squatorbalance)

    mujoco.mj_resetData(model, data)
    data.ctrl = ctrl0
    data.qpos = qpos0

    A = np.zeros((2*nv, 2*nv))
    B = np.zeros((2*nv, nu))
    epsilon = 1e-6
    flg_centered = True
    mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)
    return A, B

def initializeLQR(squatorbalance=1, balance_cost=100, important_joint_cost=3.5, other_joint_cost=0.2):
    # Get initial state and control
    ctrl0, qpos0 = calculateActuatorValues(squatorbalance)
    
    # Construct system matrices
    A, B = constructK(squatorbalance)
    
    # Construct Q and R matrices
    Q = constructQ(squatorbalance, balance_cost, important_joint_cost, other_joint_cost)
    R = np.eye(model.nu)
    
    # Solve Riccati equation
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
    # Compute optimal feedback gain
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    return K, ctrl0, qpos0

# Initialize LQR controller before main loop
K, ctrl0, qpos0 = initializeLQR()
data.qpos = qpos0

while not glfw.window_should_close(window):
    if not paused:
        # Compute state difference
        dq = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
        dx = np.hstack((dq, data.qvel)).T
        
        # Compute control
        data.ctrl = ctrl0 - K @ dx
        
        # Add some noise for realism
        noise = 0.0 * np.random.randn(model.nu)
        data.ctrl += noise
        
        mujoco.mj_step(model, data)

    viewport = mujoco.MjrRect(0, 0, 0, 0)
    viewport.width, viewport.height = glfw.get_framebuffer_size(window)

    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene)
    mujoco.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
