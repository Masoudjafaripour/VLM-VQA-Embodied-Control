import os
import time
import numpy as np
import imageio.v2 as imageio
import mujoco
import mujoco.viewer

MODEL_PATH = "models/panda/franka_emika_panda/scene.xml"
VIDEO_PATH = "outputs/pick_place_franka_real.mp4"

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Loaded model")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"actuator {i}: {name}")

if model.nu != 8:
    raise ValueError(f"Expected 8 actuators for Panda menagerie model, got {model.nu}")

# --------------------------------------------------
# Video setup
# --------------------------------------------------
os.makedirs("outputs", exist_ok=True)
renderer = mujoco.Renderer(model, height=480, width=640)
writer = imageio.get_writer(VIDEO_PATH, fps=30)

def save_frame():
    renderer.update_scene(data)
    frame = renderer.render()
    writer.append_data(frame)

# --------------------------------------------------
# Utility
# --------------------------------------------------
def step_sim(viewer=None, n=1, save_video=True):
    for _ in range(n):
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if save_video:
            save_frame()

def hold_ctrl(ctrl, steps, viewer=None, save_video=True):
    data.ctrl[:] = ctrl
    step_sim(viewer=viewer, n=steps, save_video=save_video)

def goto_ctrl(target_ctrl, duration_sec, viewer=None, save_video=True):
    """
    Smoothly move from current ctrl to target ctrl using real physics stepping.
    """
    start_ctrl = data.ctrl.copy()
    n_steps = int(duration_sec / model.opt.timestep)
    n_steps = max(n_steps, 1)

    for k in range(n_steps):
        alpha = (k + 1) / n_steps
        data.ctrl[:] = (1 - alpha) * start_ctrl + alpha * target_ctrl
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if save_video:
            save_frame()

def settle(seconds, viewer=None, save_video=True):
    n_steps = int(seconds / model.opt.timestep)
    step_sim(viewer=viewer, n=n_steps, save_video=save_video)

def get_body_pos(name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise ValueError(f"Body '{name}' not found")
    return data.xpos[bid].copy()

# --------------------------------------------------
# Conservative arm waypoints
# 7 arm joints + 1 gripper actuator
# These are example joint-space targets for a front workspace.
# You will likely need slight tuning for your exact scene/table height.
# --------------------------------------------------
GRIP_OPEN   = 0.04
GRIP_CLOSED = 0.00

HOME = np.array([
    0.00, -0.60,  0.00, -2.00, 0.00, 1.40, 0.70, GRIP_OPEN
])

ABOVE_CUBE = np.array([
    0.22, -0.40,  0.00, -1.85, 0.00, 1.55, 0.82, GRIP_OPEN
])

AT_CUBE = np.array([
    0.22, -0.25,  0.00, -1.72, 0.00, 1.57, 0.82, GRIP_OPEN
])

GRASP = np.array([
    0.22, -0.25,  0.00, -1.72, 0.00, 1.57, 0.82, GRIP_CLOSED
])

LIFT = np.array([
    0.22, -0.48,  0.00, -1.92, 0.00, 1.50, 0.82, GRIP_CLOSED
])

ABOVE_GOAL = np.array([
   -0.18, -0.40,  0.00, -1.85, 0.00, 1.55, 0.82, GRIP_CLOSED
])

AT_GOAL = np.array([
   -0.18, -0.25,  0.00, -1.72, 0.00, 1.57, 0.82, GRIP_CLOSED
])

RELEASE = np.array([
   -0.18, -0.25,  0.00, -1.72, 0.00, 1.57, 0.82, GRIP_OPEN
])

RETREAT = np.array([
   -0.18, -0.45,  0.00, -1.90, 0.00, 1.50, 0.82, GRIP_OPEN
])

# --------------------------------------------------
# Main
# --------------------------------------------------
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Start from home-ish controls
data.ctrl[:] = HOME
settle(1.0, viewer=None, save_video=False)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Let the viewer appear
        settle(0.5, viewer=viewer, save_video=True)

        print("Move to HOME")
        goto_ctrl(HOME, duration_sec=2.0, viewer=viewer)
        settle(0.5, viewer=viewer)

        print("Move ABOVE_CUBE")
        goto_ctrl(ABOVE_CUBE, duration_sec=2.0, viewer=viewer)
        settle(0.4, viewer=viewer)

        print("Move AT_CUBE")
        goto_ctrl(AT_CUBE, duration_sec=1.5, viewer=viewer)
        settle(0.5, viewer=viewer)

        print("Close gripper")
        goto_ctrl(GRASP, duration_sec=1.2, viewer=viewer)
        settle(1.0, viewer=viewer)

        print("Lift")
        goto_ctrl(LIFT, duration_sec=1.8, viewer=viewer)
        settle(0.6, viewer=viewer)

        print("Move ABOVE_GOAL")
        goto_ctrl(ABOVE_GOAL, duration_sec=2.2, viewer=viewer)
        settle(0.5, viewer=viewer)

        print("Move AT_GOAL")
        goto_ctrl(AT_GOAL, duration_sec=1.6, viewer=viewer)
        settle(0.5, viewer=viewer)

        print("Open gripper")
        goto_ctrl(RELEASE, duration_sec=1.0, viewer=viewer)
        settle(1.0, viewer=viewer)

        print("Retreat")
        goto_ctrl(RETREAT, duration_sec=1.6, viewer=viewer)
        settle(0.5, viewer=viewer)

        print("Return HOME")
        goto_ctrl(HOME, duration_sec=2.2, viewer=viewer)
        settle(2.0, viewer=viewer)

finally:
    writer.close()
    renderer.close()

print(f"Done. Video saved to: {VIDEO_PATH}")