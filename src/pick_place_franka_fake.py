import os
import time
import numpy as np
import imageio.v2 as imageio
import mujoco
import mujoco.viewer

MODEL_PATH = "models/panda/franka_emika_panda/scene.xml"
VIDEO_PATH = "outputs/pick_place_franka_safe.mp4"

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# -----------------------------
# Video / renderer
# -----------------------------
os.makedirs("outputs", exist_ok=True)
renderer = mujoco.Renderer(model, height=480, width=640)
video_writer = imageio.get_writer(VIDEO_PATH, fps=30)

def save_frame():
    renderer.update_scene(data)
    pixels = renderer.render()
    video_writer.append_data(pixels)

# -----------------------------
# Names you may need to adjust
# -----------------------------
CUBE_BODY_NAME = "cube"

# In many Panda models, the hand body is called "hand"
# If this fails, print body names and change it.
GRIPPER_BODY_NAME = "hand"

# -----------------------------
# Helpers
# -----------------------------
def body_id(name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise ValueError(f"Body '{name}' not found.")
    return bid

cube_bid = body_id(CUBE_BODY_NAME)
gripper_bid = body_id(GRIPPER_BODY_NAME)

# Find freejoint qpos start index for cube
cube_jnt_adr = None
for j in range(model.njnt):
    j_body = model.jnt_bodyid[j]
    if j_body == cube_bid and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
        cube_jnt_adr = model.jnt_qposadr[j]
        break

if cube_jnt_adr is None:
    raise ValueError("Cube freejoint not found. Make sure cube has <freejoint/>.")

def set_cube_pose(pos, quat=None):
    """Set cube world pose via freejoint qpos."""
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
    data.qpos[cube_jnt_adr:cube_jnt_adr+3] = pos
    data.qpos[cube_jnt_adr+3:cube_jnt_adr+7] = quat

def get_body_pos(bid):
    return data.xpos[bid].copy()

def move_qpos_smooth(target_qpos_arm, target_grip, steps, viewer, carry_cube=False):
    """
    Smoothly interpolate the Panda arm joints and gripper.
    Panda here is assumed to have 7 arm joints + 2 finger joints in qpos[0:9].
    """
    start = data.qpos[:9].copy()
    target = start.copy()
    target[:7] = target_qpos_arm
    target[7] = target_grip
    target[8] = target_grip

    for i in range(steps):
        alpha = (i + 1) / steps
        data.qpos[:9] = (1 - alpha) * start + alpha * target
        mujoco.mj_forward(model, data)

        if carry_cube:
            # Soft grasp: keep cube slightly below gripper center
            gpos = get_body_pos(gripper_bid)
            cube_pos = gpos + np.array([0.0, 0.0, -0.06])
            set_cube_pose(cube_pos)
            mujoco.mj_forward(model, data)

        viewer.sync()
        save_frame()
        time.sleep(0.01)

def hold_steps(n, viewer, carry_cube=False):
    for _ in range(n):
        mujoco.mj_forward(model, data)
        if carry_cube:
            gpos = get_body_pos(gripper_bid)
            cube_pos = gpos + np.array([0.0, 0.0, -0.06])
            set_cube_pose(cube_pos)
            mujoco.mj_forward(model, data)
        viewer.sync()
        save_frame()
        time.sleep(0.01)

# -----------------------------
# Safe conservative waypoints
# These are chosen to keep the arm in front of the robot
# and reduce self-collision risk.
# You may tune slightly for your exact scene.
# -----------------------------
HOME        = np.array([ 0.00, -0.60,  0.00, -2.00,  0.00,  1.40,  0.70])
ABOVE_CUBE  = np.array([ 0.25, -0.35,  0.00, -1.90,  0.00,  1.55,  0.85])
AT_CUBE     = np.array([ 0.25, -0.20,  0.00, -1.75,  0.00,  1.55,  0.85])
LIFT        = np.array([ 0.25, -0.45,  0.00, -1.95,  0.00,  1.50,  0.85])
ABOVE_GOAL  = np.array([-0.20, -0.35,  0.00, -1.90,  0.00,  1.55,  0.85])
AT_GOAL     = np.array([-0.20, -0.20,  0.00, -1.75,  0.00,  1.55,  0.85])

GRIP_OPEN = 0.04
GRIP_CLOSED = 0.00

# -----------------------------
# Main
# -----------------------------
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Set cube initial pose explicitly
set_cube_pose(np.array([0.55, 0.00, 0.03]))
mujoco.mj_forward(model, data)

print("Loaded model.")
print("nq =", model.nq, "nu =", model.nu)
print("Running safe scripted pick-and-place...")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Save a few initial frames
        for _ in range(10):
            viewer.sync()
            save_frame()
            time.sleep(0.01)

        # Start open at home
        move_qpos_smooth(HOME, GRIP_OPEN, 200, viewer)
        hold_steps(40, viewer)

        # Move above cube
        move_qpos_smooth(ABOVE_CUBE, GRIP_OPEN, 220, viewer)
        hold_steps(30, viewer)

        # Move down to cube
        move_qpos_smooth(AT_CUBE, GRIP_OPEN, 180, viewer)
        hold_steps(30, viewer)

        # Close gripper
        move_qpos_smooth(AT_CUBE, GRIP_CLOSED, 120, viewer)
        hold_steps(20, viewer)

        # Lift while carrying cube
        move_qpos_smooth(LIFT, GRIP_CLOSED, 220, viewer, carry_cube=True)
        hold_steps(30, viewer, carry_cube=True)

        # Move above goal while carrying cube
        move_qpos_smooth(ABOVE_GOAL, GRIP_CLOSED, 260, viewer, carry_cube=True)
        hold_steps(30, viewer, carry_cube=True)

        # Move down to place
        move_qpos_smooth(AT_GOAL, GRIP_CLOSED, 180, viewer, carry_cube=True)
        hold_steps(20, viewer, carry_cube=True)

        # Open gripper and leave cube there
        gpos = get_body_pos(gripper_bid)
        place_pos = gpos + np.array([0.0, 0.0, -0.06])
        set_cube_pose(place_pos)
        mujoco.mj_forward(model, data)
        move_qpos_smooth(AT_GOAL, GRIP_OPEN, 120, viewer)
        hold_steps(30, viewer)

        # Retreat
        move_qpos_smooth(ABOVE_GOAL, GRIP_OPEN, 180, viewer)
        hold_steps(60, viewer)

        # Return home
        move_qpos_smooth(HOME, GRIP_OPEN, 220, viewer)
        hold_steps(300, viewer)

finally:
    video_writer.close()
    renderer.close()

print(f"Done. Video saved to: {VIDEO_PATH}")