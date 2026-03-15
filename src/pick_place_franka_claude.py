import os
import numpy as np
import imageio.v2 as imageio
import mujoco
import mujoco.viewer

MODEL_PATH = "models/panda/franka_emika_panda/scene.xml"
VIDEO_PATH = "outputs/pick_place_franka_fixed.mp4"

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

# -----------------------------
# Physics tuning for reliable grasping
# Higher friction and lower timestep reduce cube slip.
# -----------------------------
model.opt.timestep    = 0.001   # 1 ms — more stable contact resolution
model.geom_friction[:] = np.array([2.0, 0.005, 0.0001])  # sliding, torsion, rolling

SIM_DT       = model.opt.timestep
RENDER_EVERY = max(1, int(1 / 60 / SIM_DT))   # render at ~60 fps

# -----------------------------
# Video / renderer
# -----------------------------
os.makedirs("outputs", exist_ok=True)
renderer     = mujoco.Renderer(model, height=480, width=640)
video_writer = imageio.get_writer(VIDEO_PATH, fps=60)

def save_frame():
    renderer.update_scene(data)
    video_writer.append_data(renderer.render())

# -----------------------------
# Helpers
# -----------------------------
def body_id(name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                 for i in range(model.nbody)]
        raise ValueError(f"Body '{name}' not found. Have: {names}")
    return bid

def actuator_id(name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                 for i in range(model.nu)]
        raise ValueError(f"Actuator '{name}' not found. Have: {names}")
    return aid

# Body & actuator IDs
cube_bid    = body_id("cube")
gripper_bid = body_id("hand")
lf_bid      = body_id("left_finger")
rf_bid      = body_id("right_finger")

ARM_ACTUATOR_NAMES = [
    "actuator1","actuator2","actuator3","actuator4",
    "actuator5","actuator6","actuator7",
]
arm_aids   = [actuator_id(n) for n in ARM_ACTUATOR_NAMES]
finger_aid = actuator_id("actuator8")   # single mimic actuator for both fingers

# Find cube freejoint qpos address (used ONCE at startup)
cube_jnt_adr = None
for j in range(model.njnt):
    if (model.jnt_bodyid[j] == cube_bid and
            model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE):
        cube_jnt_adr = model.jnt_qposadr[j]
        break
if cube_jnt_adr is None:
    raise ValueError("Cube freejoint not found.")

def set_cube_pose(pos, quat=None):
    """Used ONCE at startup to place the cube. Never called again."""
    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[cube_jnt_adr:cube_jnt_adr+3]   = pos
    data.qpos[cube_jnt_adr+3:cube_jnt_adr+7] = quat
    data.qvel[cube_jnt_adr:cube_jnt_adr+6]   = 0.0

# -----------------------------
# Contact diagnostic
# Counts contacts on the cube to verify grasp quality.
# -----------------------------
def cube_contact_count():
    count = 0
    for i in range(data.ncon):
        c  = data.contact[i]
        g1 = model.geom_bodyid[c.geom1]
        g2 = model.geom_bodyid[c.geom2]
        if cube_bid in (g1, g2):
            count += 1
    return count

def check_grasp(min_contacts=2):
    n = cube_contact_count()
    status = "OK  " if n >= min_contacts else "WARN"
    print(f"  [{status}] {n} contacts on cube (need >= {min_contacts})")
    if n < min_contacts:
        print("        -> Grasp may be weak. Try tuning AT_CUBE waypoint or friction.")
    return n >= min_contacts

# -----------------------------
# Control helpers
# -----------------------------
def set_arm(q):
    for i, aid in enumerate(arm_aids):
        data.ctrl[aid] = q[i]

def set_grip(width):
    """width = total gripper opening in metres."""
    data.ctrl[finger_aid] = width

# -----------------------------
# Simulation helpers  (no fake carry, no weld, no teleportation)
# -----------------------------
def smoothstep(a):
    return a * a * (3 - 2 * a)

def move_to(target_arm, target_grip, duration_s, viewer):
    """
    Interpolate ctrl targets and drive with mj_step.
    The cube moves ONLY because contact forces from the fingers act on it.
    No weld constraints, no kinematic carry, no qpos hacking.
    """
    n          = int(duration_s / SIM_DT)
    start_arm  = np.array([data.ctrl[aid] for aid in arm_aids])
    start_grip = data.ctrl[finger_aid]

    for i in range(n):
        a = smoothstep((i + 1) / n)
        set_arm(  (1 - a) * start_arm  + a * np.array(target_arm))
        set_grip( (1 - a) * start_grip + a * target_grip)
        mujoco.mj_step(model, data)          # full dynamics integration
        if i % RENDER_EVERY == 0:
            viewer.sync()
            save_frame()

def hold(duration_s, viewer):
    n = int(duration_s / SIM_DT)
    for i in range(n):
        mujoco.mj_step(model, data)
        if i % RENDER_EVERY == 0:
            viewer.sync()
            save_frame()

# -----------------------------
# Waypoints  (7-DOF arm, joint space)
# AT_CUBE: fingers must straddle the cube — tune joint2/joint4 for your scene.
# -----------------------------
HOME        = np.array([ 0.00, -0.60,  0.00, -2.00,  0.00,  1.40,  0.70])
ABOVE_CUBE  = np.array([ 0.25, -0.35,  0.00, -1.90,  0.00,  1.55,  0.85])
AT_CUBE     = np.array([ 0.25, -0.15,  0.00, -1.70,  0.00,  1.55,  0.85])
LIFT        = np.array([ 0.25, -0.50,  0.00, -2.00,  0.00,  1.50,  0.85])
ABOVE_GOAL  = np.array([-0.20, -0.35,  0.00, -1.90,  0.00,  1.55,  0.85])
AT_GOAL     = np.array([-0.20, -0.15,  0.00, -1.70,  0.00,  1.55,  0.85])

GRIP_OPEN   = 0.08   # fully open — wider than cube
GRIP_CLOSED = 0.00   # fully closed — squeeze the cube

# -----------------------------
# Main
# -----------------------------
print("=== Actuators ===")
for i in range(model.nu):
    print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)}")
print("=== Bodies ===")
for i in range(model.nbody):
    print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")

mujoco.mj_resetData(model, data)
set_cube_pose(np.array([0.55, 0.00, 0.03]))   # initial placement only
set_arm(HOME)
set_grip(GRIP_OPEN)
mujoco.mj_forward(model, data)                 # compute initial kinematics

print(f"\ndt={SIM_DT:.4f}s  render_every={RENDER_EVERY} steps")
print("Running fully contact-based pick-and-place (no fake grasp)...\n")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:

        print("1. Settling...")
        hold(1.5, viewer)

        print("2. Moving to home...")
        move_to(HOME, GRIP_OPEN, 2.0, viewer)
        hold(0.3, viewer)

        print("3. Moving above cube...")
        move_to(ABOVE_CUBE, GRIP_OPEN, 2.0, viewer)
        hold(0.3, viewer)

        print("4. Descending to cube (gripper open)...")
        move_to(AT_CUBE, GRIP_OPEN, 1.5, viewer)
        hold(0.5, viewer)

        print("5. Closing gripper — contact grasp...")
        move_to(AT_CUBE, GRIP_CLOSED, 1.2, viewer)
        hold(0.6, viewer)
        check_grasp(min_contacts=2)

        print("6. Lifting — cube carried by contact forces only...")
        move_to(LIFT, GRIP_CLOSED, 2.0, viewer)
        hold(0.5, viewer)
        print(f"   Contacts during lift: {cube_contact_count()}")

        print("7. Moving to above goal...")
        move_to(ABOVE_GOAL, GRIP_CLOSED, 2.5, viewer)
        hold(0.3, viewer)

        print("8. Lowering to place...")
        move_to(AT_GOAL, GRIP_CLOSED, 1.5, viewer)
        hold(0.5, viewer)

        print("9. Opening gripper — cube released under gravity...")
        move_to(AT_GOAL, GRIP_OPEN, 1.0, viewer)
        hold(0.8, viewer)

        print("10. Retreating...")
        move_to(ABOVE_GOAL, GRIP_OPEN, 1.5, viewer)
        hold(0.3, viewer)

        print("11. Returning home...")
        move_to(HOME, GRIP_OPEN, 2.0, viewer)
        hold(1.5, viewer)

finally:
    video_writer.close()
    renderer.close()

print(f"\nDone. Video saved to: {VIDEO_PATH}")