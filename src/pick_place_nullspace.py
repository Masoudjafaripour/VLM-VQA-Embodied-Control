import os
import time
import numpy as np
import imageio.v2 as imageio
import mujoco
import mujoco.viewer


# ============================================================
# Config
# ============================================================
MODEL_PATH = "external/mjctrl/franka_emika_panda/scene.xml"
VIDEO_PATH = "outputs/pick_place_nullspace.mp4"

# DiffIK / controller params
integration_dt = 0.1
damping = 1e-4
Kpos = 0.95
Kori = 0.95
gravity_compensation = True
dt = 0.002
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
max_angvel = 0.785

# Scene names
SITE_NAME = "attachment_site"
MOCAP_NAME = "target"
CUBE_BODY_NAME = "cube"
HOME_KEY_NAME = "home"

# Gripper command
# If open/close is reversed in your model, swap these two values.
GRIP_OPEN = 0.04
GRIP_CLOSED = 0.00

# Fixed end-effector orientation.
# We keep the target orientation constant to avoid weird wrist gymnastics.
# This uses the initial mocap quaternion as the default orientation.
USE_INITIAL_MOCAP_QUAT = True

# Task geometry
APPROACH_Z = 0.12
GRASP_Z = 0.035
LIFT_Z = 0.16
GOAL_POS = np.array([0.45, -0.20, 0.02])   # tune if needed
POS_TOL = 0.012
WAIT_AFTER_GRASP = 0.8
WAIT_AFTER_RELEASE = 0.6

# Video
FPS = 30


# ============================================================
# Helper functions
# ============================================================
def body_id_or_fail(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        raise ValueError(f"Body '{name}' not found in model.")
    return bid


def site_id_or_fail(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid == -1:
        raise ValueError(f"Site '{name}' not found in model.")
    return sid


def actuator_id_or_fail(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid == -1:
        raise ValueError(f"Actuator '{name}' not found in model.")
    return aid


def key_id_or_fail(model, name):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if kid == -1:
        raise ValueError(f"Keyframe '{name}' not found in model.")
    return kid


def quat_wxyz_from_mat9(mat9):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat9)
    return quat


def set_mocap_pose(data, mocap_id, pos, quat):
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = quat


def save_frame(renderer, data, writer):
    renderer.update_scene(data)
    frame = renderer.render()
    writer.append_data(frame)


def pos_reached(current, target, tol=POS_TOL):
    return np.linalg.norm(current - target) < tol


# ============================================================
# Main
# ============================================================
def main():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    os.makedirs("outputs", exist_ok=True)

    print("Loaded model.")
    print("nq =", model.nq, "nv =", model.nv, "nu =", model.nu)
    for i in range(model.nu):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"actuator {i}: {nm}")

    # --------------------------------------------------------
    # IDs / names
    # --------------------------------------------------------
    site_id = site_id_or_fail(model, SITE_NAME)
    cube_bid = body_id_or_fail(model, CUBE_BODY_NAME)
    key_id = key_id_or_fail(model, HOME_KEY_NAME)

    mocap_body_id = body_id_or_fail(model, MOCAP_NAME)
    mocap_id = model.body(MOCAP_NAME).mocapid[0]
    if mocap_id < 0:
        raise ValueError(f"Body '{MOCAP_NAME}' exists but is not a mocap body.")

    joint_names = [f"joint{i}" for i in range(1, 8)]
    dof_ids = np.array([model.joint(name).id for name in joint_names], dtype=int)
    arm_actuator_ids = np.array([actuator_id_or_fail(model, name) for name in joint_names], dtype=int)

    # Use last actuator as gripper actuator
    gripper_actuator_id = model.nu - 1

    # Home posture from keyframe
    q0 = model.key(HOME_KEY_NAME).qpos.copy()

    # --------------------------------------------------------
    # Preallocate arrays
    # --------------------------------------------------------
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # --------------------------------------------------------
    # Reset to home
    # --------------------------------------------------------
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    # Keep initial orientation as default downward-ish grasp orientation
    if USE_INITIAL_MOCAP_QUAT:
        target_quat = data.mocap_quat[mocap_id].copy()
    else:
        target_quat = quat_wxyz_from_mat9(data.site(site_id).xmat)

    # Initial gripper command
    gripper_cmd = GRIP_OPEN

    # --------------------------------------------------------
    # Task state machine
    # --------------------------------------------------------
    state = "move_above_cube"
    state_enter_time = data.time

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------
    renderer = mujoco.Renderer(model, height=480, width=640)
    writer = imageio.get_writer(VIDEO_PATH, fps=FPS)

    print("Starting pick-and-place with nullspace DiffIK...")

    try:
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            # Keep a short startup period
            for _ in range(20):
                viewer.sync()
                save_frame(renderer, data, writer)
                time.sleep(0.01)

            while viewer.is_running():
                step_start = time.time()

                # -----------------------------
                # Read current scene state
                # -----------------------------
                cube_pos = data.xpos[cube_bid].copy()
                ee_pos = data.site(site_id).xpos.copy()

                # -----------------------------
                # Pick-and-place state machine
                # -----------------------------
                if state == "move_above_cube":
                    target_pos = cube_pos + np.array([0.0, 0.0, APPROACH_Z])
                    gripper_cmd = GRIP_OPEN

                    if pos_reached(ee_pos, target_pos):
                        state = "move_to_cube"
                        state_enter_time = data.time
                        print("State -> move_to_cube")

                elif state == "move_to_cube":
                    target_pos = cube_pos + np.array([0.0, 0.0, GRASP_Z])
                    gripper_cmd = GRIP_OPEN

                    if pos_reached(ee_pos, target_pos):
                        state = "close_gripper"
                        state_enter_time = data.time
                        print("State -> close_gripper")

                elif state == "close_gripper":
                    target_pos = cube_pos + np.array([0.0, 0.0, GRASP_Z])
                    gripper_cmd = GRIP_CLOSED

                    if data.time - state_enter_time > WAIT_AFTER_GRASP:
                        state = "lift"
                        state_enter_time = data.time
                        print("State -> lift")

                elif state == "lift":
                    # Lift relative to current cube position.
                    # If grasp succeeded, cube should come with the gripper.
                    target_pos = cube_pos + np.array([0.0, 0.0, LIFT_Z])
                    gripper_cmd = GRIP_CLOSED

                    if pos_reached(ee_pos, target_pos):
                        state = "move_above_goal"
                        state_enter_time = data.time
                        print("State -> move_above_goal")

                elif state == "move_above_goal":
                    target_pos = GOAL_POS + np.array([0.0, 0.0, APPROACH_Z])
                    gripper_cmd = GRIP_CLOSED

                    if pos_reached(ee_pos, target_pos):
                        state = "move_to_goal"
                        state_enter_time = data.time
                        print("State -> move_to_goal")

                elif state == "move_to_goal":
                    target_pos = GOAL_POS + np.array([0.0, 0.0, GRASP_Z])
                    gripper_cmd = GRIP_CLOSED

                    if pos_reached(ee_pos, target_pos):
                        state = "open_gripper"
                        state_enter_time = data.time
                        print("State -> open_gripper")

                elif state == "open_gripper":
                    target_pos = GOAL_POS + np.array([0.0, 0.0, GRASP_Z])
                    gripper_cmd = GRIP_OPEN

                    if data.time - state_enter_time > WAIT_AFTER_RELEASE:
                        state = "retreat"
                        state_enter_time = data.time
                        print("State -> retreat")

                elif state == "retreat":
                    target_pos = GOAL_POS + np.array([0.0, 0.0, APPROACH_Z])
                    gripper_cmd = GRIP_OPEN

                    if pos_reached(ee_pos, target_pos):
                        state = "done"
                        state_enter_time = data.time
                        print("State -> done")

                elif state == "done":
                    target_pos = GOAL_POS + np.array([0.0, 0.0, APPROACH_Z])
                    gripper_cmd = GRIP_OPEN

                else:
                    raise ValueError(f"Unknown state: {state}")

                # -----------------------------
                # Move mocap target
                # -----------------------------
                set_mocap_pose(data, mocap_id, target_pos, target_quat)

                # -----------------------------
                # DiffIK: compute desired end-effector twist
                # -----------------------------
                dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
                twist[:3] = Kpos * dx / integration_dt

                mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
                mujoco.mju_negQuat(site_quat_conj, site_quat)
                mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
                mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
                twist[3:] *= Kori / integration_dt

                # -----------------------------
                # Jacobian
                # -----------------------------
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

                # -----------------------------
                # Damped least-squares IK
                # -----------------------------
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

                # -----------------------------
                # Nullspace posture term
                # Bias toward home configuration while preserving main task
                # -----------------------------
                # dq += (eye - np.linalg.pinv(jac) @ jac) @ (Kn * (q0[dof_ids] - data.qpos[dof_ids]))

                # --- Nullspace posture term (arm joints only)

                J_arm = jac[:, dof_ids]                     # 6 x 7
                N = np.eye(len(dof_ids)) - np.linalg.pinv(J_arm) @ J_arm   # 7 x 7

                dq_null = N @ (Kn * (q0[dof_ids] - data.qpos[dof_ids]))

                dq[dof_ids] += dq_null

                # -----------------------------
                # Clamp joint velocity
                # -----------------------------
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

                # -----------------------------
                # Integrate to desired joint positions
                # -----------------------------
                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)

                # Clip only valid joint ranges where possible
                if model.jnt_range.shape[0] >= len(dof_ids):
                    q_clip = q.copy()
                    q_clip[dof_ids] = np.clip(q[dof_ids], model.jnt_range[dof_ids, 0], model.jnt_range[dof_ids, 1])
                    q = q_clip

                # -----------------------------
                # Send control
                # Arm is position-controlled.
                # Gripper uses the last actuator.
                # -----------------------------
                data.ctrl[arm_actuator_ids] = q[dof_ids]
                data.ctrl[gripper_actuator_id] = gripper_cmd

                # -----------------------------
                # Step physics
                # -----------------------------
                mujoco.mj_step(model, data)

                # -----------------------------
                # Sync + save video
                # -----------------------------
                viewer.sync()
                save_frame(renderer, data, writer)

                # -----------------------------
                # Real-time pacing
                # -----------------------------
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                # Keep the done state visible for a bit
                if state == "done" and (data.time - state_enter_time) > 2.0:
                    break

    finally:
        writer.close()
        renderer.close()

    print(f"Done. Video saved to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()