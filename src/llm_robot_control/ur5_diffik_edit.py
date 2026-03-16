# This code is from https://github.com/kevinzakka/mjctrl -- not written by me, I only changed it a bit

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import imageio.v2 as imageio

os.makedirs("outputs/ur5_images", exist_ok=True)
# clear this dir
for f in os.listdir("outputs/ur5_images"):
    os.remove(os.path.join("outputs/ur5_images", f))

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 1.0

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

# Define a trajectory for the end-effector site to follow.
def get_target_xyz(t: float, r: float = 0.1, h: float = 0.5, k: float = 0.0, z0: float = 0.4, f: float = 0.5) -> np.ndarray:
    x = r * np.cos(3 * np.pi * f * t) + h
    y = r * np.sin(3 * np.pi * f * t) + k
    z = z0
    return np.array([x, y, z])

def get_target_xyz(
    t: float,
    A: np.ndarray = np.array([0.45, -0.20, 0.40]),   # pick
    B: np.ndarray = np.array([0.45,  0.20, 0.40]),   # place
    speed: float = 1,                             # smaller = slower
) -> np.ndarray:
    alpha = min(t * speed, 1.0)   # goes from 0 to 1
    return (1 - alpha) * A + alpha * B

def plan_waypoints(A: np.ndarray,
                   B: np.ndarray,
                   O_center: np.ndarray,
                   O_radius: float,
                   clearance: float = 0.08) -> list[np.ndarray]:
    """
    Return waypoints from A to B while avoiding one spherical obstacle.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    O_center = np.asarray(O_center, dtype=float)

    AB = B - A
    AB_norm = np.linalg.norm(AB)
    if AB_norm < 1e-8:
        return [A.copy(), B.copy()]

    u = AB / AB_norm

    # Closest point from obstacle center to line segment A->B
    t = np.dot(O_center - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0.0, 1.0)
    P = A + t * AB

    dist = np.linalg.norm(O_center - P)
    safe_dist = O_radius + clearance

    # If straight line is safe, no detour needed
    if dist >= safe_dist:
        return [A.copy(), B.copy()]

    # Build a perpendicular detour direction
    away = P - O_center
    away_norm = np.linalg.norm(away)

    if away_norm < 1e-8:
        # obstacle center lies almost exactly on line, choose any perpendicular
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tmp, u)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        detour_dir = np.cross(u, tmp)
        detour_dir /= np.linalg.norm(detour_dir)
    else:
        detour_dir = away / away_norm

    # Detour waypoint: push away from obstacle
    W = O_center + detour_dir * safe_dist

    # Optional: keep detour roughly between A and B in height
    W[2] = max(A[2], B[2], O_center[2] + 0.05)

    return [A.copy(), W, B.copy()]

def get_target_xyz(t: float,
                   waypoints: list[np.ndarray],
                   speed: float = 1.5) -> np.ndarray:
    """
    Move smoothly along piecewise-linear waypoint path.
    """
    if len(waypoints) == 1:
        return waypoints[0].copy()

    pts = [np.asarray(p, dtype=float) for p in waypoints]
    seg_lens = [np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1)]
    total_len = sum(seg_lens)

    if total_len < 1e-8:
        return pts[-1].copy()

    s = min(t * speed, total_len)

    acc = 0.0
    for i, L in enumerate(seg_lens):
        if s <= acc + L:
            alpha = (s - acc) / max(L, 1e-8)
            return (1 - alpha) * pts[i] + alpha * pts[i+1]
        acc += L

    return pts[-1].copy()

A = np.array([0.45, -0.20, 0.2])
B = np.array([0.45,  0.20, 0.4])
O = np.array([0.45,  0.00, 0.4])

O_radius = 0.2
waypoints = plan_waypoints(A, B, O_center=O, O_radius=0.06)


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("external/mjctrl/universal_robots_ur5e/scene.xml")

    data = mujoco.MjData(model)

    # Override the simulation timestep.
    model.opt.timestep = dt

    # End-effector site we wish to control, in this case a site attached to the last
    # link (wrist_3_link) of the robot.
    site_id = model.site("attachment_site").id

    # Name of bodies we wish to apply gravity compensation to.
    body_names = [
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]
    body_ids = [model.body(name).id for name in body_names]
    if gravity_compensation:
        model.body_gravcomp[body_ids] = 1.0

    # Get the dof and actuator ids for the joints we wish to control.
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    # Note that actuator names are the same as joint names in this case.
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_id = model.key("home").id

    # Mocap body we will control with our mouse.
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    renderer = mujoco.Renderer(model, height=480, width=640)

    os.makedirs("outputs", exist_ok=True)
    video_writer = imageio.get_writer("outputs/ur5_video.mp4", fps=30)

    with mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False) as viewer:
        # Reset the simulation to the initial keyframe.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Initialize the camera view to that of the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        viewer.cam.azimuth = 180
        viewer.cam.elevation = -20
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.5, 0.0, 0.3]

        # Toggle site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        last_vlm_time = -0.5
        vlm_period = 0.5   # seconds
        while viewer.is_running():
            step_start = time.time()

            # Set the target position of the end-effector site.
            data.mocap_pos[mocap_id] = get_target_xyz(data.time, waypoints, speed=0.5)

            # --- capture image for VLM ---
            renderer.update_scene(data)
            img = renderer.render()
            video_writer.append_data(img)

            from PIL import Image
            pil_img = Image.fromarray(img)

            if data.time - last_vlm_time >= vlm_period:
                renderer.update_scene(data)
                img = renderer.render()

                from PIL import Image
                pil_img = Image.fromarray(img)

                # send pil_img to VLM here
                print(last_vlm_time, data.time, "Captured image for VLM")
                save_path = f"outputs/ur5_images/img_{data.time:.2f}.png"
                pil_img.save(save_path)
                last_vlm_time = data.time

            # Position error.
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            # Orientation error.
            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            # Get the Jacobian with respect to the end-effector site.
            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

            # Solve system of equations: J @ dq = error.
            dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            # Scale down joint velocities if they exceed maximum.
            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set the control signal.
            np.clip(q, *model.jnt_range.T, out=q)
            data.ctrl[actuator_ids] = q[dof_ids]

            # Step the simulation.
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    video_writer.close()


if __name__ == "__main__":
    main()
