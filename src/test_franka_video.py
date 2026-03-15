import os
import time
import numpy as np
import imageio.v2 as imageio
import mujoco
import mujoco.viewer

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "models/panda/franka_emika_panda/scene.xml"
VIDEO_PATH = "outputs/franka_motion.mp4"

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Model loaded.")
print("nq =", model.nq)   # number of position coordinates
print("nv =", model.nv)   # number of velocity coordinates
print("nu =", model.nu)   # number of actuators

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Reset state
mujoco.mj_resetData(model, data)

# Save the initial configuration
base_qpos = data.qpos.copy()

# Offscreen renderer for video saving
renderer = mujoco.Renderer(model, height=480, width=640)

# Simulation / animation settings
fps = 30
duration_sec = 10
total_frames = fps * duration_sec

# We will animate the first few Panda joints directly through qpos.
# This is not a proper controller yet, but it is the easiest way
# to guarantee visible motion and a saved video.
with imageio.get_writer(VIDEO_PATH, fps=fps) as writer:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_wall = time.time()

        for frame_idx in range(total_frames):
            t = frame_idx / fps

            # Start from original pose each frame
            data.qpos[:] = base_qpos

            # Animate arm joints safely if they exist
            # Panda usually has at least 7 arm joints in qpos[0:7]
            if model.nq >= 7:
                data.qpos[0] = base_qpos[0] + 0.6 * np.sin(2.0 * np.pi * 0.20 * t)
                data.qpos[1] = base_qpos[1] + 0.4 * np.sin(2.0 * np.pi * 0.15 * t)
                data.qpos[3] = base_qpos[3] + 0.5 * np.sin(2.0 * np.pi * 0.18 * t)
                data.qpos[5] = base_qpos[5] + 0.3 * np.sin(2.0 * np.pi * 0.12 * t)

            # Animate fingers if present (commonly last 2 qpos entries)
            # This makes the gripper open/close visually.
            if model.nq >= 9:
                finger = 0.02 + 0.02 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 0.50 * t))
                data.qpos[7] = finger
                data.qpos[8] = finger

            # Recompute forward kinematics after manually changing qpos
            mujoco.mj_forward(model, data)

            # Update viewer
            viewer.sync()

            # Render offscreen frame and save to video
            renderer.update_scene(data)
            pixels = renderer.render()
            writer.append_data(pixels)

            # Slow wall-clock execution so the window stays visible in real time
            elapsed = time.time() - start_wall
            target_elapsed = (frame_idx + 1) / fps
            sleep_time = target_elapsed - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

print(f"Done. Video saved to: {VIDEO_PATH}")