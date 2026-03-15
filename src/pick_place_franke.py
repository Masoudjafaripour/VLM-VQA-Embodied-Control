import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("models/panda/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

print("Number of actuators (nu):", model.nu)
for i in range(model.nu):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

# 7 arm joints + 1 gripper actuator
home = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.04])
close_gripper = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.00])
open_gripper  = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7, 0.04])

def run_ctrl(target_ctrl, steps, viewer):
    for _ in range(steps):
        data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)
        viewer.sync()

with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetData(model, data)

    print("Move to home")
    run_ctrl(home, 300, viewer)
    pause = input("Press Enter to continue...")

    print("Close gripper")
    run_ctrl(close_gripper, 300, viewer)

    print("Open gripper")
    run_ctrl(open_gripper, 300, viewer)