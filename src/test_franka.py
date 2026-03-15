import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("models/panda/franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()