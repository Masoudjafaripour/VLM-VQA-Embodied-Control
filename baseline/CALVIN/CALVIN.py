"""Compact demo: step one task in each CALVIN scene and save a frame.

Run with the vla_venv (created for this repo, with calvin_env + pybullet installed):
    vla_venv/bin/python baseline/CALVIN/CALVIN.py
"""
import os

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from PIL import Image

CALVIN_ROOT = os.path.expanduser("~/calvin_repo")
CONF_DIR = os.path.join(CALVIN_ROOT, "calvin_env", "conf")

# CALVIN's 4 tabletop scenes (A/B/C/D differ in object colors/positions) are the
# standard splits used for the benchmark's zero-shot generalization evaluation.
SCENES = ["calvin_scene_A", "calvin_scene_B", "calvin_scene_C", "calvin_scene_D"]
# One representative task per scene, picked from conf/tasks/new_playtable_tasks.yaml
SCENE_TASKS = {
    "calvin_scene_A": "rotate_red_block_right",
    "calvin_scene_B": "push_blue_block_left",
    "calvin_scene_C": "open_drawer",
    "calvin_scene_D": "lift_pink_block_table",
}
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def make_env(scene: str):
    with initialize_config_dir(version_base=None, config_dir=CONF_DIR):
        cfg = compose(config_name="config_data_collection.yaml", overrides=["cameras=static_and_gripper", f"scene={scene}"])
        cfg.env["use_egl"] = False
        cfg.env["show_gui"] = False
        cfg.env["use_vr"] = False
        cfg.env["use_scene_info"] = True
    env = hydra.utils.instantiate(cfg.env)
    tasks = hydra.utils.instantiate(cfg.tasks)
    return env, tasks


def demo_scene(scene: str, task_name: str, n_steps: int = 5):
    env, tasks = make_env(scene)
    env.reset()
    start_info = env.get_info()

    for _ in range(n_steps):
        action = np.concatenate([np.random.uniform(-1, 1, size=6), np.random.choice([-1, 1], size=1)])
        env.step(action)

    end_info = env.get_info()
    completed = tasks.get_task_info(start_info, end_info)

    frame = env.render(mode="rgb_array").astype(np.uint8)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{scene}_{task_name}.png")
    Image.fromarray(frame).save(out_path)

    env.close()
    return completed, out_path


def main():
    print(f"CALVIN scenes: {SCENES}")
    for scene in SCENES:
        task_name = SCENE_TASKS[scene]
        completed, out_path = demo_scene(scene, task_name)
        print(f"[{scene}] target task: {task_name!r} (random actions -> completed: {completed or 'none'})")
        print(f"  -> saved frame to {out_path}")


if __name__ == "__main__":
    main()
