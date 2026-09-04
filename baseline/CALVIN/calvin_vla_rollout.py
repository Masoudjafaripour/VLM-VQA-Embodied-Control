"""Rollout a CALVIN task with a pluggable VLM/VLA policy.

Any policy that implements `get_action(image, instruction) -> np.ndarray[7]`
can drive the env (see baseline/Models/VLAs.py for the interface and the full
set of VLA wrappers evaluated in this repo). Swap DummyPolicy for OpenVLAPolicy
(or your own VLM/VLA wrapper) to close the loop with a real model.

Run with the vla_venv:
    vla_venv/bin/python baseline/CALVIN/calvin_vla_rollout.py
"""
import os
import sys

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Models.VLAs import DummyPolicy, OpenVLAPolicy, Policy  # noqa: E402

CALVIN_ROOT = os.path.expanduser("~/calvin_repo")
CONF_DIR = os.path.join(CALVIN_ROOT, "calvin_env", "conf")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# CALVIN task names read like snake_case instructions (e.g. "lift_red_block_table").
# The real benchmark pairs each with crowd-sourced language annotations (see
# calvin_models' annotations); this turns the task name itself into a stand-in
# instruction so the demo needs no extra dataset download.
TASK_TO_INSTRUCTION = {
    "move_slider_left": "move the slider left",
    "open_drawer": "open the drawer",
    "lift_red_block_table": "lift the red block",
    "rotate_blue_block_right": "rotate the blue block right",
    "push_pink_block_left": "push the pink block left",
}


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


def rollout(policy: Policy, scene: str = "calvin_scene_D", task_name: str = "lift_red_block_table", n_steps: int = 20):
    instruction = TASK_TO_INSTRUCTION[task_name]
    env, tasks = make_env(scene)
    env.reset()
    start_info = env.get_info()

    frames = [env.render(mode="rgb_array").astype(np.uint8)]
    for _ in range(n_steps):
        action = policy.get_action(frames[-1], instruction)
        env.step(action)
        frames.append(env.render(mode="rgb_array").astype(np.uint8))

    end_info = env.get_info()
    completed = task_name in tasks.get_task_info(start_info, end_info)
    env.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    gif_path = os.path.join(OUT_DIR, f"{scene}_{task_name}_rollout.gif")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    return instruction, gif_path, completed


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["dummy", "openvla"], default="dummy")
    parser.add_argument("--scene", default="calvin_scene_D")
    parser.add_argument("--task", default="lift_red_block_table", choices=list(TASK_TO_INSTRUCTION))
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.policy == "openvla":
        policy = OpenVLAPolicy(model_id=args.model_id, device=args.device, unnorm_key=args.unnorm_key)
    else:
        policy = DummyPolicy()

    instruction, gif_path, success = rollout(policy, scene=args.scene, task_name=args.task, n_steps=args.n_steps)
    print(f"instruction: {instruction!r}")
    print(f"success: {success}")
    print(f"rollout saved to: {gif_path}")


if __name__ == "__main__":
    main()
