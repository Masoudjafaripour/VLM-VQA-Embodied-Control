"""Rollout a CALVIN task with a pluggable VLM/VLA policy.

Any policy that implements `get_action(image, instruction) -> np.ndarray[7]`
can drive the env. Swap DummyPolicy for OpenVLAPolicy (or your own VLM/VLA
wrapper) to close the loop with a real model.

Run with the vla_venv:
    vla_venv/bin/python baseline/CALVIN/calvin_vla_rollout.py
"""
import os

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from PIL import Image

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


class Policy:
    """Interface every VLM/VLA policy plugged into this rollout must implement."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        raise NotImplementedError


class DummyPolicy(Policy):
    """Random-action placeholder so the rollout runs without any model weights."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        action_displacement = np.random.uniform(-1, 1, size=6)
        action_gripper = np.random.choice([-1, 1], size=1)
        return np.concatenate([action_displacement, action_gripper])


class OpenVLAPolicy(Policy):
    """Real VLA policy via OpenVLA (https://github.com/openvla/openvla).

    Needs: pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10"
    "accelerate==0.25.0" (transformers>=5 dropped AutoModelForVision2Seq, which
    OpenVLA's remote code relies on), and a GPU for practical speed.
    `unnorm_key` must match a dataset OpenVLA was trained/fine-tuned on; OpenVLA
    is not released with a CALVIN-specific head, so plug in your own fine-tuned
    checkpoint's key here.
    """

    def __init__(self, model_id: str = "openvla/openvla-7b", device: str = "cuda", unnorm_key: str = "bridge_orig"):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.unnorm_key = unnorm_key
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to(self.device, dtype=self.torch.bfloat16)
        return self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)


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
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.policy == "openvla":
        policy = OpenVLAPolicy(model_id=args.model_id, device=args.device)
    else:
        policy = DummyPolicy()

    instruction, gif_path, success = rollout(policy, scene=args.scene, task_name=args.task, n_steps=args.n_steps)
    print(f"instruction: {instruction!r}")
    print(f"success: {success}")
    print(f"rollout saved to: {gif_path}")


if __name__ == "__main__":
    main()
