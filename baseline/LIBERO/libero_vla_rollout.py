"""Rollout a LIBERO task with a pluggable VLM/VLA policy.

Any policy that implements `get_action(image, instruction) -> np.ndarray[7]`
can drive the env. Swap DummyPolicy for OpenVLAPolicy (or your own VLM/VLA
wrapper) to close the loop with a real model.

Run with the vla_venv:
    MUJOCO_GL=egl vla_venv/bin/python baseline/LIBERO/libero_vla_rollout.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


class Policy:
    """Interface every VLM/VLA policy plugged into this rollout must implement."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        raise NotImplementedError


class DummyPolicy(Policy):
    """Random-action placeholder so the rollout runs without any model weights."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        return np.random.uniform(-1, 1, size=7)


class OpenVLAPolicy(Policy):
    """Real VLA policy via OpenVLA (https://github.com/openvla/openvla).

    Needs: pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10"
    "accelerate==0.25.0" (transformers>=5 dropped AutoModelForVision2Seq, which
    OpenVLA's remote code relies on), and a GPU for practical speed.
    `unnorm_key` must match the LIBERO suite you roll out on (see OpenVLA's
    model card for the available keys, e.g. "libero_spatial", "libero_goal").
    """

    def __init__(self, model_id: str = "openvla/openvla-7b", device: str = "cuda", unnorm_key: str = "libero_spatial"):
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


def rollout(policy: Policy, suite_name: str = "libero_spatial", task_id: int = 0, n_steps: int = 20):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl_file = suite.get_task_bddl_file_path(task_id)

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
    obs = env.reset()
    init_states = suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[0])

    frames = [obs["agentview_image"][::-1]]
    done = False
    for _ in range(n_steps):
        action = policy.get_action(obs["agentview_image"][::-1], task.language)
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1])
        if done:
            break
    env.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    gif_path = os.path.join(OUT_DIR, f"{suite_name}_task{task_id}_rollout.gif")
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
    return task.language, gif_path, done


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["dummy", "openvla"], default="dummy")
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.policy == "openvla":
        policy = OpenVLAPolicy(model_id=args.model_id, device=args.device, unnorm_key=args.suite)
    else:
        policy = DummyPolicy()

    language, gif_path, success = rollout(policy, suite_name=args.suite, task_id=args.task_id, n_steps=args.n_steps)
    print(f"task: {language!r}")
    print(f"success: {success}")
    print(f"rollout saved to: {gif_path}")


if __name__ == "__main__":
    main()
