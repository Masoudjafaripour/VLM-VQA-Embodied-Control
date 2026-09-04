"""Rollout a LIBERO task with a pluggable VLM/VLA policy.

Any policy that implements `get_action(image, instruction) -> np.ndarray[7]`
can drive the env (see baseline/Models/VLAs.py for the interface and the full
set of VLA wrappers evaluated in this repo). Swap DummyPolicy for OpenVLAPolicy
(or your own VLM/VLA wrapper) to close the loop with a real model.

Run with the vla_venv:
    MUJOCO_GL=egl vla_venv/bin/python baseline/LIBERO/libero_vla_rollout.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Models.VLAs import DummyPolicy, OpenVLAPolicy, Policy  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


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
