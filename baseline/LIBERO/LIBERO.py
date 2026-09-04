"""Compact demo: step one task from each LIBERO benchmark suite and save a frame.

Run with the vla_venv (created for this repo, with libero + robosuite + mujoco installed):
    MUJOCO_GL=egl vla_venv/bin/python baseline/LIBERO/LIBERO.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def demo_task(suite_name: str, task_id: int = 0, n_steps: int = 5) -> tuple[str, str]:
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl_file = suite.get_task_bddl_file_path(task_id)

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
    env.reset()
    init_states = suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[0])

    for _ in range(n_steps):
        obs, reward, done, info = env.step(np.zeros(7))

    frame = obs["agentview_image"][::-1]  # robosuite camera images render upside down
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{suite_name}_task{task_id}.png")
    Image.fromarray(frame).save(out_path)

    env.close()
    return task.language, out_path


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    print("Available LIBERO task suites:")
    for name, suite_cls in benchmark_dict.items():
        try:
            print(f"  - {name}: {suite_cls().get_num_tasks()} tasks")
        except KeyError:
            continue  # 'libero_100' is an aggregate alias, not directly instantiable
    print()

    for suite_name in SUITES:
        language, out_path = demo_task(suite_name)
        print(f"[{suite_name}] task 0: {language!r}")
        print(f"  -> saved frame to {out_path}")


if __name__ == "__main__":
    main()
