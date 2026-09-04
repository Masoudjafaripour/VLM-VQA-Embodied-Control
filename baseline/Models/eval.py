"""Evaluate VLA policies (see VLAs.py) on the LIBERO and CALVIN baselines in
this repo. Saves one result JSON per policy/benchmark under results/<policy>/,
plus a bar chart comparing success rates across everything evaluated.

OpenVLA and DummyPolicy run under vla_venv; SmolVLA/pi0/pi0.5 need the
separate vla_venv_lerobot (see README.md) - run this script once per venv
with the policies that venv supports.

Run:
    MUJOCO_GL=egl vla_venv/bin/python baseline/Models/eval.py --policies dummy openvla --benchmarks libero calvin
    vla_venv_lerobot/bin/python baseline/Models/eval.py --policies smolvla --benchmarks libero calvin
"""
import argparse
import glob
import json
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")  # headless - no display attached
import matplotlib.pyplot as plt
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from CALVIN.calvin_vla_rollout import rollout as calvin_rollout  # noqa: E402
from LIBERO.libero_vla_rollout import rollout as libero_rollout  # noqa: E402
from Models.VLAs import POLICY_REGISTRY  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

LIBERO_TASKS = [("libero_spatial", 0), ("libero_object", 0), ("libero_goal", 0), ("libero_10", 0)]
CALVIN_TASKS = [
    ("calvin_scene_A", "rotate_blue_block_right"),
    ("calvin_scene_B", "push_pink_block_left"),
    ("calvin_scene_C", "open_drawer"),
    ("calvin_scene_D", "lift_red_block_table"),
]
BENCHMARKS = {"libero": (libero_rollout, LIBERO_TASKS), "calvin": (calvin_rollout, CALVIN_TASKS)}


def evaluate(policy_name: str, benchmark_name: str, n_steps: int, episodes: int) -> dict:
    rollout_fn, tasks = BENCHMARKS[benchmark_name]
    policy = POLICY_REGISTRY[policy_name]()  # built once - VLA weights are too heavy to reload per episode
    gif_dir = os.path.join(RESULTS_DIR, policy_name, benchmark_name)
    os.makedirs(gif_dir, exist_ok=True)

    successes, n, gif_paths = 0, 0, []
    trials = [(t1, t2) for (t1, t2) in tasks for _ in range(episodes)]
    for task_arg1, task_arg2 in tqdm(trials, desc=f"{policy_name} on {benchmark_name}", unit="episode"):
        if hasattr(policy, "reset"):
            policy.reset()
        _, gif_path, success = rollout_fn(policy, task_arg1, task_arg2, n_steps=n_steps)
        # each rollout script always writes to the same fixed path in its own outputs/ dir,
        # so copy it here before the next episode overwrites it
        saved_gif = os.path.join(gif_dir, f"episode{n}_{task_arg1}_{task_arg2}{os.path.splitext(gif_path)[1]}")
        shutil.copy(gif_path, saved_gif)
        gif_paths.append(saved_gif)
        successes += int(bool(success))
        n += 1
    return {
        "policy": policy_name,
        "benchmark": benchmark_name,
        "status": "ran",
        "episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "gifs": gif_paths,
    }


def save_result(result: dict) -> str:
    policy_dir = os.path.join(RESULTS_DIR, result["policy"])
    os.makedirs(policy_dir, exist_ok=True)
    out_path = os.path.join(policy_dir, f"{result['benchmark']}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def plot_results():
    """Bar chart of success rate per policy/benchmark, built from every result
    JSON on disk under results/ (not just the policies/benchmarks just run)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("status") == "ran":
            rows.append(r)
    if not rows:
        return None

    policies = sorted({r["policy"] for r in rows})
    benchmarks = sorted({r["benchmark"] for r in rows})
    width = 0.8 / len(benchmarks)

    fig, ax = plt.subplots(figsize=(1.5 * len(policies) + 2, 4))
    for i, benchmark in enumerate(benchmarks):
        entries = [next((r for r in rows if r["policy"] == p and r["benchmark"] == benchmark), None) for p in policies]
        rates = [e["success_rate"] if e else 0 for e in entries]
        x = [j + i * width for j in range(len(policies))]
        bars = ax.bar(x, rates, width=width, label=benchmark)
        for bar, e in zip(bars, entries):
            if e is None:
                continue
            # label every bar with successes/episodes, even 0-height ones, so the
            # chart still says something when every rate happens to be 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{e['successes']}/{e['episodes']}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks([j + width * (len(benchmarks) - 1) / 2 for j in range(len(policies))])
    ax.set_xticklabels(policies)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1)
    ax.set_title("VLA success rate by benchmark")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "summary_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["dummy"], choices=list(POLICY_REGISTRY))
    parser.add_argument("--benchmarks", nargs="+", default=["libero", "calvin"], choices=list(BENCHMARKS))
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for policy_name in args.policies:
        for benchmark_name in args.benchmarks:
            try:
                result = evaluate(policy_name, benchmark_name, args.n_steps, args.episodes)
            except Exception as e:
                result = {"policy": policy_name, "benchmark": benchmark_name, "status": "not_run", "reason": str(e)}
            out_path = save_result(result)
            print(result, "->", out_path)

    plot_path = plot_results()
    if plot_path:
        print(f"summary plot saved to {plot_path}")


if __name__ == "__main__":
    main()
