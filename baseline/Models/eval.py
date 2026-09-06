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


def evaluate(policy_name: str, benchmark_name: str, n_steps: int, episodes: int, libero_suites: list[str] | None = None) -> dict:
    rollout_fn, tasks = BENCHMARKS[benchmark_name]
    if benchmark_name == "libero" and libero_suites:
        # a single OpenVLAPolicy/Pi0Policy/... instance is built once below from one
        # checkpoint, which is only meaningful for the LIBERO suite it was fine-tuned
        # on - sweeping every suite with it by default would silently mix in wrong numbers
        tasks = [t for t in tasks if t[0] in libero_suites]
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
    """Per-episode outcome dots for every (policy, benchmark) result JSON on disk
    under results/ (not just the ones just run). A bar chart of success *rate*
    is unreadable at the trial counts these smoke tests use - a 0/4 result is a
    zero-height bar, i.e. an empty-looking plot - so this plots one marker per
    episode instead, which stays visible regardless of the outcome."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("status") == "ran":
            rows.append(r)
    if not rows:
        return None

    rows.sort(key=lambda r: (r["policy"], r["benchmark"]))
    max_episodes = max(r["episodes"] for r in rows)

    fig, ax = plt.subplots(figsize=(max(6, max_episodes + 3), 0.6 * len(rows) + 1.5))
    for i, r in enumerate(rows):
        n, k = r["episodes"], r["successes"]
        for j in range(n):
            ok = j < k
            ax.scatter(j, i, marker="o" if ok else "x", s=140, linewidths=2, color="tab:green" if ok else "tab:red", zorder=3)
        ax.text(n + 0.4, i, f"{k}/{n}", va="center", fontsize=9)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['policy']} / {r['benchmark']}" for r in rows])
    ax.set_xlabel("episode")
    ax.set_xlim(-0.6, max_episodes + 1.6)
    ax.set_xticks(range(max_episodes))
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()
    ax.set_title("VLA rollout outcomes per episode (● success   ✕ fail)")
    fig.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "summary_plot.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", nargs="+", default=["dummy"], choices=list(POLICY_REGISTRY))
    parser.add_argument("--benchmarks", nargs="+", default=["libero", "calvin"], choices=list(BENCHMARKS))
    parser.add_argument(
        "--libero-suites",
        nargs="+",
        default=["libero_spatial"],
        choices=[t[0] for t in LIBERO_TASKS],
        help="which LIBERO suites to run when 'libero' is in --benchmarks (default: just the suite the default "
        "OpenVLA checkpoint is fine-tuned on - pass all 4 only if your policy/checkpoint isn't suite-specific)",
    )
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for policy_name in args.policies:
        for benchmark_name in args.benchmarks:
            try:
                result = evaluate(policy_name, benchmark_name, args.n_steps, args.episodes, args.libero_suites)
            except Exception as e:
                result = {"policy": policy_name, "benchmark": benchmark_name, "status": "not_run", "reason": str(e)}
            out_path = save_result(result)
            print(result, "->", out_path)

    plot_path = plot_results()
    if plot_path:
        print(f"summary plot saved to {plot_path}")


if __name__ == "__main__":
    main()
