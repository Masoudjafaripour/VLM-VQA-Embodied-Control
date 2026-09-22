"""
DAgger (Ross, Gordon & Bagnell, 2011) on a 2-link planar arm reaching task.

Env:      2-link arm, base fixed at origin, reaches a randomly sampled goal
          point in its workspace. Expert = closed-form inverse kinematics.
Modes:    --mode state   -> MLP over (joint angles, goal)
          --mode vision  -> CNN over a rendered RGB image only
          --mode both    -> CNN + state MLP fused
          --mode all     -> trains all three and compares them

Usage:
    python DAgger_2D_policy.py --mode all --iterations 8
"""

import argparse
import os
from dataclasses import dataclass, field

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Arm2DEnv:
    """2-link planar arm. Angles wrap in [-pi, pi]; base at the origin."""

    def __init__(self, l1=1.0, l2=1.0, dt=0.1, max_steps=50,
                 max_angular_speed=0.35, goal_eps=0.08, image_size=64, seed=0):
        self.l1, self.l2 = l1, l2
        self.dt = dt
        self.max_steps = max_steps
        self.max_speed = max_angular_speed
        self.goal_eps = goal_eps
        self.image_size = image_size
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reach = l1 + l2
        self.theta = np.zeros(2)
        self.goal = np.zeros(2)
        self.t = 0

    def forward_kinematics(self, theta):
        t1, t2 = theta
        x1, y1 = self.l1 * np.cos(t1), self.l1 * np.sin(t1)
        x2 = x1 + self.l2 * np.cos(t1 + t2)
        y2 = y1 + self.l2 * np.sin(t1 + t2)
        return np.array([x1, y1]), np.array([x2, y2])

    def inverse_kinematics(self, xy):
        x, y = xy
        r2 = x * x + y * y
        cos_t2 = np.clip((r2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        t2 = np.arccos(cos_t2)  # elbow-down solution
        k1 = self.l1 + self.l2 * np.cos(t2)
        k2 = self.l2 * np.sin(t2)
        t1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        return np.array([t1, t2])

    def reset(self):
        self.theta = self.rng.uniform(-np.pi, np.pi, size=2)
        goal_theta = self.rng.uniform(-np.pi, np.pi, size=2)
        _, self.goal = self.forward_kinematics(goal_theta)
        self.t = 0
        return self._obs()

    def step(self, action):
        action = np.clip(action, -self.max_speed, self.max_speed)
        self.theta = np.arctan2(np.sin(self.theta + action * self.dt),
                                 np.cos(self.theta + action * self.dt))
        self.t += 1
        _, ee = self.forward_kinematics(self.theta)
        dist = float(np.linalg.norm(ee - self.goal))
        done = dist < self.goal_eps or self.t >= self.max_steps
        return self._obs(), done, {"dist": dist, "success": dist < self.goal_eps}

    def expert_action(self):
        target = self.inverse_kinematics(self.goal)
        diff = np.arctan2(np.sin(target - self.theta), np.cos(target - self.theta))
        return np.clip(diff / self.dt, -self.max_speed, self.max_speed)

    def state_vector(self):
        # sin/cos encoding avoids the -pi/pi wrap discontinuity
        return np.array([np.sin(self.theta[0]), np.cos(self.theta[0]),
                          np.sin(self.theta[1]), np.cos(self.theta[1]),
                          self.goal[0], self.goal[1]], dtype=np.float32)

    def render_rgb(self):
        """Fast rasterized RGB observation (cv2), used during training/rollouts."""
        s = self.image_size
        img = np.full((s, s, 3), 255, dtype=np.uint8)
        scale = s / (2.2 * self.reach)

        def to_px(p):
            return (int(s / 2 + p[0] * scale), int(s / 2 - p[1] * scale))

        base = to_px((0, 0))
        j1, ee = self.forward_kinematics(self.theta)
        j1_px, ee_px = to_px(j1), to_px(ee)
        goal_px = to_px(self.goal)
        cv2.line(img, base, j1_px, (30, 30, 200), 3)
        cv2.line(img, j1_px, ee_px, (200, 60, 30), 3)
        cv2.circle(img, base, 4, (0, 0, 0), -1)
        cv2.circle(img, j1_px, 3, (0, 0, 0), -1)
        cv2.circle(img, ee_px, 4, (0, 150, 0), -1)
        cv2.drawMarker(img, goal_px, (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=10, thickness=1)
        return img

    def _obs(self):
        return {"state": self.state_vector(), "image": self.render_rgb()}


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, out_dim), nn.ReLU(),
        )

    def forward(self, img):
        return self.net(img)


class StatePolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state=None, image=None):
        return self.net(state)


class VisionPolicy(nn.Module):
    def __init__(self, action_dim=2):
        super().__init__()
        self.enc = ConvEncoder(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, state=None, image=None):
        return self.head(self.enc(image))


class FusionPolicy(nn.Module):
    def __init__(self, state_dim=6, action_dim=2):
        super().__init__()
        self.enc = ConvEncoder(128)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 32, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, state=None, image=None):
        return self.head(torch.cat([self.enc(image), self.state_mlp(state)], dim=-1))


def make_policy(mode):
    return {"state": StatePolicy, "vision": VisionPolicy, "both": FusionPolicy}[mode]().to(DEVICE)


def policy_action(policy, obs, mode):
    with torch.no_grad():
        state = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(DEVICE) if mode != "vision" else None
        image = None
        if mode != "state":
            img = obs["image"].astype(np.float32).transpose(2, 0, 1) / 255.0
            image = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
        return policy(state=state, image=image).squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------
# DAgger
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    states: list = field(default_factory=list)
    images: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    iters: list = field(default_factory=list)  # DAgger iteration each sample was collected in

    def add(self, obs, action, iteration=0):
        self.states.append(obs["state"])
        self.images.append(obs["image"])
        self.actions.append(action)
        self.iters.append(iteration)

    def __len__(self):
        return len(self.actions)

    def tensors(self):
        states = torch.from_numpy(np.stack(self.states)).float().to(DEVICE)
        images = torch.from_numpy(np.stack(self.images).astype(np.float32).transpose(0, 3, 1, 2) / 255.0).to(DEVICE)
        actions = torch.from_numpy(np.stack(self.actions)).float().to(DEVICE)
        return states, images, actions


def rollout(env, policy, mode, beta, dataset, iteration=0, max_steps=None):
    """Roll out beta*expert + (1-beta)*policy, always labeling with the expert."""
    obs = env.reset()
    done = False
    while not done:
        expert_a = env.expert_action()
        dataset.add(obs, expert_a, iteration=iteration)
        if policy is None or np.random.rand() < beta:
            act = expert_a
        else:
            act = policy_action(policy, obs, mode)
        obs, done, info = env.step(act)
    return info


def train_policy(policy, dataset, epochs, batch_size, mode, lr=1e-3, val_frac=0.15):
    states, images, actions = dataset.tensors()
    n = len(dataset)
    perm = torch.randperm(n, device=DEVICE)
    n_val = max(1, int(n * val_frac)) if n > 4 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def split(idx):
        s = states[idx] if mode != "vision" else None
        im = images[idx] if mode != "state" else None
        return s, im, actions[idx]

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    for _ in range(epochs):
        epoch_perm = train_idx[torch.randperm(len(train_idx), device=DEVICE)]
        batch_losses = []
        for i in range(0, len(epoch_perm), batch_size):
            idx = epoch_perm[i:i + batch_size]
            s, im, a = split(idx)
            pred = policy(state=s, image=im)
            loss = loss_fn(pred, a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        if n_val > 0:
            with torch.no_grad():
                s, im, a = split(val_idx)
                val_losses.append(loss_fn(policy(state=s, image=im), a).item())
        else:
            val_losses.append(train_losses[-1])
    return policy, train_losses, val_losses


def evaluate(env, policy, mode, n_episodes):
    successes, dists = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done, info = False, {}
        while not done:
            act = policy_action(policy, obs, mode)
            obs, done, info = env.step(act)
        successes.append(info["success"])
        dists.append(info["dist"])
    return float(np.mean(successes)), float(np.mean(dists))


def dagger(mode, iterations, episodes_per_iter, epochs, eval_episodes, seed, image_size,
           max_steps=50, log_fn=print):
    env = Arm2DEnv(image_size=image_size, seed=seed, max_steps=max_steps)
    policy = make_policy(mode)
    dataset = Dataset()
    history = []
    loss_log = {"train": [], "val": [], "boundaries": []}  # boundaries = epoch index where each DAgger iter starts

    for it in range(iterations):
        beta = 1.0 if it == 0 else max(0.0, 0.6 ** it)  # first round: pure expert
        policy_frac = 1.0 - beta  # fraction of rollout actions taken by the learned policy
        for _ in range(episodes_per_iter):
            rollout(env, policy if it > 0 else None, mode, beta, dataset, iteration=it)

        loss_log["boundaries"].append(len(loss_log["train"]))
        policy, train_losses, val_losses = train_policy(policy, dataset, epochs=epochs, batch_size=128, mode=mode)
        loss_log["train"].extend(train_losses)
        loss_log["val"].extend(val_losses)

        succ, dist = evaluate(env, policy, mode, eval_episodes)
        history.append({"iter": it, "beta": beta, "policy_frac": policy_frac, "n_data": len(dataset),
                         "success": succ, "avg_dist": dist, "final_train_loss": train_losses[-1],
                         "final_val_loss": val_losses[-1]})
        log_fn(f"[{mode}] iter {it}: policy_frac={policy_frac:.2f} data={len(dataset):5d} "
               f"success={succ:.2f} avg_dist={dist:.3f} val_loss={val_losses[-1]:.4f}")

    return policy, dataset, history, env, loss_log


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def rollout_from_state(env, actor, mode, max_frames):
    """Roll out from env's *current* theta/goal (no reset), so a policy run and an expert
    (ground-truth) run can be replayed from the exact same start state for comparison.
    `actor` is either a policy module or the string "expert"."""
    obs = env._obs()
    frames_theta = [env.theta.copy()]
    done, info = False, {}
    while not done and len(frames_theta) < max_frames:
        act = env.expert_action() if actor == "expert" else policy_action(actor, obs, mode)
        obs, done, info = env.step(act)
        frames_theta.append(env.theta.copy())
    return frames_theta, info


def build_demo_episode(env, l1, l2, dt, max_speed, goal_eps, image_size, seed, policy, mode, max_frames):
    """One episode for the GIF: policy rollout + a ground-truth expert rollout replayed
    from the identical start state/goal, for a direct visual comparison."""
    policy_env = Arm2DEnv(l1=l1, l2=l2, dt=dt, max_steps=max_frames, max_angular_speed=max_speed,
                           goal_eps=goal_eps, image_size=image_size, seed=seed)
    policy_env.reset()
    theta0, goal = policy_env.theta.copy(), policy_env.goal.copy()
    policy_frames, policy_info = rollout_from_state(policy_env, policy, mode, max_frames)

    expert_env = Arm2DEnv(l1=l1, l2=l2, dt=dt, max_steps=max_frames, max_angular_speed=max_speed,
                           goal_eps=goal_eps, image_size=image_size, seed=seed)
    expert_env.theta, expert_env.goal, expert_env.t = theta0.copy(), goal.copy(), 0
    expert_frames, expert_info = rollout_from_state(expert_env, "expert", mode, max_frames)

    return {
        "env": policy_env, "theta0": theta0, "goal": goal,
        "policy_frames": policy_frames, "policy_info": policy_info,
        "expert_ee_path": [policy_env.forward_kinematics(t)[1] for t in expert_frames],
    }


def render_train_val_gif(env, policy, mode, out_path, val_seed_offset=999, demo_max_steps=150):
    """Side-by-side GIF: an in-distribution ('train') episode next to a held-out
    ('val') episode drawn from an independently seeded env the policy never trained on.
    Each panel shows the policy rollout (start/end markers + solid trace) against the
    ground-truth expert path from the same start/goal (dashed), so you can see how far
    the learned policy deviates from the expert demonstration.
    Demo episodes run for `demo_max_steps` (independent of the shorter training horizon)
    so the arm has time to reach the goal, or to clearly show it failing to."""
    max_frames = demo_max_steps + 1
    common = dict(l1=env.l1, l2=env.l2, dt=env.dt, max_speed=env.max_speed, goal_eps=env.goal_eps,
                  image_size=env.image_size, policy=policy, mode=mode, max_frames=max_frames)
    train_ep = build_demo_episode(env, seed=env.seed, **common)
    val_ep = build_demo_episode(env, seed=env.seed + val_seed_offset, **common)
    episodes = [train_ep, val_ep]

    n_frames = max(len(ep["policy_frames"]) for ep in episodes)
    for ep in episodes:
        ep["policy_frames"] += [ep["policy_frames"][-1]] * (n_frames - len(ep["policy_frames"]))

    reach = env.l1 + env.l2
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    panels = []
    for ax, ep, label in zip(axes, episodes, ["train", "val (held-out)"]):
        ax.set_xlim(-reach * 1.1, reach * 1.1)
        ax.set_ylim(-reach * 1.1, reach * 1.1)
        ax.set_aspect("equal")
        ax.add_patch(plt.Circle((0, 0), reach, fill=False, linestyle="--", color="gray", alpha=0.5))
        gt = np.array(ep["expert_ee_path"])
        ax.plot(gt[:, 0], gt[:, 1], "--", color="black", alpha=0.6, linewidth=1.5, label="G.T. (expert)")
        ax.plot(*ep["goal"], marker="*", color="black", markersize=14, label="goal")
        start_ee = ep["env"].forward_kinematics(ep["theta0"])[1]
        ax.plot(*start_ee, marker="o", markerfacecolor="white", markeredgecolor="black",
                markersize=8, label="start")
        end_ee = ep["env"].forward_kinematics(ep["policy_frames"][-1])[1]
        end_marker, = ax.plot(*end_ee, marker="X", color="#2ca02c", markersize=9,
                               linestyle="None", label="end (policy)")
        end_marker.set_visible(False)
        link1, = ax.plot([], [], "-o", color="#1f77b4", linewidth=4, markersize=6)
        link2, = ax.plot([], [], "-o", color="#d62728", linewidth=4, markersize=6)
        trace, = ax.plot([], [], "-", color="#2ca02c", alpha=0.6, linewidth=1.5, label="policy")
        ax.set_title(f"{label}: {'success' if ep['policy_info'].get('success') else 'timeout'}")
        ax.legend(loc="upper right", fontsize=6)
        panels.append((link1, link2, trace, end_marker, [], []))
    fig.suptitle(f"DAgger arm ({mode})")

    def update(i):
        artists = []
        for ep, (link1, link2, trace, end_marker, tx, ty) in zip(episodes, panels):
            theta = ep["policy_frames"][i]
            j1, ee = ep["env"].forward_kinematics(theta)
            link1.set_data([0, j1[0]], [0, j1[1]])
            link2.set_data([j1[0], ee[0]], [j1[1], ee[1]])
            tx.append(ee[0]); ty.append(ee[1])
            trace.set_data(tx, ty)
            if i == n_frames - 1:
                end_marker.set_visible(True)
            artists += [link1, link2, trace, end_marker]
        return artists

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer="pillow", fps=20)
    plt.close(fig)


def plot_loss_curve(mode, loss_log, out_path):
    """Train vs val loss over all epochs, with dashed lines marking each DAgger iteration."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(len(loss_log["train"]))
    ax.plot(epochs, loss_log["train"], label="train loss")
    ax.plot(epochs, loss_log["val"], label="val loss")
    for b in loss_log["boundaries"]:
        ax.axvline(b, color="gray", linestyle="--", alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("epoch (concatenated across DAgger iterations)")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"Training / validation loss ({mode})")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_data_distribution(dataset, env, mode, out_path):
    """Coverage of the aggregated DAgger dataset: visited end-effector positions and joint
    configs (colored by which DAgger iteration collected them), sampled goals, and the
    expert action labels. Iteration-0 (pure expert) points cover only the expert's own
    trajectories; later points show the policy-visited states DAgger adds on top."""
    states = np.stack(dataset.states)
    actions = np.stack(dataset.actions)
    iters = np.array(dataset.iters)
    theta1 = np.arctan2(states[:, 0], states[:, 1])
    theta2 = np.arctan2(states[:, 2], states[:, 3])
    goals = states[:, 4:6]
    ee = np.stack([env.forward_kinematics(np.array([t1, t2]))[1] for t1, t2 in zip(theta1, theta2)])

    reach = env.l1 + env.l2
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    sc = axes[0, 0].scatter(ee[:, 0], ee[:, 1], c=iters, cmap="viridis", s=4, alpha=0.5)
    axes[0, 0].add_patch(plt.Circle((0, 0), reach, fill=False, linestyle="--", color="gray", alpha=0.5))
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_title("Visited end-effector positions")
    fig.colorbar(sc, ax=axes[0, 0], label="DAgger iteration")

    axes[0, 1].scatter(goals[:, 0], goals[:, 1], s=4, alpha=0.3, color="black")
    axes[0, 1].add_patch(plt.Circle((0, 0), reach, fill=False, linestyle="--", color="gray", alpha=0.5))
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_title("Sampled goal positions")

    axes[1, 0].scatter(theta1, theta2, c=iters, cmap="viridis", s=4, alpha=0.5)
    axes[1, 0].set_xlim(-np.pi, np.pi); axes[1, 0].set_ylim(-np.pi, np.pi)
    axes[1, 0].set_xlabel("θ1"); axes[1, 0].set_ylabel("θ2")
    axes[1, 0].set_title("Visited joint-angle configs")

    axes[1, 1].hist2d(actions[:, 0], actions[:, 1], bins=40, cmap="viridis")
    axes[1, 1].set_xlabel("Δθ1"); axes[1, 1].set_ylabel("Δθ2")
    axes[1, 1].set_title("Expert action labels")

    fig.suptitle(f"Aggregated dataset distribution ({mode}, n={len(dataset)})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dagger_diagnostics(histories, out_path):
    """Success rate, distance, expert/policy action mix, and dataset growth vs DAgger iteration."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for mode, hist in histories.items():
        iters = [h["iter"] for h in hist]
        axes[0, 0].plot(iters, [h["success"] for h in hist], marker="o", label=mode)
        axes[0, 1].plot(iters, [h["avg_dist"] for h in hist], marker="o", label=mode)
        axes[1, 0].plot(iters, [h["policy_frac"] for h in hist], marker="o", label=mode)
        axes[1, 1].plot(iters, [h["n_data"] for h in hist], marker="o", label=mode)
    axes[0, 0].set_title("Success rate"); axes[0, 0].set_ylabel("success rate")
    axes[0, 1].set_title("Avg final distance to goal"); axes[0, 1].set_ylabel("distance")
    axes[1, 0].set_title("Rollout action mix"); axes[1, 0].set_ylabel("fraction from policy (1-β)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].set_title("Aggregated dataset size"); axes[1, 1].set_ylabel("# (state, action) pairs")
    for ax in axes.flat:
        ax.set_xlabel("DAgger iteration")
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["state", "vision", "both", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--episodes-per-iter", type=int, default=15,
                         help="rollouts labeled by the expert per DAgger iteration; raise this to collect more training samples")
    parser.add_argument("--max-steps", type=int, default=50,
                         help="training/eval episode horizon; raise this to collect more samples per episode")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-max-steps", type=int, default=150,
                         help="episode horizon used only for the rendered demo GIFs")
    default_outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser.add_argument("--outdir", type=str, default=default_outdir)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--infer", action="store_true",
                         help="skip training; load a checkpoint and evaluate/render only")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="policy .pt path for --infer (default: {outdir}/{mode}_policy.pt)")
    args = parser.parse_args()

    if args.infer:
        assert args.mode != "all", "--infer requires a single --mode (state|vision|both)"
        ckpt = args.checkpoint or os.path.join(args.outdir, f"{args.mode}_policy.pt")
        env = Arm2DEnv(image_size=args.image_size, seed=args.seed, max_steps=args.max_steps)
        policy = make_policy(args.mode)
        policy.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        policy.eval()
        succ, dist = evaluate(env, policy, args.mode, args.eval_episodes)
        print(f"[{args.mode}] loaded {ckpt}: success={succ:.2f} avg_dist={dist:.3f}")
        if not args.no_viz:
            render_train_val_gif(env, policy, args.mode, os.path.join(args.outdir, f"{args.mode}_infer.gif"),
                                 demo_max_steps=args.demo_max_steps)
        return

    modes = ["state", "vision", "both"] if args.mode == "all" else [args.mode]
    histories = {}
    for mode in modes:
        print(f"\n=== training mode={mode} ===")
        policy, dataset, history, env, loss_log = dagger(
            mode, args.iterations, args.episodes_per_iter, args.epochs,
            args.eval_episodes, args.seed, args.image_size, max_steps=args.max_steps,
        )
        histories[mode] = history
        os.makedirs(args.outdir, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(args.outdir, f"{mode}_policy.pt"))
        if not args.no_viz:
            render_train_val_gif(env, policy, mode, os.path.join(args.outdir, f"{mode}_episode.gif"),
                                 demo_max_steps=args.demo_max_steps)
            plot_loss_curve(mode, loss_log, os.path.join(args.outdir, f"{mode}_loss.png"))
            plot_data_distribution(dataset, env, mode, os.path.join(args.outdir, f"{mode}_data_distribution.png"))

    if not args.no_viz:
        plot_dagger_diagnostics(histories, os.path.join(args.outdir, "dagger_diagnostics.png"))

    print(f"\nDone. Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()
