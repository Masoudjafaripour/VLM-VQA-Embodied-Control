"""
DAgger (Ross, Gordon & Bagnell, 2011) on a 3-DOF spherical-shoulder arm reaching task.
3D analog of DAgger_2D_policy.py: same DAgger machinery, three data modes, and
train/val comparison GIFs, but a 3D arm with closed-form 3D IK.

Env:      shoulder azimuth (phi) + shoulder elevation (theta) + elbow (gamma).
          phi rotates the arm's vertical plane around the z-axis; theta/gamma
          are a standard 2-link IK problem solved inside that plane.
Modes:    --mode state   -> MLP over (joint angles, goal)
          --mode vision  -> CNN over a rendered RGB image (fixed isometric camera)
          --mode both    -> CNN + state MLP fused
          --mode all     -> trains all three and compares them

Usage:
    python DAgger_3D_policy.py --mode all --iterations 8
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ISO_COS, ISO_SIN = np.cos(np.pi / 6), np.sin(np.pi / 6)  # cheap fixed isometric camera


def project_iso(p):
    """Fixed isometric projection (no per-call trig) used for the fast cv2 render."""
    x, y, z = p
    return (x - y) * ISO_COS, (x + y) * ISO_SIN - z


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class Arm3DEnv:
    """3-DOF arm: angles = (phi, theta, gamma) = (shoulder azimuth, shoulder
    elevation, elbow). FK/IK reduce to the 2D 2-link case inside the plane
    picked by phi, so IK stays closed-form."""

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
        self.angles = np.zeros(3)  # (phi, theta, gamma)
        self.goal = np.zeros(3)
        self.t = 0

    def forward_kinematics(self, angles):
        phi, theta, gamma = angles
        r1, z1 = self.l1 * np.cos(theta), self.l1 * np.sin(theta)
        r2 = r1 + self.l2 * np.cos(theta + gamma)
        z2 = z1 + self.l2 * np.sin(theta + gamma)
        elbow = np.array([r1 * np.cos(phi), r1 * np.sin(phi), z1])
        ee = np.array([r2 * np.cos(phi), r2 * np.sin(phi), z2])
        return elbow, ee

    def inverse_kinematics(self, xyz):
        x, y, z = xyz
        phi = np.arctan2(y, x)
        r = np.hypot(x, y)
        cos_gamma = np.clip((r ** 2 + z ** 2 - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        gamma = np.arccos(cos_gamma)  # elbow-down solution
        k1 = self.l1 + self.l2 * np.cos(gamma)
        k2 = self.l2 * np.sin(gamma)
        theta = np.arctan2(z, r) - np.arctan2(k2, k1)
        return np.array([phi, theta, gamma])

    def reset(self):
        self.angles = self.rng.uniform(-np.pi, np.pi, size=3)
        goal_angles = self.rng.uniform(-np.pi, np.pi, size=3)
        _, self.goal = self.forward_kinematics(goal_angles)
        self.t = 0
        return self._obs()

    def step(self, action):
        action = np.clip(action, -self.max_speed, self.max_speed)
        self.angles = np.arctan2(np.sin(self.angles + action * self.dt),
                                  np.cos(self.angles + action * self.dt))
        self.t += 1
        _, ee = self.forward_kinematics(self.angles)
        dist = float(np.linalg.norm(ee - self.goal))
        done = dist < self.goal_eps or self.t >= self.max_steps
        return self._obs(), done, {"dist": dist, "success": dist < self.goal_eps}

    def expert_action(self):
        target = self.inverse_kinematics(self.goal)
        diff = np.arctan2(np.sin(target - self.angles), np.cos(target - self.angles))
        return np.clip(diff / self.dt, -self.max_speed, self.max_speed)

    def state_vector(self):
        s = np.sin(self.angles)
        c = np.cos(self.angles)
        return np.array([s[0], c[0], s[1], c[1], s[2], c[2],
                          self.goal[0], self.goal[1], self.goal[2]], dtype=np.float32)

    def render_rgb(self):
        """Fast rasterized RGB observation (cv2, fixed isometric camera)."""
        s = self.image_size
        img = np.full((s, s, 3), 255, dtype=np.uint8)
        scale = s / (4.5 * self.reach)

        def to_px(p):
            sx, sy = project_iso(p)
            return (int(s / 2 + sx * scale), int(s / 2 - sy * scale))

        base = to_px((0, 0, 0))
        elbow, ee = self.forward_kinematics(self.angles)
        elbow_px, ee_px = to_px(elbow), to_px(ee)
        goal_px = to_px(self.goal)
        cv2.line(img, base, elbow_px, (30, 30, 200), 3)
        cv2.line(img, elbow_px, ee_px, (200, 60, 30), 3)
        cv2.circle(img, base, 4, (0, 0, 0), -1)
        cv2.circle(img, elbow_px, 3, (0, 0, 0), -1)
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
    def __init__(self, state_dim=9, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state=None, image=None):
        return self.net(state)


class VisionPolicy(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.enc = ConvEncoder(128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, action_dim))

    def forward(self, state=None, image=None):
        return self.head(self.enc(image))


class FusionPolicy(nn.Module):
    def __init__(self, state_dim=9, action_dim=3):
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


def rollout(env, policy, mode, beta, dataset, iteration=0):
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
    env = Arm3DEnv(image_size=image_size, seed=seed, max_steps=max_steps)
    policy = make_policy(mode)
    dataset = Dataset()
    history = []
    loss_log = {"train": [], "val": [], "boundaries": []}

    for it in range(iterations):
        beta = 1.0 if it == 0 else max(0.0, 0.6 ** it)
        policy_frac = 1.0 - beta
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
    """Roll out from env's *current* angles/goal (no reset), so a policy run and an
    expert (ground-truth) run can be replayed from the exact same start state."""
    obs = env._obs()
    frames = [env.angles.copy()]
    done, info = False, {}
    while not done and len(frames) < max_frames:
        act = env.expert_action() if actor == "expert" else policy_action(actor, obs, mode)
        obs, done, info = env.step(act)
        frames.append(env.angles.copy())
    return frames, info


def build_demo_episode(env, l1, l2, dt, max_speed, goal_eps, image_size, seed, policy, mode, max_frames):
    """One episode for the GIF: policy rollout + a ground-truth expert rollout replayed
    from the identical start state/goal, for a direct visual comparison."""
    policy_env = Arm3DEnv(l1=l1, l2=l2, dt=dt, max_steps=max_frames, max_angular_speed=max_speed,
                           goal_eps=goal_eps, image_size=image_size, seed=seed)
    policy_env.reset()
    angles0, goal = policy_env.angles.copy(), policy_env.goal.copy()
    policy_frames, policy_info = rollout_from_state(policy_env, policy, mode, max_frames)

    expert_env = Arm3DEnv(l1=l1, l2=l2, dt=dt, max_steps=max_frames, max_angular_speed=max_speed,
                           goal_eps=goal_eps, image_size=image_size, seed=seed)
    expert_env.angles, expert_env.goal, expert_env.t = angles0.copy(), goal.copy(), 0
    expert_frames, expert_info = rollout_from_state(expert_env, "expert", mode, max_frames)

    return {
        "env": policy_env, "angles0": angles0, "goal": goal,
        "policy_frames": policy_frames, "policy_info": policy_info,
        "expert_ee_path": [policy_env.forward_kinematics(a)[1] for a in expert_frames],
    }


def render_train_val_gif(env, policy, mode, out_path, val_seed_offset=999, demo_max_steps=150):
    """Side-by-side 3D GIF: an in-distribution ('train') episode next to a held-out
    ('val') episode from an independently seeded env. Each panel shows the policy
    rollout (start/end markers + solid trace) against the ground-truth expert path
    from the same start/goal (dashed)."""
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
    fig = plt.figure(figsize=(9, 4.5))
    panels = []
    for i, (ep, label) in enumerate(zip(episodes, ["train", "val (held-out)"])):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.set_xlim(-reach, reach); ax.set_ylim(-reach, reach); ax.set_zlim(-reach, reach)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=22, azim=45)
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        ax.plot_wireframe(reach * np.cos(u) * np.sin(v), reach * np.sin(u) * np.sin(v), reach * np.cos(v),
                           color="gray", alpha=0.1, linewidth=0.5)
        gt = np.array(ep["expert_ee_path"])
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "--", color="black", alpha=0.6, linewidth=1.5, label="G.T. (expert)")
        ax.plot(*ep["goal"], marker="*", color="black", markersize=14, label="goal")
        start_ee = ep["env"].forward_kinematics(ep["angles0"])[1]
        ax.plot(*start_ee, marker="o", markerfacecolor="white", markeredgecolor="black",
                markersize=8, label="start")
        end_ee = ep["env"].forward_kinematics(ep["policy_frames"][-1])[1]
        end_marker, = ax.plot([end_ee[0]], [end_ee[1]], [end_ee[2]], marker="X", color="#2ca02c",
                               markersize=9, linestyle="None", label="end (policy)")
        end_marker.set_visible(False)
        link1, = ax.plot([], [], [], "-o", color="#1f77b4", linewidth=4, markersize=6)
        link2, = ax.plot([], [], [], "-o", color="#d62728", linewidth=4, markersize=6)
        trace, = ax.plot([], [], [], "-", color="#2ca02c", alpha=0.6, linewidth=1.5, label="policy")
        ax.set_title(f"{label}: {'success' if ep['policy_info'].get('success') else 'timeout'}")
        ax.legend(loc="upper right", fontsize=6)
        panels.append((link1, link2, trace, end_marker, [], [], []))
    fig.suptitle(f"DAgger 3D arm ({mode})")

    def update(i):
        for ep, (link1, link2, trace, end_marker, tx, ty, tz) in zip(episodes, panels):
            angles = ep["policy_frames"][i]
            j1, ee = ep["env"].forward_kinematics(angles)
            link1.set_data([0, j1[0]], [0, j1[1]]); link1.set_3d_properties([0, j1[2]])
            link2.set_data([j1[0], ee[0]], [j1[1], ee[1]]); link2.set_3d_properties([j1[2], ee[2]])
            tx.append(ee[0]); ty.append(ee[1]); tz.append(ee[2])
            trace.set_data(tx, ty); trace.set_3d_properties(tz)
            if i == n_frames - 1:
                end_marker.set_visible(True)
        return []

    anim = matplotlib.animation.FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
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
    """Coverage of the aggregated DAgger dataset: visited end-effector/goal positions
    in 3D (colored by which DAgger iteration collected them), shoulder-elevation vs
    elbow-angle coverage, and the expert action labels (theta/gamma components)."""
    states = np.stack(dataset.states)
    actions = np.stack(dataset.actions)
    iters = np.array(dataset.iters)
    phi = np.arctan2(states[:, 0], states[:, 1])
    theta = np.arctan2(states[:, 2], states[:, 3])
    gamma = np.arctan2(states[:, 4], states[:, 5])
    goals = states[:, 6:9]
    ee = np.stack([env.forward_kinematics(a)[1] for a in zip(phi, theta, gamma)])

    fig = plt.figure(figsize=(10, 9))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    sc = ax1.scatter(ee[:, 0], ee[:, 1], ee[:, 2], c=iters, cmap="viridis", s=4, alpha=0.5)
    ax1.set_title("Visited end-effector positions")
    fig.colorbar(sc, ax=ax1, label="DAgger iteration", shrink=0.6)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(goals[:, 0], goals[:, 1], goals[:, 2], s=4, alpha=0.3, color="black")
    ax2.set_title("Sampled goal positions")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(theta, gamma, c=iters, cmap="viridis", s=4, alpha=0.5)
    ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(-np.pi, np.pi)
    ax3.set_xlabel("theta (shoulder elevation)"); ax3.set_ylabel("gamma (elbow)")
    ax3.set_title("Visited elevation/elbow configs")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist2d(actions[:, 1], actions[:, 2], bins=40, cmap="viridis")
    ax4.set_xlabel("Δtheta"); ax4.set_ylabel("Δgamma")
    ax4.set_title("Expert action labels (theta/gamma)")

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
    default_outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "3D")
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
        env = Arm3DEnv(image_size=args.image_size, seed=args.seed, max_steps=args.max_steps)
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
