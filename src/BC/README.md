# DAgger on a 2-link planar arm

[DAgger_2D_policy.py](DAgger_2D_policy.py) implements DAgger (Ross, Gordon & Bagnell, 2011)
for a 2-link planar arm reaching task, with state-based, vision-based, and fused policies.

## Environment

Base fixed at the origin, joint angles `θ = (θ1, θ2)`, link lengths `l1, l2`.

**Forward kinematics:**
```
x1 = l1 cos θ1,                 y1 = l1 sin θ1
x2 = x1 + l2 cos(θ1+θ2),        y2 = y1 + l2 sin(θ1+θ2)   (end-effector)
```

**Inverse kinematics** (elbow-down), target `(x, y)`:
```
θ2 = arccos( (x²+y² − l1² − l2²) / (2 l1 l2) )
θ1 = atan2(y, x) − atan2(l2 sin θ2, l1 + l2 cos θ2)
```

**Dynamics:** kinematic integrator, `θ_{t+1} = θ_t + a_t·dt`, action `a = Δθ/dt` clipped to
`max_angular_speed`. Episode ends when the end-effector is within `goal_eps` of the goal or
`max_steps` is reached.

## Expert & DAgger

The expert is closed-form IK plus a P-controller toward the target joint angles:
```
a*(s) = clip( wrap(θ*_goal − θ) / dt,  −a_max, a_max )
```

Each DAgger iteration `i` rolls out a mixture policy `π̃_i = β_i·π* + (1−β_i)·π_i`
(`β_i = 0.6^i`, `β_0 = 1`), **always labels visited states with the expert action**,
aggregates into `D ← D ∪ {(s, a*(s))}`, and retrains `π_{i+1}` from scratch on all of `D`:
```
π_{i+1} = argmin_π  (1/|D|) Σ_{(s,a*)∈D}  ‖π(s) − a*‖²
```
This is the no-regret reduction from the paper: querying the expert on the *policy's own*
state distribution (not just the expert's) is what fixes the compounding-error problem in
plain behavior cloning.

## Data modes

| `--mode`  | Input                          | Network                     |
|-----------|---------------------------------|------------------------------|
| `state`   | `(sinθ1,cosθ1,sinθ2,cosθ2,goal_x,goal_y)` | MLP                |
| `vision`  | rendered 64×64 RGB image only  | CNN                          |
| `both`    | image + state, fused            | CNN + MLP → concat → MLP     |

## Outputs (`results/` by default)

- `{mode}_policy.pt` — trained weights
- `{mode}_episode.gif` — side-by-side rollout of the final policy: an in-distribution episode (left) vs. a held-out episode from an independently seeded env the policy never trained on (right, `--infer` saves this as `{mode}_infer.gif`). Each panel marks the **start** (hollow circle) and **end** (green ✕) of the policy's path, and overlays the **ground-truth expert trajectory** (black dashed) replayed from the same start state/goal, so you can see directly how far the learned policy deviates from the demonstration.
- `{mode}_loss.png` — train/val MSE loss (dashed lines mark each DAgger retrain)
- `{mode}_data_distribution.png` — coverage of the aggregated dataset: visited end-effector positions and joint configs (colored by which DAgger iteration collected them — each `env.reset()` draws a fresh random start pose and goal, so no two episodes repeat), sampled goal positions, and expert action labels
- `dagger_diagnostics.png` — success rate, avg distance, policy/expert action mix, dataset size vs. iteration

## Commands

Train one mode:
```bash
python DAgger_2D_policy.py --mode state --iterations 8
python DAgger_2D_policy.py --mode vision --iterations 8
python DAgger_2D_policy.py --mode both --iterations 8
```

Train and compare all three:
```bash
python DAgger_2D_policy.py --mode all --iterations 8
```

Inference from a saved checkpoint (no training):
```bash
python DAgger_2D_policy.py --mode state --infer --eval-episodes 20 \
    --checkpoint results/state_policy.pt --demo-max-steps 150
```

Collect more training samples (dataset size ≈ `episodes_per_iter × avg_episode_length × iterations`,
aggregated across rounds — DAgger never discards old data):
```bash
python DAgger_2D_policy.py --mode all --iterations 8 --episodes-per-iter 30 --max-steps 80
```

Useful flags: `--episodes-per-iter` (default 15) — rollouts labeled by the expert per DAgger
round, `--max-steps` (default 50) — training/eval episode horizon, `--epochs`, `--eval-episodes`,
`--outdir`, `--checkpoint <path>` (defaults to `{outdir}/{mode}_policy.pt`), `--no-viz` to skip
plots/GIFs, `--demo-max-steps` (default 150) — episode horizon used only for the rendered GIFs,
independent of `--max-steps`, so the demo has time to reach the goal or clearly show it failing to.
