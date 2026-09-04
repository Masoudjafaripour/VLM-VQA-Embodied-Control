# VLA Evaluation

A common `Policy` interface plus wrappers for well-known VLAs, evaluated against the [LIBERO](../LIBERO/) and [CALVIN](../CALVIN/) baselines in this repo.

## Files

| File | What it does |
|---|---|
| [`VLAs.py`](VLAs.py) | The `Policy` interface (`get_action(image, instruction) -> action`) plus one wrapper class per VLA, and `POLICY_REGISTRY` mapping a CLI name to each. |
| [`eval.py`](eval.py) | Runs a policy across a fixed set of LIBERO/CALVIN tasks (`tqdm` progress bar per run), writes one result JSON + one rollout GIF per episode under [`results/<policy>/`](results), and renders `results/summary_plot.png` (matplotlib, `Agg` backend - headless, success/episode counts annotated on every bar) from every result on disk. |
| [`results/`](results) | Eval output, one subfolder per policy: JSON + GIFs per benchmark, plus the shared `summary_plot.png`. Real runs from this environment (see below) - not a benchmark leaderboard. |

`LIBERO.py`/`libero_vla_rollout.py` and `CALVIN.py`/`calvin_vla_rollout.py` import `Policy`/`DummyPolicy`/`OpenVLAPolicy` from here rather than redefining them.

## Models covered

| Model | Status here | Why |
|---|---|---|
| OpenVLA | **ran** (real 7B checkpoint, GPU) | `openvla/openvla-7b-finetuned-libero-spatial` on `libero_spatial`, verified in the actual LIBERO rollout loop |
| SmolVLA | **`get_action()` validated** | `lerobot/smolvla_base` loads and produces a real action through our adapter; not yet run through the full LIBERO/CALVIN loop (different venv, see below) |
| pi0 / pi0.5 | **weights loaded**, blocked before inference | `lerobot/pi0_base` / `pi05_base` (14GB, 14.5GB) downloaded and their state dicts loaded successfully; both fail one step later building the tokenizer, which pulls `google/paligemma-3b-pt-224` - a **gated** HF repo needing an approved, authenticated HF token |
| Octo | wired, not run | JAX - needs its own venv, would risk the CUDA-pinned torch builds above |
| RT-1 | wired, not run | TensorFlow - same reasoning as Octo, plus checkpoints ship as a GCS bucket, not a pip package |
| RT-2 | never runnable | Google never released weights |
| CoT-VLA | never runnable | No public checkpoint |

## Two venvs, on purpose

- **`vla_venv`** (repo root) - LIBERO, CALVIN, OpenVLA. Pinned to `torch==2.5.1+cu121` / `torchvision==0.20.1` (LIBERO's init-state loading breaks on torch≥2.6) and `transformers==4.40.1` (OpenVLA's remote code needs the since-removed `AutoModelForVision2Seq`).
- **`vla_venv_lerobot`** (repo root) - SmolVLA, pi0, pi0.5. `lerobot` requires `torch>=2.7`, which conflicts directly with the pin above, so it lives in its own venv:
  ```bash
  python3 -m venv vla_venv_lerobot
  source vla_venv_lerobot/bin/activate
  pip install "lerobot[smolvla,pi]" matplotlib tqdm
  ```
  This venv does **not** have LIBERO/CALVIN installed, so `eval.py` here can only exercise these policies' `get_action()` directly, not the full benchmark rollout - installing `calvin_env`'s and LIBERO's own deps into this venv too (following the recipes in the [LIBERO](../LIBERO/README.md#environment-setup) and [CALVIN](../CALVIN/README.md#environment-setup) READMEs) would close that gap.

Octo (JAX) and RT-1 (TensorFlow) would each need a *third* venv; not created here.

## Running

```bash
# dummy + OpenVLA, both benchmarks (vla_venv)
MUJOCO_GL=egl vla_venv/bin/python baseline/Models/eval.py --policies dummy openvla --benchmarks libero calvin --n-steps 30 --episodes 2

# SmolVLA/pi0/pi0.5 (vla_venv_lerobot) - see the venv note above on why this only
# exercises get_action(), not a full rollout, until LIBERO/CALVIN are installed there too
vla_venv_lerobot/bin/python baseline/Models/eval.py --policies smolvla --benchmarks libero calvin
```

`eval.py --help` lists all flags. Each `(policy, benchmark)` run writes `results/<policy>/<benchmark>.json`, a GIF per episode under `results/<policy>/<benchmark>/`, and regenerates `results/summary_plot.png` from *every* result JSON currently on disk, so plots stay complete across separate runs (and across the two venvs). Add a new VLA by writing a class in `VLAs.py` with `get_action(image, instruction) -> np.ndarray` and registering it in `POLICY_REGISTRY`.

## Sample results

Generated in this environment, under [`results/`](results):

```
results/
├── dummy/
│   ├── libero.json                     success_rate: 0.0 (4 episodes, random actions)
│   ├── libero/episode{0..3}_*.gif      one rollout GIF per task
│   ├── calvin.json                     success_rate: 0.0 (4 episodes, random actions)
│   └── calvin/episode{0..3}_*.gif
├── openvla/
│   ├── libero.json                     success_rate: 0.0 (2 episodes, 15-step smoke test, real 7B GPU inference)
│   └── libero/episode{0,1}_*.gif
├── smolvla/get_action_check.json       status: get_action_validated (real weights, single forward pass)
├── pi0/status.json                     status: not_run - gated tokenizer repo (see table above)
├── pi0.5/status.json                   status: not_run - gated tokenizer repo
├── octo/{libero,calvin}.json           status: not_run - needs its own JAX venv
├── rt1/{libero,calvin}.json            status: not_run - needs its own TF venv
├── rt2/{libero,calvin}.json            status: not_run - no public checkpoint
├── cot-vla/{libero,calvin}.json        status: not_run - no public checkpoint
└── summary_plot.png                    bar chart, success/episode count labeled on every bar
```

0% success rates are expected here - these are short smoke-test rollouts (5-15 steps, 1-2 episodes) proving the plumbing works end-to-end, not reproductions of published benchmark numbers (which need full-length episodes, many trials, and - for a fair OpenVLA/pi0/pi0.5 number - the checkpoint fine-tuned for that specific suite). The GIFs are the more useful artifact at this trial count: `results/openvla/libero/episode0_libero_spatial_0.gif` shows directed, non-random arm motion toward the table over its 15 steps (too short to complete the pick-and-place), versus `results/dummy/*`, which visibly flails.
