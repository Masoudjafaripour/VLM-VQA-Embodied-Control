# VLA Evaluation

A common `Policy` interface plus wrappers for well-known VLAs, evaluated against the [LIBERO](../LIBERO/) and [CALVIN](../CALVIN/) baselines in this repo.

## Files

| File | What it does |
|---|---|
| [`VLAs.py`](VLAs.py) | The `Policy` interface (`get_action(image, instruction) -> action`) plus one wrapper class per VLA, and `POLICY_REGISTRY` mapping a CLI name to each. |
| [`eval.py`](eval.py) | Runs a policy across a fixed set of LIBERO/CALVIN tasks (`tqdm` progress bar per run), writes one result JSON + one rollout GIF per episode under [`results/<policy>/`](results), and renders `results/summary_plot.png` (matplotlib, `Agg` backend - headless) from every result on disk. |
| [`results/`](results) | Eval output, one subfolder per policy: JSON + GIFs per benchmark, plus the shared `summary_plot.png`. Real runs from this environment (see below) - not a benchmark leaderboard. |

`LIBERO.py`/`libero_vla_rollout.py` and `CALVIN.py`/`calvin_vla_rollout.py` import `Policy`/`DummyPolicy`/`OpenVLAPolicy` from here rather than redefining them.

## Models covered

| Model | Status here | Why |
|---|---|---|
| OpenVLA | **ran** (real 7B checkpoint, GPU) | `openvla/openvla-7b-finetuned-libero-spatial` on `libero_spatial`, verified in the actual LIBERO rollout loop |
| SmolVLA | **`get_action()` validated** | `lerobot/smolvla_base` loads and produces a real action through our adapter; not yet run through the full LIBERO/CALVIN loop (different venv, see below) |
| pi0 / pi0.5 | **weights loaded**, blocked before inference | `lerobot/pi0_base` / `pi05_base` (14GB, 14.5GB) downloaded and their state dicts loaded successfully; both fail one step later building the tokenizer, which pulls `google/paligemma-3b-pt-224` - a **gated** HF repo needing an approved, authenticated HF token |
| RT-1 | **`get_action()` validated** | The real `rt1main` checkpoint (a stateful `tf_agents` `TFPolicy` SavedModel, not a bare forward-pass model) loads and produces real, evolving 7-dim actions across sequential steps with policy state correctly threaded through and a real Universal Sentence Encoder instruction embedding; not yet run through the full LIBERO/CALVIN loop (third venv, see below) |
| Octo | wired, not run | JAX - needs its own venv, would risk the CUDA-pinned torch builds above |
| RT-2 | never runnable | Google never released weights |
| CoT-VLA | never runnable | No public checkpoint |

## Three venvs, on purpose

- **`vla_venv`** (repo root) - LIBERO, CALVIN, OpenVLA. Pinned to `torch==2.5.1+cu121` / `torchvision==0.20.1` (LIBERO's init-state loading breaks on torch≥2.6) and `transformers==4.40.1` (OpenVLA's remote code needs the since-removed `AutoModelForVision2Seq`).
- **`vla_venv_lerobot`** (repo root) - SmolVLA, pi0, pi0.5. `lerobot` requires `torch>=2.7`, which conflicts directly with the pin above, so it lives in its own venv:
  ```bash
  python3 -m venv vla_venv_lerobot
  source vla_venv_lerobot/bin/activate
  pip install "lerobot[smolvla,pi]" matplotlib tqdm
  ```
- **`vla_venv_tf`** (repo root) - RT-1. TensorFlow, not torch at all - a fully separate stack:
  ```bash
  python3 -m venv vla_venv_tf
  source vla_venv_tf/bin/activate
  pip install tensorflow tensorflow_probability tf-agents --no-deps
  pip install gin-config tf-keras pillow "gym<=0.23.0,>=0.17.0" tensorflow_hub "setuptools<81" matplotlib tqdm
  ```
  `tf-agents` needs `--no-deps` on Python 3.12 (its pinned `pygame==2.1.3` fails to build there; its other pins - `gym`, `pillow`, `tensorflow-probability~=0.23` - are all satisfied loosely enough by newer versions). `setuptools<81` is needed because `tensorflow_hub` still imports the now-removed `pkg_resources`.

  Checkpoint - the repo ships it via **git-lfs**, so a plain `git clone` leaves LFS pointer stubs instead of real weights. Without `git-lfs` installed, fetch the real files from GitHub's LFS media endpoint directly:
  ```bash
  git clone https://github.com/google-research/robotics_transformer.git ~/rt1_repo
  CKPT=~/rt1_repo/trained_checkpoints/rt1main
  MEDIA="https://media.githubusercontent.com/media/google-research/robotics_transformer/master/trained_checkpoints/rt1main"
  curl -sL "$MEDIA/variables/variables.data-00000-of-00001" -o "$CKPT/variables/variables.data-00000-of-00001"
  ```
  (`saved_model.pb`, `variables/variables.index`, `fingerprint.pb` and `assets/metadata.textproto` are small enough that git clones them as real files, not LFS pointers - only the actual weight tensor needs the LFS fetch.)

None of `vla_venv`/`vla_venv_lerobot`/`vla_venv_tf` have the other two venvs' benchmark/model packages installed, so `eval.py` run from `vla_venv_lerobot` or `vla_venv_tf` can only exercise a policy's `get_action()` directly (see the sample-result JSONs below), not the full LIBERO/CALVIN rollout loop - installing `calvin_env`'s and LIBERO's own deps into those venvs too (following the recipes in the [LIBERO](../LIBERO/README.md#environment-setup) and [CALVIN](../CALVIN/README.md#environment-setup) READMEs) would close that gap.

Octo (JAX) would need a *fourth* venv; not created here.

## Running

```bash
# OpenVLA on LIBERO - restricted by default to libero_spatial, the one suite the
# default checkpoint (openvla-7b-finetuned-libero-spatial) is actually fine-tuned on
MUJOCO_GL=egl vla_venv/bin/python baseline/Models/eval.py --policies openvla --benchmarks libero --episodes 3

# same, but sweeping every LIBERO suite with one checkpoint (numbers on the 3
# mismatched suites won't be meaningful, but the plumbing still runs)
MUJOCO_GL=egl vla_venv/bin/python baseline/Models/eval.py --policies openvla --benchmarks libero --libero-suites libero_spatial libero_object libero_goal libero_10

# dummy sanity-checks the harness on both benchmarks with no weights needed
MUJOCO_GL=egl vla_venv/bin/python baseline/Models/eval.py --policies dummy --benchmarks libero calvin

# SmolVLA/pi0/pi0.5 (vla_venv_lerobot) - see the venv note above on why this only
# exercises get_action(), not a full rollout, until LIBERO/CALVIN are installed there too
vla_venv_lerobot/bin/python baseline/Models/eval.py --policies smolvla --benchmarks libero calvin

# RT-1 (vla_venv_tf) - same get_action()-only caveat as above
vla_venv_tf/bin/python -c "
from Models.VLAs import RT1Policy
import numpy as np
policy = RT1Policy()
policy.reset()
print(policy.get_action(np.zeros((256, 256, 3), dtype=np.uint8), 'lift the red block'))
"
```

`eval.py --help` lists all flags. Each `(policy, benchmark)` run writes `results/<policy>/<benchmark>.json`, a GIF per episode under `results/<policy>/<benchmark>/`, and regenerates `results/summary_plot.png` from *every* result JSON currently on disk, so plots stay complete across separate runs (and across all three venvs) - though `eval.py` itself only runs where it can import the LIBERO/CALVIN rollout modules, i.e. from `vla_venv` right now (see the venv note above). Add a new VLA by writing a class in `VLAs.py` with `get_action(image, instruction) -> np.ndarray` and registering it in `POLICY_REGISTRY`.

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
├── rt1/
│   ├── get_action_check.json           status: get_action_validated (real checkpoint, 3 sequential steps, real actions)
│   └── {libero,calvin}.json            status: not_run - TF isn't installed in vla_venv (RT-1 needs vla_venv_tf)
├── octo/{libero,calvin}.json           status: not_run - needs its own JAX venv
├── rt2/{libero,calvin}.json            status: not_run - no public checkpoint
├── cot-vla/{libero,calvin}.json        status: not_run - no public checkpoint
└── summary_plot.png                    ● success / ✕ fail marker per episode, one row per (policy, benchmark)
```

0% success rates are expected here - these are short smoke-test rollouts (5-15 steps, 1-2 episodes) proving the plumbing works end-to-end, not reproductions of published benchmark numbers (which need full-length episodes, many trials, and - for a fair OpenVLA/pi0/pi0.5 number - the checkpoint fine-tuned for that specific suite). The GIFs are the more useful artifact at this trial count: `results/openvla/libero/episode0_libero_spatial_0.gif` shows directed, non-random arm motion toward the table over its 15 steps (too short to complete the pick-and-place), versus `results/dummy/*`, which visibly flails.
