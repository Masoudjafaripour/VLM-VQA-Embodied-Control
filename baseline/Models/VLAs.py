"""VLA policy wrappers evaluated on the LIBERO and CALVIN baselines in this repo.

Every policy implements the same interface:

    class Policy:
        def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray: ...

`image` is an (H, W, 3) uint8 frame, `instruction` a natural-language task string.
See README.md for per-model install commands, which venv each needs, checkpoint
choices, and which of these actually ran here vs. are wired-but-unexecuted or
have no public weights at all.
"""
import os

import numpy as np


class Policy:
    """Interface every VLA wrapper in this file implements."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        raise NotImplementedError


class DummyPolicy(Policy):
    """Random 7-dim action, no weights needed. Sanity-checks the eval loop."""

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        return np.concatenate([np.random.uniform(-1, 1, size=6), np.random.choice([-1, 1], size=1)])


# --------------------------------------------------------------------------
# OpenVLA - https://github.com/openvla/openvla (venv: vla_venv)
# --------------------------------------------------------------------------
class OpenVLAPolicy(Policy):
    """7B VLA (Prismatic VLM backbone) with per-dataset action-unnormalization heads.

    Needs: pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10"
    "accelerate==0.25.0" in vla_venv (transformers>=5 dropped AutoModelForVision2Seq,
    which OpenVLA's remote code relies on). Use the benchmark-specific fine-tuned
    checkpoint for real results, e.g. "openvla/openvla-7b-finetuned-libero-spatial"
    (also -object/-goal/-10), rather than the base "openvla/openvla-7b" checkpoint.
    """

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b-finetuned-libero-spatial",
        device: str = "cuda",
        unnorm_key: str = "libero_spatial",
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.unnorm_key = unnorm_key
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device)

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        from PIL import Image

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to(self.device, dtype=self.torch.bfloat16)
        return self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)


# --------------------------------------------------------------------------
# HuggingFace LeRobot policies: SmolVLA, pi0, pi0.5 (venv: vla_venv_lerobot -
# kept SEPARATE from OpenVLA's vla_venv, since lerobot needs torch>=2.7 while
# LIBERO's init-state loading needs torch<2.6 - the two can't share a venv)
# --------------------------------------------------------------------------
class _LeRobotPolicy(Policy):
    """Shared get_action() adapter for LeRobot-hosted policies.

    Broadcasts our single camera frame to every camera slot the checkpoint
    expects and feeds a zero proprioceptive state, since this repo's rollout
    scripts don't track real robot state - fine for a smoke-test rollout, not
    for reproducing published numbers (use a LIBERO/CALVIN-finetuned checkpoint
    and wire in real proprioception for that).
    """

    def __init__(self, policy_cls, repo_id: str, device: str = "cuda"):
        import torch
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.device = device
        self.policy = policy_cls.from_pretrained(repo_id).eval().to(device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, pretrained_path=repo_id)
        self.policy.reset()

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        img = self.torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        obs = {cam: img for cam in self.policy.config.image_features}
        obs["observation.state"] = self.torch.zeros(1, self.policy.config.robot_state_feature.shape[0], device=self.device)
        obs["task"] = [instruction]
        batch = self.preprocessor(obs)
        action = self.postprocessor(self.policy.select_action(batch))
        return action.squeeze(0).float().cpu().numpy()

    def reset(self):
        """Clear the action-chunk queue lerobot policies keep between episodes."""
        self.policy.reset()


class SmolVLAPolicy(_LeRobotPolicy):
    """450M VLA from HuggingFace LeRobot: https://huggingface.co/lerobot/smolvla_base"""

    def __init__(self, repo_id: str = "lerobot/smolvla_base", device: str = "cuda"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _Impl

        super().__init__(_Impl, repo_id, device)


class Pi0Policy(_LeRobotPolicy):
    """Physical Intelligence pi0, ported to PyTorch in LeRobot. Use the
    LIBERO-finetuned checkpoint "lerobot/pi0_libero_finetuned_v044" (14GB) for
    real LIBERO numbers; "lerobot/pi0_base" below is a generalist checkpoint.
    """

    def __init__(self, repo_id: str = "lerobot/pi0_base", device: str = "cuda"):
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as _Impl

        super().__init__(_Impl, repo_id, device)


class Pi05Policy(_LeRobotPolicy):
    """Physical Intelligence pi0.5. LIBERO-finetuned checkpoint:
    "lerobot/pi05_libero_finetuned_v044" (14.5GB); "lerobot/pi05_base" is the
    generalist checkpoint.
    """

    def __init__(self, repo_id: str = "lerobot/pi05_base", device: str = "cuda"):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as _Impl

        super().__init__(_Impl, repo_id, device)


# --------------------------------------------------------------------------
# Octo - https://github.com/octo-models/octo (JAX). Needs its own venv: JAX
# alongside the CUDA-pinned torch builds above risks breaking them.
# --------------------------------------------------------------------------
class OctoPolicy(Policy):
    """Needs its own venv:
        git clone https://github.com/octo-models/octo && cd octo && pip install -e .
    """

    def __init__(self, model_path: str = "hf://rail-berkeley/octo-base-1.5"):
        try:
            from octo.model.octo_model import OctoModel
        except ImportError as e:
            raise NotImplementedError("octo is not installed - it needs its own JAX venv, see README.md") from e
        self.model = OctoModel.load_pretrained(model_path)

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        import jax

        task = self.model.create_tasks(texts=[instruction])
        obs = {"image_primary": image[None, None], "timestep_pad_mask": np.array([[True]])}
        actions = self.model.sample_actions(obs, task, rng=jax.random.PRNGKey(0))
        return np.asarray(actions[0, 0])


# --------------------------------------------------------------------------
# RT-1 - https://github.com/google-research/robotics_transformer (TensorFlow).
# Public checkpoints, but a third framework alongside JAX/torch above; needs
# its own venv (vla_venv_tf, see README.md).
#
# The checkpoint isn't a bare forward-pass model: it's a tf_agents TFPolicy
# SavedModel, exported with 'action' and 'get_initial_state' signatures. It's
# stateful (keeps a 6-frame image/action-token history across calls) and was
# trained on real-robot observations this repo's sim envs don't produce -
# proprioceptive/workspace-calibration fields below (gripper_closed,
# workspace_bounds, orientation_box, ...) are zero-filled placeholders, and the
# instruction is embedded with the Universal Sentence Encoder RT-1 was trained
# against, not learned end-to-end from raw text like OpenVLA/SmolVLA.
# --------------------------------------------------------------------------
class RT1Policy(Policy):
    """Needs its own venv (see README.md for the full recipe, including why
    tf-agents needs --no-deps on Python 3.12 and the setuptools<81 pin
    tensorflow_hub needs for pkg_resources).
    Checkpoint: clone https://github.com/google-research/robotics_transformer
    and fetch trained_checkpoints/rt1main via git-lfs (or the
    media.githubusercontent.com LFS endpoint directly, if git-lfs isn't installed).
    """

    IMAGE_HW = (256, 320)  # (height, width) - RT-1's real-robot camera aspect ratio, not square like LIBERO/CALVIN

    def __init__(
        self,
        checkpoint_dir: str = "~/rt1_repo/trained_checkpoints/rt1main",
        use_model: str = "https://tfhub.dev/google/universal-sentence-encoder-large/5",
    ):
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            import tf_agents  # noqa: F401 - import side effect: registers the TypeSpec RT-1's SavedModel needs
        except ImportError as e:
            raise NotImplementedError("tensorflow/tf_agents/tensorflow_hub are not installed - RT-1 needs its own TF venv, see README.md") from e

        self.tf = tf
        self.model = tf.saved_model.load(checkpoint_dir)
        self.use = hub.load(use_model)
        self.policy_state = None

    def reset(self):
        self.policy_state = None

    def get_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        tf = self.tf
        first_step = self.policy_state is None
        if first_step:
            self.policy_state = self.model.signatures["get_initial_state"](batch_size=tf.constant(1))

        image_resized = tf.image.resize(image, self.IMAGE_HW, method="bilinear")
        image_resized = tf.cast(image_resized, tf.uint8)
        instruction_embedding = self.use([instruction])[0]

        zeros = lambda *shape: tf.zeros((1, *shape), dtype=tf.float32)  # noqa: E731 - placeholders for real-robot-only fields, see class docstring
        observation = {
            "image": image_resized[tf.newaxis, ...],
            "natural_language_instruction": tf.constant([instruction]),
            "natural_language_embedding": instruction_embedding[tf.newaxis, ...],
            "gripper_closed": zeros(1),
            "gripper_closedness_commanded": zeros(1),
            "height_to_bottom": zeros(1),
            "base_pose_tool_reached": zeros(7),
            "workspace_bounds": zeros(3, 3),
            "robot_orientation_positions_box": zeros(3, 3),
            "orientation_box": zeros(2, 3),
            "orientation_start": zeros(4),
            "src_rotation": zeros(4),
            "vector_to_go": zeros(3),
            "rotation_delta_to_go": zeros(3),
        }
        inputs = {f"0/observation/{k}": v for k, v in observation.items()}
        inputs["0/step_type"] = tf.constant([0 if first_step else 1], dtype=tf.int32)
        inputs["0/reward"] = tf.constant([0.0], dtype=tf.float32)
        inputs["0/discount"] = tf.constant([1.0], dtype=tf.float32)
        inputs.update({f"1/{k}": v for k, v in self.policy_state.items()})

        result = self.model.signatures["action"](**inputs)
        self.policy_state = {k[len("state/"):]: v for k, v in result.items() if k.startswith("state/")}

        world_vector = result["action/world_vector"][0].numpy()
        rotation_delta = result["action/rotation_delta"][0].numpy()
        gripper = np.atleast_1d(result["action/gripper_closedness_action"][0].numpy())[:1]
        return np.concatenate([world_vector, rotation_delta, gripper])


# --------------------------------------------------------------------------
# RT-2 and CoT-VLA: no public checkpoints exist for either. Kept in the
# registry so the results table can report *why* they weren't run, not just
# that they weren't.
# --------------------------------------------------------------------------
class RT2Policy(Policy):
    """RT-2 (Brohan et al., 2023) - Google never released model weights."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RT-2 has no public checkpoint (Google never released weights)")


class CoTVLAPolicy(Policy):
    """CoT-VLA (chain-of-thought VLA) - no public checkpoint has been released."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CoT-VLA has no public checkpoint")


POLICY_REGISTRY = {
    "dummy": DummyPolicy,
    "openvla": OpenVLAPolicy,
    "smolvla": SmolVLAPolicy,
    "pi0": Pi0Policy,
    "pi0.5": Pi05Policy,
    "octo": OctoPolicy,
    "rt1": RT1Policy,
    "rt2": RT2Policy,
    "cot-vla": CoTVLAPolicy,
}
