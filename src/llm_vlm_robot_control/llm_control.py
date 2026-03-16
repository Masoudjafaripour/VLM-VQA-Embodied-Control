import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   # text-only 3B planner
XML_PATH = "external/mjctrl/franka_emika_panda/scene.xml"

integration_dt: float = 0.1
damping: float = 1e-4
Kpos: float = 0.95
Kori: float = 0.95
gravity_compensation: bool = True
dt: float = 0.002
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])
max_angvel = 0.785

REACH_POS_THRESH = 0.015
REACH_ORI_THRESH = 0.08
MAX_SUBGOAL_STEPS = 1000


# ============================================================
# Data structures
# ============================================================

@dataclass
class Subgoal:
    position: List[float]               # [x, y, z]
    yaw_deg: float = 0.0                # simple orientation control
    reason: str = ""


# ============================================================
# LLM Planner
# ============================================================

class QwenPlanner:
    def __init__(self, model_name: str = MODEL_NAME):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()

    def build_prompt(self, instruction: str, scene: Dict[str, Any]) -> str:
        return f"""
You are a robot motion planner.

Task:
Convert the user instruction and current robot scene into a short sequence of safe end-effector subgoals.

Rules:
- Output ONLY valid JSON.
- Use this schema:
{{
  "subgoals": [
    {{
      "position": [x, y, z],
      "yaw_deg": 0.0,
      "reason": "short text"
    }}
  ]
}}
- Keep 2 to 6 subgoals only.
- The robot end-effector should avoid obstacles.
- Use a safe height first if needed.
- Do not output torques or joint angles.
- Positions must be in meters.

Instruction:
{instruction}

Scene:
{json.dumps(scene, indent=2)}
""".strip()

    @torch.no_grad()
    def plan(self, instruction: str, scene: Dict[str, Any]) -> List[Subgoal]:
        prompt = self.build_prompt(instruction, scene)

        messages = [
            {"role": "system", "content": "You are a precise robot planner that returns JSON only."},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,      # start deterministic
            temperature=1.0,
            top_p=1.0,
        )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        data = self._extract_json(generated)
        subgoals = []
        for item in data.get("subgoals", []):
            subgoals.append(
                Subgoal(
                    position=item["position"],
                    yaw_deg=float(item.get("yaw_deg", 0.0)),
                    reason=item.get("reason", ""),
                )
            )
        return subgoals

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # very simple robust extraction
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Planner did not return JSON.\nRaw output:\n{text}")

        raw = text[start:end + 1]
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from planner:\n{raw}") from e


# ============================================================
# Scene extraction
# ============================================================

def extract_scene_state(model: mujoco.MjModel,
                        data: mujoco.MjData,
                        site_id: int,
                        obstacle_names: Optional[List[str]] = None,
                        target_name: str = "target") -> Dict[str, Any]:
    ee_pos = data.site(site_id).xpos.copy().tolist()

    scene = {
        "ee_position": [round(x, 4) for x in ee_pos],
        "obstacles": [],
        "notes": "All coordinates are in world frame, meters."
    }

    if obstacle_names:
        for name in obstacle_names:
            body_id = model.body(name).id
            pos = data.xpos[body_id].copy().tolist()
            scene["obstacles"].append({
                "name": name,
                "position": [round(x, 4) for x in pos]
            })

    # if target is a mocap body, use mocap position instead
    try:
        mocap_id = model.body(target_name).mocapid[0]
        if mocap_id != -1:
            scene["current_target_marker"] = [round(x, 4) for x in data.mocap_pos[mocap_id].tolist()]
    except Exception:
        pass

    return scene


# ============================================================
# Orientation helper
# ============================================================

def yaw_deg_to_quat(yaw_deg: float) -> np.ndarray:
    # Simple yaw-only quaternion around z-axis
    yaw = np.deg2rad(yaw_deg)
    half = yaw / 2.0
    qw = np.cos(half)
    qz = np.sin(half)
    return np.array([qw, 0.0, 0.0, qz], dtype=np.float64)


# ============================================================
# DiffIK step (adapted from your code)
# ============================================================

def diffik_step(model: mujoco.MjModel,
                data: mujoco.MjData,
                site_id: int,
                mocap_id: int,
                q0: np.ndarray,
                dof_ids: np.ndarray,
                actuator_ids: np.ndarray,
                jac: np.ndarray,
                diag: np.ndarray,
                eye_task: np.ndarray,
                twist: np.ndarray,
                site_quat: np.ndarray,
                site_quat_conj: np.ndarray,
                error_quat: np.ndarray) -> None:
    dx = data.mocap_pos[mocap_id] - data.site(site_id).xpos
    twist[:3] = Kpos * dx / integration_dt

    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
    mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    twist[3:] *= Kori / integration_dt

    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # Only arm joints should be commanded.
    J = jac[:, dof_ids]  # shape: (6, 7)

    # Damped least squares in arm space.
    dq = J.T @ np.linalg.solve(J @ J.T + diag, twist)

    # Nullspace posture term.
    dq += (np.eye(len(dof_ids)) - np.linalg.pinv(J) @ J) @ (Kn * (q0[dof_ids] - data.qpos[dof_ids]))

    dq_abs_max = np.abs(dq).max()
    if dq_abs_max > max_angvel:
        dq *= max_angvel / dq_abs_max

    q = data.qpos.copy()
    qvel_full = np.zeros(model.nv)
    qvel_full[dof_ids] = dq

    mujoco.mj_integratePos(model, q, qvel_full, integration_dt)

    for j in dof_ids:
        joint_id = j
        low, high = model.jnt_range[joint_id]
        q[joint_id] = np.clip(q[joint_id], low, high)

    data.ctrl[actuator_ids] = q[dof_ids]
    mujoco.mj_step(model, data)


# ============================================================
# Goal check
# ============================================================

def reached_subgoal(data: mujoco.MjData, site_id: int, mocap_id: int) -> bool:
    pos_err = np.linalg.norm(data.mocap_pos[mocap_id] - data.site(site_id).xpos)

    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)

    ori_vel = np.zeros(3)
    mujoco.mju_quat2Vel(ori_vel, error_quat, 1.0)
    ori_err = np.linalg.norm(ori_vel)

    return pos_err < REACH_POS_THRESH and ori_err < REACH_ORI_THRESH


# ============================================================
# High-level task
# ============================================================

def run_llm_control_demo():
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to MuJoCo >= 3.1.0"

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    site_name = "attachment_site"
    site_id = model.site(site_name).id

    joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    key_name = "home"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos.copy()

    mocap_name = "target"
    mocap_id = model.body(mocap_name).mocapid[0]

    planner = QwenPlanner(MODEL_NAME)

    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye_task = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)

    # Example language instruction.
    instruction = (
        "Move to the goal while avoiding obstacles. "
        "First rise to a safe height, then go around the obstacle from the right, then descend."
    )

    obstacle_names = []  # add body names from your XML if available

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # ----------------------------------------------------
        # Ask the LLM for a plan
        # ----------------------------------------------------
        scene = extract_scene_state(model, data, site_id, obstacle_names=obstacle_names)
        subgoals = planner.plan(instruction, scene)

        print("\nLLM PLAN:")
        for i, g in enumerate(subgoals):
            print(f"{i}: pos={g.position}, yaw={g.yaw_deg}, reason={g.reason}")

        current_idx = 0
        steps_on_current = 0

        if not subgoals:
            raise RuntimeError("Planner returned no subgoals.")

        # Initialize first goal
        data.mocap_pos[mocap_id] = np.array(subgoals[0].position, dtype=np.float64)
        data.mocap_quat[mocap_id] = yaw_deg_to_quat(subgoals[0].yaw_deg)

        while viewer.is_running():
            step_start = time.time()

            # Move to next subgoal if current reached
            if reached_subgoal(data, site_id, mocap_id):
                current_idx += 1
                steps_on_current = 0

                if current_idx >= len(subgoals):
                    print("Task complete.")
                    break

                goal = subgoals[current_idx]
                print(f"Switching to subgoal {current_idx}: {goal}")
                data.mocap_pos[mocap_id] = np.array(goal.position, dtype=np.float64)
                data.mocap_quat[mocap_id] = yaw_deg_to_quat(goal.yaw_deg)

            # Safety timeout per subgoal
            if steps_on_current > MAX_SUBGOAL_STEPS:
                print("Subgoal timeout. Replanning...")
                scene = extract_scene_state(model, data, site_id, obstacle_names=obstacle_names)
                scene["failed_subgoal"] = {
                    "position": subgoals[current_idx].position,
                    "yaw_deg": subgoals[current_idx].yaw_deg,
                    "reason": subgoals[current_idx].reason
                }
                scene["instruction"] = instruction

                subgoals = planner.plan("Replan from the current state and continue the task.", scene)
                current_idx = 0
                steps_on_current = 0

                if not subgoals:
                    raise RuntimeError("Replanner returned no subgoals.")

                data.mocap_pos[mocap_id] = np.array(subgoals[0].position, dtype=np.float64)
                data.mocap_quat[mocap_id] = yaw_deg_to_quat(subgoals[0].yaw_deg)

            diffik_step(
                model=model,
                data=data,
                site_id=site_id,
                mocap_id=mocap_id,
                q0=q0,
                dof_ids=dof_ids,
                actuator_ids=actuator_ids,
                jac=jac,
                diag=diag,
                eye_task=eye_task,
                twist=twist,
                site_quat=site_quat,
                site_quat_conj=site_quat_conj,
                error_quat=error_quat,
            )

            steps_on_current += 1
            viewer.sync()

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    run_llm_control_demo()