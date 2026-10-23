"""Runner for SRMAV student DAgger training and deployment export."""

from __future__ import annotations

import torch
import wandb
from rsl_rl.models import MLPModel
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.srmaV.rl.network.student_actor import SRMAVStudentActorModel


class SRMAVStudentRunner(MjlabOnPolicyRunner):
  """Persist DAgger stage state and export the deployable student policy."""

  env: RslRlVecEnvWrapper

  def get_inference_policy(self, device: str | None = None) -> MLPModel:
    policy = super().get_inference_policy(device)
    if isinstance(policy, SRMAVStudentActorModel):
      policy.bind_debug_env(self.env.unwrapped)
    return policy

  def save(self, path: str, infos=None) -> None:
    """Save the algorithm stage state without dropping its ``srmaV`` metadata."""
    saved_dict = self.alg.save()
    saved_infos = dict(saved_dict.get("infos") or {})
    saved_infos.update(infos or {})
    saved_infos["env_state"] = {
      "common_step_counter": self.env.unwrapped.common_step_counter
    }
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = saved_infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"]:
      self.logger.save_model(path, self.current_learning_iteration)

    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)
      run_name: str = (wandb.run.name if is_wandb and wandb.run else "local") or "local"
      metadata = get_base_metadata(self.env.unwrapped, run_name, obs_group="actor")
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if is_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as exc:
      print(f"[WARN] ONNX export failed (training continues): {exc}")
