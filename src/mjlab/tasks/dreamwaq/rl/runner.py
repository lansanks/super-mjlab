import wandb
from rsl_rl.models import MLPModel
from rsl_rl.utils import WandbLogWriter

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.dreamwaq.rl.network.actor import DreamwaqActorModel


class DreamwaqOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def get_inference_policy(self, device: str | None = None) -> MLPModel:
    """Return the play policy and expose its velocity estimate to the viewer."""
    policy = super().get_inference_policy(device)
    if isinstance(policy, DreamwaqActorModel):
      # The viewer's debug visualizer draws the CENet estimate from this buffer.
      policy.bind_debug_env(self.env.unwrapped)
    return policy

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      is_wandb = isinstance(self.logger.writer, WandbLogWriter)
      run_name: str = wandb.run.name if is_wandb and wandb.run else "local"  # type: ignore[assignment]
      # The exported model consumes the CENet observation history (six frames),
      # not the 45-dim actor observation, so describe that group instead.
      actor = self.alg.actor
      obs_group = (
        actor.cenet_group if isinstance(actor, DreamwaqActorModel) else "actor"
      )
      metadata = get_base_metadata(self.env.unwrapped, run_name, obs_group=obs_group)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if is_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as e:
      print(f"[WARN] ONNX export failed (training continues): {e}")
