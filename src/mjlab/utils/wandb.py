"""WandB utilities."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Sequence

_DEFAULT_BASE_URL = "https://api.wandb.ai"
_PROBE_BODY = b'{"query": "query Preflight { viewer { id } }", "variables": {}}'


def ensure_wandb_reachable(timeout: float = 5.0) -> bool:
  """Check that the W&B API answers, falling back to offline logging if it does not.

  Training runs last for hours, so an unreachable API at startup should not kill
  them, but ``wandb.init`` raises ``CommError`` in that case. When the probe
  fails, ``WANDB_MODE`` is set to ``offline``: metrics are written next to the
  run and can be uploaded later with ``wandb sync``.

  The endpoint is probed with a minimal GraphQL query, which returns ``403``
  with an HTML body when the exit IP is blocked (a proxy node W&B rejects) and
  refuses connections when the network is down. Both cases fall back to offline.

  Returns:
    True when the API answered, i.e. online logging can proceed.
  """
  mode = os.environ.get("WANDB_MODE", "").strip().lower()
  if mode and mode != "online":
    # An explicit offline/disabled run was requested; nothing to check.
    return False

  base_url = os.environ.get("WANDB_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
  reason = "connection failed"
  try:
    request = urllib.request.Request(
      f"{base_url}/graphql",
      data=_PROBE_BODY,
      method="POST",
      headers={"Content-Type": "application/json", "User-Agent": "mjlab"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
      if 200 <= response.status < 300:
        return True
      reason = f"HTTP {response.status}"
  except urllib.error.HTTPError as exc:
    if 200 <= exc.code < 300:
      return True
    reason = f"HTTP {exc.code}"
  except Exception as exc:  # noqa: BLE001
    reason = f"{type(exc).__name__}: {exc}"

  os.environ["WANDB_MODE"] = "offline"
  print(f"[WARN] W&B API unreachable ({reason}), logging offline to disk.")
  print("[WARN] Upload the run later with `uv run wandb sync wandb/run-*`.")
  if reason.startswith("HTTP 403"):
    print(
      "[WARN] A 403 means this network exit is blocked by W&B; try another "
      "proxy node to restore online logging."
    )
  return False


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run.

  Note: This function stores tags in wandb.config._wandb_tags if the run is not yet
  initialized, allowing them to be retrieved later. If the run is already initialized,
  tags are added directly.
  """
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
    else:
      # Store tags to be added when run is initialized.
      # This is a workaround for lazy wandb initialization in rsl_rl 3.1.0.
      current_tags = os.environ.get("WANDB_TAGS", "")
      all_tags = set(current_tags.split(",") if current_tags else [])
      all_tags.update(tags)
      os.environ["WANDB_TAGS"] = ",".join(sorted(all_tags))
  except ImportError:
    pass
