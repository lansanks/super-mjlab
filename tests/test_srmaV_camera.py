"""Tests for the Black SRMAV camera sensor."""

import torch
from conftest import get_test_device

import mjlab.tasks.srmaV.config.black  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.srmaV.config.black.env_cfg import black_srmaV_env_cfg


def test_black_srmaV_play_has_depth_camera() -> None:
  train_sensors = {
    sensor.name for sensor in (black_srmaV_env_cfg().scene.sensors or ())
  }
  assert "black_depth_camera" not in train_sensors
  assert "srmaV_teacher_scan" not in train_sensors

  cfg = black_srmaV_env_cfg(play=True)
  sensors = {sensor.name: sensor for sensor in (cfg.scene.sensors or ())}
  camera = sensors["black_depth_camera"]

  assert isinstance(camera, CameraSensorCfg)
  assert camera.parent_body == "robot/d435_link"
  assert camera.data_types == ("rgb", "depth")
  assert (camera.width, camera.height) == (160, 90)


def test_black_srmaV_depth_camera_renders() -> None:
  cfg = black_srmaV_env_cfg(play=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    observations = env.observation_manager.compute()
    height_map = observations["teacher_height_map"]
    assert isinstance(height_map, torch.Tensor)
    assert height_map.shape == (1, 187)
    assert bool(height_map.isfinite().all())
    data = env.scene["black_depth_camera"].data
    assert data.rgb is not None
    assert data.rgb.shape == (1, 90, 160, 3)
    assert data.rgb.any()
    assert data.depth is not None
    assert data.depth.shape == (1, 90, 160, 1)
    assert bool(data.depth.isfinite().all())
    assert float(data.depth.median()) > 0.1
  finally:
    env.close()
