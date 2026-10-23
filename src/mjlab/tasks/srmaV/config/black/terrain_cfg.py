"""Black SRMAV stair, rough and broken-bridge terrain set."""

from mjlab.terrains import BoxBrokenBridgeTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv, random_rough

BLACK_SRMAV_BRIDGE_TERRAIN = "broken_bridge"

BLACK_SRMAV_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=4,
  curriculum=True,
  sub_terrains={
    "stairs_up": pyramid_stairs(
      proportion=1.0,
      step_height_range=(0.0, 0.25),
    ),
    "stairs_down": pyramid_stairs_inv(
      proportion=1.0,
      step_height_range=(0.0, 0.25),
    ),
    "random_rough": random_rough(proportion=1.0),
    BLACK_SRMAV_BRIDGE_TERRAIN: BoxBrokenBridgeTerrainCfg(
      proportion=1.0,
      gap_width_range=(0.10, 0.35),
      num_gaps=4,
      deck_between_gaps=1.2,
      end_platform_width=1.0,
      pit_depth=1.0,
    ),
  },
  add_lights=True,
)
