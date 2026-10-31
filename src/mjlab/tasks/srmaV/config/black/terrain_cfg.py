"""Black SRMAV stair, rough and broken-bridge terrain set."""

from mjlab.terrains import BoxBrokenBridgeTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv, random_rough

# Several bridge columns are generated so the pit depth is drawn per patch from
# a distribution instead of being pinned to a single value. Every bridge column
# shares the same gap layout so the bridge reward and command stay valid.
BLACK_SRMAV_BRIDGE_TERRAINS = (
  "broken_bridge",
  "broken_bridge_b",
  "broken_bridge_c",
  "broken_bridge_d",
)
BLACK_SRMAV_BRIDGE_TERRAIN = BLACK_SRMAV_BRIDGE_TERRAINS[0]

BLACK_SRMAV_BRIDGE_DEPTH_RANGE = (0.2, 1.0)


def _broken_bridge_cfg() -> BoxBrokenBridgeTerrainCfg:
  """Create one bridge column with per-patch pit-depth randomization."""
  return BoxBrokenBridgeTerrainCfg(
    proportion=1.0,
    gap_width_range=(0.10, 0.35),
    num_gaps=4,
    deck_between_gaps=1.2,
    end_platform_width=1.0,
    pit_depth_range=BLACK_SRMAV_BRIDGE_DEPTH_RANGE,
  )


BLACK_SRMAV_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=3 + len(BLACK_SRMAV_BRIDGE_TERRAINS),
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
    **{name: _broken_bridge_cfg() for name in BLACK_SRMAV_BRIDGE_TERRAINS},
  },
  add_lights=True,
)
