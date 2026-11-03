"""Black SRMAM stair, rough and broken-bridge terrain set."""

from copy import deepcopy

from mjlab.terrains import (
  BoxBrokenBridgeTerrainCfg,
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.config import (
  ROUGH_TERRAINS_CFG,
  pyramid_stairs,
  pyramid_stairs_inv,
  random_rough,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg

# Several bridge columns are generated so the pit depth is drawn per patch from
# a distribution instead of being pinned to a single value. Every bridge column
# shares the same gap layout so the bridge reward and command stay valid.
BLACK_SRMAM_BRIDGE_TERRAINS = (
  "broken_bridge",
  "broken_bridge_b",
  "broken_bridge_c",
  "broken_bridge_d",
)
BLACK_SRMAM_BRIDGE_TERRAIN = BLACK_SRMAM_BRIDGE_TERRAINS[0]

BLACK_SRMAM_BRIDGE_DEPTH_RANGE = (0.2, 1.0)


def _broken_bridge_cfg() -> BoxBrokenBridgeTerrainCfg:
  """Create one bridge column with per-patch pit-depth randomization."""
  return BoxBrokenBridgeTerrainCfg(
    proportion=1.0,
    gap_width_range=(0.10, 0.35),
    num_gaps=4,
    deck_between_gaps=1.2,
    end_platform_width=1.0,
    pit_depth_range=BLACK_SRMAM_BRIDGE_DEPTH_RANGE,
  )


BLACK_SRMAM_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=3 + len(BLACK_SRMAM_BRIDGE_TERRAINS),
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
    **{name: _broken_bridge_cfg() for name in BLACK_SRMAM_BRIDGE_TERRAINS},
  },
  add_lights=True,
)


def make_srmaM_loco_terrains_cfg() -> TerrainGeneratorCfg:
  """Return SRMA's rough map as SRMA's Black task trained on it.

  ``ROUGH_TERRAINS_CFG`` is the shared velocity preset (flat, both stair
  pyramids, both slope pyramids, random rough, waves). SRMA's Black task flips
  it into curriculum mode and raises the stair steps to 25 cm; this factory
  reproduces that while leaving the shared preset untouched.
  """
  cfg = deepcopy(ROUGH_TERRAINS_CFG)
  cfg.curriculum = True
  stairs = cfg.sub_terrains["pyramid_stairs"]
  inverse_stairs = cfg.sub_terrains["pyramid_stairs_inv"]
  assert isinstance(stairs, BoxPyramidStairsTerrainCfg)
  assert isinstance(inverse_stairs, BoxInvertedPyramidStairsTerrainCfg)
  stairs.step_height_range = (0.0, 0.25)
  inverse_stairs.step_height_range = (0.0, 0.25)
  return cfg


# Step run per column, widest to narrowest. The run is fixed inside a column, so
# width - not the rise - is what separates the four difficulty bands.
SRMAM_STAIRS_STEP_WIDTHS = (0.42, 0.36, 0.30, 0.26)


def make_srmaM_stairs_terrains_cfg(
  step_height_range: tuple[float, float] = (0.15, 0.30),
  step_widths: tuple[float, ...] = SRMAM_STAIRS_STEP_WIDTHS,
) -> TerrainGeneratorCfg:
  """Return a stairs-only curriculum: four runs, each ascending and descending.

  Ten rows interpolate the step height from ``step_height_range[0]`` on row 0 to
  ``step_height_range[1]`` on row 9, so the rise is the only difficulty axis
  inside a column. Across columns the run is held fixed at 42, 36, 30 and 26 cm:
  the same rise gets progressively steeper (30 cm over 42 cm is 36 degrees; 30 cm
  over 26 cm is 49), instead of the run growing with the rise and cancelling part
  of the difficulty increase.
  """
  up: dict[str, SubTerrainCfg] = {}
  down: dict[str, SubTerrainCfg] = {}
  for width in step_widths:
    tag = f"w{round(width * 100):02d}"
    up[f"stairs_up_{tag}"] = pyramid_stairs(
      proportion=1.0,
      step_height_range=step_height_range,
      step_width=width,
    )
    down[f"stairs_down_{tag}"] = pyramid_stairs_inv(
      proportion=1.0,
      step_height_range=step_height_range,
      step_width=width,
    )
  sub_terrains = {**up, **down}
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=len(sub_terrains),
    curriculum=True,
    sub_terrains=sub_terrains,
    add_lights=True,
  )
