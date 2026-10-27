"""State layout for the Go1 velocity SuperMimic world model.

The world-model state is a flat vector with a fixed layout:

======  ==================================================================
Slice   Meaning
======  ==================================================================
0:3     base position (world)
3:9     base orientation, 6-D rotation representation
9:12    base linear velocity (world)
12:15   base angular velocity (world)
15:27   joint positions (12)
27:39   joint velocities (12)
39:43   foot contact flags (4)
43:47   foot air time (4)
47:51   foot height above terrain (4)
======  ==================================================================

The first 39 entries are predicted as dynamics (position/orientation/velocity/
joint increments); the last 12 are auxiliary contact quantities predicted by
separate heads so that the reward head can see contact state.
"""

from __future__ import annotations

STATE_DIM = 51
DYN_DIM = 39
CONTACT_DIM = 4
AIR_DIM = 4
FH_DIM = 4

POS = slice(0, 3)
ROT6 = slice(3, 9)
LIN = slice(9, 12)
ANG = slice(12, 15)
Q = slice(15, 27)
QD = slice(27, 39)
CONTACT = slice(39, 43)
AIR = slice(43, 47)
FH = slice(47, 51)
