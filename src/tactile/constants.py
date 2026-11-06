"""Every measurement the tactile layout and rule checker use, with its source.

Lengths are millimetres. Unless marked otherwise, each value comes from BANA
*Guidelines and Standards for Tactile Graphics, 2022* (section in the
comment), converted from the inches BANA states first. Values that are not
from that document are grouped at the end under "Not from BANA" so they can be
reviewed; do not add a spacing value without a source.
"""

from __future__ import annotations

MM_PER_INCH = 25.4

# --- Page (BANA §3.4.2) ------------------------------------------------------------
PAGE_WIDTH = 11.5 * MM_PER_INCH   # 11.5 x 11 inch paper, §3.4.2
PAGE_HEIGHT = 11.0 * MM_PER_INCH
CELLS_PER_LINE = 40               # §3.4.2.1
LINES_PER_PAGE = 25               # §3.4.2.2

# --- Braille cell (paper braille) --------------------------------------------------
# BANA "Size and Spacing of Braille Characters" (brailleauthority.org/node/63),
# which quotes Library of Congress Specification 800 §3.2 nominal values; the
# 2022 guidelines refer to it from §3.4.1.
DOT_DIAMETER = 0.057 * MM_PER_INCH      # base diameter
DOT_PITCH = 0.092 * MM_PER_INCH         # between dots in the same cell
CELL_PITCH = 0.245 * MM_PER_INCH        # between corresponding dots in adjacent cells
LINE_PITCH = 0.400 * MM_PER_INCH        # between corresponding dots in adjacent lines
LABEL_HEIGHT = 2 * DOT_PITCH + DOT_DIAMETER   # derived: three rows of dots

# --- Spacing between components ----------------------------------------------------
LABEL_MIN_GAP = 1 / 8 * MM_PER_INCH   # labels to any component, minimum, §3.4.3.12, §6.6.6.6
LABEL_MAX_GAP = 1 / 4 * MM_PER_INCH   # label to its own component, maximum, §3.4.3.12, §5.10.5
# Labels are placed at the minimum distance; anything between the two is allowed.
LABEL_GAP = LABEL_MIN_GAP
# Clearance between the vertical heading and the top value label: the 1/8
# inch label-to-component minimum (§3.4.3.12). §6.6.6.6 needs no blank line
# after the heading.
Y_HEADING_CLEARANCE = LABEL_MIN_GAP
# Gaps between bars in a set and between sets (and before the first set),
# measured between the bars' raised outer edges.
# Bars in a set are joined: §6.6.6.7 says bars are "separated or joined in
# braille as they are in print", and grouped bars are joined in print charts
# such as PlotQA's. This also keeps sets distinguishable from each other.
BAR_GAP = 0.0
# Between sets §6.6.6.7 gives no number ("separated enough to be able to read
# the background grid"); by project decision the §3.4.3.12 1/8 inch
# component spacing is reused. Must stay larger than BAR_GAP.
GROUP_GAP = LABEL_MIN_GAP

# --- Bars, grid and lines ----------------------------------------------------------
BAR_MIN_WIDTH = 3 / 8 * MM_PER_INCH     # §6.6.6.7 ("3/8 inch (1 centimeter)")
BAR_MAX_WIDTH = 1.0 * MM_PER_INCH       # §6.6.6.7
GRID_MIN_SPACING = 3 / 8 * MM_PER_INCH  # §3.4.3.9, §6.6.1.1
PRIMARY_LINE_MIN_LENGTH = 1 / 2 * MM_PER_INCH   # §3.4.3.2
TICK_HALF_LENGTH = 1 / 4 * MM_PER_INCH          # each side of the axis, §6.6.6.5

# --- Keys --------------------------------------------------------------------------
KEY_SAMPLE_WIDTH = 1.0 * MM_PER_INCH    # §3.4.3.14, §5.9.1.1
KEY_SAMPLE_HEIGHT = 1 / 2 * MM_PER_INCH  # §3.4.3.14, §5.9.1.2
KEY_SAMPLE_LINES = 2                    # a sample needs two braille lines, §5.9.1.2
KEY_TEXT_CELL = 6                       # explanation after a texture, §5.8.4.3
KEY_RUNOVER_CELL = 8                    # its runovers, §5.8.4.3
LETTER_KEY_RUNOVER_CELL = 3             # runovers after a letter/number key, §5.8.4.5
LETTER_KEY_MIN_CELLS = 2                # §5.8.1.1 (2-cell keys preferred)
LETTER_KEY_MAX_CELLS = 3                # §5.8.1.1
MAX_AREA_TEXTURES = 5                   # §3.2.2

# --- Labels --------------------------------------------------------------------------
Y_HEADING_MAX_CELLS = 20                # §6.6.6.6
TN_FIRST_CELL = 7                       # transcriber's note starts in cell 7, §5.6.2
TN_RUNOVER_CELL = 5                     # and runs over in cell 5, §5.6.2

# --- Not from BANA (review these) ----------------------------------------------------
# BANA gives no numbers for these. Its line styles (Appendices F and I) and
# texture palettes (Appendices E and H) are picture samples per production
# method, with the instruction that textures and lines in one graphic be
# "tactually distinct" and that embossed results be proofread on hard copy.
# Line widths and strengths. BANA ranks the lines (grid weakest, axes
# stronger, §6.6.6.5) but gives no widths.
GRID_STRENGTH, LEAD_STRENGTH, BAR_STRENGTH, AXIS_STRENGTH = 1, 1, 2, 3
GRID_WIDTH, BAR_OUTLINE_WIDTH, AXIS_WIDTH = 0.5, 1.0, 1.5
# Texture fills: tile size and mark size in mm. The dot fill uses dots
# smaller than braille dots (§6.8.3.3 describes "fine dots that are smaller
# than braille dots"); the exact sizes are not from BANA.
TEXTURE_TILE = 4.0
TEXTURE_LINE_WIDTH = 0.6
DOT_TEXTURE_TILE = 3.0
DOT_TEXTURE_DIAMETER = 0.8
# Upper limit on grid intervals, to keep the grid sparse. A count, not a
# spacing; BANA only says some grid lines may be omitted (§6.6.1.1).
MAX_GRID_INTERVALS = 8

TEXTURES = ("diagonal", "dots", "horizontal", "crosshatch", "vertical")
