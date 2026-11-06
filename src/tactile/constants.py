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

# --- Spacing between components ----------------------------------------------------
LABEL_MIN_GAP = 1 / 8 * MM_PER_INCH   # labels to any component, minimum, §3.4.3.12, §6.6.6.6
LABEL_MAX_GAP = 1 / 4 * MM_PER_INCH   # label to its own component, maximum, §3.4.3.12, §5.10.5
# Labels are placed at the minimum distance; anything between the two is allowed.
LABEL_GAP = LABEL_MIN_GAP

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
MAX_AREA_TEXTURES = 5                   # §3.2.2

# --- Labels --------------------------------------------------------------------------
Y_HEADING_MAX_CELLS = 20                # §6.6.6.6

# --- Not from BANA (review these) ----------------------------------------------------
# Braille dot and cell sizes: Library of Congress Specification 800, nominal
# paper braille. BANA refers to a separate size-and-spacing document for these
# (§3.4.1), which was not available when this was written.
DOT_DIAMETER = 0.057 * MM_PER_INCH
DOT_PITCH = 0.092 * MM_PER_INCH         # between dots in a cell
CELL_PITCH = 0.245 * MM_PER_INCH        # between cells
LINE_PITCH = 0.400 * MM_PER_INCH        # between braille lines
LABEL_HEIGHT = 2 * DOT_PITCH + DOT_DIAMETER
# Line widths and strengths. BANA ranks the lines (grid weakest, axes
# stronger, §6.6.6.5) but gives no widths; production-specific line styles are
# in its appendices F and I, which this code does not follow yet.
GRID_STRENGTH, LEAD_STRENGTH, BAR_STRENGTH, AXIS_STRENGTH = 1, 1, 2, 3
GRID_WIDTH, BAR_OUTLINE_WIDTH, AXIS_WIDTH = 0.5, 1.0, 1.5
# Upper limit on grid intervals, to keep the grid sparse. A count, not a
# spacing; BANA only says some grid lines may be omitted (§6.6.1.1).
MAX_GRID_INTERVALS = 8
# Space between sets of bars. Invented; replaced by a sourced value in the
# bar-spacing fix.
GROUP_GAP = 8.0

TEXTURES = ("diagonal", "dots", "horizontal", "crosshatch", "vertical")
