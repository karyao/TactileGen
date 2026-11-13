# Tactile output (vertical bar charts)

`src/tactile/` turns a `VerticalBarChart` (see [contracts](contracts.md)) into a
one-page tactile graphic and checks it against BANA rules. This is the last
stage of the pipeline: chart image → YOLO → OCR → chart data → **tactile page**.

```text
VerticalBarChart -> layout (mm geometry + braille) -> BANA rule check -> SVG
```

## Try it

No extra dependencies beyond the standard library are needed.

```bash
python -m src.tactile --chart tests/fixtures/tactile/rainfall.json \
  --output output/tactile/rainfall.svg --show-print
python -m src.tactile --chart tests/fixtures/tactile/crowded.json \
  --output output/tactile/crowded.svg --show-print --show-violations
python -m unittest tests.test_tactile
```

Open the SVG in a browser. `--show-print` adds small grey print text under
each braille label for sighted reviewers; `--show-violations` draws red boxes
around elements that break a rule (hover a box for the rule). `--strict`
exits with status 1 when any rule fails. The three demo charts in
`tests/fixtures/tactile/` are made-up example data, not PlotQA charts.

## What the layout does

Page and braille sizes:

- 11.5 x 11 inch graphics paper; content limited to 40 cells by 25 lines
  (BANA §3.4.2).
- Braille is drawn as dots at the paper-braille sizes on BANA's "Size and
  Spacing of Braille Characters" page (Library of Congress Specification 800):
  1.45 mm dots, 2.34 mm between dots, 6.22 mm between cells, 10.16 mm between
  lines. No braille font is needed.
- Line widths and texture sizes are not from BANA: its line styles and texture
  palettes (Appendices E, F, H, I) are picture samples without measurements.
  They are marked for review in `src/tactile/constants.py`.

Placement, following BANA *Guidelines and Standards for Tactile Graphics, 2022*:

| Element | Placement | BANA |
| --- | --- | --- |
| Title | Centred heading at the top, then a blank line | §5.3.1 |
| Key | Before the graph, in left-to-right bar order, with every legend entry in legend order (an entry whose bars are not visible gets a sample and name but draws no bars); 1 x 1/2 inch texture samples from cell 1; explanation in cell 6 level with the sample top, runovers in cell 8 | §6.6.6.4, §3.4.3.14, §5.8.4.3-4 |
| Textures | At most five area textures; the dot fill uses 0.8 mm dots, smaller than braille dots | §3.2.2, §6.8.3.3 |
| Vertical values | Right-justified, numeric indicator, dots 2-5 level with the grid line, 4 mm from the axis | §6.6.6.2, §6.6.6.6 |
| Grid | Horizontal grid lines (least distinct, dashed) instead of tick marks; at least 3/8 inch apart | §6.6.6.3, §6.6.6.5, §3.4.3.9 |
| Bars | 3/8 to 1 inch wide, sized from the number of categories and series; bars in a set are joined (as in print); 1/8 inch between sets (BANA gives no number for this gap, so the §3.4.3.12 spacing is reused); distinct textures per series | §6.6.6.7, §3.4.3.12 |
| Category labels | Centred under each set of bars. Overflow order when they do not fit: (1) full names; (2) alphabetic keys of 2-3 letters from the name, with a dot 3 or 6 letter, never aa/ac/cc/ca; (3) numeric keys with the numeric indicator and no period; (4) flag the page for review. Keyed names are listed in the key in left-to-right order: symbol in cell 1, a blank cell, the name, runovers in cell 3 | §6.6.6.6, §5.8.1, §5.7.3, §5.8.4.5 |
| Headings | Vertical heading above the values (≤ 20 cells per line), at least 1/8 inch above the top value label (`Y_HEADING_CLEARANCE`; no blank line needed); horizontal heading below the values, left-justified with the first one | §6.6.6.6, §3.4.3.12 |

A legend entry with no visible bars (`BarSeries.has_bars` false) is keyed like any other series
and takes no bar slot. The page gets a transcriber's note before the key, "No bars are visible for:
<names>." (never "zero": the data may be below the axis), on the page that carries the key. Because the key and
note take space, the plot is shorter than for the same chart without them. When that leaves less than
`MIN_PLOT_HEIGHT` (5 × `GRID_MIN_SPACING`, 47.6 mm: room for 5 grid intervals; not from BANA), the title, notes and
series key go on a key page before the graph page, the same key page a divided graph uses, and the
graph page keeps the height. A single graph page with a short plot and no series key is unchanged.

The layout never drops content to make a chart fit. If a chart breaks a
limit (for example 36 bars on one page), the page is still produced and the
rule checker reports every problem.

## Graphs too dense for one page

`layout_vertical_bar_chart_pages` (and the CLI, unless `--one-page` is given)
divides the categories into the fewest consecutive sections whose pages pass
every rule (BANA §3.9). Each section uses the same value scale. With several
series, a key page comes first (§6.6.6.4: the key goes before the graphic page
when it cannot share it) holding the title, a transcriber's note giving the
division and number of parts (§3.9.2), and the series key; the graph pages then
keep their height for the plot. With one series, the first graph page holds
the title and that note instead. The title is not repeated (§5.3.3). Each
graph page has a section title such as "Part 2 of 2: July to December." in
transcriber's note indicators (§3.9.4; notes start in cell 7 with runovers in
cell 5, §5.6.2; indicators from UEB Rule 3.27) and its own category key. Keys
are chosen across the whole graph so they match between pages. The CLI writes
`name-1.svg`, `name-2.svg` and so on; `crowded.json` gives a key page and two
graph pages of six months each. §3.9.1 calls for an overview of the
whole graph first; none is generated, so part 1 reports `overview_first`.

## Rule checker

`src.tactile.check_layout(layout)` returns a list of violations, each with the
rule, BANA section and the elements involved. It reads only the layout, so it
also works on layouts built another way.

| Rule | Checks | BANA |
| --- | --- | --- |
| `bar_width` | Bars 3/8 to 1 inch wide | §6.6.6.7 |
| `bar_spacing` | Bars in a set do not overlap; at least 1/8 inch between sets | §6.6.6.7, §3.4.3.12 |
| `label_clearance` | Labels at least 1/8 inch (3 mm) from every other component | §3.4.3.12, §6.6.6.6 |
| `label_owner_distance` | A label at most 1/4 inch (6 mm) from its component | §3.4.3.12, §5.10.5 |
| `fits_page` | Everything inside 40 cells x 25 lines | §3.4.2 |
| `grid_spacing` | Grid lines at least 3/8 inch apart | §3.4.3.9, §6.6.1.1 |
| `line_hierarchy` | Grid and lead lines weaker than axis lines | §6.6.6.5, §3.4.3.8 |
| `primary_line_length` | Axis lines at least 1/2 inch long | §3.4.3.2 |
| `tick_length` | Tick marks cross the axis 1/4 inch each side | §6.6.6.5 |
| `numeric_indicator` | Numeric values use the numeric indicator | §6.6.6.2 |
| `all_bars_labeled` | No bar label omitted | §6.6.1.3 |
| `key_before_graph`, `key_order` | Key above the graph, in bar order | §6.6.6.4 |
| `distinct_textures` | One texture per series, matching its key sample | §6.6.6.7 |
| `key_sample_size` | Key samples 1 x 1/2 inch | §3.4.3.14 |
| `y_heading_length` | Vertical heading lines at most 20 cells | §6.6.6.6 |
| `texture_count` | At most five area textures | §3.2.2 |
| `key_symbols` | Valid, distinct alphabetic or numeric keys | §5.8.1.1-3, §5.8.1.6 |
| `x_label_overflow` | Reports pages whose labels fit in no overflow mode | §6.6.6.6, §5.8 |
| `overview_first` | A divided graph needs an overview first (always reported on part 1; not generated) | §3.9.1 |
| `heading_placement` | Horizontal heading left-justified with the first value; vertical heading aligned with the first cell of the values (not centred on the plot) | §6.6.6.6 |
| `key_format` | Key explanation in cell 6, level with its sample | §5.8.4.3, §5.8.4.4 |

Passing every rule means these checks passed, not that the page is fully
BANA compliant. Many guidelines (texture choice, whether a graph needs to be
simplified) need a trained transcriber's judgement.

## Braille

`src/tactile/braille.py` is a small uncontracted UEB subset: letters,
capitals (single and word), digits with the numeric indicator, decimal point
and numeric comma, the grade 1 indicator after a digit, minus sign and common
punctuation. Anything else raises `ValueError` rather than guessing.

Two liblouis checks in `tests/test_tactile.py` use the UEB grade 1 table
(`en-ueb-g1.ctb`): a forward check that liblouis produces the same braille
for the demo labels, and a round-trip check that back-translating the braille
of every label (demos plus abbreviation- and number-keyed layouts, about 70
labels) gives the original text. Both skip when the `louis` Python bindings
are not installed. **Neither has been run yet**: liblouis was not available
on the development machine.

## Not done yet

- Full braille translation (contracted UEB, Nemeth) through liblouis.
- Embosser formats (BRF, or a graphics embosser's own format) and PDF. The SVG
  can be printed on swell paper or imported into embosser software.
- An overview page before a divided graph (§3.9.1); divided graphs are flagged
  for review instead.
- A two-column key (§5.8.4.9). BANA sets no minimum plot height, so a page can
  still pass every rule with a short plot.
- The "Key:" heading and transcriber's note indicators around a
  transcriber-made key (§5.8.4.2).
- Texture designs from BANA's texture palettes (Appendices E and H), and a
  test that the dot fill is distinguishable from braille by touch.
- Negative values, horizontal bars, stacked bars, line graphs and histograms.
- Reading `VerticalBarSpec` from `feat/reconstruct-vbar`, and the transcriber's
  notes BANA requires in some cases (for example omitted numeric indicators).
- Review by a certified transcriber, and testing with braille readers.

## Sources

- BANA, *Guidelines and Standards for Tactile Graphics, 2022*:
  <https://www.brailleauthority.org/guidelines-and-standards-tactile-graphics>
- BANA, Size and Spacing of Braille Characters (quoting Library of Congress
  Specification 800): <https://www.brailleauthority.org/node/63>
