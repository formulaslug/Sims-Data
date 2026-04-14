## Battery thermal schematics

This diagram is a **lumped thermal network** view of a battery pack: nodes represent temperatures (cells, case, ambient), and arrows show **where conductive or convective heat transfer is modeled** between those nodes.

![Battery pack thermal schematic](Battery_schematics.png)

### What the blocks represent

- **Battery_case** — The enclosure around the cell stack. In a simulation it is usually a single thermal mass (or a small set of masses) whose temperature couples perimeter cells and exchanges heat with the outside.
- **AIR** — The external fluid region on the **left and right** sides of the case. It is the sink (or source) for heat that leaves the pack through the case sides.

### Cell layout and naming

- Cells are arranged in a **3 columns × 10 rows** grid inside the case (30 cells total).
- The pack is split into **five segments**, **SEG0** through **SEG4**. Each segment has **six cells** in a **2×3** arrangement (two rows of three).
- Within a segment, cells are labeled **CELL0 … CELL5** following the pattern in the figure: top row **0, 1, 2** and bottom row **5, 4, 3** (so the horizontal order of indices differs between rows).

Full cell IDs follow: **`SEG<segment>_CELL<index>`** (for example `SEG0_CELL0`, `SEG1_CELL3`).

### How to read the arrows (heat transfer paths)

Arrows indicate **modeled heat flow directions** in the thermal circuit—not necessarily instantaneous physical flow at every instant (heat can still move “backward” through a resistance if the temperature gradient reverses). In practice:

1. **Cell ↔ cell (double-headed arrows)**  
   - **Horizontal:** Between neighbors in the same row (e.g. `SEG0_CELL0` ↔ `SEG0_CELL1` ↔ `SEG0_CELL2`, and similarly on the bottom row).  
   - **Vertical:** Between the two rows within a segment, and **between the bottom of one segment and the top of the next** (e.g. down the stack from SEG0 through SEG4).  
   These paths represent **in-pack conduction** (and any equivalent series path through gaps, frames, or busbars, depending on how the model is parameterized).

2. **Cell → Battery_case (single-headed arrows from cells outward)**  
   **Perimeter cells** connect to the case on the **outer boundary** of the grid:  
   - **Top** row of SEG0 → case above  
   - **Bottom** row of SEG4 → case below  
   - **Left** column (the cells on the left edge of the 3-wide grid) → case on the left  
   - **Right** column (the cells on the right edge) → case on the right  

   Interior cells only “see” the case indirectly through other cells unless the model adds extra paths. These links are the **cell-to-housing** thermal resistances.

3. **Battery_case → AIR (single-headed arrows from case to the side AIR blocks)**  
   Large arrows from the **vertical sides** of the **Battery_case** to **AIR** represent **heat rejection** from the pack shell to the ambient at the sides—typically **convection** (and any explicit **radiation** if folded into the same coefficient). This is where pack heat leaves to the environment in this topology.

### Summary

| Path | Typical meaning |
|------|------------------|
| Cell ↔ cell | Conduction / in-pack coupling between neighboring cells |
| Cell → case | Conduction from cell stack to inner surface of the housing |
| Case → air | Convection (± radiation) from outer case to ambient |

Together, these paths define how heat generated in each cell propagates **laterally and vertically through the stack**, then **out through the case** to **air** on the sides—matching the structure shown in the schematic.
