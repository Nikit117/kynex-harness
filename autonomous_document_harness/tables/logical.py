"""Logical tables: header hierarchies, fragments, continuation reconstruction.

Native grids, DI cells and vision readings all arrive flattened. This module
recovers structure WITHOUT domain knowledge: header paths from multi-row
headers and spans, banner/group labels, and multi-page continuations.

Never merges fragments without evidence. Never invents values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PhysicalTableFragment:
    doc_id: str
    page: Optional[int]
    sheet: Optional[str]
    index: int
    header_rows: List[List[str]]
    body_rows: List[List[str]]
    bbox: Optional[list] = None
    method: str = ""
    title: str = ""


@dataclass
class LogicalTable:
    doc_id: str
    header_paths: List[List[str]]
    rows: List[Dict[str, Any]]  # {"values": [...], "fragment": i, "row": r}
    fragments: List[PhysicalTableFragment] = field(default_factory=list)
    title: str = ""
    continuation_confidence: float = 0.0


def _norm(cell: str) -> str:
    return " ".join(str(cell or "").split()).strip()


def resolve_header_paths(header_rows: List[List[str]]) -> List[List[str]]:
    """Multi-row/merged headers -> per-column paths, inheriting blanks.

    Blank cells inherit from the LEFT (grouped columns); deeper rows refine.
    Example: [["System", "DCS", "DCS"], ["Area", "AI", "AO"]]
             -> [["System"], ["DCS", "AI"], ["DCS", "AO"]]
    """
    rows = [[_norm(c) for c in r] for r in header_rows if any(_norm(c) for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    grid = [r + [""] * (width - len(r)) for r in rows]
    for r in grid:  # left-inherit within a row (merged/grouped headers)
        for j in range(1, width):
            if not r[j]:
                r[j] = r[j - 1]
    paths: List[List[str]] = []
    for j in range(width):
        path: List[str] = []
        for i in range(len(grid)):
            label = grid[i][j]
            if label and (not path or path[-1].lower() != label.lower()):
                path.append(label)
        paths.append(path or [f"col_{j + 1}"])
    return paths


def apply_banner_labels(
    header_rows: List[List[str]], banner: List[str]
) -> List[List[str]]:
    """Banner/group row above headers becomes each column's path root.

    Example: banner ["", "DCS", "", "SIS", ""] over headers
    ["Area", "AI", "AO", "AI", "AO"] ->
    [["Area"], ["DCS", "AI"], ["DCS", "AO"], ["SIS", "AI"], ["SIS", "AO"]]
    """
    banner = [_norm(c) for c in banner]
    if not any(banner):
        return resolve_header_paths(header_rows)
    width = max(len(banner), max((len(r) for r in header_rows), default=0))
    banner = (banner + [""] * width)[:width]
    for j in range(1, width):  # merged banner cells span groups
        if not banner[j]:
            banner[j] = banner[j - 1]
    paths = resolve_header_paths(header_rows)
    while len(paths) < width:
        paths.append([f"col_{len(paths) + 1}"])
    out = []
    for j in range(width):
        root = banner[j]
        path = list(paths[j])
        if root and (not path or path[0].lower() != root.lower()):
            path = [root] + path
        out.append(path)
    return out


def split_header_body(
    rows: List[List[str]], header_depth: int = 1
) -> Tuple[List[List[str]], List[List[str]]]:
    """Split grid into header rows + body rows (depth heuristic, capped at 3)."""
    depth = max(1, min(header_depth, 3, len(rows) - 1 if len(rows) > 1 else 1))
    return rows[:depth], rows[depth:]


def sheet_grid(sheet, max_rows: int = 31) -> List[List[str]]:
    """Column-aligned string grid for a parsed sheet (empties preserved).

    Never drop empty cells: positional alignment is what makes banner/span
    reasoning valid.
    """
    grid: List[List[str]] = []
    for r in range(1, min(sheet.max_row + 1, max_rows + 1)):
        grid.append(
            [
                ""
                if sheet.values.get((r, c)) is None
                else str(sheet.values.get((r, c)))
                for c in range(1, sheet.max_col + 1)
            ]
        )
    return [row for row in grid if any(row)]


def detect_header_layout(grid: List[List[str]]) -> Tuple[Optional[List[str]], int]:
    """Locate the header row in a raw grid; return (banner_or_None, header_idx).

    Title rows and blank rows above the header are skipped. A banner/group row
    directly above the header becomes the hierarchy root.
    """
    if not grid:
        return None, 0

    def _codes(row: List[str]) -> int:
        import re

        return sum(
            1 for c in row if re.fullmatch(r"[A-Za-z%]{1,8}", str(c or "").strip())
        )

    def _labels(row: List[str]) -> set:
        import re

        return {
            str(c).strip()
            for c in row
            if str(c).strip()
            and not re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?%?", str(c).strip())
        }

    for h in range(min(4, len(grid))):
        if _codes(grid[h]) >= 3:
            if h > 0 and 1 <= len(_labels(grid[h - 1])) <= 3:
                # Banner rows carry MULTIPLE group labels; a lone title cell
                # (single non-empty cell) is a title, never a banner root.
                nonempty = sum(1 for c in grid[h - 1] if str(c).strip())
                if nonempty >= 2:
                    return [str(c) for c in grid[h - 1]], h
            return None, h
    return None, 0


def _header_signature(paths: List[List[str]]) -> Tuple[str, ...]:
    return tuple(">".join(p).lower() for p in paths)


def group_continuations(fragments: List[PhysicalTableFragment]) -> List[List[int]]:
    """Group fragment indexes into logical tables (conservative).

    Signals: same sheet+adjacent implicit order, equal width, identical header
    signatures, continuation labels (continued/cont'd), shared titles.
    """
    groups: List[List[int]] = []
    for i, frag in enumerate(fragments):
        placed = False
        sig = _header_signature(resolve_header_paths(frag.header_rows))
        for group in groups:
            first = fragments[group[0]]
            prev = fragments[group[-1]]
            if frag.sheet != first.sheet:
                continue
            same_page_order = (
                frag.page is not None
                and prev.page is not None
                and frag.page == prev.page + 1
            )
            adjacent_sheet = frag.sheet is not None and frag.sheet == prev.sheet
            if not (same_page_order or adjacent_sheet):
                continue
            width_ok = len(resolve_header_paths(frag.header_rows)) == len(
                resolve_header_paths(first.header_rows)
            )
            sig_ok = (
                not sig
                or not _header_signature(resolve_header_paths(first.header_rows))
                or sig == _header_signature(resolve_header_paths(first.header_rows))
            )
            continued = any(
                "continu" in _norm(c).lower()
                for row in frag.header_rows + frag.body_rows[:1]
                for c in row
            )
            titled = bool(frag.title) and frag.title == first.title
            if width_ok and (sig_ok or continued or titled):
                group.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def reconstruct(fragments: List[PhysicalTableFragment]) -> List[LogicalTable]:
    """Fragments -> logical tables. Singletons pass through untouched."""
    tables = []
    for group in group_continuations(fragments):
        frags = [fragments[i] for i in group]
        paths = resolve_header_paths(frags[0].header_rows)
        rows = []
        for position, frag in enumerate(frags):
            for r, row in enumerate(frag.body_rows):
                rows.append({"values": list(row), "fragment": position, "row": r})
        confidence = 1.0 if len(frags) == 1 else 0.75
        tables.append(
            LogicalTable(
                doc_id=frags[0].doc_id,
                header_paths=paths,
                rows=rows,
                fragments=frags,
                title=frags[0].title,
                continuation_confidence=confidence,
            )
        )
    return tables


def fragments_from_grid(
    doc_id: str,
    page: Optional[int],
    sheet: Optional[str],
    index: int,
    rows: List[List[str]],
    header_depth: int = 1,
    bbox: Optional[list] = None,
    method: str = "",
    title: str = "",
    banner: Optional[List[str]] = None,
) -> PhysicalTableFragment:
    headers, body = split_header_body(
        [[_norm(c) for c in r] for r in rows], header_depth
    )
    if banner and any(_norm(c) for c in banner):
        # Banner/group labels become the top header row: resolve_header_paths
        # left-inherits them across groups and stacks refinement below.
        banner_row = [_norm(c) for c in banner]
        if any(banner_row) and (not headers or banner_row != headers[0]):
            headers = [banner_row] + headers
    return PhysicalTableFragment(
        doc_id=doc_id,
        page=page,
        sheet=sheet,
        index=index,
        header_rows=headers,
        body_rows=body,
        bbox=bbox,
        method=method,
        title=title,
    )
