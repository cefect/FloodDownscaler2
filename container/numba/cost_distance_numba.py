
"""
Numba-accelerated multi-source cost-distance + allocation for 2D rasters.

Goal:
- Given a 2D cost surface and a boolean mask of "source" cells, compute for every cell
  the index of its least-cost source (multi-source Dijkstra), using 8-neighbour moves.
- This can be used as a fast replacement for WhiteboxTools cost_distance + cost_allocation
  in the "fill nodata by least-cost nearest neighbour" workflow.

Notes:
- Single-core (no prange/parallel). Focus is predictable speed and portability.
- Uses a custom binary heap with decrease-key so heap size is bounded by n_cells.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

try:
    import numba as nb
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This module requires numba. Install e.g. `pip install numba`."
    ) from e


@nb.njit(inline="always", cache=True)
def _heap_sift_up(heap_d, heap_i, heap_pos, pos: int) -> int:
    """Sift-up for a binary min-heap (decrease-key). Returns final position."""
    i = heap_i[pos]
    d = heap_d[pos]
    while pos > 0:
        parent = (pos - 1) // 2
        if heap_d[parent] <= d:
            break
        # move parent down
        heap_d[pos] = heap_d[parent]
        heap_i[pos] = heap_i[parent]
        heap_pos[heap_i[pos]] = pos
        pos = parent
    heap_d[pos] = d
    heap_i[pos] = i
    heap_pos[i] = pos
    return pos


@nb.njit(inline="always", cache=True)
def _heap_sift_down(heap_d, heap_i, heap_pos, pos: int, heap_size: int) -> int:
    """Sift-down for a binary min-heap. Returns final position."""
    i = heap_i[pos]
    d = heap_d[pos]
    while True:
        left = 2 * pos + 1
        if left >= heap_size:
            break
        right = left + 1
        child = left
        if right < heap_size and heap_d[right] < heap_d[left]:
            child = right
        if heap_d[child] >= d:
            break
        heap_d[pos] = heap_d[child]
        heap_i[pos] = heap_i[child]
        heap_pos[heap_i[pos]] = pos
        pos = child
    heap_d[pos] = d
    heap_i[pos] = i
    heap_pos[i] = pos
    return pos


@nb.njit(inline="always", cache=True)
def _heap_push(heap_d, heap_i, heap_pos, heap_size: int, idx: int, dist: float) -> int:
    """Insert idx with dist. Assumes idx not currently in heap."""
    pos = heap_size
    heap_size += 1
    heap_d[pos] = dist
    heap_i[pos] = idx
    heap_pos[idx] = pos
    _heap_sift_up(heap_d, heap_i, heap_pos, pos)
    return heap_size


@nb.njit(inline="always", cache=True)
def _heap_decrease_key(heap_d, heap_i, heap_pos, idx: int, new_dist: float):
    """Decrease the key for an existing idx in heap."""
    pos = heap_pos[idx]
    heap_d[pos] = new_dist
    _heap_sift_up(heap_d, heap_i, heap_pos, pos)


@nb.njit(inline="always", cache=True)
def _heap_pop(heap_d, heap_i, heap_pos, heap_size: int) -> Tuple[int, float, int]:
    """Pop min item. Returns (idx, dist, new_heap_size)."""
    idx = heap_i[0]
    dist = heap_d[0]
    heap_size -= 1
    heap_pos[idx] = -2  # mark as finalized/visited

    if heap_size > 0:
        last_idx = heap_i[heap_size]
        last_dist = heap_d[heap_size]
        heap_i[0] = last_idx
        heap_d[0] = last_dist
        heap_pos[last_idx] = 0
        _heap_sift_down(heap_d, heap_i, heap_pos, 0, heap_size)

    return idx, dist, heap_size


@nb.njit(cache=True)
def multi_source_cost_allocation(
    cost: np.ndarray,
    source_mask: np.ndarray,
    dr: np.ndarray,
    dc: np.ndarray,
    dl: np.ndarray,
    has_cost_nodata: bool,
    cost_nodata: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-source Dijkstra on a 2D grid.

    Parameters
    ----------
    cost:
        2D float array of per-unit-distance traversal costs (>=0). Non-finite values are treated as barriers.
    source_mask:
        2D bool array; True marks source cells.
    dr, dc:
        1D int arrays (len K) of neighbour row/col offsets.
    dl:
        1D float array (len K) of move distances for each neighbour.
    has_cost_nodata, cost_nodata:
        If has_cost_nodata True, any cell with cost==cost_nodata is treated as barrier.

    Returns
    -------
    src_idx:
        1D int64 array (flattened) containing the flattened index of the source cell
        assigned to each cell; -1 for unreachable / barrier cells.
    dist:
        1D float64 array of accumulated cost distance from each cell to its assigned source.
        np.inf for unreachable / barrier cells.
    """
    nrows, ncols = cost.shape
    n = nrows * ncols

    dist = np.empty(n, dtype=np.float64)
    src = np.empty(n, dtype=np.int64)

    # heap storage (bounded by n because we use decrease-key)
    heap_d = np.empty(n, dtype=np.float64)
    heap_i = np.empty(n, dtype=np.int64)
    heap_pos = np.empty(n, dtype=np.int64)  # -1 not in heap, -2 finalized

    # init
    for i in range(n):
        dist[i] = math.inf
        src[i] = -1
        heap_pos[i] = -1

    # initialize sources
    heap_size = 0
    for r in range(nrows):
        for c in range(ncols):
            if source_mask[r, c]:
                idx = r * ncols + c
                # Skip sources that are barriers in cost raster.
                cv = cost[r, c]
                if not math.isfinite(cv):
                    continue
                if has_cost_nodata and cv == cost_nodata:
                    continue
                if cv < 0.0:
                    continue

                dist[idx] = 0.0
                src[idx] = idx
                heap_size = _heap_push(heap_d, heap_i, heap_pos, heap_size, idx, 0.0)

    # early out: no sources
    if heap_size == 0:
        return src, dist

    kmax = dr.shape[0]

    while heap_size > 0:
        idx, dcur, heap_size = _heap_pop(heap_d, heap_i, heap_pos, heap_size)

        r = idx // ncols
        c = idx - r * ncols  # faster than % for numba in some cases

        # if current is a barrier (shouldn't happen), skip expansion
        cv = cost[r, c]
        if not math.isfinite(cv):
            continue
        if has_cost_nodata and cv == cost_nodata:
            continue
        if cv < 0.0:
            continue

        for k in range(kmax):
            rr = r + dr[k]
            cc = c + dc[k]
            if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
                continue

            nidx = rr * ncols + cc

            # finalized nodes cannot be improved under Dijkstra with non-negative edges
            if heap_pos[nidx] == -2:
                continue

            ncv = cost[rr, cc]
            if not math.isfinite(ncv):
                continue
            if has_cost_nodata and ncv == cost_nodata:
                continue
            if ncv < 0.0:
                continue

            step = 0.5 * (cv + ncv) * dl[k]
            nd = dcur + step

            if nd < dist[nidx]:
                dist[nidx] = nd
                src[nidx] = src[idx]

                if heap_pos[nidx] == -1:
                    heap_size = _heap_push(heap_d, heap_i, heap_pos, heap_size, nidx, nd)
                else:
                    _heap_decrease_key(heap_d, heap_i, heap_pos, nidx, nd)

    return src, dist


@nb.njit(cache=True)
def apply_allocation_values(
    values: np.ndarray,
    src_idx: np.ndarray,
    nodata_value: float,
) -> np.ndarray:
    """
    Create an allocated raster by assigning each cell the value of its source.

    Parameters
    ----------
    values:
        1D float array of source values (flattened original raster).
    src_idx:
        1D int array of source indices (-1 for unreachable).
    nodata_value:
        Value to use for unreachable cells.

    Returns
    -------
    out:
        1D float array.
    """
    n = src_idx.shape[0]
    out = np.empty(n, dtype=values.dtype)
    for i in range(n):
        s = src_idx[i]
        if s >= 0:
            out[i] = values[s]
        else:
            out[i] = nodata_value
    return out


def neighbor_offsets_and_lengths(
    res_x: float,
    res_y: float,
    use_diagonal: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (dr, dc, dl) arrays for 4- or 8-neighbour movement.

    res_x/res_y: pixel size in x/y units (sign ignored).
    """
    dx = abs(float(res_x))
    dy = abs(float(res_y))
    diag = math.hypot(dx, dy)

    if use_diagonal:
        # Clockwise starting at E (matches many backlink conventions, but not required here).
        dr = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.int8)
        dc = np.array([1, 1, 0, -1, -1, -1, 0, 1], dtype=np.int8)
        dl = np.array([dx, diag, dy, diag, dx, diag, dy, diag], dtype=np.float64)
    else:
        dr = np.array([0, 1, 0, -1], dtype=np.int8)
        dc = np.array([1, 0, -1, 0], dtype=np.int8)
        # choose dx for E/W and dy for N/S
        dl = np.array([dx, dy, dx, dy], dtype=np.float64)

    return dr, dc, dl
