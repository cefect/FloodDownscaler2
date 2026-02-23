
"""
Drop-in replacement for `_distance_fill_cost_wbt` using a Numba-accelerated
multi-source cost-distance allocation (no WhiteboxTools; no temp GeoTIFFs).

Expected behaviour:
- Treat cells with valid WSE values as "sources".
- Compute least-cost connectivity on the provided non-negative cost surface.
- Fill missing WSE values by assigning the WSE of the least-cost source.

This mirrors the typical WhiteboxTools pipeline:
  CostDistance(source=wse, cost=cost) -> backlink
  CostAllocation(source=wse, backlink=backlink) -> allocated wse

but does it in-memory and cross-platform.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import xarray as xr
try:
    import rioxarray  # noqa: F401  (registers the .rio accessor)
except Exception:  # pragma: no cover
    rioxarray = None

from cost_distance_numba import (
    apply_allocation_values,
    multi_source_cost_allocation,
    neighbor_offsets_and_lengths,
)


def _as_numpy_2d(da: xr.DataArray) -> np.ndarray:
    """Get a dense in-memory numpy 2D array (no band dim)."""
    arr = np.asarray(da.data)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


def _build_source_mask(wse: np.ndarray, nodata) -> np.ndarray:
    """
    Determine which cells are sources (valid).

    WhiteboxTools uses "positive, non-zero" convention in some tools;
    in your existing code you add an increment for negative WSE so those
    cells still count as sources. Here we treat any finite, non-nodata
    value as a source.
    """
    if nodata is None:
        return np.isfinite(wse)
    try:
        nodata_f = float(nodata)
    except Exception:
        return np.isfinite(wse)

    if math.isnan(nodata_f):
        return np.isfinite(wse)

    return np.isfinite(wse) & (wse != nodata_f)


def _cost_nodata_info(cost_xr: xr.DataArray):
    """Return (has_cost_nodata, cost_nodata_float)."""
    cn = None
    try:
        cn = cost_xr.rio.nodata
    except Exception:
        cn = None

    if cn is None:
        return False, np.nan

    try:
        cn_f = float(cn)
    except Exception:
        return False, np.nan

    if math.isnan(cn_f):
        return False, np.nan

    return True, cn_f


def _distance_fill_cost_numba(
    wse_xr: xr.DataArray,
    cost_xr: xr.DataArray,
    log=None,
    out_dir: Optional[str] = None,
    use_diagonal: bool = True,
) -> xr.DataArray:
    """
    In-memory/Numba version of `_distance_fill_cost_wbt`.

    Parameters
    ----------
    wse_xr:
        2D DataArray with spatial coords and .rio metadata. Missing values indicated by nodata or NaN.
    cost_xr:
        2D DataArray with non-negative traversal costs. NoData/NaN treated as barriers.
    log:
        Optional logger with .debug/.warning methods.
    out_dir:
        Ignored (kept for signature compatibility). You can add debug dumps here if needed.
    use_diagonal:
        If True, use 8-neighbour movement; if False, use 4-neighbour.

    Returns
    -------
    xr.DataArray:
        Filled WSE with the same metadata as input.
    """
    # --------- basic validation ----------
    if wse_xr.shape != cost_xr.shape:
        raise ValueError(f"Shape mismatch: wse {wse_xr.shape} vs cost {cost_xr.shape}")

    # Extract nodata for output consistency
    try:
        nodata = wse_xr.rio.nodata
    except Exception:
        nodata = None

    wse = _as_numpy_2d(wse_xr).astype(np.float64, copy=False)
    cost = _as_numpy_2d(cost_xr).astype(np.float64, copy=False)

    # WhiteboxTools requires non-negative costs; keep the same assumption.
    cmin = np.nanmin(cost)
    if cmin < 0.0:
        raise ValueError(f"Expected non-negative costs, but min(cost)={cmin}")

    # --------- build masks and neighbor geometry ----------
    source_mask = _build_source_mask(wse, nodata)

    has_cost_nodata, cost_nodata = _cost_nodata_info(cost_xr)

    # Pixel sizes (x,y). rioxarray returns (xres, yres) where yres is typically negative.
    try:
        res_x, res_y = wse_xr.rio.resolution()
    except Exception:
        # Fall back to unit grid if not georeferenced
        res_x, res_y = 1.0, 1.0

    dr, dc, dl = neighbor_offsets_and_lengths(res_x, res_y, use_diagonal=use_diagonal)

    if log is not None:
        log.debug(
            f"numba cost-allocation: shape={wse.shape}, use_diagonal={use_diagonal}, "
            f"res=({res_x},{res_y}), sources={int(source_mask.sum())}"
        )

    # --------- run multi-source Dijkstra ----------
    src_idx, _dist = multi_source_cost_allocation(
        cost=cost,
        source_mask=source_mask,
        dr=dr,
        dc=dc,
        dl=dl,
        has_cost_nodata=has_cost_nodata,
        cost_nodata=cost_nodata,
    )

    # If nodata is None, choose NaN for float outputs.
    if nodata is None:
        nodata_value = np.nan
    else:
        nodata_value = float(nodata) if not (isinstance(nodata, float) and math.isnan(nodata)) else np.nan

    filled_flat = apply_allocation_values(wse.ravel(), src_idx, nodata_value)
    filled = filled_flat.reshape(wse.shape)

    # --------- back to xarray with metadata ----------
    out = wse_xr.copy(deep=True)
    out.data = filled.astype(wse_xr.dtype, copy=False) if np.issubdtype(wse_xr.dtype, np.floating) else filled

    # preserve attrs
    out.attrs = wse_xr.attrs.copy()

    # preserve CRS and nodata
    try:
        out = out.rio.write_crs(wse_xr.rio.crs, inplace=False)
    except Exception:
        pass

    if nodata is not None:
        try:
            out = out.rio.write_nodata(nodata, inplace=False)
        except Exception:
            pass

    if log is not None:
        log.debug("finished (numba)")

    return out
