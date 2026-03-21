"""Shared validation helpers for CostGrow notebook workflows."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rasterio.warp import Resampling, reproject
from skimage.metrics import structural_similarity

METRIC_ORDER = [
    "psnr",
    "ssim",
    "rmse_m",
    "rmse_wet_m",
    "csi",
    "mae_m",
    "bias_m",
    "obs_wet_fraction",
    "pred_wet_fraction",
    "valid_pixel_count",
]


def _shared_valid_mask(obs_arr, pred_arr, domain_mask):
    """Return pixels finite in both arrays and inside the domain mask."""
    assert obs_arr.shape == pred_arr.shape == domain_mask.shape, "shape mismatch"
    return domain_mask & np.isfinite(obs_arr) & np.isfinite(pred_arr)


def wse_to_depth_m(wse_arr, dem_arr, domain_mask):
    """Convert WSE to depth meters with dry cells treated as zero depth in-domain."""
    assert wse_arr.shape == dem_arr.shape == domain_mask.shape, "shape mismatch"
    depth = np.full_like(dem_arr, np.nan, dtype="float64")
    if not np.any(domain_mask):
        return depth

    wse_filled = np.where(np.isfinite(wse_arr), wse_arr, dem_arr)
    depth_local = np.maximum(wse_filled - dem_arr, 0.0)
    depth[domain_mask] = depth_local[domain_mask]
    return depth


def reproject_to_profile(src_arr, src_profile, dst_profile, resampling):
    """Reproject source array onto a destination profile grid."""
    assert src_arr.ndim == 2, "src_arr must be 2D"
    assert "height" in dst_profile and "width" in dst_profile, "dst_profile missing shape keys"

    src_nodata = -9999.0
    dst_nodata = -9999.0
    src_clean = np.where(np.isfinite(src_arr), src_arr, src_nodata).astype("float64")
    dst = np.full((dst_profile["height"], dst_profile["width"]), dst_nodata, dtype="float64")

    reproject(
        source=src_clean,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=src_nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=dst_nodata,
        resampling=resampling,
    )

    dst[dst == dst_nodata] = np.nan
    return dst


def build_bilinear_depth_baseline(lores_wse_arr, lores_wse_profile, dem_profile, dem_arr, domain_mask):
    """Build a bilinear-upsampled WSE baseline and its depth raster on DEM grid."""
    assert lores_wse_arr.ndim == 2 and dem_arr.ndim == 2, "input rasters must be 2D"
    assert dem_arr.shape == domain_mask.shape, "domain_mask shape mismatch"

    bilinear_wse = reproject_to_profile(
        src_arr=lores_wse_arr,
        src_profile=lores_wse_profile,
        dst_profile=dem_profile,
        resampling=Resampling.bilinear,
    )
    bilinear_wse[~domain_mask] = np.nan

    bilinear_depth = wse_to_depth_m(
        wse_arr=bilinear_wse,
        dem_arr=dem_arr,
        domain_mask=domain_mask,
    )
    return bilinear_wse, bilinear_depth


def metric_psnr(obs_arr, pred_arr, domain_mask):
    """Compute PSNR over shared-valid pixels."""
    valid = _shared_valid_mask(obs_arr, pred_arr, domain_mask)
    if not np.any(valid):
        return np.nan

    mse = float(np.mean((pred_arr[valid] - obs_arr[valid]) ** 2))
    if np.isclose(mse, 0.0):
        return np.inf

    max_val = float(np.max(obs_arr[valid]))
    if max_val <= 0.0:
        max_val = 1.0
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def metric_ssim(obs_arr, pred_arr, domain_mask):
    """Compute SSIM over shared-valid pixels using zero-fill outside domain."""
    valid = _shared_valid_mask(obs_arr, pred_arr, domain_mask)
    if np.count_nonzero(valid) < 9:
        return np.nan

    obs_img = np.where(valid, obs_arr, 0.0)
    pred_img = np.where(valid, pred_arr, 0.0)
    obs_valid = obs_arr[valid]

    data_range = float(obs_valid.max() - obs_valid.min())
    if data_range <= 0.0:
        data_range = 1.0

    return float(structural_similarity(obs_img, pred_img, data_range=data_range))


def metric_rmse(obs_arr, pred_arr, domain_mask):
    """Compute RMSE over shared-valid pixels."""
    valid = _shared_valid_mask(obs_arr, pred_arr, domain_mask)
    if not np.any(valid):
        return np.nan
    return float(np.sqrt(np.mean((pred_arr[valid] - obs_arr[valid]) ** 2)))


def metric_rmse_wet(obs_arr, pred_arr, domain_mask, threshold_m):
    """Compute RMSE on wet observed pixels where obs > threshold_m."""
    valid = _shared_valid_mask(obs_arr, pred_arr, domain_mask)
    wet = valid & (obs_arr > threshold_m)
    if not np.any(wet):
        return np.nan
    return float(np.sqrt(np.mean((pred_arr[wet] - obs_arr[wet]) ** 2)))


def metric_csi(obs_arr, pred_arr, domain_mask, threshold_m):
    """Compute CSI for wet/dry classification at the provided threshold."""
    valid = _shared_valid_mask(obs_arr, pred_arr, domain_mask)
    if not np.any(valid):
        return np.nan

    obs_wet = obs_arr[valid] >= threshold_m
    pred_wet = pred_arr[valid] >= threshold_m

    tp = np.sum(pred_wet & obs_wet)
    fp = np.sum(pred_wet & ~obs_wet)
    fn = np.sum(~pred_wet & obs_wet)
    denom = tp + fp + fn

    if denom == 0:
        return np.nan
    return float(tp / denom)


def compute_validation_metrics(obs_depth_m, pred_depth_m, domain_mask, wet_thresh_m, csi_thresh_m):
    """Compute modelz-style scalar validation metrics for one prediction raster.

    Parameters
    ----------
    obs_depth_m : numpy.ndarray
        Observed/reference depth raster in meters.
    pred_depth_m : numpy.ndarray
        Predicted depth raster in meters.
    domain_mask : numpy.ndarray
        Boolean mask of valid analysis pixels.
    wet_thresh_m : float
        Wet threshold used for wet-only RMSE.
    csi_thresh_m : float
        Threshold used for CSI and wet-fraction summaries.

    Returns
    -------
    dict
        Metrics keyed by names in ``METRIC_ORDER``.
    """
    assert obs_depth_m.shape == pred_depth_m.shape == domain_mask.shape, "shape mismatch"

    valid = _shared_valid_mask(obs_depth_m, pred_depth_m, domain_mask)
    if not np.any(valid):
        return {
            "psnr": np.nan,
            "ssim": np.nan,
            "rmse_m": np.nan,
            "rmse_wet_m": np.nan,
            "csi": np.nan,
            "mae_m": np.nan,
            "bias_m": np.nan,
            "obs_wet_fraction": np.nan,
            "pred_wet_fraction": np.nan,
            "valid_pixel_count": 0,
        }

    obs_vals = obs_depth_m[valid]
    pred_vals = pred_depth_m[valid]
    mae_m = float(np.mean(np.abs(pred_vals - obs_vals)))
    bias_m = float(np.mean(pred_vals - obs_vals))

    return {
        "psnr": metric_psnr(obs_depth_m, pred_depth_m, domain_mask),
        "ssim": metric_ssim(obs_depth_m, pred_depth_m, domain_mask),
        "rmse_m": metric_rmse(obs_depth_m, pred_depth_m, domain_mask),
        "rmse_wet_m": metric_rmse_wet(obs_depth_m, pred_depth_m, domain_mask, threshold_m=wet_thresh_m),
        "csi": metric_csi(obs_depth_m, pred_depth_m, domain_mask, threshold_m=csi_thresh_m),
        "mae_m": mae_m,
        "bias_m": bias_m,
        "obs_wet_fraction": float(np.mean(obs_vals >= csi_thresh_m)),
        "pred_wet_fraction": float(np.mean(pred_vals >= csi_thresh_m)),
        "valid_pixel_count": int(valid.sum()),
    }


def build_metrics_table(obs_depth_m, pred_depth_by_name, domain_mask, wet_thresh_m, csi_thresh_m, delta_pairs=None):
    """Build a metrics DataFrame for one or more named predictions."""
    assert isinstance(pred_depth_by_name, dict) and len(pred_depth_by_name) > 0, "pred_depth_by_name must be a non-empty dict"

    metrics_by_name = {}
    for name, pred_depth_m in pred_depth_by_name.items():
        metrics_by_name[name] = compute_validation_metrics(
            obs_depth_m=obs_depth_m,
            pred_depth_m=pred_depth_m,
            domain_mask=domain_mask,
            wet_thresh_m=wet_thresh_m,
            csi_thresh_m=csi_thresh_m,
        )

    metrics_df = pd.DataFrame(metrics_by_name)
    metrics_df = metrics_df.loc[[k for k in METRIC_ORDER if k in metrics_df.index]]

    if delta_pairs:
        for left, right in delta_pairs:
            if left in metrics_df.columns and right in metrics_df.columns:
                metrics_df[f"Delta_{left}_minus_{right}"] = metrics_df[left] - metrics_df[right]

    return metrics_df


def plot_hist_raster_grid(plot_specs, title, bins=60, metrics_df=None):
    """Plot histogram+raster diagnostics and append metrics text for each panel.

    Parameters
    ----------
    plot_specs : list[dict]
        Plot configuration dictionaries with:
        `name` (str), `arr` (numpy.ndarray-like), `cmap` (str),
        `use_dry_mask` (bool), `dry_thresh` (float|None),
        `is_depth` (bool), and optional `metric_key` (str).
    title : str
        Figure-level title.
    bins : int
        Histogram bin count.
    metrics_df : pandas.DataFrame or None
        Metrics table from `build_metrics_table` (metrics as index, run names as columns).
    """
    # Build a two-column layout: histogram on the left, raster on the right.
    fig, axes = plt.subplots(nrows=len(plot_specs), ncols=2, figsize=(10, 4 * len(plot_specs)))
    if len(plot_specs) == 1:
        axes = np.array([axes])

    for row_idx, spec in enumerate(plot_specs):
        # Pull required config for the current row.
        name = spec["name"]
        arr = np.asarray(spec["arr"])
        cmap = spec["cmap"]
        use_dry_mask = bool(spec["use_dry_mask"])
        dry_thresh = spec["dry_thresh"]
        is_depth = bool(spec.get("is_depth", False))
        metric_key = spec.get("metric_key", name)

        # Extract finite values for robust stats and histogram generation.
        vals = arr[np.isfinite(arr)]

        # For depth histograms only, mask exact zero-depth values to reduce dry-cell dominance.
        if is_depth:
            vals_hist = vals[~np.isclose(vals, 0.0)]
        else:
            vals_hist = vals

        ax_hist = axes[row_idx, 0]
        ax_raster = axes[row_idx, 1]

        if vals_hist.size:
            ax_hist.hist(vals_hist, bins=bins, color="steelblue", alpha=0.9)
            if use_dry_mask and dry_thresh is not None:
                ax_hist.axvline(dry_thresh, color="red", linestyle="--", linewidth=1.5)
            stat_text = (
                f"shape: {arr.shape}\n"
                f"hist_n: {vals_hist.size:,}\n"
                f"min: {vals_hist.min():.3f}\n"
                f"max: {vals_hist.max():.3f}\n"
                f"mean: {vals_hist.mean():.3f}\n"
                f"std: {vals_hist.std():.3f}"
            )
        else:
            stat_text = f"shape: {arr.shape}\n(no finite values)"

        # Append all metric rows for this panel when a matching column exists.
        if metrics_df is not None and metric_key in metrics_df.columns:
            metric_text_l = []
            for metric_name in metrics_df.index:
                metric_val = metrics_df.at[metric_name, metric_key]
                if pd.isna(metric_val):
                    metric_text_l.append(f"{metric_name}: nan")
                elif "count" in metric_name:
                    metric_text_l.append(f"{metric_name}: {int(metric_val):,}")
                else:
                    metric_text_l.append(f"{metric_name}: {float(metric_val):.6f}")
            stat_text = stat_text + "\n" + "\n".join(metric_text_l)

        ax_hist.set_title(f"{name} histogram")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(color="lightgrey", linestyle="-", linewidth=0.7)
        ax_hist.text(
            0.98,
            0.95,
            stat_text,
            transform=ax_hist.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        # Optionally hide dry cells in map view to emphasize wet structure.
        if use_dry_mask and dry_thresh is not None:
            raster_arr = np.ma.masked_where(arr < dry_thresh, arr)
        else:
            raster_arr = arr

        im = ax_raster.imshow(raster_arr, cmap=cmap)
        ax_raster.set_title(f"{name} raster")
        ax_raster.set_axis_off()
        fig.colorbar(im, ax=ax_raster, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()
