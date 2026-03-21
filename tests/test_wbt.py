"""
Simple WhiteboxTools smoke tests with temporary GeoTIFF inputs.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def test_wbt_spinup():
    """WhiteboxTools should initialize and report a valid version string."""
    from ..fdsc.hp.wbt import wbt

    version = wbt.version()

    assert isinstance(version, str)
    assert "WhiteboxTools" in version


@pytest.mark.parametrize(
    "shape, source_idx",
    [
        pytest.param((5, 5), (2, 2), id="small_square_center_source"),
        pytest.param((6, 4), (1, 2), id="small_rectangular_source"),
    ],
)
def test_wbt_cost_distance_returns_zero(tmp_path, shape, source_idx):
    """cost_distance should run cleanly and assign zero cost at the source cell."""
    from ..fdsc.hp.wbt import wbt

    source_fp = tmp_path / "source.tif"
    cost_fp = tmp_path / "cost.tif"
    accum_fp = tmp_path / "accum.tif"
    backlink_fp = tmp_path / "backlink.tif"

    source_ar = np.zeros(shape, dtype=np.float32)
    source_ar[source_idx] = 1.0
    cost_ar = np.ones(shape, dtype=np.float32)

    profile = dict(
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, float(shape[0]), 1, 1),
        nodata=-9999.0,
    )

    # write temporary source and cost rasters used by WhiteboxTools
    with rasterio.open(source_fp, "w", **profile) as dst:
        dst.write(source_ar, 1)
    with rasterio.open(cost_fp, "w", **profile) as dst:
        dst.write(cost_ar, 1)

    # point WBT at the temporary output directory for this test
    wbt.set_working_dir(str(tmp_path))
    return_code = wbt.cost_distance(str(source_fp), str(cost_fp), str(accum_fp), str(backlink_fp))

    # validate return code and source-cell cost accumulation
    with rasterio.open(accum_fp) as src:
        accum_ar = src.read(1)

    assert return_code == 0
    assert float(accum_ar[source_idx]) == pytest.approx(0.0)
