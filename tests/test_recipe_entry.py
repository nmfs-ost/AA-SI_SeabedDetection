# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Smoke tests for the recipe entry point.

These check the output contract a seafloor detection step relies on (a 1-D
line in metres on the input's ping grid), not that HDBSCAN converges well.
"""

import numpy as np
import pytest
import xarray as xr

from seabed_detection import detect_seafloor_hdbscan


def test_returns_line_on_the_input_ping_grid(synthetic_ds_Sv):
    """The result is a 1-D metre-valued line covering every input ping."""
    seafloor_depth = detect_seafloor_hdbscan(
        synthetic_ds_Sv, min_cluster_size=20, min_samples=5, offset_m=1.0
    )

    assert isinstance(seafloor_depth, xr.DataArray)
    assert seafloor_depth.dims == ("ping_time",)
    assert seafloor_depth.name == "seafloor_depth"
    assert seafloor_depth.attrs["units"] == "m"
    np.testing.assert_array_equal(
        seafloor_depth["ping_time"].values, synthetic_ds_Sv["ping_time"].values
    )
    assert np.isfinite(seafloor_depth.values).all()


def test_line_lands_on_the_seabed_band(synthetic_ds_Sv):
    """The deepest cluster is picked, so the line sits near the 20 m band."""
    seafloor_depth = detect_seafloor_hdbscan(
        synthetic_ds_Sv, min_cluster_size=20, min_samples=5, offset_m=1.0
    )

    assert 10.0 < float(seafloor_depth.median()) < 30.0


def test_range_window_limits_what_is_clustered(synthetic_ds_Sv):
    """The range window narrows the vertical axis but keeps every ping."""
    seafloor_depth = detect_seafloor_hdbscan(
        synthetic_ds_Sv,
        min_cluster_size=20,
        min_samples=5,
        range_sample_start=10,
        range_sample_end=28,
    )

    assert seafloor_depth.dims == ("ping_time",)
    np.testing.assert_array_equal(
        seafloor_depth["ping_time"].values, synthetic_ds_Sv["ping_time"].values
    )
    # Nothing outside the 10 m to 28 m window can be reported.
    assert 10.0 <= float(seafloor_depth.min()) <= float(seafloor_depth.max()) <= 28.0


def test_depth_is_required(synthetic_ds_Sv):
    """Without depth on ds_Sv and without echodata there is no vertical axis."""
    ds_Sv = synthetic_ds_Sv.drop_vars("depth")

    with pytest.raises(ValueError, match="no depth"):
        detect_seafloor_hdbscan(ds_Sv, min_cluster_size=20, min_samples=5)
