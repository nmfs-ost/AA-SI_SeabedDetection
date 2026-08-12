# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Pytest configuration and fixtures.

Add shared fixtures and pytest configuration here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Example fixture:
# @pytest.fixture
# def sample_data():
#     """Provide sample data for tests."""
#     return {"key": "value"}


@pytest.fixture
def synthetic_ds_Sv():
    """Two channel Sv dataset shaped like echopype's, with a flat seabed.

    Background water is around -90 dB and everything from 20 m down is around
    -30 dB, so a clustering run has an obvious deepest feature to find. Depth
    is supplied as echopype's (channel, ping_time, range_sample) variable
    rather than as a coordinate, matching what a workflow passes in after
    add_depth.
    """
    rng = np.random.default_rng(0)
    n_channel, n_ping, n_range = 2, 40, 30
    seabed_top_m = 20

    Sv = np.full((n_channel, n_ping, n_range), -90.0)
    Sv[:, :, seabed_top_m:] = -30.0
    Sv += rng.normal(0.0, 0.5, Sv.shape)

    depth_values = np.arange(n_range, dtype=float)
    depth = np.broadcast_to(depth_values, Sv.shape).copy()

    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), Sv),
            "depth": (("channel", "ping_time", "range_sample"), depth),
        },
        coords={
            "channel": ["ch18", "ch38"],
            "ping_time": pd.date_range("2024-01-01", periods=n_ping, freq="1s").values,
            "range_sample": np.arange(n_range),
        },
    )
