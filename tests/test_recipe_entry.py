# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Smoke tests for the recipe entry point.

These check the output contract a seafloor detection step relies on (a 1-D
line in metres on the input's ping grid), not that HDBSCAN converges well.
"""

import numpy as np
import pandas as pd
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


def _multi_channel_ds(frequencies_hz, seabed_top_m=20, n_ping=40, n_range=30):
    """Sv dataset whose channel order follows frequencies_hz as given."""
    rng = np.random.default_rng(0)
    n_channel = len(frequencies_hz)
    Sv = np.full((n_channel, n_ping, n_range), -90.0)
    Sv[:, :, seabed_top_m:] = -30.0
    Sv += rng.normal(0.0, 0.5, Sv.shape)
    depth = np.broadcast_to(np.arange(n_range, dtype=float), Sv.shape).copy()
    return xr.Dataset(
        {
            "Sv": (("channel", "ping_time", "range_sample"), Sv),
            "depth": (("channel", "ping_time", "range_sample"), depth),
            "frequency_nominal": (("channel",), np.array(frequencies_hz, dtype=float)),
        },
        coords={
            "channel": [f"ch{int(f / 1000)}" for f in frequencies_hz],
            "ping_time": pd.date_range("2024-01-01", periods=n_ping, freq="1s").values,
            "range_sample": np.arange(n_range),
        },
    )


class TestFeatureChannelSelection:
    """Which channels become features must not depend on raw file ordering."""

    # An EK80 file can present its channels unsorted; HB2407 is 18, 70, 200,
    # 120, 38. Taking the first two as they arrive would silently pair 18 with
    # 70 rather than 18 with 38.
    UNSORTED = [18000, 70000, 200000, 120000, 38000]

    def test_count_selects_by_ascending_frequency_not_file_order(self):
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds(self.UNSORTED)
        ordered, n = _order_feature_channels(ds_Sv, None, 2)

        assert n == 2
        assert [str(c) for c in ordered["channel"].values[:2]] == ["ch18", "ch38"]

    def test_every_channel_is_kept_only_reordered(self):
        """prepare_features drops rows NaN in any channel, so none may be lost."""
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds(self.UNSORTED)
        ordered, _ = _order_feature_channels(ds_Sv, [38, 70], 2)

        assert ordered.sizes["channel"] == ds_Sv.sizes["channel"]
        assert set(str(c) for c in ordered["channel"].values) == {
            "ch18", "ch38", "ch70", "ch120", "ch200"
        }

    def test_explicit_frequencies_lead_in_the_order_given(self):
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds(self.UNSORTED)
        ordered, n = _order_feature_channels(ds_Sv, [38, 70], 2)

        assert n == 2
        assert [str(c) for c in ordered["channel"].values[:2]] == ["ch38", "ch70"]

    def test_channels_may_be_named_by_label(self):
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds(self.UNSORTED)
        ordered, n = _order_feature_channels(ds_Sv, ["ch120", "ch18"], 2)

        assert n == 2
        assert [str(c) for c in ordered["channel"].values[:2]] == ["ch120", "ch18"]

    def test_single_channel_runs_end_to_end(self):
        """A 38 kHz only run is valid; it simply has no difference features."""
        ds_Sv = _multi_channel_ds(self.UNSORTED)

        seafloor_depth = detect_seafloor_hdbscan(
            ds_Sv, min_cluster_size=20, min_samples=5, feature_channels=[38]
        )

        assert seafloor_depth.dims == ("ping_time",)
        assert seafloor_depth.attrs["num_feature_channels"] == 1
        assert np.isfinite(seafloor_depth.values).all()

    def test_explicit_selection_runs_end_to_end(self):
        ds_Sv = _multi_channel_ds(self.UNSORTED)

        seafloor_depth = detect_seafloor_hdbscan(
            ds_Sv, min_cluster_size=20, min_samples=5, feature_channels=[38, 70]
        )

        assert seafloor_depth.dims == ("ping_time",)
        assert seafloor_depth.attrs["num_feature_channels"] == 2

    def test_missing_frequency_is_rejected(self):
        ds_Sv = _multi_channel_ds(self.UNSORTED)

        with pytest.raises(ValueError, match="no channel at 333 kHz"):
            detect_seafloor_hdbscan(
                ds_Sv, min_cluster_size=20, min_samples=5, feature_channels=[333]
            )

    def test_repeated_channel_is_rejected(self):
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds(self.UNSORTED)
        with pytest.raises(ValueError, match="more than once"):
            _order_feature_channels(ds_Sv, [38, 38], 2)

    def test_asking_for_more_channels_than_exist_is_rejected(self):
        from seabed_detection.recipe_entry import _order_feature_channels

        ds_Sv = _multi_channel_ds([18000, 38000])
        with pytest.raises(ValueError, match="exceeds the 2 channels"):
            _order_feature_channels(ds_Sv, None, 5)
