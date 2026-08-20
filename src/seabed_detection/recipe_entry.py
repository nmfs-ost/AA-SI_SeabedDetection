# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Single callable entry point for the HDBSCAN seabed detector.

Detection is a three call chain ending in a pandas Series. A workflow step can
only name one callable, so this module composes the chain and converts the
result to the 1-D (ping_time,) DataArray such steps return.
"""

import echopype as ep
import numpy as np
import xarray as xr

from seabed_detection.data_preprocessing import prepare_features
from seabed_detection.hdbscan_seabed_detection import hdbscan_seabed_detection
from seabed_detection.seabed_export import extract_seabed_line

DEPTH_COORD = "depth (meters)"


def _with_depth_coordinate(ds_Sv, echodata=None):
    """Give ds_Sv the 1-D depth coordinate prepare_features expects.

    The EK60/EK80 scripts build a "depth (meters)" coordinate after calibration.
    A dataset from a workflow instead carries echopype's depth variable on
    (channel, ping_time, range_sample), so the axis is taken from the first
    channel and first ping and range_sample is renamed to match. That drops
    per-ping variation in depth, the same assumption the scripts make.

    Args:
        ds_Sv: Calibrated Sv dataset.
        echodata: Source EchoData, used to add depth when ds_Sv has none.

    Returns:
        Dataset whose vertical dimension is named "depth (meters)" and measured
        in metres, with Sv ordered (channel, ping_time, depth).

    Raises:
        ValueError: If depth cannot be established from ds_Sv or echodata.
    """
    if DEPTH_COORD not in ds_Sv.coords:
        if "depth" not in ds_Sv:
            if echodata is None:
                raise ValueError(
                    "ds_Sv has no depth; run echopype.consolidate.add_depth "
                    "upstream or pass echodata"
                )
            ds_Sv = ep.consolidate.add_depth(ds_Sv, echodata)

        depth = ds_Sv["depth"]
        leading = {dim: 0 for dim in depth.dims if dim != "range_sample"}
        depth_values = depth.isel(leading).values

        ds_Sv = ds_Sv.assign_coords(range_sample=("range_sample", depth_values))
        ds_Sv = ds_Sv.rename(range_sample=DEPTH_COORD)

    return ds_Sv.transpose("channel", "ping_time", DEPTH_COORD, ...)


def _limit_range(ds_Sv, range_sample_start, range_sample_end):
    """Keep only a window of the vertical axis.

    This is the range limit prepare_features documents as its remedy for the
    memory cost of clustering a full grid, applied here so the caller can set
    it without editing that module.

    Args:
        ds_Sv: Calibrated Sv dataset.
        range_sample_start: First range sample to keep.
        range_sample_end: One past the last range sample to keep, or None.

    Returns:
        Dataset narrowed to the window, or unchanged when the window is open.
    """
    if range_sample_start == 0 and range_sample_end is None:
        return ds_Sv

    dim = "range_sample" if "range_sample" in ds_Sv.dims else DEPTH_COORD
    return ds_Sv.isel({dim: slice(range_sample_start, range_sample_end)})


def _channel_frequencies_khz(ds_Sv):
    """Nominal frequency of each channel in kHz, in the dataset's channel order.

    Args:
        ds_Sv: Calibrated Sv dataset.

    Returns:
        List of frequencies in kHz, or None when ds_Sv carries no
        frequency_nominal.
    """
    if "frequency_nominal" not in ds_Sv:
        return None
    freq = ds_Sv["frequency_nominal"]
    spare = [dim for dim in freq.dims if dim != "channel"]
    if spare:
        freq = freq.isel({dim: 0 for dim in spare})
    return [int(round(float(v) / 1000)) for v in np.atleast_1d(freq.values)]


def _resolve_channel(wanted, labels, frequencies_khz):
    """Index of the channel named by a label or a frequency in kHz."""
    if isinstance(wanted, str):
        for index, label in enumerate(labels):
            if str(label) == wanted:
                return index
        raise ValueError(f"no channel labelled {wanted!r}; have {[str(x) for x in labels]}")

    if frequencies_khz is None:
        raise ValueError(
            "ds_Sv has no frequency_nominal, so channels can only be chosen by label"
        )
    target = int(round(float(wanted)))
    matches = [i for i, khz in enumerate(frequencies_khz) if khz == target]
    if not matches:
        raise ValueError(f"no channel at {target} kHz; have {frequencies_khz}")
    if len(matches) > 1:
        raise ValueError(f"{target} kHz matches {len(matches)} channels; choose by label")
    return matches[0]


def _order_feature_channels(ds_Sv, feature_channels, num_feature_channels):
    """Put the channels used as clustering features first.

    hdbscan_seabed_detection reads the first ``num_channel_chosen_for_features``
    columns, so which channels become features depends on the order they sit in.
    The EK60/EK80 scripts sort by frequency before building features; a dataset
    arriving from a workflow carries whatever order the raw file had, so sort
    here too rather than relying on it.

    Every channel is kept, only reordered. prepare_features drops a row where
    any channel is NaN, so subsetting to the selected channels would quietly
    relax that and change which cells are clustered.

    Args:
        ds_Sv: Calibrated Sv dataset.
        feature_channels: Channels to use, as labels or frequencies in kHz, in
            the order they should be fed. None selects by count instead.
        num_feature_channels: How many channels to use when feature_channels is
            None, taken in ascending frequency.

    Returns:
        Tuple of the reordered dataset and the number of leading channels to
        use as features.

    Raises:
        ValueError: If a requested channel is missing, ambiguous, repeated, or
            if more channels are asked for than the dataset holds.
    """
    frequencies_khz = _channel_frequencies_khz(ds_Sv)
    n_channels = ds_Sv.sizes["channel"]

    if feature_channels:
        labels = list(ds_Sv["channel"].values)
        chosen = []
        for wanted in feature_channels:
            index = _resolve_channel(wanted, labels, frequencies_khz)
            if index in chosen:
                raise ValueError(f"channel {wanted!r} requested more than once")
            chosen.append(index)
        rest = [i for i in range(n_channels) if i not in chosen]
        return ds_Sv.isel(channel=chosen + rest), len(chosen)

    if num_feature_channels > n_channels:
        raise ValueError(
            f"num_feature_channels={num_feature_channels} exceeds the "
            f"{n_channels} channels in ds_Sv"
        )
    if frequencies_khz is not None:
        ds_Sv = ds_Sv.isel(channel=np.argsort(frequencies_khz, kind="stable"))
    return ds_Sv, num_feature_channels


def detect_seafloor_hdbscan(
    ds_Sv,
    echodata=None,
    min_cluster_size=300,
    min_samples=300,
    num_feature_channels=2,
    feature_channels=None,
    offset_m=1.0,
    range_sample_start=0,
    range_sample_end=None,
    gen_min_span_tree=False,
    core_dist_n_jobs=4,
    plot=False,
):
    """Detect seafloor depth by clustering Sv features with HDBSCAN.

    Clusters every non-NaN (ping_time, depth) cell on its per-channel Sv
    values, the dB differences against the first channel, and its ping and
    depth position. The cluster with the greatest median depth is taken as the
    seabed, and its shallowest point per ping becomes the line.

    Args:
        ds_Sv: Calibrated Sv dataset, ideally already through add_depth.
        echodata: Source EchoData, only used when ds_Sv carries no depth.
        min_cluster_size: Smallest number of points HDBSCAN accepts as a
            cluster.
        min_samples: HDBSCAN neighbourhood size; larger values label more
            points as noise.
        num_feature_channels: Number of frequency channels used as
            clustering features, taken in ascending frequency. Ignored when
            feature_channels is given.
        feature_channels: Explicit channels to use as features, as labels or
            frequencies in kHz, e.g. [38, 70]. The first one is the baseline
            the dB differences are measured against. A single channel is
            allowed and simply contributes no difference features.
        offset_m: Metres subtracted from the detected line to move it up
            through the water column.
        range_sample_start: First range sample to cluster.
        range_sample_end: One past the last range sample to cluster, or None
            for everything below range_sample_start. Clustering cost grows with
            pings times range samples, so a survey recorded well past the
            seabed needs this to stay tractable.
        gen_min_span_tree: Keep HDBSCAN's minimum spanning tree. Off here
            because nothing in this path reads it and it is held for the life
            of the fit.
        core_dist_n_jobs: Processes HDBSCAN uses for core distances. Each one
            holds its own copy of the tree, so 1 trades run time for a lower
            peak.
        plot: Show the diagnostic echogram and per-cluster plots. Off by
            default because the plots block on an interactive window, which
            would hang an automated run.

    Returns:
        xr.DataArray: Seafloor depth in metres, dims (ping_time,), on ds_Sv's
        ping_time coordinate.
    """
    ds_limited = _limit_range(ds_Sv, range_sample_start, range_sample_end)
    ds_ordered, n_feature_channels = _order_feature_channels(
        ds_limited, feature_channels, num_feature_channels
    )
    ds_prepared = _with_depth_coordinate(ds_ordered, echodata)

    (
        Sv_data,
        Sv_clean,
        Ch,
        T,
        R,
        depth_values,
        depths_clean,
        ping_time_vals,
        pings_clean,
    ) = prepare_features(ds_prepared)

    labels, probabilities, df_results = hdbscan_seabed_detection(
        Sv_data,
        Sv_clean,
        Ch,
        T,
        R,
        depth_values,
        depths_clean,
        ping_time_vals,
        pings_clean,
        min_cluster_size,
        min_samples,
        n_feature_channels,
        plot=plot,
        gen_min_span_tree=gen_min_span_tree,
        core_dist_n_jobs=core_dist_n_jobs,
    )

    seabed_line = extract_seabed_line(df_results, ping_time_vals, offset_m=offset_m)

    return xr.DataArray(
        seabed_line.values,
        coords={"ping_time": ds_Sv["ping_time"]},
        dims=["ping_time"],
        name="seafloor_depth",
        attrs={
            "long_name": "Seafloor depth from HDBSCAN clustering",
            "units": "m",
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "num_feature_channels": n_feature_channels,
            "offset_m": offset_m,
        },
    )
