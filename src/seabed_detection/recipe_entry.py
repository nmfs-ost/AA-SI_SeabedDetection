# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Single callable entry point for the HDBSCAN seabed detector.

Detection is a three call chain ending in a pandas Series. A workflow step can
only name one callable, so this module composes the chain and converts the
result to the 1-D (ping_time,) DataArray such steps return.
"""

import echopype as ep
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


def detect_seafloor_hdbscan(
    ds_Sv,
    echodata=None,
    min_cluster_size=300,
    min_samples=300,
    num_feature_channels=2,
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
        num_feature_channels: Number of frequency channels, taken in the
            dataset's channel order, used as clustering features.
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
    ds_prepared = _with_depth_coordinate(ds_limited, echodata)

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
        num_feature_channels,
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
            "num_feature_channels": num_feature_channels,
            "offset_m": offset_m,
        },
    )
