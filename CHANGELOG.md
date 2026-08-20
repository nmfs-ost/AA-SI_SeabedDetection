# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure from NOAA Fisheries AA-SI Python template
- `seabed_detection.detect_seafloor_hdbscan`, one callable that runs
  `prepare_features`, `hdbscan_seabed_detection`, and `extract_seabed_line` and
  returns seafloor depth as a 1-D `(ping_time,)` `xr.DataArray` in metres. This
  is what lets another package drive the detector, since a caller such as a
  workflow step can only name a single function. It takes a dataset carrying
  echopype's `depth` variable as well as one that already has the
  `depth (meters)` coordinate the EK60/EK80 scripts build, and can add depth
  itself from an `EchoData`.
- `range_sample_start` / `range_sample_end` arguments on
  `detect_seafloor_hdbscan`, exposing the range limit `prepare_features`
  documents as its remedy for the memory cost of a full grid. 
- `plot` argument on `hdbscan_seabed_detection`, defaulting to `True` so
  interactive use is unchanged. Passing `False` skips the plots, which otherwise
  block on an open window and would hang an automated run.
- `feature_channels` on `detect_seafloor_hdbscan`, naming the channels used as
  clustering features by frequency in kHz (e.g. `[38, 70]`) or by label, in the
  order they are fed, the first being the dB difference baseline. A single
  channel is allowed and contributes no difference features.
  `num_channel_chosen_for_features` still applies when no list is given.
- Smoke tests for the new entry point.

### Changed
- `data_preprocessing.py`, `hdbscan_seabed_detection.py`, and `seabed_export.py`
  moved from the repo root into `src/seabed_detection/`, so they ship with the
  installed package. `main.py` imports them from there now.
- Declared `scikit-learn`, `matplotlib`, and `seaborn`, which
  `hdbscan_seabed_detection.py` already imports.
- `detect_seafloor_hdbscan` sorts channels by `frequency_nominal` before
  selecting features, as `EK60_processing.py` and `EK80_processing.py` already
  do. Raw channel order is not always ascending, so without the sort an EK80
  survey stored as 18, 70, 200, 120, 38 kHz would pair 18 with 70 rather than
  18 with 38.

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Security
- Nothing yet

## [0.1.0] - YYYY-MM-DD

### Added
- Initial release
- Basic package structure with src layout
- Development tooling (pytest, black, pylint, pre-commit)

<!--
=============================================================================
CHANGELOG GUIDELINES
=============================================================================

When adding entries, use the following categories:
- Added: for new features
- Changed: for changes in existing functionality
- Deprecated: for soon-to-be removed features
- Removed: for now removed features
- Fixed: for any bug fixes
- Security: in case of vulnerabilities

Each release should have a version number and date in the format:
## [X.Y.Z] - YYYY-MM-DD

Link definitions should be added at the bottom (optional)
