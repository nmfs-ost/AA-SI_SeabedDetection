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
  documents as its remedy for the memory cost of a full grid. Clustering an
  entire survey grid exhausts memory rather than merely running slowly: 1548
  pings by 15207 range samples is 23.5 million points, which was killed after
  26 minutes on a 16 GB machine, while around half a million points completes
  in under a minute.
- `plot` argument on `hdbscan_seabed_detection`, defaulting to `True` so
  interactive use is unchanged. Passing `False` skips the plots, which otherwise
  block on an open window and would hang an automated run.
- Smoke tests for the new entry point.

On tuning: `extract_seabed_line` picks the cluster with the greatest median
depth, and a bottom multiple sits at roughly twice the seabed's depth, so it
wins whenever clustering resolves it as its own cluster. On a 100 ping EK60 test
file with a seabed near 200 m, `min_cluster_size`/`min_samples` of 300/300 and
900/70 both returned about 400 m, while 5000/1000 returned 197 m. Expect to tune
the two per survey and to compare the line against another detector.

### Changed
- `data_preprocessing.py`, `hdbscan_seabed_detection.py`, and `seabed_export.py`
  moved from the repo root into `src/seabed_detection/`, so they ship with the
  installed package. `main.py` imports them from there now.
- Declared `scikit-learn`, `matplotlib`, and `seaborn`, which
  `hdbscan_seabed_detection.py` already imports.

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- The wheel build target pointed at `src/AA-SI_SeabedDetection`, which does not
  exist, so an install produced a package with nothing in it.
- `echopype>=1.11.1` pinned a version that was never released, leaving the
  dependency unresolvable. Corrected to `>=0.11.1`.
- Coverage was configured for `src/SI_SeabedDetection`, which does not exist, so
  `pytest --cov` measured nothing.
- `__init__.py` looked up the version of `mypackagename`, so `__version__` always
  fell back to `0.0.0.dev`. It now reads `seabed-detection`, the name in
  `pyproject.toml`.
- `tests/test_package.py` imported `mypackagename`, so the suite could not run.
- The plots raised `AttributeError` on matplotlib 3.9 and newer, which removed
  `matplotlib.cm.get_cmap`.

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
