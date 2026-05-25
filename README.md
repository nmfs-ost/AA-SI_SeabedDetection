# Seabed Detection based on HDBSCAN 

<!-- markdownlint-disable MD033 MD041 -->

## Description 
This project uses machine learning technique called HDBSCAN to automate the Seabed detection.

## Requirements
Pythton 3.12
Echopype


## Usage
- Input: EK60 or EK80 raw data
- Output: Clusters of the acoustic data including the seabed cluster

## Getting Started
---


### Running the code
- Provide the directory path to your raw file, DATA_DIR.

- Provide the name of .raw file, file_name.

- Run the main.py.

### Hyperparameter tuning:
- min_cluster_size: Indicates the minimum size of any grouped data points by HDBSCAN to be called a cluster. 

## Technologies
Python, Pandas, Scikit-learn, Echopype, 


<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Getting Started](#getting-started) •
[Customization Checklist](#customization-checklist) •
[Development](#development) •
[Project Structure](#project-structure)

</div>

---

### Installation

```bash
# Clone the repository
git clone https://github.com/nmfs-ost/AA_SI-SeabedDetection.git
cd Seabed_Detection

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
pytest
pytest --cov=mypackagename
```

### Code Quality

```bash
black src/ tests/
pylint src/mypackagename
pre-commit run --all-files
```

### Building

```bash
pip install build
python -m build
```

---

## Project Structure

```
├── .gitignore
├── .pre-commit-config.yaml
├── .pylintrc
├── CHANGELOG.md
├── LICENSE
├── NOTICE
├── pyproject.toml
├── README.md
├── src/
│   └── seabed_detection/
│       └── __init__.py
└── tests/
    ├── conftest.py
    └── test_package.py
```

---

## License

This template uses the Apache License 2.0. Verify this license meets your project requirements before use.

---

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
