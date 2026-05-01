# Seabed Detection based on HDBSCAN 

## Description 
This project uses machine learning technique called HDBSCAN to automate the Seabed detection.

## Requirements
Pythton 3.12
Echopype

## Installation 

## Usage
- Input: EK60 or EK80 raw data
- Output: Clusters of the acoustic data including the seabed cluster

### Running the code
- Provide the directory path to your raw file, DATA_DIR.

- Provide the name of .raw file, file_name.

- Run the main.py.

### Hyperparameter tuning:
- min_cluster_size: Indicates the minimum size of any grouped data points by HDBSCAN to be called a cluster. 

## Technologies
Python, Pandas, Scikit-learn, Echopype, 

## Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an 'as is' basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or its bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise does not imply endorsement or favoring by the Department of Commerce. Use of DOC seals or logos shall not suggest endorsement by DOC or the U.S. Government.