# ===========================
# Seabed Detection Algorithm
# based on HDBSCAN Technique 
# ===========================

# %matplotlib inline 
import time 
import echopype as ep 
from pathlib import Path 
from netcdf_conversion import raw_to_netcdf
from data_preprocessing import prepare_features
from hdbscan_seabed_detection import hdbscan_seabed_detection


start_time = time.time()
# =====================================================================
# =================== raw data to NetCDF Converssion ==================
# ed: created ed object
sonar_model = 'EK60' # 'EK80'
DATA_DIR = Path('/home/user/Project/DATA') 
file_name =  "D20160725-T212129.raw" # "D20090405-T114914.raw"  #
ed = raw_to_netcdf(sonar_model, DATA_DIR, file_name)
# =====================================================================


# =====================================================================
# ======================== Data calibration ===========================
# Sv: calibrated backscatter value
Sv = ep.calibrate.compute_Sv(ed) #, encode_mode = "power", waveform_mode = "CW")
# =====================================================================


# =======================================================================
# ============ Data Preprocessing for Machine Learning ==================
Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean = prepare_features(Sv, ed)
# =======================================================================


# ====== Seabed detection =====================
# Apply HDBSCAN
print(f'Initiate Seabed Detection')
min_cluster_size = 900 # 1100 #900 is best so far
labels, probabilities = hdbscan_seabed_detection(Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean, min_cluster_size)
# =============================================


end_time = time.time()
print(f'Running time of HDBSCAN for Seabed Detection is {end_time - start_time: 0.2f} seconds.')