# ===========================
# Seabed Detection Algorithm
# based on HDBSCAN Technique 
# ===========================

# %matplotlib inline 
import time 
import pandas as pd 
import echopype as ep 
import numpy as np
import xarray as xr
from pathlib import Path 
from netcdf_conversion import raw_to_netcdf
from EK60_processing import calibrate_EK60
from EK80_processing import calibrate_EK80
from data_preprocessing import prepare_features
from hdbscan_seabed_detection import hdbscan_seabed_detection
from seabed_export import extract_seabed_line, export_to_evl


start_time = time.time()
# =====================================================================
# ================================ Inputs =============================
sonar_model = 'EK60' #ed.sonar_model
encode_mode = 'power'

depth_offset = 0
depth_limit = 500
# low_res_spacing = 1.0

file_name = "D20090403-T095648.raw" # "HB-D20240503-T033715.raw" #"HB-D20220419-T040908.raw" # "D20151021-T180531.raw" #"DY2303_EK80-D20230216-T145249.raw" # "DY2303_EK80-D20230216-T140111.raw" # "DY2303_EK80-D20230222-T092840.raw" #"DY2303_EK80-D20230220-T144308.raw" # "HB-D20221114-T174241.raw" #"HB-D20221113-T031848.raw" #"HB-D20221113-T024714.raw" # "DY2303_EK80-D20230219-T073128.raw" # "D20090405-T114914.raw" # "DY2303_EK80-D20230221-T144452.raw" # "DY2303_EK80-D20230219-T073128.raw" # "0-D20110909-T202410.raw" #HB-D20210909-T193739.raw" #"DY2303_EK80-D20230219-T073128.raw" # "D20160725-T212129.raw" # "D20160718-T225425.raw" # "D20160720-T151416.raw" # "0-D20110909-T193325.raw" # "0-D20110915-T173905.raw" # "0-D20110915-T195505.raw" #"D20160725-T212129.raw" #"D20160725-T212129.raw, 900" # "0-D20110915-T195505.raw" # "D20160725-T212129.raw (parameter: 900)" , D20090405-T114914.raw

min_cluster_size = 300 # 30 # 300 #200 #300 # 400 # 300 #200 # 170 # 150 || #70 #50 #400 #100, 900 # 1500 # 3000, 6000 #100 #900 #70 # 500 #500 # 900 # 1100 #900 is best so far
min_samples = 300 #100 # 150 # 200 # min_cluster_size 
#300, 150
# 300, 50
# 200, 50
# 150, 50
# 150, 30
# 150, 20
# cluster_selection_epsilon = 
# cluster_selection_method
# =====================================================================


# =====================================================================
# ======================== Input Data Directory =======================
if sonar_model == 'EK60':
    DATA_DIR = Path('/home/user/downloads/HB0901_bbox_-68p74_40p62_-68p72_40p624') #HB2202_bbox_-69p79_42p09_-69p78_42p11') #Project/Data') #Data_NCEI')  #
elif sonar_model == 'EK80' and encode_mode == 'power': 
    DATA_DIR = Path('/home/user/downloads/HB2402_bbox_-68p79_42p22_-68p58_42p23') # HB2202_bbox_-69p79_42p09_-69p78_42p11') # Data_AFSC/DY2303') # Data_NCEI')
else: print('No data exists.')
# =====================================================================



# =====================================================================
# =================== raw data to NetCDF Converssion ==================
ed = raw_to_netcdf(sonar_model, DATA_DIR, file_name)
# =====================================================================

# =====================================================================
# ======================== Data calibration ===========================
# Sv: calibrated backscatter value
# Sv = ep.calibrate.compute_Sv(ed) #, encode_mode = "power", waveform_mode = "CW")
# =====================================================================
ed_list = [ed]
if sonar_model == "EK60":
    Sv = calibrate_EK60(ed_list, depth_offset)
elif sonar_model == "EK80": 
    Sv = calibrate_EK80(ed_list, depth_offset, depth_limit, encode_mode)

print(f'Calibrated dataset has its channels sorted: {Sv.channel.values}')
# =======================================================================



# =======================================================================
# ============ Data Preprocessing for Machine Learning ==================
Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean = prepare_features(Sv, ed)
# =======================================================================


# =======================================================================
# ============= Seabed detection ========================================
# Apply HDBSCAN
print(f'Initiate Seabed Detection')
labels, probabilities, df_results = hdbscan_seabed_detection(Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean, min_cluster_size, min_samples)
# =======================================================================


# =======================================================================
# ============================== Automated Line Extraction ==============
print("Extracting 1m offset seabed line...")
seabed_line = extract_seabed_line(df_results, ping_time_vals, offset_m = 1.0)
# =======================================================================


# =======================================================================
# =================================== EVL file generation ===============
print("Exporting to .evl format...")
evl_filename = file_name.replace(".raw", "_seabed_offset.evl")
export_to_evl(seabed_line, output_filename = DATA_DIR / evl_filename)
# =====================================================================


end_time = time.time()
print(f'Running time of HDBSCAN for Seabed Detection is {end_time - start_time: 0.2f} seconds.')