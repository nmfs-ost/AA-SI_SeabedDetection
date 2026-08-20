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
from netcdf_conversion import raw_to_netcdf_list
# from netcdf_conversion_temp import raw_to_netcdf
from EK60_processing import calibrate_EK60
from EK80_processing import calibrate_EK80
from seabed_detection.data_preprocessing import prepare_features
from seabed_detection.hdbscan_seabed_detection import hdbscan_seabed_detection
from seabed_detection.seabed_export import extract_seabed_line, export_to_evl


start_time = time.time()
# =====================================================================
# ================================ Inputs =============================
sonar_model = 'EK80' #ed.sonar_model
encode_mode = 'power'


total_files = 1 #18
print(f"Total number of .raw files to process is {total_files}.") 

depth_offset = 0
depth_limit = 500

low_resolution = False # True # 
low_res_spacing = 1.0

# file_name = "HDD_Henry_B_Bigelow_HB2407_Echosounder_Data_Raw_D20241106-T090205.raw" #"D20090403-T095648.raw" # "HB-D20240503-T033715.raw" #"HB-D20220419-T040908.raw" # "D20151021-T180531.raw" #"DY2303_EK80-D20230216-T145249.raw" # "DY2303_EK80-D20230216-T140111.raw" # "DY2303_EK80-D20230222-T092840.raw" #"DY2303_EK80-D20230220-T144308.raw" # "HB-D20221114-T174241.raw" #"HB-D20221113-T031848.raw" #"HB-D20221113-T024714.raw" # "DY2303_EK80-D20230219-T073128.raw" # "D20090405-T114914.raw" # "DY2303_EK80-D20230221-T144452.raw" # "DY2303_EK80-D20230219-T073128.raw" # "0-D20110909-T202410.raw" #HB-D20210909-T193739.raw" #"DY2303_EK80-D20230219-T073128.raw" # "D20160725-T212129.raw" # "D20160718-T225425.raw" # "D20160720-T151416.raw" # "0-D20110909-T193325.raw" # "0-D20110915-T173905.raw" # "0-D20110915-T195505.raw" #"D20160725-T212129.raw" #"D20160725-T212129.raw, 900" # "0-D20110915-T195505.raw" # "D20160725-T212129.raw (parameter: 900)" , D20090405-T114914.raw
num_channel_chosen_for_features = 2
min_cluster_size = 900 #700                                      # 6000  #300 # 30 # 300 #200 #300 # 400 # 300 #200 # 170 # 150 || #70 #50 #400 #100, 900 # 1500 # 3000, 6000 #100 #900 #70 # 500 #500 # 900 # 1100 #900 is best so far
min_samples = 70 # min_cluster_size #   
# 1000, 200
#100 # 150 # 200 # min_cluster_size 
#300, 150 1000, 30
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
    data_directory = Path('/home/user/downloads/HB0901_bbox_-68p74_40p62_-68p72_40p624') #HB2202_bbox_-69p79_42p09_-69p78_42p11') #Project/Data') #Data_NCEI')  #
elif sonar_model == 'EK80' and encode_mode == 'power': 
    data_directory = Path('/home/user/Demo') #Project/Data_AFSC/DY2303') #Data_NCEI') # Demo') # downloads/HB2402_bbox_-68p79_42p22_-68p58_42p23') # HB2202_bbox_-69p79_42p09_-69p78_42p11') # Data_AFSC/DY2303') # Data_NCEI')
else: print('No data exists.')
# =====================================================================




# ====================================================================================
#                           Generate Echodata Object 
# ====================================================================================
# A list of EchoData objects
print('Create a list of echodata objects.')
ed_list = raw_to_netcdf_list(sonar_model, total_files, data_directory)
# ====================================================================================

# # =====================================================================
# # =================== raw data to NetCDF Converssion ==================
# ed = raw_to_netcdf(sonar_model, data_directory, file_name)
# ed_list = [ed]
# # =====================================================================



# =====================================================================
# ======================== Data calibration ===========================
# Sv: calibrated backscatter value
# Sv = ep.calibrate.compute_Sv(ed) #, encode_mode = "power", waveform_mode = "CW")
# =====================================================================
if sonar_model == "EK60":
    Sv = calibrate_EK60(ed_list, depth_offset)
elif sonar_model == "EK80": 
    Sv = calibrate_EK80(ed_list, depth_offset, depth_limit, encode_mode,  low_resolution, low_res_spacing)

print(f'Calibrated dataset has its channels sorted: {Sv.channel.values}')
# =======================================================================



# =======================================================================
# ============ Data Preprocessing for Machine Learning ==================
Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean = prepare_features(Sv)
# =======================================================================



# # =======================================================================
# # ============= Seabed detection ========================================
# # Apply HDBSCAN
# print(f'Initiate Seabed Detection')
# labels, probabilities, df_results = hdbscan_seabed_detection(Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean, min_cluster_size, min_samples)


# # =============== KEEP NOISE DATAPOINTS =================================
# print("Isolating datapoints labeled as noise...")
# # 1. Keep the metadata (ping_time, depth, label) for noise points
# df_noise = df_results[df_results['cluster_label'] == -1].copy()

# # 2. (Optional) Keep the actual acoustic backscatter values for the noise points
# noise_mask = (labels == -1)
# Sv_noise = Sv_clean[noise_mask]

# print(f"Successfully kept {len(df_noise)} noise datapoints.")
# # =======================================================================
# # =======================================================================






# =======================================================================
# ============= First Pass: Seabed detection ============================
print(f'Initiate First Pass: Seabed Detection')
labels_1, probabilities_1, df_results_1 = hdbscan_seabed_detection(
    Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, 
    ping_time_vals, pings_clean, min_cluster_size, min_samples, num_channel_chosen_for_features
)

# # =============== Isolate Noise Datapoints ==============================
# print("Isolating datapoints labeled as noise...")

# # Create a boolean mask where the label is -1 (noise)
# noise_mask = (labels_1 == -1)

# # Filter the input arrays to ONLY contain the noise points
# Sv_noise = Sv_clean[noise_mask]
# pings_noise = pings_clean[noise_mask]
# depths_noise = depths_clean[noise_mask]

# print(f"Successfully isolated {len(Sv_noise)} noise datapoints.")
# # =======================================================================


# # =======================================================================
# # ============= Second Pass: HDBSCAN on Noise Data ======================
# print(f'Initiate Second Pass: HDBSCAN on Isolated Noise')

# # Define new hyperparameters for the noise pass. 
# # Noise is sparser, so these usually need to be significantly smaller 
# # than your primary seabed detection parameters.
# noise_min_cluster_size = 10000 #1000 # 500  # Adjust as needed (e.g., 100, 500)
# noise_min_samples = 100 # 170 # 300 #500       # 50   # Adjust as needed 

# # Call the function again, but replace Sv_clean, depths_clean, 
# # and pings_clean with your isolated noise arrays.
# labels_2, probabilities_2, df_results_2 = hdbscan_seabed_detection(
#     Sv_data,              # Original dataset (needed for the baseline plots)
#     Sv_noise,             # Passed as the new "Sv_clean"
#     Ch, T, R,             # Original dimensions
#     depth_values,         # Original depth grid 
#     depths_noise,         # The y-coordinates of just the noise points
#     ping_time_vals,       # Original ping_time grid
#     pings_noise,          # The x-coordinates of just the noise points
#     noise_min_cluster_size, 
#     noise_min_samples
# )
# # =======================================================================

# # =======================================================================
# # ============================== Automated Line Extraction ==============
# print("Extracting 1m offset seabed line from First Pass...")
# # Note: We use df_results_1 here because that contains the actual seabed!
# seabed_line = extract_seabed_line(df_results_1, ping_time_vals, offset_m = 1.0)
# # =======================================================================



# # =======================================================================
# # ============================== Automated Line Extraction ==============
# print("Extracting 1m offset seabed line...")
# seabed_line = extract_seabed_line(df_results, ping_time_vals, offset_m = 1.0)
# # =======================================================================



# # =======================================================================
# # =================================== EVL file generation ===============
# print("Exporting to .evl format...")
# evl_filename = file_name.replace(".raw", "_seabed_offset.evl")
# export_to_evl(seabed_line, output_filename = data_directory / evl_filename)
# # =====================================================================


end_time = time.time()
print(f'Running time of HDBSCAN for Seabed Detection is {end_time - start_time: 0.2f} seconds.')