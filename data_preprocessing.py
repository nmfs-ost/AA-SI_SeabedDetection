# ==================================================
# Preprocess data 
# for ML based processing

# input: ed, calibrated Sv
# output: DataFrame ready for processing by HDBSCAN
# ==================================================

import numpy as np
import time 
import echopype as ep 
import xarray as xr 


"""
Data Preparation  
EK60 data comes with ping_time, range_sample, channel, and Sv values. 
It is basically a three dimensional data, with its dimensions being:
channel,
ping_time, 
range_sample. 

This data needs to be restructured to be suitable (tabular) for HDBSCAN clustering algorithm. 
Table to consist of as many rows as of ping_time * range_sample.
rows: each row is denoting one cell in the Echogram(ping_time, range_sample)
Columns = Sv values at various frequency channels 
"""


"""
Restructure Sv data into a DataFrame suitable for passing into the HDBSCAN algorithm.
        
Sv_data: Sv data with coordinations
    - 'channels': Data array of channels (e.g., 18kHz, 38kHz, ...)
    - 'ping_time': Data array of ping_times
    - 'range_sample': Data array of range_samples
    and the Sv values across these dimensions.
    - Sv: 3D array of channel, ping_time, and range_sample of Sv values in dB
    Returns:
        DataFrame (in tabular format) with rows representing each (ping_time, range_sample) in the echogram, 
        and columns reprenting the frequency channels. The values per each row and column are the Sv values.
"""


start_time = time.time()

def prepare_features(Sv, ed):
    # ====== Volume Backscattering Strength in dB =======           
    Sv_original = Sv.Sv              # shape = (Ch, T, R)
    # ===================================================
    

    # ===== Limit the echo_range as needed ===================================
    # To address the memory issue, selecte k depths rather than the whole data
    # k_min = 0
    # k = 1100 # 900 # 1100  # 890 # 1100 
    # Sv_data = Sv_original.isel(range_sample = slice(k_min, k))  
    # print(f'Depth data is limited to the first {k-k_min} samples.')                       
    Sv_data = Sv_original
    # =========================================================================
          

    # ==== Compute depth values =====================================
    Sv_with_depth = ep.consolidate.add_depth(Sv, echodata = ed)
    # Sv_with_depth['depth'].shape     # (4, 1209, 2665)
    # Sv_with_depth['depth'].values    # (4, 1209, 2665)
    depth_values = Sv_with_depth.depth[0][0][:].values   
    # depth_values = Sv_with_depth.depth[0][0][k_min:k].values 
    # ===============================================================


    # ===== Sv_data to hold depth values rather than range_sample ===================
    # For plotting the originial Sv with actual depth values
    Sv_data = Sv_data.assign_coords(range_sample = ( 'range_sample' ,  depth_values) )
    # Rename the coordinate of range_sample to depth (meters)
    Sv_data = Sv_data.rename(range_sample = "depth (meters)")
    # ===============================================================================


    # ===== Obtain number of channels, ping_times, and echo_range ========     
    Ch, T, R = Sv_data.shape   
    # print(f'Total number of channels: {Ch}')
    # print(f'Total number of ping_times: {T}')
    # print(f'Total number of depths: {R}')
    # ====================================================================


    # ===== Conversion of Sv from xarray to numpy array
    # Transpose and Convert Sv to numpy array
    Sv_np = Sv_data.transpose('ping_time','depth (meters)','channel').values
    # ======================================================================


    # ===== Built the table structure: Form the table rows ==============
    # match the reshaped data's row order
    pings = np.repeat(Sv_data.ping_time.values, R)     # lenght T*R
    depths = np.tile(depth_values, T)                        # length T*R
    # ===================================================================


    # ===== Reshape Sv to (T*R, Ch) =====================================
    # Sv_np is three dimensional
    # Flatten Sv_np to two dimensional array
    Sv_reshaped = Sv_np.reshape(T*R, Ch)
    # ===================================================================


    # ===== Remove rows with NaN values on any channel ======================
    # mask returns a 1D boolean array where each value is True if that row has at least one NaN in any channel
    mask = ~np.isnan(Sv_reshaped).any(axis = 1) # axis = 1 referes to columns
    # Drop any row (pint_time, depth_value) if any channel value is missing (axis = 1). 
    Sv_clean = Sv_reshaped[mask]           

    print(f'Original number of rows:{T*R}')
    print(f'Number of rows after removing NaNs:{Sv_clean.shape[0]}') # Total number of raws (ping_time * range_sample) 

    # Coordinate mapping arrays for ping_time and range_sample 
    # Apply mask to coordinates 
    pings_clean = pings[mask]
    depths_clean = depths[mask]
    ping_time_vals = Sv_data.ping_time.values



    end_time = time.time()
    print(f"Running time of preprocessing step is {end_time - start_time} seconds.")

    return Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean


    
    