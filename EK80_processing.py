
import numpy as np
import echopype as ep 
import xarray as xr
import matplotlib.pyplot as plt
from echopype.commongrid import resample_to_geometry



def calibrate_EK80(ed_list, depth_offset, depth_limit, encode_mode, low_resolution, low_res_spacing):
     
    cw_list, cw_original_list = [], []
    frq = 38000 #120000
    # low_res_spacing = 1 # 10.0
    # low_resolution =  True #False #

    print("\nCalibrating, regridding, and adding depth per file...")

    for ed in ed_list:
        # ---------------- CW Pipeline ----------------
        try:
            # Calibrate
            Sv_cw = ep.calibrate.compute_Sv(ed, waveform_mode='CW', encode_mode='power')

            Sv_cw = Sv_cw.sortby('frequency_nominal')

            Sv_cw_original = Sv_cw

            # Count the number of valid range_samples per channel
            # Select the first ping and count the non-NaN Sv values across the range_sample dimension
            valid_samples = Sv_cw['Sv'].isel(ping_time=0).count(dim='range_sample')
            for ch, count in zip(Sv_cw.channel.values, valid_samples.values):
                print(f"Channel {ch}: {count} valid samples")

            Sv_cw_orignal = Sv_cw
            # Regrid to one of the existing frequency channels
            selected_ch_cw = Sv_cw_orignal.channel.sel(channel=Sv_cw_orignal.frequency_nominal == frq).item()
            Sv_cw_orignal = resample_to_geometry(Sv_cw_orignal, target_variable="Sv", target_channel=selected_ch_cw)
    
            # =================================================================
            # Custom Regrid Block
            # =================================================================
            # Find the max range across channels for this specific file
            max_range = float(Sv_cw["echo_range"].max(skipna=True).compute().values)
            
            # Create the 1D target range
            target_range = np.arange(0, max_range + low_res_spacing, low_res_spacing)

            # Pad with NaNs so it temporarily fits the existing range_sample dimension
            n_range_sample = Sv_cw.sizes["range_sample"]
            target_range_padded = np.full(n_range_sample, np.nan)
            target_range_padded[: len(target_range)] = target_range

            # Create the 2D target grid
            target_grid = xr.DataArray(
                np.tile(target_range_padded, (Sv_cw.sizes["ping_time"], 1)),
                dims=("ping_time", "range_sample"),
                coords={
                    "ping_time": Sv_cw["ping_time"],
                    "range_sample": Sv_cw["range_sample"],
                },
                name="echo_range",
            )
            

            # Resample to the custom low-resolution grid
            Sv_cw = resample_to_geometry(
                Sv_cw, 
                target_variable="Sv", 
                target_grid=target_grid
            )
            

            # CRITICAL: Slice off the padded NaNs to actually shrink the array 
            # size in memory before appending it to the list.
            Sv_cw = Sv_cw.isel(range_sample=slice(0, len(target_range)))
            # =================================================================
            

            # Add depth (using the new low-res geometry)
            Sv_cw = ep.consolidate.add_depth(Sv_cw, ed, depth_offset=depth_offset)
            Sv_cw_original = ep.consolidate.add_depth(Sv_cw_original, ed, depth_offset=depth_offset)


            # Find the max depth per channel
            max_depths = Sv_cw['depth'].max(dim=['ping_time', 'range_sample'], skipna=True)
            for ch, max_val in zip(max_depths.channel.values, max_depths.values):
                print(f"Channel {ch}: Maximum depth is {max_val:.2f} meters")
            

            # Add the computed BB data to the list
            cw_list.append(Sv_cw)
            cw_original_list.append(Sv_cw_original)

        except Exception as e:
            print(f"Skipped CW for a file due to error: {e}")

    print("\nStitching processed datasets with Xarray...")

    Sv_cw = xr.concat(cw_list, dim='ping_time')    
    Sv_cw_original = xr.concat(cw_original_list, dim='ping_time')

    print("Stitching complete! Ready for plotting.")
    
    print(f"Final regridded depth size for HDBSCAN: {Sv_cw.sizes['range_sample']}")

    # print(f"CW channels are {Sv_cw.channel.values}")
    # print(f"CW ping_times are {Sv_cw.ping_time.values}")
    # print(f"Number of CW pings (continuous): {Sv_cw.ping_time.size}")
    # =================================================================

    freq_1d_cw = Sv_cw.frequency_nominal.isel(ping_time=0)
    cw_channels_sorted = Sv_cw.sortby(freq_1d_cw).channel.values
    print(f"Sorted CW channels:\n{cw_channels_sorted}")

    # ========================================================
    #          Plotting based on range_sample 
    # ======================================================== 
    # Sv_bb_data = Sv_bb.Sv
    # Sv_cw_data = Sv_cw.Sv
    # for bb, cw in zip(bb_channels_sorted, cw_channels_sorted):
    #     # --- BB Plotting ---
    #     plt.figure(figsize=(12,6))
    #     Sv_bb_data.sel(channel = bb).plot(
    #     x='ping_time', 
    #     y='range_sample',       
    #     vmin=-80, vmax=-30, cmap='viridis', 
    #     yincrease=False  
    #     )
    #     # plt.ylim(80000)
    #     plt.title(f"FM signal, channel {bb}")
    #     plt.ylabel("range_sample") 
    #     plt.xlabel("ping_time")
    #     plt.tight_layout()
    #     plt.show()
    #     # --- CW Plotting ---
    #     plt.figure(figsize=(12,6))
    #     Sv_cw_data.sel(channel = cw).plot(
    #     x='ping_time', 
    #     y='range_sample',       
    #     vmin=-80, vmax=-30, cmap='viridis', 
    #     yincrease=False  
    #     )
    #     plt.title(f"CW signal, channel {cw}")
    #     plt.ylabel("range_sample") 
    #     plt.xlabel("ping_time")
    #     plt.tight_layout()
    #     plt.show()
    # # =======================================================


    # =================================================
    #         Ploting Sv values with Depth
    # =================================================
    print("*" * 80)
    print("Plot echograms of CW according to depth.")
    print("*" * 80)

    for cw in cw_channels_sorted:
        if low_resolution: 
            # ======== CW Plotting ================
            Sv_cw_channel = Sv_cw.sel(channel= cw)

            # Assign 'depth' as a coordinate. xarray needs it to be a coordinate to map it to the Y-axis properly.
            Sv_cw_channel = Sv_cw_channel.assign_coords(depth=Sv_cw_channel.depth)

            Sv_cw_clean = Sv_cw_channel.dropna(dim='range_sample', subset=['depth'], how='all')

            fig, ax = plt.subplots(figsize=(14, 6))
            # When passing a 2D coordinate to 'y', xarray automatically uses pcolormesh
            Sv_cw_clean['Sv'].plot(
                x='ping_time', 
                y='depth', 
                yincrease=False,   
                cmap='viridis',     
                vmin=-80,           
                vmax=-30,           
                ax=ax
            )
            # ax.set_ylim(1300,0)
            ax.set_title(f"CW Echogram - Channel: {cw}")
            ax.set_ylabel("Depth (m)")
            ax.set_xlabel("Ping Time")

            plt.tight_layout()
            plt.show()


        # ======== CW Original Plotting ================
        Sv_cw_channel = Sv_cw_original.sel(channel= cw)

        # Assign 'depth' as a coordinate. xarray needs it to be a coordinate to map it to the Y-axis properly.
        Sv_cw_channel = Sv_cw_channel.assign_coords(depth=Sv_cw_channel.depth)

        Sv_cw_clean = Sv_cw_channel.dropna(dim='range_sample', subset=['depth'], how='all')

        fig, ax = plt.subplots(figsize=(14, 6))
        # When passing a 2D coordinate to 'y', xarray automatically uses pcolormesh
        Sv_cw_clean['Sv'].plot(
            x='ping_time', 
            y='depth', 
            yincrease=False,   
            cmap='viridis',     
            vmin=-80,           
            vmax=-30,           
            ax=ax
        )
        # ax.set_ylim(1300,0)
        ax.set_title(f"Original CW Echogram - Channel: {cw}")
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Ping Time")

        plt.tight_layout()
        plt.show()
       



    if low_resolution:
        chosen_Sv = Sv_cw 
    else:
        chosen_Sv = Sv_cw_original
    # everything in the below code block to change to chosen_Sv from Sv_cw_origiv=nal
    # ===============================================================================
    # ===== Sv_cw_original to hold depth values rather than range_sample ============
    print("Replacing 'range_sample' coordinate with actual 'depth (meters)'...")
    
    # Extract 1D depth values from the first channel and first ping 
    # (Assuming depth structure is constant across pings for the array shape)
    depth_values = chosen_Sv.depth[0][0][:].values   
    
    # Assign new coordinates and rename the dimension
    chosen_Sv = chosen_Sv.assign_coords(range_sample=('range_sample', depth_values))
    chosen_Sv = chosen_Sv.rename(range_sample="depth (meters)")
    # ===============================================================================


    if chosen_Sv == Sv_cw:
        print("Returning the low resolution calibrated signal for further processing for seabed detection.")
    elif chosen_Sv == Sv_cw_original:
        print("No resolution reduction. Returning the original calibrated signal for further processing for seabed detection.")
    
    
    # Sort the dimensions to ensure they are strictly monotonic for xarray plotting
    chosen_Sv = chosen_Sv.sortby("ping_time")
    chosen_Sv = chosen_Sv.sortby("depth (meters)")
    
    return chosen_Sv # Sv_cw_original # Sv_cw chosen_Sv







# def calibrate_EK80(ed_list, depth_offset, depth_limit):
     
#     cw_list = []
#     frq = 120000

#     print("\nCalibrating, regridding, and adding depth per file...")

#     for ed in ed_list:

#         # ---------------- CW Pipeline ----------------
#         try:
#             # Calibrate
#             Sv_cw = ep.calibrate.compute_Sv(ed, waveform_mode='CW', encode_mode='power')
#             Sv_cw = Sv_cw.sortby('frequency_nominal')

#             # Regrid
#             # selected_ch_cw = Sv_cw.channel.sel(channel=Sv_cw.frequency_nominal == frq).item()
#             # Sv_cw = resample_to_geometry(Sv_cw, target_variable="Sv", target_channel=selected_ch_cw)
            
#             # Add depth 
#             Sv_cw = ep.consolidate.add_depth(Sv_cw, ed, depth_offset=5)

#             # Add the computed BB data to the list
#             cw_list.append(Sv_cw)
#         except Exception as e:
#             print(f"Skipped CW for a file due to error: {e}")

#     print("\nStitching processed datasets with Xarray...")
#     Sv_cw = xr.concat(cw_list, dim='ping_time')

#     print("Stitching complete! Ready for plotting.")
    
#     # print(f"FM channels are {Sv_bb.channel.values}")
#     # print(f"FM ping_times are {Sv_bb.ping_time.values}")
#     # print(f"Number of BB pings (continuous): {Sv_bb.ping_time.size}")

#     # print(f"CW channels are {Sv_cw.channel.values}")
#     # print(f"CW ping_times are {Sv_cw.ping_time.values}")
#     # print(f"Number of CW pings (continuous): {Sv_cw.ping_time.size}")
#     # =================================================================


#     freq_1d_cw = Sv_cw.frequency_nominal.isel(ping_time=0)
#     cw_channels_sorted = Sv_cw.sortby(freq_1d_cw).channel.values
#     print(f"Sorted CW channels:\n{cw_channels_sorted}")

#     # ========================================================
#     #          Plotting based on range_sample 
#     # ======================================================== 
#     # Sv_bb_data = Sv_bb.Sv
#     # Sv_cw_data = Sv_cw.Sv
#     # for bb, cw in zip(bb_channels_sorted, cw_channels_sorted):
#     #     # --- BB Plotting ---
#     #     plt.figure(figsize=(12,6))
#     #     Sv_bb_data.sel(channel = bb).plot(
#     #     x='ping_time', 
#     #     y='range_sample',       
#     #     vmin=-80, vmax=-30, cmap='viridis', 
#     #     yincrease=False  
#     #     )
#     #     # plt.ylim(80000)
#     #     plt.title(f"FM signal, channel {bb}")
#     #     plt.ylabel("range_sample") 
#     #     plt.xlabel("ping_time")
#     #     plt.tight_layout()
#     #     plt.show()
#     #     # --- CW Plotting ---
#     #     plt.figure(figsize=(12,6))
#     #     Sv_cw_data.sel(channel = cw).plot(
#     #     x='ping_time', 
#     #     y='range_sample',       
#     #     vmin=-80, vmax=-30, cmap='viridis', 
#     #     yincrease=False  
#     #     )
#     #     plt.title(f"CW signal, channel {cw}")
#     #     plt.ylabel("range_sample") 
#     #     plt.xlabel("ping_time")
#     #     plt.tight_layout()
#     #     plt.show()
#     # # =======================================================


#     # =================================================
#     #         Ploting Sv values with Depth
#     # =================================================
#     print("*" * 80)
#     print("Plot echograms of CW and FM according to depth.")
#     print("*" * 80)

#     for cw in cw_channels_sorted:
#         # ======== CW Plotting ================
#         Sv_cw_channel = Sv_cw.sel(channel= cw)

#         # Assign 'depth' as a coordinate. xarray needs it to be a coordinate to map it to the Y-axis properly.
#         Sv_cw_channel = Sv_cw_channel.assign_coords(depth=Sv_cw_channel.depth)

#         Sv_cw_clean = Sv_cw_channel.dropna(dim='range_sample', subset=['depth'], how='all')

#         fig, ax = plt.subplots(figsize=(14, 6))
#         # When passing a 2D coordinate to 'y', xarray automatically uses pcolormesh
#         Sv_cw_clean['Sv'].plot(
#             x='ping_time', 
#             y='depth', 
#             yincrease=False,   
#             cmap='viridis',     
#             vmin=-80,           
#             vmax=-30,           
#             ax=ax
#         )
#         # ax.set_ylim(depth_limit,0)
#         ax.set_title(f"CW Echogram - Channel: {cw}")
#         ax.set_ylabel("Depth (m)")
#         ax.set_xlabel("Ping Time")

#         plt.tight_layout()
#         plt.show()

#         # # ========= FM Plotting =============
#         # Sv_bb_channel = Sv_bb.sel(channel= bb)

#         # # Assign 'depth' as a coordinate. xarray needs it to be a coordinate to map it to the Y-axis properly.
#         # Sv_bb_channel = Sv_bb_channel.assign_coords(depth=Sv_bb_channel.depth)
        
#         # Sv_bb_clean = Sv_bb_channel.dropna(dim='range_sample', subset=['depth'], how='all')

#         # fig, ax = plt.subplots(figsize=(14, 6))
#         # # When passing a 2D coordinate to 'y', xarray automatically uses pcolormesh
#         # Sv_bb_clean['Sv'].plot(
#         #     x='ping_time', 
#         #     y='depth', 
#         #     yincrease=False,   
#         #     cmap='viridis',     
#         #     vmin=-80,           
#         #     vmax=-30,           
#         #     ax=ax
#         # )
#         # ax.set_ylim(1300,0)
#         # ax.set_title(f"FM Echogram - Channel: {bb}")
#         # ax.set_ylabel("Depth (m)")
#         # ax.set_xlabel("Ping Time")

#         # plt.tight_layout()
#         # plt.show()

#     return Sv_cw #, Sv_bb





