
import echopype as ep 
import xarray as xr
from pathlib import Path 
import matplotlib.pyplot as plt


def calibrate_EK60(ed_list, depth_offset):
    print("\nCalibrating, and adding depth per file...")
    cw_list = []
    for ed in ed_list:
        # ---------------- CW Pipeline ----------------
        try:
            # Calibrate
            Sv_cw = ep.calibrate.compute_Sv(ed)
            Sv_cw = Sv_cw.sortby('frequency_nominal')

            # Add depth 
            Sv_cw = ep.consolidate.add_depth(Sv_cw, ed, depth_offset)
            # Add the computed cw data to the list
            cw_list.append(Sv_cw)
        except Exception as e:
            print(f"Skipped CW for a file due to error: {e}")

    print("\nStitching processed datasets with Xarray...")
    Sv_cw = xr.concat(cw_list, dim='ping_time')

    print("Stitching complete! Ready for plotting.")

    # print(f"CW channels are {Sv_cw.channel.values}")
    # print(f"CW ping_times are {Sv_cw.ping_time.values}")
    # print(f"Number of CW pings (continuous): {Sv_cw.ping_time.size}")
    # =================================================================

    freq_1d_cw = Sv_cw.frequency_nominal.isel(ping_time=0)
    cw_channels_sorted = Sv_cw.sortby(freq_1d_cw).channel.values
    # print(f"Sorted CW channels:\n{cw_channels_sorted}")

    # ========================================================
    #          Plotting based on range_sample 
    # ======================================================== 
    # Sv_cw_data = Sv_cw.Sv
    # for ch_cw in cw_channels_sorted:
    #     # --- CW Plotting ---
    #     plt.figure(figsize=(12,6))
    #     Sv_cw_data.sel(channel = ch_cw).plot(
    #     x='ping_time', 
    #     y='range_sample',       
    #     vmin=-80, vmax=-30, cmap='viridis', 
    #     yincrease=False  
    #     )
    #     plt.title(f"CW signal, channel {ch_cw}")
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
        Sv_cw_channel = Sv_cw.sel(channel= cw)

        # Assign 'depth' as a coordinate. xarray needs it to be a coordinate to map it to the Y-axis properly.
        Sv_cw_channel = Sv_cw_channel.assign_coords(depth=Sv_cw_channel.depth)

        fig, ax = plt.subplots(figsize=(14, 6))
        # When passing a 2D coordinate to 'y', xarray automatically uses pcolormesh
        Sv_cw_channel['Sv'].plot(
            x='ping_time', 
            y='depth', 
            yincrease=False,   
            cmap='viridis',     
            vmin=-80,           
            vmax=-30,           
            ax=ax
        )

        ax.set_title(f"Echogram - Channel: {cw}")
        ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Ping Time")

        plt.tight_layout()
        plt.show()

    return Sv_cw