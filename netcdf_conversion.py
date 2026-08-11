
import echopype as ep 
import xarray as xr
from pathlib import Path 
import glob
import os
from sys import exit 
from echopype.commongrid import resample_to_geometry


def raw_to_netcdf_list(model, num, data_directory):

    def createOutDir(outputdir):
        # create the output directory
        success = True
        if outputdir.exists():
            print('Directory %s exists' % outputdir)
            success = True
        else:
            try:
                outputdir.mkdir()
                success = True
            except OSError:
                print('Unable to create output directory %s' % outputdir)
                success = False
                #exit()
            else:
                print('Output directory created %s' % outputdir)
                success = True
        return success

    # =====================================================================
    #   return a list of individual EchoData objects
    # =====================================================================
    
    outdir = data_directory / 'netCDF4_Files'
    # print(data_directory)
    # print(outdir)
    dc = createOutDir(outdir)
    if not dc:  exit()  





    
    # *********************************************************************
    # file_name = 'D20250727-T033835.raw' #'D20250727-T040754.raw' #'SetteSE2403Bigeye-D20240320-T032338.raw' # "D20090405-T114914.raw" #
    # ed = raw_to_netcdf(sonar_model, file_name)

    # Find all .raw files, sort them by time, and grab the first 15
    all_raw_files = sorted(glob.glob(os.path.join(data_directory, '*.raw')))
    file_list = all_raw_files[:num] # [:15]
    for f in file_list:
        print(f'The file to be processed is {f}.')

    if isinstance(file_list, str):
        file_list = [file_list]

    full_paths = [Path(data_directory + f) if not str(f).startswith('/') else Path(f) for f in file_list]
    print(full_paths)

    print(f'Opening {len(full_paths)} files individually...')
    ed_list = [ep.open_raw(str(p), sonar_model=model) for p in full_paths]

    for ed in ed_list:
        ed.to_netcdf(save_path=str(outdir))

    return ed_list
    # ======================================================================




# # Beam_group1 and Beam_group2 exist in the ed object: 
# # 1) FM data is located in Beam_group1
# # 2) CW data is located in Beam_group2
# beam1 = ed['Sonar/Beam_group1']
# beam2 = ed['Sonar/Beam_group2'] 
# # ===========================================================================

# # ========================= channel names ===================================
# bb_channels = beam1.channel.values
# cw_channels = beam2.channel.values  
# print(f"FM channels: {bb_channels}")
# print(f"CW channels: {cw_channels}")
# """ 
# ['WBT 401014-15 ES38-7_2' 
#  'WBT 401025-15 ES120-7C_8'
#  'WBT 401028-15 ES18-11mk2_7' 
#  'WBT 401045-15 ES70-7C_8'
#  'WBT 401061-15 ES200-7C_1']

# ['WBT 401014-15 ES38-7_5' 
#  'WBT 401025-15 ES120-7C_5'
#  'WBT 401028-15 ES18-11mk2_3' 
#  'WBT 401045-15 ES70-7C_5'
#  'WBT 401061-15 ES200-7C_2']
# """
# # ============================================================================

# # ==========================================
# # pulse length(duration) for each ping_time
# # ==========================================
# print("=== Transmit Pulse Durations ===")

# # Check the first Broadband channel (Ping index 0)
# bb_pulse_durations = beam1['transmit_duration_nominal']
# ch_bb = bb_channels[0]
# pd_bb = bb_pulse_durations.sel(channel=ch_bb).isel(ping_time=0).values
# print(f"BB Pulse Duration ({ch_bb}): {pd_bb * 1000:.2f} milliseconds")

# # Check the first Continuous Wave channel (Ping index 1)
# cw_pulse_durations = beam2['transmit_duration_nominal']
# ch_cw = cw_channels[0]
# pd_cw = cw_pulse_durations.sel(channel=ch_cw).isel(ping_time=1).values
# print(f"CW Pulse Duration ({ch_cw}): {pd_cw * 1000:.2f} milliseconds")

