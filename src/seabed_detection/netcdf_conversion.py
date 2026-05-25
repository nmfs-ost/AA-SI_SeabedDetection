# -*- coding: utf-8 -*-
"""
Convert_EK60-to-netcdf
Use echopype to convert Simrad EK60 raw files to netCDF
jech  
"""

def raw_to_netcdf(model, directory_path, filename):

    import echopype as ep
    from echopype import open_raw
    from pathlib import Path
    import json
    from sys import exit 

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
    
    # filename = Path('/home/user/Project/DATA/' + file_name) #D20090405-T114914.raw') #SetteSE2403Bigeye-D20240320-T033313.raw')  #') # 
    full_path = directory_path / filename
    dataDirectory = Path(filename).parent
    outdir = dataDirectory / 'netCDF4_Files'
    # print(dataDirectory)
    # print(outdir)

    dc = createOutDir(outdir)
    if not dc:  exit()

    # convert to netCDF4    
    print('Converting: ', filename)
    ed = open_raw(str(full_path), sonar_model=model)
    # Henry B. Bigelow ICES code is 33HH
    ed['Platform']['platform_name'] = 'Henry B. Bigelow'
    ed['Platform']['platform_type'] = 'SHIPC'
    ed['Platform']['platform_code_ICES'] = '33HH'
    # the to_netcdf function seems to have an error catch, so I don't use "try"
    ed.to_netcdf(save_path=str(outdir))
    
    return ed

