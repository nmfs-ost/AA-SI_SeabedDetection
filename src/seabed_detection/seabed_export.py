# ==================================================
# Seabed Line Extraction and EVL Export
# ==================================================

import pandas as pd
import numpy as np

def extract_seabed_line(df_results, ping_time_vals, offset_m = 1.0):
    """
    Identifies the seabed cluster, extracts the upper boundary, 
    and applies an upward offset.
    """
    # Filter out the noise (label = -1)
    valid_clusters = df_results[df_results['cluster_label'] != -1]
    
    # idea: The seabed is generally the continuous cluster with the greatest depth.
    # Calculate the median depth for each cluster to find the deepest feature.
    cluster_stats = valid_clusters.groupby('cluster_label')['depth (meters)'].median()
    seabed_cluster_id = cluster_stats.idxmax()
    print(f"Automatic Selection: Cluster {seabed_cluster_id} identified as the Seabed.")
    
    # Extract the upper boundary (minimum depth) of the seabed per ping
    seabed_points = valid_clusters[valid_clusters['cluster_label'] == seabed_cluster_id]
    seabed_top = seabed_points.groupby('ping_time')['depth (meters)'].min()
    
    # Apply the offset (subtract offset_m to move the line *up* in the water column)
    seabed_top_offset = seabed_top - offset_m
    
    # Reindex to ensure have a depth value for *every* ping time in the original file
    # Prevents gaps in the Echoview line. We use linear interpolation for missing data.
    seabed_line = seabed_top_offset.reindex(ping_time_vals)
    seabed_line = seabed_line.interpolate(method='linear').bfill().ffill()
    
    return seabed_line

def export_to_evl(seabed_line, output_filename = "seabed_line.evl" ):
    """
    Exports a pandas Series of (ping_time, depth) to an Echoview .evl file.
    """
    with open(output_filename, 'w') as f:
        # EVL Header
        f.write("EVL 3\n")
        # Number of data points
        f.write(f"{len(seabed_line)}\n")
        
        # Data rows
        for ping_time, depth in seabed_line.items():
            # Convert numpy datetime64 to python datetime to format strings
            pt = pd.to_datetime(ping_time)
            
            # Echoview expects: YYYYMMDD HHMMSS.ssss
            date_str = pt.strftime("%Y%m%d")
            time_str = pt.strftime("%H%M%S")
            
            # Extract fractional seconds (up to 4 digits for EVL)
            ms = int(pt.microsecond / 100) 
            
            # 3 is the Echoview line status code for "unverified" 
            # (standard for machine-generated lines before manual review)
            f.write(f"{date_str} {time_str}.{ms:04d} {depth:.4f} 3\n")
            
    print(f"Successfully exported Echoview line file to: {output_filename}")