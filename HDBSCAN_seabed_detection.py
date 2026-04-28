# ===========================================
# Seabed Detection 
# HDBSCAN algorithm implementation

# input: Sv dataframe
# Output: multiple Clusters including seabed 
# ==========================================



# Goal:
# Apply HDBSCAN over the tabularized data frame (Sv_clean) for seabed detection.
# 
# Initial step: Feature selection
# For the HDBSCAN technique we need to choose features (attributes in the data) to be used by
# the Machine learning technique for learning.
# Features options to choose from: frequency channels, dB differencing, depth, and ping_time (6 features)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
import echopype as ep
import xarray as xr 
import sklearn
# from optimal_eps import compute_optimal_eps
import hdbscan
import seaborn as sns 

# 900
# 29,232 more noise
# 49,528
# 253,6328
 
# 1000
# 23,372 noise data
# 50,356 cluster 0/ seabed
# 2541360 

# 1100


def HDBSCAN_seabed_detection(Sv_data, Sv_clean, Ch, T, R, depth_values, depths_clean, ping_time_vals, pings_clean):

    # ===== Feature selection ========================
    ch_x = 2
    Sv_ml = Sv_clean[:, :ch_x]

    # === Add dB differrencing as features ============
    # Add Sv differences to the numpy array, Sv_ml
    ref = Sv_ml[:,0].reshape(-1, 1)
    target_cols = Sv_ml[:, 1:]
    Sv_ml = np.hstack([Sv_ml, target_cols - ref])
    # Sv_ml = np.hstack( [ Sv_ml , Sv_ml[:,1].reshape(-1, 1) - ref ] )


    # ===== Add ping_time and depth as features =======================
    # Need index of the ping_times rather than the actual values
    ping_times_idx = np.arange(pings_clean.size)
    ping_times_idx_col = ping_times_idx.reshape(-1,1)
    depths_col = depths_clean.reshape(-1,1)
    # Add two columns to the numpy array, Sv_ml
    Sv_ml = np.hstack( [ Sv_ml, ping_times_idx_col, depths_col] )
    # ==================================================================

    num_features = Sv_ml.shape[1]
    print(f'Total number of features is set to {num_features}.')








    # Sv_clean_linear = 10 ** (Sv_clean / 10)
    # print("Backscatter values transferred to linear format.")

    # ===== Normalize the data ==========================================================================
    scaler = StandardScaler() 
    Sv_scaled = scaler.fit_transform(Sv_ml) # (Sv_clean_linear)
    print("The data has been scaled and ready for processing by HDBSCAN by seabed detection.")
    # ===================================================================================================

    # print("Ran successfully up to this line.")
    # compute_optimal_eps(minPoints, Sv_scaled)
    # print("Ran the compute eps successfully.")

    # , metric = 'canberra'

    # ===== Hyperparameter tuing ================================================
    # Set min_cluster_size 
    # The only parameter we really need to set is 'min_cluster_size'.
    # We set it to x, meaning each cluster has to have a min number of x points.
    min_cluster_size = 1100 #900 is best so far
    print(f'min_cluster_size for seabed detection is set to {min_cluster_size}')
    # ===========================================================================
    
    # 300, 300: seabed detection
    # 500, 500: seabed detection
    # 10000, 500 tree size good
    # 10000, 100 

    # 4 features
    # 2000, 2000

    # ===== Create the model and fit to the data ==========================================================
    clusterer = hdbscan.HDBSCAN(min_cluster_size, min_samples = min_cluster_size, gen_min_span_tree = True) 
    clusterer.fit(Sv_scaled)

    # 'labels_' contains the cluster assignment for each point.
    # Noise points are labeled -1.
    labels = clusterer.labels_

    # 'probabilities_' contains the "confidence" that a point belongs to its cluster.
    # Lower probabilities mean it's more of an outlier.
    probabilities = clusterer.probabilities_
    # =====================================================================================================

    clusterer.condensed_tree_.plot()

    # Get the number of clusters found (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Estimated number of clusters: {n_clusters}")

    n_noise = list(labels).count(-1) 
    print(f"Total number of outliers is equal to {n_noise}.")


    
    
    # =============== Store the results into a DataFrame to easily filter, plot, and map back to the echogram

    df_results = pd.DataFrame({
        'ping_time': pings_clean , 
        'depth (meters)': depths_clean, 
        'cluster_label': labels} 
        )

    print(f'First five rows of the data along with the clustered labels: ')
    print(df_results.head())


    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # # Visualization of the outcome
    # * Echogram type of plotting for the clusterred data (range_sample versus ping_time with cluster labels on the graph instead of the back scattered values.)
    # * Side by side plot of the original data echogram versus the clustered data
    # * Scatter plot for range_sample versus ping_time 
    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    # Reconstruct an xarray DataArray of cluster labels, so we can plot like an echogram. 
    # This helps make results visually comparable to the original Sv image.

    # # ================= Echogram - cluster plot =================

    # # All points as NaN
    cluster_grid = np.full( (T,R), np.nan)    # Two dimensional numpy array with T rows and R columns

    sorted_indices = np.argsort(ping_time_vals)
    ping_time_vals_sorted = ping_time_vals[sorted_indices]
    t_indices = np.searchsorted(ping_time_vals_sorted, pings_clean )
    t_indices = sorted_indices[t_indices]

    # sorted_indices = np.argsort(depth_values)
    # true_depths_sorted = depth_values[sorted_indices]
    r_indices = np.searchsorted(depth_values, depths_clean)
    # r_indices = sorted_indices[r_indices]


    cluster_grid[t_indices, r_indices] = labels

    # DataArray
    cluster_da = xr.DataArray(
        cluster_grid,
        coords = {"ping_time": ping_time_vals, "depth (meters)": depth_values},
        dims = ["ping_time", "depth (meters)"]
    )



    import matplotlib.patches as mpatches
    unique_labels = np.unique(labels)

    # A categorical colormap with enough colors
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))

    # legends
    legend_handles = []
    for i , lbl in enumerate(unique_labels):
        color = cmap(i)
        name = "Noise" if lbl == -1 else f'Cluster {lbl}'
        legend_handles.append(mpatches.Patch(color = color, label = name))


    plt.figure(figsize = (12,6))
    cluster_da.plot(x = "ping_time", y = "depth (meters)", cmap = cmap, vmin = -1, vmax = len(unique_labels)-1, add_colorbar = False) # -1 for noise; cmap: color map; tab20: categorical color map; vmin: min value of the color map
    # plt.title("HDBSCAN Clusters")
    plt.title( f"HDBSCAN, Features = {num_features},  min_cluster_size = {min_cluster_size}")

    plt.gca().invert_yaxis()           # gca: get current axes
    plt.legend(handles = legend_handles, title = "Cluster labels", bbox_to_anchor = (1.05, 1), loc = "upper left")
    plt.show()
    # like an echogram but each pixel is colored by its cluster ID


    # # ==================== Sv echogram vs HDBSCAN cluster echogram =========================
    fig, axes = plt.subplots(1, 2, figsize=(18,7), sharey = True)

    Sv_data[0].plot(x = "ping_time", y = "depth (meters)", ax = axes[0], cmap="viridis", add_colorbar = True, vmin = -150, vmax = 0)
    axes[0].invert_yaxis()
    axes[0].set_title("Original Sv (dB)")
    plt.gca().invert_yaxis()

    cluster_da.plot(x = "ping_time", y = "depth (meters)", ax = axes[1], cmap=cmap, vmin = -1,vmax = len(unique_labels)-1, add_colorbar = False)
    # axes[1].set_title("HDBSCAN Clusters")
    axes[1].set_title(f"HDBSCAN, Features = {num_features}, min_cluster_size = {min_cluster_size}")

    axes[1].invert_yaxis()
    axes[1].legend(handles = legend_handles, title = "Cluster labels", bbox_to_anchor = (1.05, 1), loc = "upper left")

    plt.tight_layout()
    plt.show()
    # ========================================================================================

    cols = ["18kHz", "38kHz", "70kHz", "120kHz", "200kHz"]
    # ===================== Plot clusters separately =====================
    for i , lbl in enumerate(unique_labels):
        # if lbl != -1: continue
        color = cmap(i)
        cluster_masked = cluster_da.where(cluster_da == lbl)
        print(len(cluster_masked))
        mask = (labels == lbl)
        print(f'Total number of data points belonging to cluster {lbl} is {len(Sv_clean[mask])}.')


        plt.figure(figsize = (12,6))
        cluster_masked.plot(x = "ping_time", y = "depth (meters)", cmap = cmap, vmin = -1, vmax = len(unique_labels)-1, add_colorbar = False) # -1 for noise; tab20: categorical color map
        plt.gca().invert_yaxis()           # gca: get current axes

        name = f'Cluster {lbl}'
        legend_handle = [ mpatches.Patch(color = color, label = name) ]
        plt.legend(handles = legend_handle, title = "Cluster labels", bbox_to_anchor = (1.05, 1), loc = "upper left")
        # plt.title(f"Echogram - {name}")
        plt.title( f"HDBSCAN Clustering, Features = {num_features}, min_cluster_size = {min_cluster_size}")

        plt.show()

        cols = ["18kHz", "38kHz", "70kHz", "120kHz", "200kHz"]
        df = pd.DataFrame(Sv_clean[mask] , columns = cols[:Ch] ) #[:,:num_features]
        sns.boxplot(df) 
        plt.title(f"Box and Whisker Plot for cluster {lbl}")
        plt.show()


    return labels, probabilities        