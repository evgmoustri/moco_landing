import os
import opensim as osim
import numpy as np
from utils import readExternalLoadsFile
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

project_path = os.getcwd()
results_path = os.path.abspath(project_path +
                                  "/Results/Compare/Flat-foot")
bw = 737.0678

# Track
results_path_track = os.path.abspath(project_path +
                                  "/Results/case_Gait2392_track/")
grf_file_track = os.path.abspath(results_path_track + "/grf_CoP.sto")
T1_header, T1_labels, T1_data = readExternalLoadsFile(grf_file_track)

T1_data = np.asarray(T1_data)
time_array = T1_data[:, 0]

filter_window = 13
filter_deg = 3
grf_fx_track = savgol_filter(T1_data[:, 7] / bw, filter_window,
                       filter_deg)
grf_fy_track = savgol_filter(T1_data[:, 8] / bw, filter_window,
                       filter_deg)
grf_fz_track = savgol_filter(T1_data[:, 9] / bw, filter_window,
                       filter_deg)

grf_mx_track = savgol_filter(T1_data[:, 16] / bw, filter_window,
                       filter_deg)
grf_my_track = savgol_filter(T1_data[:, 17] / bw, filter_window,
                       filter_deg)
grf_mz_track = savgol_filter(T1_data[:, 18] / bw, filter_window,
                       filter_deg)

# -----------------------------------------------------------
#  Flat-foot
# -----------------------------------------------------------
results_path_flat = os.path.abspath(project_path +
                               "/Results/case_Gait2392_track_flat-foot/")

# GRF

grf_file_flat = os.path.abspath(results_path_flat + "/grf_CoP.sto")
T2_header, T2_labels, T2_data = readExternalLoadsFile(grf_file_flat)

T2_data = np.asarray(T2_data)
time_array = T2_data[:, 0]

grf_fx_flat = savgol_filter(T2_data[:, 7] / bw, filter_window,
                       filter_deg)
grf_fy_flat = savgol_filter(T2_data[:, 8] / bw, filter_window,
                       filter_deg)
grf_fz_flat = savgol_filter(T2_data[:, 9] / bw, filter_window,
                       filter_deg)

grf_mx_flat = savgol_filter(T2_data[:, 16] / bw, filter_window,
                       filter_deg)
grf_my_flat = savgol_filter(T2_data[:, 17] / bw, filter_window,
                       filter_deg)
grf_mz_flat = savgol_filter(T2_data[:, 18] / bw, filter_window,
                       filter_deg)

fig1, axs1 = plt.subplots(2, 3, figsize=(15, 15))
fig1.suptitle('Flat-foot')

axs1[0, 0].plot(time_array, grf_fx_track, label='Fore-foot')
axs1[0, 0].plot(time_array, grf_fx_flat, label='Flat-foot')
axs1[0, 0].set_ylabel('Fx (N/BW)', fontsize=16)
axs1[0, 0].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 0].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 0].legend(loc="upper right")

axs1[0, 1].plot(time_array, grf_fy_track, label='Fore-foot')
axs1[0, 1].plot(time_array, grf_fy_flat, label='Flat-foot')
axs1[0, 1].set_title('GRF', fontsize=24, fontweight='bold')
axs1[0, 1].set_ylabel('Fy (N/BW)', fontsize=16)
axs1[0, 1].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 1].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 1].legend(loc="upper right")

axs1[0, 2].plot(time_array, grf_fz_track, label='Fore-foot')
axs1[0, 2].plot(time_array, grf_fz_flat, label='Flat-foot')
axs1[0, 2].set_ylabel('Fz (N/BW)', fontsize=16)
axs1[0, 2].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 2].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 2].legend(loc="upper right")

axs1[1, 0].plot(time_array, grf_mx_track, label='Fore-foot')
axs1[1, 0].plot(time_array, grf_mx_flat, label='Flat-foot')
axs1[1, 0].set_ylabel('Mx (Nm/BW)', fontsize=16)
axs1[1, 0].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 0].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 0].legend(loc="upper right")

axs1[1, 1].plot(time_array, grf_my_track, label='Fore-foot')
axs1[1, 1].plot(time_array, grf_my_flat, label='Flat-foot')
axs1[1, 1].set_title('GRF', fontsize=24, fontweight='bold')
axs1[1, 1].set_ylabel('My (Nm/BW)', fontsize=16)
axs1[1, 1].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 1].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 1].legend(loc="upper right")

axs1[1, 2].plot(time_array, grf_mz_track, label='Fore-foot')
axs1[1, 2].plot(time_array, grf_mz_flat, label='Flat-foot')
axs1[1, 2].set_ylabel('Mz (Nm/BW)', fontsize=16)
axs1[1, 2].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 2].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 2].legend(loc="upper right")

plt.savefig(results_path + '/GRF_flat.png')


# -----------------------------------------------------------
#  JRA
# -----------------------------------------------------------
# Track

jr_file_track = os.path.abspath(results_path_track + '/JR_in_Child_2392.sto')
T1_header, T1_labels, T1_data = readExternalLoadsFile(jr_file_track)

T1_data = np.asarray(T1_data)
time_array = T1_data[:, 0]

jr_fx_track = savgol_filter(T1_data[:, 4] / bw, filter_window,
                       filter_deg)
jr_fy_track = savgol_filter(T1_data[:, 5] / bw, filter_window,
                       filter_deg)
jr_fz_track = savgol_filter(T1_data[:, 6] / bw, filter_window,
                       filter_deg)

jr_mx_track = savgol_filter(T1_data[:, 1] / bw, filter_window,
                       filter_deg)
jr_my_track = savgol_filter(T1_data[:, 2] / bw, filter_window,
                       filter_deg)
jr_mz_track = savgol_filter(T1_data[:, 3] / bw, filter_window,
                       filter_deg)

# Flat-foot

jr_file_flat = os.path.abspath(results_path_flat + '/JR_in_Child_2392.sto')
T2_header, T2_labels, T2_data = readExternalLoadsFile(jr_file_flat)

T2_data = np.asarray(T2_data)
time_array = T2_data[:, 0]

jr_fx_flat = savgol_filter(T2_data[:, 4] / bw, filter_window,
                       filter_deg)
jr_fy_flat = savgol_filter(T2_data[:, 5] / bw, filter_window,
                       filter_deg)
jr_fz_flat = savgol_filter(T2_data[:, 6] / bw, filter_window,
                       filter_deg)

jr_mx_flat = savgol_filter(T2_data[:, 1] / bw, filter_window,
                       filter_deg)
jr_my_flat = savgol_filter(T2_data[:, 2] / bw, filter_window,
                       filter_deg)
jr_mz_flat = savgol_filter(T2_data[:, 3] / bw, filter_window,
                       filter_deg)

fig1, axs1 = plt.subplots(2, 3, figsize=(15, 15))

axs1[0, 0].plot(time_array, jr_fx_track, label='Fore-foot')
axs1[0, 0].plot(time_array, jr_fx_flat, label='Flat-foot')
axs1[1, 0].set_ylabel('Fx (N/BW)', fontsize=16)
axs1[1, 0].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 0].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 0].legend(loc="upper right")

axs1[0, 1].plot(time_array, jr_fy_track, label='Fore-foot')
axs1[0, 1].plot(time_array, jr_fy_flat, label='Flat-foot')
axs1[0, 1].set_ylabel('FY (N/BW)', fontsize=16)
axs1[0, 1].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 1].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 1].legend(loc="upper right")

axs1[0, 2].plot(time_array, jr_fz_track, label='Fore-foot')
axs1[0, 2].plot(time_array, jr_fz_flat, label='Flat-foot')
axs1[0, 2].set_ylabel('FZ (N/BW)', fontsize=16)
axs1[0, 2].set_xlabel('Time (sec)', fontsize=16)
axs1[0, 2].axvline(x=0.2, linestyle='--', color='red')
axs1[0, 2].legend(loc="upper right")

axs1[1, 0].plot(time_array, jr_mx_track, label='Fore-foot')
axs1[1, 0].plot(time_array, jr_mx_flat, label='Flat-foot')
axs1[1, 0].set_ylabel('Mx (Nm/BW)', fontsize=16)
axs1[1, 0].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 0].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 0].legend(loc="upper right")

axs1[1, 1].plot(time_array, jr_my_track, label='Fore-foot')
axs1[1, 1].plot(time_array, jr_my_flat, label='Flat-foot')
axs1[1, 1].set_ylabel('My (Nm/BW)', fontsize=16)
axs1[1, 1].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 1].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 1].legend(loc="upper right")

axs1[1, 2].plot(time_array, jr_mz_track, label='Fore-foot')
axs1[1, 2].plot(time_array, jr_mz_flat, label='Flat-foot')
axs1[1, 2].set_ylabel('Mz (Nm/BW)', fontsize=16)
axs1[1, 2].set_xlabel('Time (sec)', fontsize=16)
axs1[1, 2].axvline(x=0.2, linestyle='--', color='red')
axs1[1, 2].legend(loc="upper right")

plt.savefig(results_path + '/JRA_flat.png')
