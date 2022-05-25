import os
import opensim as osim
import numpy as np
from utils import readExternalLoadsFile
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib as mpl
from matplotlib import cm, rc
import matplotlib.gridspec as gridspec
from collections import OrderedDict
import statistics


cmaps = OrderedDict()

project_path = os.getcwd()
results_path = os.path.abspath(project_path +
                               "/Results/Compare/Trunk")
bw = 737.0678
scenarios = []
fx = []
fy = []
fz = []
mx = []
my = []
mz = []

filter_window = 13
filter_deg = 3

results_path_trunk = os.path.abspath(project_path +
                                     "/Results/Trunk/")

sub_directories = os.listdir(results_path_trunk)

my_colors = ['red', 'salmon', 'black', 'gray',
             'purple', 'orange', 'tan', 'goldenrod', 'yellow', 'olive',
             'darkseagreen', 'lightseagreen', 'seagreen']

scenarios = ['rot_-30', 'rot_-25', 'rot_-20',
             'rot_-15', 'rot_-10', 'rot_-5', 'track', 'rot_5', 'rot_10',
             'rot_15', 'rot_20', 'rot_25','rot_25']

scenarios_2 = ['rot_-30', 'rot_-25', 'rot_-20',
             'rot_-15', 'rot_-10', 'rot_-5', 'rot_0', 'rot_5', 'rot_10',
             'rot_15', 'rot_20', 'rot_25','rot_25']

# -----------------------------------------------------------
#  GRF
# -----------------------------------------------------------
for sen in scenarios:
    grf_file = os.path.abspath(results_path_trunk + '/' + sen +
                               "/grf_CoP.sto")
    T2_header, T2_labels, T2_data = readExternalLoadsFile(grf_file)
    T2_data = np.asarray(T2_data)
    time_array = T2_data[:, 0]

    grf_fx = savgol_filter(T2_data[:, 7] / bw, filter_window,
                           filter_deg)
    grf_fy = savgol_filter(T2_data[:, 8] / bw, filter_window,
                           filter_deg)
    grf_fz = savgol_filter(T2_data[:, 9] / bw, filter_window,
                           filter_deg)

    grf_mx = savgol_filter(T2_data[:, 16] / bw, filter_window,
                           filter_deg)
    grf_my = savgol_filter(T2_data[:, 17] / bw, filter_window,
                           filter_deg)
    grf_mz = savgol_filter(T2_data[:, 18] / bw, filter_window,
                           filter_deg)

    fx.append(grf_fx)
    fy.append(grf_fy)
    fz.append(grf_fz)
    mx.append(grf_mx)
    my.append(grf_my)
    mz.append(grf_mz)

fx = np.asarray(fx).T
fy = np.asarray(fy).T
fz = np.asarray(fz).T
mx = np.asarray(mx).T
my = np.asarray(my).T
mz = np.asarray(mz).T

colormap = cm.get_cmap('coolwarm', len(scenarios))
colors = colormap(np.linspace(0, 1, len(scenarios)))
nr = 2
ncol = 3

# Fx
handles = []
labels = []
fig1 = plt.figure(constrained_layout=True, figsize=(18, 8))
spec1 = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig1)
ax1 = fig1.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    line, = ax1.plot(time_array, fx[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax1.set_ylabel('Fx (N/BW)', fontsize=16)
ax1.set_xlabel('Time (sec)', fontsize=16)
ax1.grid(True)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg1.get_lines():
    line.set_linewidth(8)

# Fy
handles = []
labels = []
ax2 = fig1.add_subplot(spec1[0, 1])
for idx, fit in enumerate(scenarios):
    line, = ax2.plot(time_array, fy[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax2.set_ylabel('Fy (N/BW)', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=16)
ax2.grid(True)
ax2.set_title('GRF (lumbar extension)', fontsize=24, fontweight='bold')
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg2.get_lines():
    line.set_linewidth(8)

# Fz
handles = []
labels = []
ax3 = fig1.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    line, = ax3.plot(time_array, fz[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax3.set_ylabel('Fz (N/BW)', fontsize=16)
ax3.set_xlabel('Time (sec)', fontsize=16)
ax3.grid(True)
lg3 = ax3.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg3.get_lines():
    line.set_linewidth(8)

# Mx
handles = []
labels = []
ax4 = fig1.add_subplot(spec1[1, 0])
for idx, fit in enumerate(scenarios):
    line, = ax4.plot(time_array, mx[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax4.set_ylabel('Mx (Nm/BW)', fontsize=16)
ax4.set_xlabel('Time (sec)', fontsize=16)
ax4.grid(True)
lg4 = ax4.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg4.get_lines():
    line.set_linewidth(8)

# My
handles = []
labels = []
ax5 = fig1.add_subplot(spec1[1, 1])
for idx, fit in enumerate(scenarios):
    line, = ax5.plot(time_array, my[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax5.set_ylabel('My (Nm/BW)', fontsize=16)
ax5.set_xlabel('Time (sec)', fontsize=16)
ax5.grid(True)
lg5 = ax5.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg5.get_lines():
    line.set_linewidth(8)

# Mz
handles = []
labels = []
ax6 = fig1.add_subplot(spec1[1, 2])
for idx, fit in enumerate(scenarios):
    line, = ax6.plot(time_array, mz[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax6.set_ylabel('Mz (Nm/BW)', fontsize=16)
ax6.set_xlabel('Time (sec)', fontsize=16)
ax6.grid(True)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg6.get_lines():
    line.set_linewidth(8)


plt.savefig(results_path + '/GRF_ext.png')
plt.show()



# -----------------------------------------------------------
#  JRA
# -----------------------------------------------------------
fx = []
fy = []
fz = []
mx = []
my = []
mz = []
for sen in scenarios:
    jra_file = os.path.abspath(results_path_trunk + '/' + sen +
                               "/JR_in_Child_2392.sto")
    T2_header, T2_labels, T2_data = readExternalLoadsFile(jra_file)
    T2_data = np.asarray(T2_data)
    time_array = T2_data[:, 0]

    jr_fx = savgol_filter(T2_data[:, 4] / bw, filter_window,
                           filter_deg)
    jr_fy = savgol_filter(T2_data[:, 5] / bw, filter_window,
                           filter_deg)
    jr_fz = savgol_filter(T2_data[:, 6] / bw, filter_window,
                           filter_deg)

    jr_mx = savgol_filter(T2_data[:, 1] / bw, filter_window,
                           filter_deg)
    jr_my = savgol_filter(T2_data[:, 2] / bw, filter_window,
                           filter_deg)
    jr_mz = savgol_filter(T2_data[:, 3] / bw, filter_window,
                           filter_deg)

    fx.append(jr_fx)
    fy.append(jr_fy)
    fz.append(jr_fz)
    mx.append(jr_mx)
    my.append(jr_my)
    mz.append(jr_mz)

fx = np.asarray(fx).T
fy = np.asarray(fy).T
fz = np.asarray(fz).T
mx = np.asarray(mx).T
my = np.asarray(my).T
mz = np.asarray(mz).T

# Fx
handles = []
labels = []
fig1 = plt.figure(constrained_layout=True, figsize=(18, 8))
spec1 = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig1)
ax1 = fig1.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    line, = ax1.plot(time_array, fx[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax1.set_ylabel('Fx (N/BW)', fontsize=16)
ax1.set_xlabel('Time (sec)', fontsize=16)
ax1.grid(True)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg1.get_lines():
    line.set_linewidth(8)

# Fy
handles = []
labels = []
ax2 = fig1.add_subplot(spec1[0, 1])
for idx, fit in enumerate(scenarios):
    line, = ax2.plot(time_array, fy[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax2.set_ylabel('Fy (N/BW)', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=16)
ax2.grid(True)
ax2.set_title('JRA (lumbar extension)', fontsize=24, fontweight='bold')
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg2.get_lines():
    line.set_linewidth(8)

# Fz
handles = []
labels = []
ax3 = fig1.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    line, = ax3.plot(time_array, fz[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax3.set_ylabel('Fz (N/BW)', fontsize=16)
ax3.set_xlabel('Time (sec)', fontsize=16)
ax3.grid(True)
lg3 = ax3.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg3.get_lines():
    line.set_linewidth(8)

# Mx
handles = []
labels = []
ax4 = fig1.add_subplot(spec1[1, 0])
for idx, fit in enumerate(scenarios):
    line, = ax4.plot(time_array, mx[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax4.set_ylabel('Mx (Nm/BW)', fontsize=16)
ax4.set_xlabel('Time (sec)', fontsize=16)
ax4.grid(True)
lg4 = ax4.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg4.get_lines():
    line.set_linewidth(8)

# My
handles = []
labels = []
ax5 = fig1.add_subplot(spec1[1, 1])
for idx, fit in enumerate(scenarios):
    line, = ax5.plot(time_array, my[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax5.set_ylabel('My (Nm/BW)', fontsize=16)
ax5.set_xlabel('Time (sec)', fontsize=16)
ax5.grid(True)
lg5 = ax5.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg5.get_lines():
    line.set_linewidth(8)

# Mz
handles = []
labels = []
ax6 = fig1.add_subplot(spec1[1, 2])
for idx, fit in enumerate(scenarios):
    line, = ax6.plot(time_array, mz[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax6.set_ylabel('Mz (Nm/BW)', fontsize=16)
ax6.set_xlabel('Time (sec)', fontsize=16)
ax6.grid(True)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg6.get_lines():
    line.set_linewidth(8)

plt.savefig(results_path + '/JRA_ext.png')
plt.show()
print('ok')

dev_fx = np.std(fx, axis=0)
mean_fx = np.mean(fx, axis=0)

sort_fx = np.sort(dev_fx)
idx = np.argsort(dev_fx)


# Posterior GRF deviation
print('Deviation of posterior GRF or each')
for i, dev in enumerate(dev_fx):
    print(str(scenarios_2[idx[i]]) + " : " + str(dev))

# -----------------------------------------------------------
#  Quadriceps force
# -----------------------------------------------------------
handles = []
labels = []
fig7 = plt.figure(constrained_layout=True, figsize=(18, 8))
spec7 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig7)
ax7 = fig7.add_subplot(spec1[0, 0])
mean_quad = []
for idx, sen in enumerate(scenarios):
    force_file = os.path.abspath(results_path_trunk + '/' + sen +
                                 "/tendon_forces.sto")
    header, labels, data = readExternalLoadsFile(force_file)
    data = np.asarray(data)
    time_array = data[:, 0]

    for idx_2, label in enumerate(labels):
        # quadriceps
        if label == '/forceset/rect_fem_l|tendon_force':
            quad_force = data[:, idx_2]
        elif label == '/forceset/vas_med_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
        elif label == '/forceset/vas_int_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
        elif label == '/forceset/vas_lat_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
    mean_quad.append(quad_force)
    line, = ax7.plot(time_array, quad_force, color=my_colors[idx])
    handles.append(line)
    labels.append(scenarios_2[idx])
ax7.set_ylabel('Activation', fontsize=16)
ax7.set_xlabel('Time (sec)', fontsize=16)
ax7.set_title('Quadriceps activation', fontsize=24, fontweight='bold')
ax7.grid(True)
lg7 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg7.get_lines():
     line.set_linewidth(8)

plt.savefig(results_path + '/Quad_force.png')
plt.show()


# Mean Quadriceps activations
mean = np.mean(mean_quad, axis=1)
sort_mean = np.sort(mean)
idx_mean = np.argsort(mean)
print('\n Mean Quadriceps activations')
for i, my_mean in enumerate(sort_mean):
    print(str(scenarios_2[idx_mean[i]]) + " : " + str(my_mean))