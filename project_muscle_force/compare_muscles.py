import os
import numpy as np
from utils import readExternalLoadsFile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
from scipy.signal import savgol_filter
from matplotlib import rc, rcParams
from matplotlib.patches import Rectangle

mls = 'solid'
c1 = 'r'
c2 = 'g'
a = 0.5

plots = 1
# plots = 0

window = 9
order = 3

in_ind = 80
fin_ind = 160
project_path = os.getcwd()


results_path = os.path.abspath(project_path +
                               "/Results/Compare/")
if not os.path.isdir(results_path):
    os.makedirs(results_path)
results_path_trunk = os.path.abspath(project_path +
                                     "/Results/")

# Internal
name = 'predict_muscles'

scenarios = ['strong_quad_strong_ham',
              'strong_quad_normal_ham',
              'strong_quad_weak_ham',
              'strong_ham_normal_quad',
              'normal',
              'weak_ham_normal_quad',
              'weak_quad_strong_ham',
              'weak_quad_normal_ham',
              'weak_quad_weak_ham'
              ]

labels_sen = ['sq_sh',
              'sq_nh',
              'sq_wh',
              'nq_sh',
              'nq_nh',
              'nq_wh',
              'wq_sh',
              'wq_nh',
              'wh_wq'
              ]


my_colors = ['royalblue', 'cadetblue', 'mediumpurple','darkkhaki', 'peru',
             'seagreen' , 'darkred', 'darkslategrey', 'black']

bw = 737.0678
fx = []
fy = []
fz = []
mx = []
my = []
mz = []
time_arrays = []
max_fy_id = []
idx_igc = []
offset = []

# -----------------------------------------------------------
#  GRF
# -----------------------------------------------------------
for sen in scenarios:
    grf_file = os.path.abspath(results_path_trunk + '/' + sen +
                               "/grf_CoP.sto")
    T2_header, T2_labels, T2_data = readExternalLoadsFile(grf_file)
    T2_data = np.asarray(T2_data)
    time_array = T2_data[:, 0]

    grf_fx = T2_data[in_ind:fin_ind, 7] / bw
    grf_fy = T2_data[in_ind:fin_ind, 8] / bw
    grf_fz = T2_data[in_ind:fin_ind, 9] / bw
    grf_mx = T2_data[in_ind:fin_ind, 16] / bw
    grf_my = T2_data[in_ind:fin_ind, 17] / bw
    grf_mz = T2_data[in_ind:fin_ind, 18] / bw

    fx.append(grf_fx)
    fy.append(grf_fy)
    fz.append(grf_fz)
    mx.append(grf_mx)
    my.append(grf_my)
    mz.append(grf_mz)

    time_array = time_array[in_ind:fin_ind]
    time_arrays.append(time_array)

    igc = grf_fy[grf_fy > 0]
    idx_igc_sen = np.asarray(np.nonzero(grf_fy > 0))
    idx_igc.append(idx_igc_sen[0][0])

max_fy_id = np.argmax(fy,axis=1)
max_fy_id[3] = 55
max_fy_id[4] = 55
max_fy_id[5] = 55

fx = np.asarray(fx).T
fy = np.asarray(fy).T
fz = np.asarray(fz).T
mx = np.asarray(mx).T
my = np.asarray(my).T
mz = np.asarray(mz).T
time_arrays = np.asarray(time_arrays).T

nr = 2
ncol = 3
rc('font', weight='bold')
rc('lines', linewidth=1.5)
# Fx
handles = []
labels = []
fig1 = plt.figure(constrained_layout=True, figsize=(18, 9))
fig1.suptitle('Drop - Landing: Muscle Forces case study',
              fontsize=26, fontweight='bold')
spec1 = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig1)
ax1 = fig1.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax1.plot(time_arrays[:,0],savgol_filter(fx[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                      marker='x',markevery=marker_on)
    # line, = ax1.plot(time_arrays[:,0],fx[:, idx], color=my_colors[idx],
    #                   marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax1.set_ylabel('Fx (pGRF)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
ax1.grid(True)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg1.get_lines():
    line.set_linewidth(8)

# Fy
handles = []
labels = []
ax2 = fig1.add_subplot(spec1[0, 1])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax2.plot(time_arrays[:,0], fy[:, idx], color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax2.set_ylabel('Fy (vGRF)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax2.grid(True)
ax2.set_title('Ground Reaction Forces and Moments (/BW)\n',
              fontsize=20, fontweight='bold')
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
extra = Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False, edgecolor='none',
                  linewidth=0)
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg2.get_lines():
    line.set_linewidth(8)

# Fz
handles = []
labels = []
ax3 = fig1.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax3.plot(time_arrays[:,0], savgol_filter(fz[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x', markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax3.set_ylabel('Fz (mGRF)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax3.xaxis.set_tick_params(labelsize=12)
ax3.yaxis.set_tick_params(labelsize=12)
ax3.grid(True)
lg3 = ax3.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='lower left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg3.get_lines():
    line.set_linewidth(8)

# Mx
handles = []
labels = []
ax4 = fig1.add_subplot(spec1[1, 0])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax4.plot(time_arrays[:,0],savgol_filter(mx[:, idx], window,
                      order,mode='nearest'),  color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax4.set_ylabel('Mx', fontsize=16, fontweight='bold')
ax4.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax4.xaxis.set_tick_params(labelsize=12)
ax4.yaxis.set_tick_params(labelsize=12)
ax4.grid(True)
lg4 = ax4.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg4.get_lines():
    line.set_linewidth(8)

# My
handles = []
labels = []
ax5 = fig1.add_subplot(spec1[1, 1])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax5.plot(time_arrays[:,0], savgol_filter(my[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax5.set_ylabel('My ', fontsize=16, fontweight='bold')
ax5.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax5.xaxis.set_tick_params(labelsize=12)
ax5.yaxis.set_tick_params(labelsize=12)
ax5.grid(True)
lg5 = ax5.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg5.get_lines():
    line.set_linewidth(8)

# Mz
handles = []
labels = []
ax6 = fig1.add_subplot(spec1[1, 2])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax6.plot(time_arrays[:,0], savgol_filter(mz[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax6.set_ylabel('Mz', fontsize=16, fontweight='bold')
ax6.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax6.xaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
ax6.grid(True)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg6.get_lines():
    line.set_linewidth(8)
plt.savefig(results_path + '/' + name + '_GRF.png',format='png')

# -----------------------------------------------------------
#  JRA
# -----------------------------------------------------------
fx_jr = []
fy_jr = []
fz_jr = []
mx_jr = []
my_jr = []
mz_jr = []
for sen in scenarios:
    jra_file = os.path.abspath(results_path_trunk + '/' + sen +
                               "/JR_in_Child_2392.sto")
    T2_header, T2_labels, T2_data = readExternalLoadsFile(jra_file)
    T2_data = np.asarray(T2_data)
    time_array = T2_data[in_ind:fin_ind, 0]

    jr_fx = T2_data[in_ind:fin_ind, 4] / bw
    jr_fy = T2_data[in_ind:fin_ind, 5] / bw
    jr_fz = T2_data[in_ind:fin_ind, 6] / bw
    jr_mx = T2_data[in_ind:fin_ind, 1] / bw
    jr_my = T2_data[in_ind:fin_ind, 2] / bw
    jr_mz = T2_data[in_ind:fin_ind, 3] / bw

    fx_jr.append(jr_fx)
    fy_jr.append(jr_fy)
    fz_jr.append(jr_fz)
    mx_jr.append(jr_mx)
    my_jr.append(jr_my)
    mz_jr.append(jr_mz)

fx_jr = np.asarray(fx_jr).T
fy_jr = np.asarray(fy_jr).T
fz_jr = np.asarray(fz_jr).T
mx_jr = np.asarray(mx_jr).T
my_jr = np.asarray(my_jr).T
mz_jr = np.asarray(mz_jr).T

# Fx
handles = []
labels = []
fig1 = plt.figure(constrained_layout=True, figsize=(18, 9))
spec1 = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig1)
fig1.suptitle('Drop - Landing: Muscle Forces case study',
              fontsize=26, fontweight='bold')
ax1 = fig1.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax1.plot(time_arrays[:,0], savgol_filter(fx_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax1.set_ylabel('Anterior force(+)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax1.grid(True)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg1.get_lines():
    line.set_linewidth(8)

# Fy
handles = []
labels = []
ax2 = fig1.add_subplot(spec1[0, 1])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax2.plot(time_arrays[:,0], savgol_filter(fy_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax2.set_ylabel('Compressive force(-)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax2.grid(True)
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
ax2.set_title('Knee Joint Reaction Forces and Moments (/BW)\n',
              fontsize=20,
              fontweight='bold')
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='lower left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg2.get_lines():
    line.set_linewidth(8)

# Fz
handles = []
labels = []
ax3 = fig1.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax3.plot(time_arrays[:,0], savgol_filter(fz_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax3.set_ylabel('Medial force(+)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax3.grid(True)
ax3.xaxis.set_tick_params(labelsize=12)
ax3.yaxis.set_tick_params(labelsize=12)
lg3 = ax3.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='lower right', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg3.get_lines():
    line.set_linewidth(8)

# Mx
handles = []
labels = []
ax4 = fig1.add_subplot(spec1[1, 0])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax4.plot(time_arrays[:,0], savgol_filter(mx_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax4.set_ylabel('Abduction moment(+)', fontsize=16, fontweight='bold')
ax4.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax4.grid(True)
ax4.xaxis.set_tick_params(labelsize=12)
ax4.yaxis.set_tick_params(labelsize=12)
lg4 = ax4.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='lower left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg4.get_lines():
    line.set_linewidth(8)

# My
handles = []
labels = []
ax5 = fig1.add_subplot(spec1[1, 1])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax5.plot(time_arrays[:,0], savgol_filter(my_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax5.set_ylabel('Internal rotation moment(-)', fontsize=16, fontweight='bold')
ax5.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax5.grid(True)
ax5.xaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
lg5 = ax5.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg5.get_lines():
    line.set_linewidth(8)

# Mz
handles = []
labels = []
ax6 = fig1.add_subplot(spec1[1, 2])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax6.plot(time_arrays[:,0], savgol_filter(mz_jr[:, idx], window,
                      order,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax6.set_ylabel('Flexion moment(+)', fontsize=16, fontweight='bold')
ax6.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax6.grid(True)
ax6.xaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg6.get_lines():
    line.set_linewidth(8)
plt.savefig(results_path + '/' + name + '_JRA.png')

# -----------------------------------------------------------
#  Muscle force
# -----------------------------------------------------------
nr = 2
nc = 3
handles = []
handles_2 = []
handles_3 = []
handles_4 = []
handles_5 = []
handles_6 = []
labels = []
mean_quad = []
mean_ham =[]
mean_ratio = []
time_arrays_2 = []
fig3 = plt.figure(constrained_layout=True, figsize=(18, 9))
spec1 = gridspec.GridSpec(ncols=nc, nrows=nr, figure=fig3)
fig3.suptitle('Drop - Landing: Muscle Forces case study',
              fontsize=26, fontweight='bold')
ax1 = fig3.add_subplot(spec1[0, 0])
ax2 = fig3.add_subplot(spec1[0, 1])
ax3 = fig3.add_subplot(spec1[0, 2])
ax4 = fig3.add_subplot(spec1[1, 0])
ax5 = fig3.add_subplot(spec1[1, 1])
ax6 = fig3.add_subplot(spec1[1, 2])

all_ham_force = []
all_quad_force = []
all_gastro_force = []
all_ta_force = []
all_sol_force = []

for idx, sen in enumerate(scenarios):
    force_file = os.path.abspath(results_path_trunk + '/' + sen +
                                 "/tendon_forces.sto")
    header, labels_2, data = readExternalLoadsFile(force_file)
    data = np.asarray(data)
    if idx == 0:
        time_array_2 = data[in_ind:fin_ind, 0]
        time_arrays_2 = np.asarray(time_array_2).T

    for idx_2, label in enumerate(labels_2):
        # quadriceps
        if label == '/forceset/rect_fem_l|tendon_force':
            quad_force = data[:, idx_2]
        elif label == '/forceset/vas_med_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
        elif label == '/forceset/vas_int_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
        elif label == '/forceset/vas_lat_l|tendon_force':
            quad_force = np.add(data[:, idx_2], quad_force)
        elif label == '/forceset/semimem_l|tendon_force':
            ham_force = data[:, idx_2]
        elif label == '/forceset/semiten_l|tendon_force':
            ham_force = np.add(data[:, idx_2], ham_force)
        elif label == '/forceset/bifemlh_l|tendon_force':
            ham_force = np.add(data[:, idx_2], ham_force)
        elif label == '/forceset/bifemsh_l|tendon_force':
            ham_force = np.add(data[:, idx_2], ham_force)
        elif label == '/forceset/med_gas_l|tendon_force':
            gastro_force = data[:, idx_2]
        elif label == '/forceset/lat_gas_l|tendon_force':
            gastro_force = np.add(data[:, idx_2], gastro_force)
        elif label == '/forceset/tib_ant_l|tendon_force':
            ta_force = data[:, idx_2]
        elif label == '/forceset/soleus_l|tendon_force':
            sol_force = data[:, idx_2]

    all_ham_force.append(ham_force)
    all_quad_force.append(quad_force)
    all_gastro_force.append(gastro_force)
    all_ta_force.append(ta_force)
    all_sol_force.append(sol_force)

    ham_force = ham_force[in_ind:fin_ind, ]
    quad_force = quad_force[in_ind:fin_ind, ]
    gastro_force = gastro_force[in_ind:fin_ind, ]
    ta_force = ta_force[in_ind:fin_ind, ]
    sol_force = sol_force[in_ind:fin_ind, ]

    qh_ratio = np.divide(quad_force,ham_force)
    gta_ratio = np.divide(gastro_force, ta_force)

    marker_on = [max_fy_id[idx]]
    line, = ax1.plot(time_arrays_2[:],
         savgol_filter(quad_force,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    line_2, = ax2.plot(time_arrays_2[:],
         savgol_filter(ham_force,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    line_3, = ax3.plot(time_arrays_2[:],
         savgol_filter(qh_ratio,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    line_4, = ax4.plot(time_arrays_2[:],
         savgol_filter(gastro_force,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    line_5, = ax5.plot(time_arrays_2[:],
         savgol_filter(ta_force,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    line_6, = ax6.plot(time_arrays_2[:],
         savgol_filter(gta_ratio,9,3,mode='nearest'), color=my_colors[idx],
                     marker='x',markevery=marker_on)
    # line, = ax1.plot(time_arrays_2[:], quad_force, color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    # marker_on = [max_fy_id[idx]]
    # line_2, = ax2.plot(time_arrays_2[:], ham_force,color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    # marker_on = [max_fy_id[idx]]
    # line_3, = ax3.plot(time_arrays_2[:], qh_ratio, color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    # marker_on = [max_fy_id[idx]]
    # line_4, = ax4.plot(time_arrays_2[:], gastro_force,  color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    # marker_on = [max_fy_id[idx]]
    # line_5, = ax5.plot(time_arrays_2[:], ta_force,  color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    # marker_on = [max_fy_id[idx]]
    # line_6, = ax6.plot(time_arrays_2[:], gta_ratio,  color=my_colors[idx],
    #                  marker='x',markevery=marker_on)

    handles.append(line)
    handles_2.append(line_2)
    handles_3.append(line_3)
    handles_4.append(line_4)
    handles_5.append(line_5)
    handles_6.append(line_6)
    labels.append(labels_sen[idx])

all_ham_force = np.asarray(all_ham_force).T
all_quad_force = np.asarray(all_quad_force).T
all_gastro_force = np.asarray(all_gastro_force).T
all_ta_force = np.asarray(all_ta_force).T
all_sol_force = np.asarray(all_sol_force).T

all_ham_force = all_ham_force[in_ind:fin_ind, ]
all_quad_force = all_quad_force[in_ind:fin_ind, ]
all_gastro_force = all_gastro_force[in_ind:fin_ind, ]
all_ta_force = all_ta_force[in_ind:fin_ind, ]
all_sol_force = all_sol_force[in_ind:fin_ind, ]
ax1.set_ylabel('Quadriceps force (N)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax1.grid(True)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                  loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

ax2.set_ylabel('Hamstrings force (N)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax2.set_title('Muscle forces and Muscle Force Ratios\n', fontsize=20,
              fontweight='bold')
ax2.grid(True)
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
lg2 = ax2.legend(handles_2, labels, prop={'weight': 'bold', 'size': 16},
                  loc='lower left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

ax3.set_ylabel('Q/H Ratio', fontsize=16, fontweight='bold')
ax3.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax3.grid(True)
ax3.xaxis.set_tick_params(labelsize=12)
ax3.yaxis.set_tick_params(labelsize=12)
lg3 = ax3.legend(handles_3, labels, prop={'weight': 'bold', 'size': 16},
                  loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

ax4.set_ylabel('Gastrocnemius force (N)', fontsize=16, fontweight='bold')
ax4.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax4.grid(True)
ax4.xaxis.set_tick_params(labelsize=12)
ax4.yaxis.set_tick_params(labelsize=12)
lg4 = ax4.legend(handles_4, labels, prop={'weight': 'bold', 'size': 16},
                  loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

ax5.set_ylabel('Tibialis anterior force (N)', fontsize=16, fontweight='bold')
ax5.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax5.grid(True)
ax5.xaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
lg5 = ax5.legend(handles_5, labels, prop={'weight': 'bold', 'size': 16},
                  loc='upper right', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

ax6.set_ylabel('GTA Ratio', fontsize=16, fontweight='bold')
ax6.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax6.grid(True)
ax6.xaxis.set_tick_params(labelsize=12)
ax6.yaxis.set_tick_params(labelsize=12)
lg6 = ax6.legend(handles_6, labels, prop={'weight': 'bold', 'size': 16},
                  loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")

for line in lg1.get_lines():
     line.set_linewidth(8)
for line in lg2.get_lines():
     line.set_linewidth(8)
for line in lg3.get_lines():
     line.set_linewidth(8)
for line in lg4.get_lines():
     line.set_linewidth(8)
for line in lg5.get_lines():
     line.set_linewidth(8)
for line in lg6.get_lines():
     line.set_linewidth(8)

plt.savefig(results_path + '/' + name + '_muscle_forces.png')

d = {}
max_vGRF = []
for idx,sen in enumerate(scenarios):
    max_fx = fx[max_fy_id[idx], idx]
    max_fy = fy[max_fy_id[idx], idx]
    max_fz = fz[max_fy_id[idx], idx]
    max_mx = mx[max_fy_id[idx], idx]
    max_my = my[max_fy_id[idx], idx]
    max_mz = mz[max_fy_id[idx], idx]

    max_fx_jr = fx_jr[max_fy_id[idx], idx]
    max_fy_jr = fy_jr[max_fy_id[idx], idx]
    max_fz_jr = fz_jr[max_fy_id[idx], idx]
    max_mx_jr = mx_jr[max_fy_id[idx], idx]
    max_my_jr = my_jr[max_fy_id[idx], idx]
    max_mz_jr = mz_jr[max_fy_id[idx], idx]

    max_quad_force = all_quad_force[max_fy_id[idx], idx]
    max_ham_force = all_ham_force[max_fy_id[idx], idx]
    max_gastro_force = all_gastro_force[max_fy_id[idx], idx]
    max_ta_force = all_ta_force[max_fy_id[idx], idx]
    max_sol_force = all_sol_force[max_fy_id[idx], idx]

    dict_i = {'scenario': str(sen),
              'fx': str(format(max_fx,".3f")),
              'max_fy': str(format(max_fy,".3f")),
              'fz': str(format(max_fz,".3f")),
              'qh_ratio': str(format(max_quad_force / max_ham_force, ".3f")),
              'gta_ratio': str(format(max_gastro_force / max_ta_force, ".3f")),

              'fx_jr': str(format(max_fx_jr,".3f")),
              'fy_jr': str(format(max_fy_jr,".3f")),
              'fz_jr': str(format(max_fz_jr,".3f")),
              'mx_jr': str(format(max_mx_jr,".3f")),
              'my_jr': str(format(max_my_jr,".3f")),
              'mz_jr': str(format(max_mz_jr,".3f"))
              }
    # d = [d, dict_i]
    with open(results_path + '/' + name + '_output.csv',
              'a') as f:
        w = csv.DictWriter(f, dict_i.keys())
        w.writeheader()
        # f.write("\n")
        w.writerow(dict_i)


# Hip, knee , ankle angles
knee_angles = []
hip_flexions = []
sol_time_arrays = []
hip_adds = []
hip_rots = []
ankle_angles = []
for sen in scenarios:
    sol_file = os.path.abspath(results_path_trunk + '/' + sen +
                               "/solution.sto")
    sol_header, sol_labels, sol_data = readExternalLoadsFile(sol_file)
    sol_data = np.asarray(sol_data)
    sol_time_array = sol_data[in_ind:fin_ind, 0]

    for idx_2, label in enumerate(sol_labels):
        if label == '/jointset/knee_l/knee_angle_l/value':
            knee_angle = sol_data[in_ind:fin_ind, idx_2]
        elif label == '/jointset/hip_l/hip_flexion_l/value':
            hip_flexion = sol_data[in_ind:fin_ind, idx_2]
        elif label == '/jointset/ankle_l/ankle_angle_l/value':
            ankle_angle = sol_data[in_ind:fin_ind, idx_2]
        elif label == '/jointset/hip_l/hip_adduction_l/value':
            hip_add = sol_data[in_ind:fin_ind, idx_2]
        elif label == '/jointset/hip_l/hip_rotation_l/value':
            hip_rot = sol_data[in_ind:fin_ind, idx_2]

    knee_angles.append(knee_angle)
    hip_flexions.append(hip_flexion)
    ankle_angles.append(ankle_angle)
    hip_adds.append(hip_add)
    hip_rots.append(hip_rot)
    sol_time_arrays.append(sol_time_array)

knee_angles = np.rad2deg(np.asarray(knee_angles).T)
hip_flexions = np.rad2deg(np.asarray(hip_flexions).T)
ankle_angles = np.rad2deg(np.asarray(ankle_angles).T)
hip_adds = np.rad2deg(np.asarray(hip_adds).T)
hip_rots = np.rad2deg(np.asarray(hip_rots).T)
sol_time_arrays = np.asarray(sol_time_arrays).T

nr = 1
ncol = 3

# Hip flexion
handles = []
labels = []
fig_sol = plt.figure(constrained_layout=True, figsize=(18, 9))
mid = (fig_sol.subplotpars.right + fig_sol.subplotpars.left + 0.015)/2
fig_sol.suptitle('Drop - Landing: Muscle Forces case study',
              fontsize=26, fontweight='bold')
spec_sol = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig_sol)
ax1 = fig_sol.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    # line, = ax1.plot(sol_time_arrays[:,0], savgol_filter(hip_flexions[:,
    #                                                         idx],
    #         9,3,mode='nearest'), color=my_colors[idx],
    #                  marker='x',markevery=marker_on)
    line, = ax1.plot(sol_time_arrays[:,0], hip_flexions[:,idx],
             color=my_colors[idx], marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])

ax1.set_ylabel('Hip flexion(+)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax1.grid(True)
ax1.xaxis.set_tick_params(labelsize=12)
ax1.yaxis.set_tick_params(labelsize=12)
time_maxfy = sol_time_arrays[max_fy_id[0],0]
time_igc = sol_time_arrays[idx_igc[0], 0]
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg1.get_lines():
    line.set_linewidth(8)

# knee angle
handles = []
labels = []
ax2 = fig_sol.add_subplot(spec1[0, 1])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax2.plot(sol_time_arrays[:,0], knee_angles[:, idx],
             color=my_colors[idx], marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax2.set_ylabel('Knee flexion(-)', fontsize=16, fontweight='bold')
ax2.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax2.grid(True)
ax2.xaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
ax2.set_title('Hip, knee and ankle angles\n',
              fontsize=20, fontweight='bold')
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='lower left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg2.get_lines():
    line.set_linewidth(8)

# Ankle angle
handles = []
labels = []
ax3 = fig_sol.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    marker_on = [max_fy_id[idx]]
    line, = ax3.plot(sol_time_arrays[:,0], ankle_angles[:, idx],
            color=my_colors[idx], marker='x',markevery=marker_on)
    handles.append(line)
    labels.append(labels_sen[idx])
ax3.set_ylabel('Ankle flexion(+)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Time (sec)', fontsize=16, fontweight='bold')
ax3.grid(True)
ax3.xaxis.set_tick_params(labelsize=12)
ax3.yaxis.set_tick_params(labelsize=12)
lg3 = ax3.legend(handles, labels, prop={'weight': 'bold', 'size': 16},
                 loc='upper left', handletextpad=0.5, handlelength=0.5,
                 title = "x: max vGRF")
for line in lg3.get_lines():
    line.set_linewidth(8)

plt.savefig(results_path + '/' + name + '_sol.png')



