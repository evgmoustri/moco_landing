import os
import numpy as np
from utils import readExternalLoadsFile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

plots = 1
# plots = 0

in_ind = 100
fin_ind = 150
project_path = os.getcwd()

name = 'predict_height_effort0.001'
results_path = os.path.abspath(project_path +
                               "/Results/Compare/predict_height_effort0.001/")
if not os.path.isdir(results_path):
    os.makedirs(results_path)

results_path_trunk = os.path.abspath(project_path +
                                     "/Results/predict_height_effort0.001")

scenarios = ['h30', 'h35', 'h40', 'h45', 'h55']

labels_sen = ['h30', 'h35', 'h40', 'h45,50', 'h55']

my_colors = ['royalblue',  'mediumpurple','cadetblue','palevioletred',
             'darkslategrey', 'darkred']


bw = 737.0678
fx = []
fy = []
fz = []
mx = []
my = []
mz = []
max_fy_id = []

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

    max_fy_id = np.argmax(fy,axis=1)

time_array = time_array[in_ind:fin_ind]

fx = np.asarray(fx).T
fy = np.asarray(fy).T
fz = np.asarray(fz).T
mx = np.asarray(mx).T
my = np.asarray(my).T
mz = np.asarray(mz).T



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
    labels.append(labels_sen[idx])
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
    labels.append(labels_sen[idx])
ax2.set_ylabel('Fy (N/BW)', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=16)
ax2.grid(True)
ax2.set_title('Ground Reaction Forces and Moments', fontsize=24,
              fontweight='bold')
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
    labels.append(labels_sen[idx])
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
    labels.append(labels_sen[idx])
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
    labels.append(labels_sen[idx])
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
    labels.append(labels_sen[idx])
ax6.set_ylabel('Mz (Nm/BW)', fontsize=16)
ax6.set_xlabel('Time (sec)', fontsize=16)
ax6.grid(True)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg6.get_lines():
    line.set_linewidth(8)


plt.savefig(results_path + '/' + name + '_GRF_ext.png')
if plots == 1:
    plt.show()

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
fig1 = plt.figure(constrained_layout=True, figsize=(18, 8))
spec1 = gridspec.GridSpec(ncols=ncol, nrows=nr, figure=fig1)
ax1 = fig1.add_subplot(spec1[0, 0])
for idx, fit in enumerate(scenarios):
    line, = ax1.plot(time_array, fx_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
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
    line, = ax2.plot(time_array, fy_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
ax2.set_ylabel('Fy (N/BW)', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=16)
ax2.grid(True)
ax2.set_title('Knee Joint Reaction Forces and Moments', fontsize=24,
              fontweight='bold')
lg2 = ax2.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg2.get_lines():
    line.set_linewidth(8)

# Fz
handles = []
labels = []
ax3 = fig1.add_subplot(spec1[0, 2])
for idx, fit in enumerate(scenarios):
    line, = ax3.plot(time_array, fz_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
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
    line, = ax4.plot(time_array, mx_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
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
    line, = ax5.plot(time_array, my_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
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
    line, = ax6.plot(time_array, mz_jr[:, idx], color=my_colors[idx])
    handles.append(line)
    labels.append(labels_sen[idx])
ax6.set_ylabel('Mz (Nm/BW)', fontsize=16)
ax6.set_xlabel('Time (sec)', fontsize=16)
ax6.grid(True)
lg6 = ax6.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                 loc='upper left', handletextpad=0.4, handlelength=0.5)
for line in lg6.get_lines():
    line.set_linewidth(8)

plt.savefig(results_path + '/' + name + '_JRA_ext.png')
if plots == 1:
    plt.show()


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

fig3 = plt.figure(constrained_layout=True, figsize=(18, 8))
spec1 = gridspec.GridSpec(ncols=nc, nrows=nr, figure=fig3)
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
    time_array = data[:, 0]

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

    ham_force = ham_force[in_ind:fin_ind,]
    quad_force = quad_force[in_ind:fin_ind,]
    gastro_force = gastro_force[in_ind:fin_ind, ]
    ta_force = ta_force[in_ind:fin_ind, ]
    sol_force = sol_force[in_ind:fin_ind, ]

    qh_ratio = np.divide(quad_force,ham_force)
    gta_ratio = np.divide(gastro_force, ta_force)

    line, = ax1.plot(time_array[in_ind:fin_ind,], quad_force,
                     color=my_colors[idx])
    line_2, = ax2.plot(time_array[in_ind:fin_ind,], ham_force,
                       color=my_colors[idx])
    line_3, = ax3.plot(time_array[in_ind:fin_ind,], qh_ratio,
                       color=my_colors[idx])
    line_4, = ax4.plot(time_array[in_ind:fin_ind, ], gastro_force,
                       color=my_colors[idx])
    line_5, = ax5.plot(time_array[in_ind:fin_ind, ], ta_force,
                       color=my_colors[idx])
    line_6, = ax6.plot(time_array[in_ind:fin_ind, ], gta_ratio,
                       color=my_colors[idx])
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

ax1.set_ylabel('Force', fontsize=16)
ax1.set_xlabel('Time (sec)', fontsize=16)
ax1.set_title('Quadriceps force', fontsize=24, fontweight='bold')
ax1.grid(True)
lg1 = ax1.legend(handles, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

ax2.set_ylabel('Force', fontsize=16)
ax2.set_xlabel('Time (sec)', fontsize=16)
ax2.set_title('Hamstrings force', fontsize=24, fontweight='bold')
ax2.grid(True)
lg2 = ax2.legend(handles_2, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

ax3.set_ylabel('Ratio', fontsize=16)
ax3.set_xlabel('Time (sec)', fontsize=16)
ax3.set_title('Q/H RATIO', fontsize=24, fontweight='bold')
ax3.grid(True)
lg3 = ax3.legend(handles_3, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

ax4.set_ylabel('Ratio', fontsize=16)
ax4.set_xlabel('Time (sec)', fontsize=16)
ax4.set_title('Gastrocnimius force', fontsize=24, fontweight='bold')
ax4.grid(True)
lg4 = ax4.legend(handles_4, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

ax5.set_ylabel('Ratio', fontsize=16)
ax5.set_xlabel('Time (sec)', fontsize=16)
ax5.set_title('Tibialis anterior force', fontsize=24, fontweight='bold')
ax5.grid(True)
lg5 = ax5.legend(handles_5, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

ax6.set_ylabel('Ratio', fontsize=16)
ax6.set_xlabel('Time (sec)', fontsize=16)
ax6.set_title('GTA RATIO', fontsize=24, fontweight='bold')
ax6.grid(True)
lg6 = ax6.legend(handles_6, labels, prop={'weight': 'bold', 'size': 10},
                  loc='upper left', handletextpad=0.4, handlelength=0.5)

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
if plots == 1:
    plt.show()

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

    # max_qh_ratio = qh_ratio[max_fy_id[idx], idx]
    # max_gta_ratio = gta_ratio[max_fy_id[idx], idx]

    dict_i = {'scenario': str(sen),
              'fx': str(max_fx),
              'max_fy': str(max_fy),
              'fz': str(max_fz),
              'mx': str(max_mx),
              'my': str(max_my),
              'mz': str(max_mz),
              'fx_jr': str(max_fx_jr),
              'fy_jr': str(max_fy_jr),
              'fz_jr': str(max_fz_jr),
              'mx_jr': str(max_mx_jr),
              'my': str(max_my_jr),
              'mz_jr': str(max_mz_jr),
              'quad_force': str(max_quad_force),
              'ham_force': str(max_ham_force),
              'gastro_force': str(max_gastro_force),
              'ta_force': str(max_ta_force),
              'sol_force': str(max_sol_force),
              'qh_ratio': str(max_quad_force/max_ham_force),
              'gta_ratio': str(max_gastro_force/max_ta_force),
              }
    # d = [d, dict_i]
    with open(results_path + '/' + name + '_output.csv',
              'a') as f:
        w = csv.DictWriter(f, dict_i.keys())
        w.writeheader()
        # f.write("\n")
        w.writerow(dict_i)


print('ok')





