import os
import opensim as osim
import numpy as np
from utils import readExternalLoadsFile, plot_muscle_forces_Mokhtazadeh, \
    plot_muscle_forces_Sritharan
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

project_path = os.getcwd()
# Trunk
at_dict = {
    1: {'scenario': 'case_Gait2392_track', 'path': os.path.abspath(
        project_path + "/Results/")},
    2: {'scenario': 'case_Gait2392_track_flat-foot', 'path': os.path.abspath(
        project_path + "/Results/")},
    3: {'scenario': 'ext_-40', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    4: {'scenario': 'ext_-35', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    5: {'scenario': 'ext_-30', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    6: {'scenario': 'ext_-25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    7: {'scenario': 'ext_-20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    8: {'scenario': 'ext_-15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    9: {'scenario': 'ext_-10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    10: {'scenario': 'ext_-5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    11: {'scenario': 'ext_5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    12: {'scenario': 'ext_10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    13: {'scenario': 'ext_15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    14: {'scenario': 'ext_20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    15: {'scenario': 'ext_25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    16: {'scenario': 'rot_-30', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    17: {'scenario': 'rot_-25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    18: {'scenario': 'rot_-20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    19: {'scenario': 'rot_-15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    20: {'scenario': 'rot_-10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    21: {'scenario': 'rot_-5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    22: {'scenario': 'rot_5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    23: {'scenario': 'rot_10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    24: {'scenario': 'rot_15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    25: {'scenario': 'rot_20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    26: {'scenario': 'rot_25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    27: {'scenario': 'rot_30', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    28: {'scenario': 'ben_-30', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    29: {'scenario': 'ben_-25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    30: {'scenario': 'ben_-20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    31: {'scenario': 'ben_-15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    32: {'scenario': 'ben_-10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    33: {'scenario': 'ben_-5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    34: {'scenario': 'ben_5', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    35: {'scenario': 'ben_10', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    36: {'scenario': 'ben_15', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    37: {'scenario': 'ben_20', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    38: {'scenario': 'ben_25', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")},
    39: {'scenario': 'ben_30', 'path': os.path.abspath(
        project_path + "/Results/Trunk/")}
}

# Hip
# t_dict = {
#     1: {'scenario': 'rot_-30', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     2: {'scenario': 'rot_-25', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     3: {'scenario': 'rot_-20', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     4: {'scenario': 'rot_-15', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     5: {'scenario': 'rot_-10', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     6: {'scenario': 'rot_-5', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     7: {'scenario': 'rot_5', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     8: {'scenario': 'rot_10', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     9: {'scenario': 'rot_15', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     10: {'scenario': 'rot_20', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     11: {'scenario': 'rot_25', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     12: {'scenario': 'rot_30', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     13: {'scenario': 'abd_-10', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     14: {'scenario': 'abd_-8', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     15: {'scenario': 'abd_-6', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     16: {'scenario': 'abd_-4', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     17: {'scenario': 'abd_-2', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     18: {'scenario': 'abd_2', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     19: {'scenario': 'abd_4', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     20: {'scenario': 'abd_6', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     21: {'scenario': 'abd_8', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#     22: {'scenario': 'abd_610', 'path': os.path.abspath(
#         project_path + "/Results/Hip/")},
#         }

model_path = os.path.abspath(project_path + "/Opensim_Models")
model_file = os.path.abspath(model_path
                             +"/Gait2392\gait2392_only_left.osim")
model = osim.Model(model_file)
muscles = model.getMuscles()
coordSet = model.getCoordinateSet()

# BW = 75.16 kg = 737.0678161524997 N
bw = 737.0678


for i in range(len(at_dict)):
    scenario = at_dict[i+1]['scenario']
    path = at_dict[i+1]['path']
    results_path = os.path.abspath(path + '/' + scenario)

    # -----------------------------------------------------------
    #  GRF
    # -----------------------------------------------------------
    grf_file = os.path.abspath(results_path + "/grf_CoP.sto")
    T2_header, T2_labels, T2_data = readExternalLoadsFile(grf_file)

    T2_data = np.asarray(T2_data)
    time_array = T2_data[:, 0]
    m_2, n_2 = T2_data.shape

    # Savagol filter parameters (smooth results)
    filter_window = 13
    filter_deg = 3
    # wrong labels in GRF_CoP.sto left<->right
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


    fig1, axs1 = plt.subplots(2, figsize=(15, 15))
    fig1.suptitle(scenario)
    axs1[0].plot(time_array, grf_fx)
    axs1[0].plot(time_array, grf_fy)
    axs1[0].plot(time_array, grf_fz)
    axs1[0].set_title('GRF forces(xBW)')
    axs1[0].axvline(x=0.2, linestyle='--', color='red')

    grf_fx_max = max(grf_fx)
    xpos = np.where(grf_fx == grf_fx_max)
    xmax = time_array[xpos]
    axs1[0].annotate('local max', xy=(xmax, grf_fx_max),
                     xytext=(xmax, grf_fx_max + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),)

    grf_fy_max = max(grf_fy)
    xpos = np.where(grf_fy == grf_fy_max)
    xmax = time_array[xpos]
    axs1[0].annotate('local max', xy=(xmax, grf_fy_max),
                     xytext=(xmax, grf_fy_max + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),)

    grf_fz_max = max(grf_fz)
    xpos = np.where(grf_fz == grf_fz_max)
    xmax = time_array[xpos]
    axs1[0].annotate(str(grf_fz_max), xy=(xmax, grf_fz_max))

    axs1[1].plot(time_array, grf_mx)
    axs1[1].plot(time_array, grf_my)
    axs1[1].plot(time_array, grf_mz)
    axs1[1].set_title('GRF moments(xBW)')
    axs1[1].axvline(x=0.2, linestyle='--', color='red')
    plt.savefig(results_path + '/GRF.png')


    # -----------------------------------------------------------
    #  JRA
    # -----------------------------------------------------------

    file = os.path.abspath(os.path.abspath(results_path + '/JR_in_Child_2392.sto'))
    T2_header, T2_labels, T2_data = readExternalLoadsFile(file)
    T2_data = np.asarray(T2_data)
    m_2, n_2 = T2_data.shape
    time_array = T2_data[:, 0]

    # Savagol filter parameters (smooth results)
    filter_window = 13
    filter_deg = 3

    fx = savgol_filter(T2_data[:, 4], filter_window, filter_deg)
    fy = savgol_filter(T2_data[:, 5], filter_window, filter_deg)
    fz = savgol_filter(T2_data[:, 6], filter_window, filter_deg)

    mx = savgol_filter(T2_data[:, 1], filter_window, filter_deg)
    my = savgol_filter(T2_data[:, 2], filter_window, filter_deg)
    mz = savgol_filter(T2_data[:, 3], filter_window, filter_deg)

    fig2, axs2 = plt.subplots(2, figsize=(15, 15))
    fig2.suptitle(scenario)
    axs2[0].plot(time_array, fx)
    axs2[0].plot(time_array, fy)
    axs2[0].plot(time_array, fz)
    axs2[0].set_title('JRA forces')
    axs2[0].axvline(x=0.2, linestyle='--', color='red')
    axs2[1].plot(time_array, mx)
    axs2[1].plot(time_array, my)
    axs2[1].plot(time_array, mz)
    axs2[1].set_title('JRA moments')
    axs2[1].axvline(x=0.2, linestyle='--', color='red')
    plt.savefig(results_path + '/JRA.png')





    # -----------------------------------------------------------
    #  Joint Torques
    # -----------------------------------------------------------
    solution = osim.MocoTrajectory(results_path + '/solution.sto')
    statesTraj = solution.exportToStatesTrajectory(model)
    MusMomentArms_ankle_angle_l = []
    MusMomentArms_knee_angle_l = []
    MusMomentArms_hip_flexion_l = []

    model.initSystem()
    for i in range(statesTraj.getSize()):
        state = statesTraj.get(i)
        model.realizeDynamics(state)

        # Muscle moment arms
        ankle = []
        knee = []
        hip = []
        for j in range(muscles.getSize()):
            mus = muscles.get(j)
            coord = coordSet.get('ankle_angle_l')
            ankle.append(mus.computeMomentArm(state, coord))
            coord = coordSet.get('knee_angle_l')
            knee.append(mus.computeMomentArm(state, coord))
            coord = coordSet.get('hip_flexion_l')
            hip.append(mus.computeMomentArm(state, coord))

        MusMomentArms_ankle_angle_l.append(ankle)
        MusMomentArms_knee_angle_l.append(knee)
        MusMomentArms_hip_flexion_l.append(hip)

    MusMomentArms_ankle_angle_l = np.asarray(MusMomentArms_ankle_angle_l)
    MusMomentArms_knee_angle_l = np.asarray(MusMomentArms_knee_angle_l)
    MusMomentArms_hip_flexion_l = np.asarray(MusMomentArms_hip_flexion_l)

    header, labels, data = readExternalLoadsFile(
        results_path + "/tendon_forces.sto")
    time_array = np.asarray(data)[:, 0]
    # drop normalized data
    MusTendonForces = np.asarray(data)[:, 2::2]
    m, n = MusTendonForces.shape

    #  Joint moments
    JM_ankle_angle_r = np.sum(np.multiply(MusMomentArms_ankle_angle_l,
                                          MusTendonForces), 1)
    JM_knee_angle_r = np.sum(np.multiply(MusMomentArms_knee_angle_l,
                                       MusTendonForces), 1)
    JM_hip_flexion_r = np.sum(np.multiply(MusMomentArms_hip_flexion_l,
                                        MusTendonForces), 1)

    norm = bw * 1.8
    fig3, axs3 = plt.subplots(3, figsize=(15, 15))
    fig3.suptitle(scenario)
    axs3[0].plot(time_array, JM_ankle_angle_r)
    axs3[0].set_title("Ankle angle torque")
    axs3[0].axvline(x=0.2, linestyle='--', color='red')
    axs3[1].plot(time_array, JM_knee_angle_r)
    axs3[1].set_title("Knee angle torque")
    axs3[1].axvline(x=0.2, linestyle='--', color='red')
    axs3[2].plot(time_array, JM_hip_flexion_r)
    axs3[2].set_title("Hip angle torque")
    axs3[2].axvline(x=0.2, linestyle='--', color='red')
    plt.savefig(results_path + '/JointTorques.png')


    # -----------------------------------------------------------
    #  Joint angles
    # -----------------------------------------------------------
    file = os.path.abspath(
        os.path.abspath(results_path + '/solution.sto'))
    T2_header, T2_labels, T2_data = readExternalLoadsFile(file)
    data = np.asarray(T2_data)
    time_array = np.asarray(data)[:, 0]

    for idx, label in enumerate(T2_labels):
        if label == "/jointset/knee_l/knee_angle_l/value":
            knee_label = label
            knee_array = np.asarray(data)[:, idx]
        if label == "/jointset/ankle_l/ankle_angle_l/value":
            ankle_label = label
            ankle_array = np.asarray(data)[:, idx]
        if label == "/jointset/hip_l/hip_flexion_l/value":
            hip_label = label
            hip_array = np.asarray(data)[:, idx]

    fig4, axs4 = plt.subplots(3, figsize=(15, 15))
    fig4.suptitle(scenario)
    axs4[0].plot(time_array, np.rad2deg(ankle_array))
    axs4[0].set_title("Ankle angle")
    axs4[0].axvline(x=0.2, linestyle='--', color='red')
    axs4[1].plot(time_array, np.rad2deg(knee_array))
    axs4[1].set_title("Knee angle")
    axs4[1].axvline(x=0.2, linestyle='--', color='red')
    axs4[2].plot(time_array, np.rad2deg(hip_array))
    axs4[2].set_title("Hip angle")
    axs4[2].axvline(x=0.2, linestyle='--', color='red')
    plt.savefig(results_path + '/JointAngles.png')

    # -----------------------------------------------------------
    #  Muscle forces
    # -----------------------------------------------------------
    plot_muscle_forces_Sritharan(results_path + "/tendon_forces.sto", scenario, results_path)
    plot_muscle_forces_Mokhtazadeh(results_path + "/tendon_forces.sto", scenario, results_path)


