import os
import opensim as osim
import numpy as np
from utils import read_GRF_JR_file, readExternalLoadsFile, \
    index_containing_substring, estimate_cop, plot_muscle_forces
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# -----------------------------------------------------------
#  case_Gait2392_inverse
# -----------------------------------------------------------
scenario = "case_Gait2392_inverse"
final_time = 0.4
project_path = os.getcwd()
results_path = os.path.abspath(project_path + "/Results/" + scenario)
if not os.path.isdir(results_path):
    os.makedirs(results_path)
reference_dir = "./Results/case_Gait2392_track"
reference_state = "./Results/case_Gait2392_track/solution.sto"
stateRef = os.path.abspath(reference_state)
estimate_cop(reference_dir)

grf = os.path.abspath("./Results/case_Gait2392_track/grf_CoP_setup.xml")
grf_file = os.path.abspath("./Results/case_Gait2392_track/grf_CoP.sto")

# -----------------------------------------------------------
#  case_Gait2392_inverse_InAct0
# -----------------------------------------------------------
# scenario = "case_Gait2392_inverse_InAct0"
# final_time = 0.4
# project_path = os.getcwd()
# results_path = os.path.abspath(project_path + "/Results/" + scenario)
# if not os.path.isdir(results_path):
#     os.makedirs(results_path)
# reference_dir = "./Results/case_Gait2392_track_InAct0"
# reference_state = "./Results/case_Gait2392_track_InAct0/solution.sto"
# stateRef = os.path.abspath(reference_state)
#
# # estimate CoP
# estimate_cop(reference_dir)
#
# grf = os.path.abspath("./Results/case_Gait2392_track_InAct0/grf_CoP_setup.xml")


model_path = os.path.abspath(project_path + "/Opensim_Models/Gait2392")
model_file = os.path.abspath(model_path
                             + "/gait2392_only_left_UPD.osim")

modelProcessor = osim.ModelProcessor(model_file)
modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
# modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
modelProcessor.append(osim.ModOpAddReserves(1))
modelProcessor.append(osim.ModOpAddExternalLoads(grf))
model = modelProcessor.process()

inverse = osim.MocoInverse()
inverse.setModel(modelProcessor)

inverse.setKinematics(osim.TableProcessor(stateRef))
inverse.set_kinematics_allow_extra_columns(True)
inverse.set_minimize_sum_squared_activations(True)
# inverse.append_output_paths('.*reaction_on_child')

inverse.set_initial_time(0)
inverse.set_final_time(0.4)
inverse.set_mesh_interval(0.005)
inverse.set_convergence_tolerance(1e-4)
inverse.set_constraint_tolerance(1e-4)

study = inverse.initialize()

inverseSolution = inverse.solve()
inverseSolution.getMocoSolution().write(results_path + '/solution.sto')
solution = inverseSolution.getMocoSolution()

# -----------------------------------------------------------
#  normalized tendon forces are essentially normalized muscle forces
# -----------------------------------------------------------
outputPaths = osim.StdVectorString()
outputPaths.append('.*tendon_force')
outputTable = study.analyze(inverseSolution.getMocoSolution(), outputPaths)
osim.STOFileAdapter.write(outputTable, results_path +
                          "/tendon_forces.sto")

report = osim.report.Report(model, results_path + "/solution.sto",
                            bilateral=True,
                            output=results_path + '/inverse_report.pdf')
report.generate()

plot_muscle_forces(results_path + "/tendon_forces.sto", scenario, results_path)

# ----------------------------------------------------
# JRA
# ----------------------------------------------------
# BW = 75.16 kg = 737.0678161524997 N
bw = 737.0678

JR_paths = osim.StdVectorString()
# JR_paths.append('.*reaction_on_parent')
JR_paths.append('.*reaction_on_child')

statesTST = solution.exportToStatesTable()
controlsTST = solution.exportToControlsTable()

JR_outputs_table = osim.analyzeSpatialVec(model, statesTST, controlsTST,
                                          JR_paths).flatten()
osim.STOFileAdapter.write(JR_outputs_table, results_path +
                          "/JR_in_Ground.sto")

# -----------------------------------------------------------
#  express in Child
# -----------------------------------------------------------
model.initSystem()
traj = inverseSolution.getMocoSolution()
headers, jr_labels, jr_data = read_GRF_JR_file(results_path +
                                               "\JR_in_Ground.sto")
jr = np.asarray(jr_data)
states = traj.exportToStatesTable()
statesTraj = osim.StatesTrajectory.createFromStatesTable(model, states)
m, n = jr.shape
num_of_joints = model.getJointSet().getSize()
knee_l = model.getJointSet().get("knee_l")
for i in range(m):
    state = statesTraj.get(i)
    model.realizeDynamics(state)
    ground = model.getGround()
    joint_name = knee_l.getName()
    knee_l_idx = index_containing_substring(jr_labels, 'knee_l')
    Torque_ground = osim.Vec3(jr[i, knee_l_idx[0]], jr[i, knee_l_idx[0] + 1],
                              jr[i, knee_l_idx[0] + 2])
    Force_ground = osim.Vec3(jr[i, knee_l_idx[0] + 3],
                             jr[i, knee_l_idx[0] + 4],
                             jr[i, knee_l_idx[0] + 5])
    child_frame = knee_l.getChildFrame()
    Torque_local = ground.expressVectorInAnotherFrame(state,
                                                      Torque_ground,
                                                      child_frame)
    Force_local = ground.expressVectorInAnotherFrame(state,
                                                     Force_ground,
                                                     child_frame)

    jr[i, knee_l_idx[0]] = Torque_local.get(0)
    jr[i, knee_l_idx[0] + 1] = Torque_local.get(1)
    jr[i, knee_l_idx[0] + 2] = Torque_local.get(2)

    jr[i, knee_l_idx[0] + 3] = Force_local.get(0)
    jr[i, knee_l_idx[0] + 4] = Force_local.get(1)
    jr[i, knee_l_idx[0] + 5] = Force_local.get(2)

first_line = headers
second_line = [jr_labels[0]] + jr_labels[knee_l_idx[0]:knee_l_idx[0] + 6]
third_line = np.column_stack((jr[:, 0], jr[:,
                                        knee_l_idx[0]: knee_l_idx[0] + 6]))

filename = results_path + "\JR_in_Child_2392.sto"
with open(filename, 'w') as out:
    out.writelines("%s" % item for item in first_line)
    out.write('\t'.join(second_line) + '\n')
    np.savetxt(out, third_line, delimiter='\t', fmt='%1.3f')

# -----------------------------------------------------------
#  Plot and save GRF
# -----------------------------------------------------------

T2_header, T2_labels, T2_data = readExternalLoadsFile(grf_file)
T2_data = np.asarray(T2_data)
m_2, n_2 = T2_data.shape
time_array = T2_data[:, 0]

# Savagol filter parameters (smooth results)
filter_window = 13
filter_deg = 3
# wrong labels in GRF_CoP.sto left<->right
grf_fx = savgol_filter(T2_data[:, 7] / bw, filter_window, filter_deg)
grf_fy = savgol_filter(T2_data[:, 8] / bw, filter_window, filter_deg)
grf_fz = savgol_filter(T2_data[:, 9] / bw, filter_window, filter_deg)

grf_mx = savgol_filter(T2_data[:, 16] / bw, filter_window, filter_deg)
grf_my = savgol_filter(T2_data[:, 17] / bw, filter_window, filter_deg)
grf_mz = savgol_filter(T2_data[:, 18] / bw, filter_window, filter_deg)

# plt.plot(time_array, fx), fy, "b", fz, "g"
plt.plot(time_array, grf_fx, "r")
plt.plot(time_array, grf_fy, "b")
plt.plot(time_array, grf_fz, "g")
plt.legend(["f_x", "f_y", "f_z"])
plt.title(scenario + "/GRF")
plt.ylabel("GRF (N/BW)")
plt.xlabel("Time (sec)")
# plt.ylim([-3000,3000])
plt.savefig(results_path + '/GRF_force_plot.png')
plt.show()

plt.plot(time_array, grf_mx, "r")
plt.plot(time_array, grf_my, "b")
plt.plot(time_array, grf_mz, "g")
plt.legend(["m_x", "m_y", "m_z"])
plt.title(scenario + "/n GRF torque " \
                     "frame")
plt.ylabel("Moment (Nm/bw)")
plt.xlabel("Time (sec)")
# plt.ylim([-500,1000])
plt.savefig(results_path + '/GRF_moment_plot.png')
plt.show()

# -----------------------------------------------------------
#  Plot and save JRA
# -----------------------------------------------------------
file_name = '/JR_in_Child.sto'

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

# plt.plot(time_array, fx), fy, "b", fz, "g"
plt.plot(time_array, fx, "r")
plt.plot(time_array, fy, "b")
plt.plot(time_array, fz, "g")
plt.legend(["f_x", "f_y", "f_z"])
plt.title(scenario + "\nKnee force occurring on tibia expressed in " \
                     "tibia frame")
plt.ylabel("Force (N)")
plt.xlabel("Time (sec)")
# plt.ylim([-3000,3000])
plt.savefig(results_path + '/JRA_force_plot.png')
plt.show()

plt.plot(time_array, mx, "r")
plt.plot(time_array, my, "b")
plt.plot(time_array, mz, "g")
plt.legend(["m_x", "m_y", "m_z"])
plt.title(scenario + "/nKnee moment occurring on tibia expressed in "
                     "tibia " \
                     "frame")
plt.ylabel("Moment (Nm)")
plt.xlabel("Time (sec)")
# plt.ylim([-500,1000])
plt.savefig(results_path + '/JRA_moment_plot.png')
plt.show()

# -----------------------------------------------------------
#  Joint Torques
# -----------------------------------------------------------

muscles = model.getMuscles()
coordSet = model.getCoordinateSet()
statesTraj = solution.exportToStatesTrajectory(model)
MusMomentArms_ankle_angle_l = []
MusMomentArms_knee_angle_l = []
MusMomentArms_hip_flexion_l = []

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

plt.plot(time_array, JM_ankle_angle_r / norm * 100)
plt.title(scenario + "Ankle angle torque")
plt.ylabel("%BW*HT")
plt.xlabel("Time")
plt.axvline(x=0.2, linestyle='--', color='r')
plt.savefig(results_path + '/Ankle_angle_torque.png')
plt.show()

plt.plot(time_array, JM_knee_angle_r / norm * 100)
plt.title(scenario + "Knee angle torque")
plt.ylabel("%BW*HT")
plt.xlabel("Time")
plt.axvline(x=0.2, linestyle='--', color='r')
plt.savefig(results_path + '/Knee_angle_torque.png')
plt.show()

plt.plot(time_array, JM_hip_flexion_r / norm * 100)
plt.title(scenario + "Hip angle torque")
plt.ylabel("%BW*HT")
plt.xlabel("Time")
plt.axvline(x=0.2, linestyle='--', color='r')
plt.savefig(results_path + '/Hip_angle_torque.png')
plt.show()