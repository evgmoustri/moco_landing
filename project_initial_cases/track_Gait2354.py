# -*- coding: utf-8 -*-
"""
@author:  Evgenia Moustridi (evgmoustridi@gmail.com)
"""

import os
import opensim as osim
import numpy as np
from utils import fix_controls, read_GRF_JR_file, index_containing_substring,\
    estimate_cop, readExternalLoadsFile
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

scenario = 'case_Gait2354_track'
data_file = "/scone_solution_left.sto"
final_time = 0.4
initial_time = 0.05

project_path = os.getcwd()
results_path = os.path.abspath(project_path + "/Results/" + scenario)
if not os.path.isdir(results_path):
    os.makedirs(results_path)

data_path = os.path.abspath(project_path + "\Data_files")
stateRef = os.path.abspath(data_path + data_file)

model_path = os.path.abspath(project_path + "\Opensim_Models")
model_file = os.path.abspath(model_path
                             +"\Gait2354\gait2354_only_left.osim")

modelProcessor = osim.ModelProcessor(model_file)
modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
modelProcessor.append(osim.ModOpAddReserves(100))
model = modelProcessor.process()

# -----------------------------------------------------------
#  Moco Track
# -----------------------------------------------------------

track = osim.MocoTrack()
track.setModel(modelProcessor)
track.setStatesReference(osim.TableProcessor(stateRef))
track.set_allow_unused_references(True)

track.set_initial_time(initial_time)
track.set_final_time(final_time)

study = track.initialize()
study.setName(scenario)
problem = study.updProblem()

################################################################################
# Control Info for case_Gait2354_track_InAct0
################################################################################
problem.setStateInfoPattern('/forceset/.*/activation',
                            osim.MocoBounds(0, 1),
                            osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Pelvis
# -----------------------------------------------------------
problem.setStateInfo("/jointset/ground_pelvis/pelvis_ty/value",
                     osim.MocoBounds(0.7, 1.25), osim.MocoInitialBounds(
        1.25),osim.MocoFinalBounds(0.75, 0.85))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tx/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tilt/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_list/value",
                     osim.MocoBounds(-0.01,0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tilt/value",
                     osim.MocoBounds(-0.01,0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tz/value",
                     osim.MocoBounds(-0.01,0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_rotation/value",
                     osim.MocoBounds(-0.01,0.01),
                     osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Lumbar
# -----------------------------------------------------------
problem.setStateInfoPattern("/jointset/.*/lumbar_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Left Leg
# -----------------------------------------------------------
problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value", osim.MocoBounds(
    0.08, 0.5), osim.MocoInitialBounds(0.087))
problem.setStateInfo("/jointset/knee_l/knee_angle_l/value", osim.MocoBounds(
    -1.6, 0), osim.MocoInitialBounds(-0.2))
problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value", osim.MocoBounds(
    -0.6, 0.8), osim.MocoInitialBounds(-0.6), osim.MocoFinalBounds(0.69,
                                                                    0.7))

# -----------------------------------------------------------
# Both Legs
# -----------------------------------------------------------
problem.setStateInfoPattern("/jointset/.*/hip_adduction_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))
problem.setStateInfoPattern("/jointset/.*/hip_rotation_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))
problem.setStateInfoPattern("/jointset/.*/subtalar_angle_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))
problem.setStateInfoPattern("/jointset/.*/mtp_angle_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))



solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(40)
solver.set_verbosity(2)
solver.set_optim_solver("ipopt")
solver.set_optim_convergence_tolerance(1e-2)
solver.set_optim_constraint_tolerance(1e-2)
# solver.set_optim_max_iterations(10)

# # Initial Activation Goal
# activationGoal = osim.MocoInitialActivationGoal()
# problem.addGoal(activationGoal)

# # Effort Goal
# effort = osim.MocoControlGoal.safeDownCast(problem.updGoal("control_effort"))
# effort.setWeight(10)


solution = study.solve()
solution.unseal()
solution.write(results_path + "/solution.sto")

osim.STOFileAdapter.write(solution.exportToStatesTable(),
                          results_path + "/states.sto")
osim.STOFileAdapter.write(solution.exportToControlsTable(),
                          results_path + "/controls.sto")
fix_controls(results_path + "/controls.sto")

report = osim.report.Report(model, results_path + "/solution.sto",
                            bilateral=True,
                            output=results_path + '/track_report.pdf')
report.generate()

# -----------------------------------------------------------
#  normalized tendon forces are essentially normalized muscle forces
# -----------------------------------------------------------
outputPaths = osim.StdVectorString()
outputPaths.append('.*tendon_force')
outputTable = study.analyze(solution, outputPaths)
osim.STOFileAdapter.write(outputTable, results_path +
                          "/tendon_forces.sto")

# -----------------------------------------------------------
#  GRF
# -----------------------------------------------------------
contact_r = osim.StdVectorString()
contact_l = osim.StdVectorString()

contact_r.append('foot_r_1')
contact_r.append('foot_r_2')
contact_r.append('foot_r_3')
contact_r.append('foot_r_4')
contact_r.append('foot_r_5')
contact_l.append('foot_l_1')
contact_l.append('foot_l_2')
contact_l.append('foot_l_3')
contact_l.append('foot_l_4')
contact_l.append('foot_l_5')

externalForcesTableFlat = osim.createExternalLoadsTableForGait(model,
                                                               solution,
                                                               contact_r,
                                                               contact_l)
osim.STOFileAdapter.write(externalForcesTableFlat, results_path + "\GRF.sto")
estimate_cop(results_path)

# ----------------------------------------------------
# JRA
# ----------------------------------------------------
# BW = 75.16 kg = 737.0678161524997 N
bw = 737.0678

JR_paths = osim.StdVectorString()
# JR_paths.append('.*reaction_on_parent')
JR_paths.append('.*reaction_on_child')

states = solution.exportToStatesTable()
controls = solution.exportToControlsTable()

JR_outputs_table = osim.analyzeSpatialVec(model, states, controls,
                                          JR_paths).flatten()
osim.STOFileAdapter.write(JR_outputs_table, results_path +
                          "/JR_in_Ground.sto")

# -----------------------------------------------------------
#  express in Child
# -----------------------------------------------------------
model.initSystem()
headers, jr_labels, jr_data = read_GRF_JR_file(results_path +
                                               "\JR_in_Ground.sto")
jr = np.asarray(jr_data)

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
grf_file = os.path.abspath("./Results/case_Gait2354_track/grf_CoP.sto")
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

