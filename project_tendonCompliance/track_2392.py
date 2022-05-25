import os
import opensim as osim
import numpy as np
from utils import fix_controls, read_GRF_JR_file, index_containing_substring,\
    estimate_cop, fix_sto_2392

project_path = os.getcwd()
data_path = os.path.abspath(project_path + "/Data_files/")
prev_data_file = "/solution.sto"
fix_sto_2392(data_path + prev_data_file)
data_file = "/edited_solution.sto"
stateRef = os.path.abspath(data_path + data_file)

model_path = os.path.abspath(project_path + "/Opensim_Models")
model_file = os.path.abspath(model_path
                             +"/Gait2392\gait2392_only_left.osim")

modelProcessor = osim.ModelProcessor(model_file)

# modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

modelProcessor.append(osim.ModOpUseImplicitTendonComplianceDynamicsDGF())

# modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
modelProcessor.append(osim.ModOpAddReserves(100))
model = modelProcessor.process()

final_time = 0.4
initial_time = 0.05

scenario = "Track_ImplicitTendonComplianceDynamics_PassiveFiberForce"
lumbar_ext_final = 0
lumbar_rot_final = 0
lumbar_ben_final = 0

results_path = os.path.abspath(project_path +
                               "/Results/" +
                               scenario)
if not os.path.isdir(results_path):
    os.makedirs(results_path)

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
# Control Info for case_Gait2392_track_InAct0
################################################################################
problem.setStateInfoPattern('/forceset/.*/activation',
                            osim.MocoBounds(0, 1),
                            osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Pelvis
# -----------------------------------------------------------
problem.setStateInfo("/jointset/ground_pelvis/pelvis_ty/value",
                     osim.MocoBounds(0.7, 1.25), osim.MocoInitialBounds(
        1.25), osim.MocoFinalBounds(0.75, 0.85))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tx/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tilt/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_list/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tilt/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tz/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_rotation/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Lumbar
# -----------------------------------------------------------
# extension
if lumbar_ext_final >= 0:
    problem.setStateInfoPattern("/jointset/back/lumbar_extension/value",
                            osim.MocoBounds(-0.01, lumbar_ext_final + 0.01),
                            osim.MocoInitialBounds(lumbar_ext_final - 0.01,
                                                   lumbar_ext_final + 0.01))
# flexion
else:
    problem.setStateInfoPattern("/jointset/back/lumbar_extension/value",
                                osim.MocoBounds(lumbar_ext_final - 0.1, 0.01),
                                osim.MocoInitialBounds(lumbar_ext_final - 0.01,
                                                       lumbar_ext_final + 0.01))
if lumbar_ben_final >= 0:
    problem.setStateInfoPattern("/jointset/back/lumbar_bending/value",
                            osim.MocoBounds(-0.01, lumbar_ben_final + 0.01),
                            osim.MocoInitialBounds(lumbar_ben_final - 0.01,
                                                   lumbar_ben_final + 0.01))
else:
    problem.setStateInfoPattern("/jointset/back/lumbar_bending/value",
                                osim.MocoBounds(lumbar_ben_final - 0.1, 0.01),
                                osim.MocoInitialBounds(lumbar_ben_final - 0.01,
                                                       lumbar_ben_final + 0.01))
if lumbar_rot_final >= 0:
    problem.setStateInfoPattern("/jointset/back/lumbar_rotation/value",
                            osim.MocoBounds(-0.01, lumbar_rot_final + 0.01),
                            osim.MocoInitialBounds(lumbar_rot_final - 0.01,
                                                   lumbar_rot_final + 0.01))
else:
    problem.setStateInfoPattern("/jointset/back/lumbar_rotation/value",
                                osim.MocoBounds(lumbar_rot_final - 0.1, 0.01),
                                osim.MocoInitialBounds(lumbar_rot_final - 0.01,
                                                       lumbar_rot_final + 0.01))

# -----------------------------------------------------------
# Left Leg
# -----------------------------------------------------------
problem.setStateInfo("/jointset/hip_l/hip_flexion_l/value", osim.MocoBounds(
    0.08, 0.5), osim.MocoInitialBounds(0.087))
problem.setStateInfo("/jointset/knee_l/knee_angle_l/value", osim.MocoBounds(
    -1.6, 0), osim.MocoInitialBounds(-0.2))
problem.setStateInfo("/jointset/ankle_l/ankle_angle_l/value",
                     osim.MocoBounds(
                         -0.6, 0.8), osim.MocoInitialBounds(-0.6))

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
solver.set_num_mesh_intervals(100)
solver.set_verbosity(2)
solver.set_optim_solver("ipopt")
solver.set_optim_convergence_tolerance(1e-2)
solver.set_optim_constraint_tolerance(1e-2)
# solver.set_optim_max_iterations(10)

solver.setGuess('bounds')

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


