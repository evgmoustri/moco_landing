import opensim as osim
import numpy as np
import os

# print(scenario)
scenario = "predict_hip_rot_20"

project_path = os.getcwd()
results_path = os.path.abspath(project_path + "/Results/" + scenario)
if not os.path.isdir(results_path):
    os.makedirs(results_path)

study = osim.MocoStudy()

problem = study.updProblem()
modelProcessor = osim.ModelProcessor(project_path +
    "/Opensim_Models/Gait2392/gait2392_only_left.osim")
modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
modelProcessor.append(osim.ModOpAddReserves(100))
model = modelProcessor.process()

problem.setModel(model)
problem.setTimeBounds(0, [0.1, 1.0])

# Set State Info
# -----------------------------------------------------------
# Pelvis
# -----------------------------------------------------------
problem.setStateInfo("/jointset/ground_pelvis/pelvis_ty/value",
                     osim.MocoBounds(0.5, 1.25),
                     osim.MocoInitialBounds(1.25),
                     osim.MocoFinalBounds(0.5, 0.85))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tx/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tilt/value",
                     osim.MocoBounds(-0.01, 0.01))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_list/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_tz/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))
problem.setStateInfo("/jointset/ground_pelvis/pelvis_rotation/value",
                     osim.MocoBounds(-0.01, 0.01),
                     osim.MocoInitialBounds(0))

# -----------------------------------------------------------
# Right Leg
# -----------------------------------------------------------

problem.setStateInfo("/jointset/hip_r/hip_flexion_r/value",
                     osim.MocoBounds(0.51, 0.53),
                     osim.MocoInitialBounds(0.523))

problem.setStateInfo("/jointset/knee_r/knee_angle_r/value",
                     osim.MocoBounds(np.deg2rad(-121), np.deg2rad(-119)),
                     osim.MocoInitialBounds(np.deg2rad(-120)))

problem.setStateInfo("/jointset/ankle_r/ankle_angle_r/value",
                     osim.MocoBounds(-0.01, 0.01), osim.MocoInitialBounds(0))

problem.setStateInfo("/jointset/hip_r/hip_rotation_r/value",
                     osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
                     osim.MocoInitialBounds(0))
#
problem.setStateInfo("/jointset/subtalar_r/subtalar_angle_r/value",
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
    -0.6, 0.8), osim.MocoInitialBounds(-0.6))

# problem.setStateInfo("/jointset/hip_l/hip_rotation_l/value",
#                      osim.MocoBounds(np.deg2rad(15), np.deg2rad(23)),
#                      osim.MocoInitialBounds(np.deg2rad(17)),
#                      # osim.MocoInitialBounds(),
#                      osim.MocoFinalBounds(np.deg2rad(15),np.deg2rad(20)))

problem.setStateInfo("/jointset/hip_l/hip_rotation_l/value",
                     osim.MocoBounds(np.deg2rad(15)))


problem.setStateInfo("/jointset/subtalar_l/subtalar_angle_l/value",
                     osim.MocoBounds(np.deg2rad(0),np.deg2rad(10)),
                     osim.MocoInitialBounds(),
                     osim.MocoFinalBounds(np.deg2rad(0),np.deg2rad(8)))

# -----------------------------------------------------------
# Lumbar
# -----------------------------------------------------------
problem.setStateInfoPattern("/jointset/.*/lumbar_.*/value",
                            osim.MocoBounds(-0.01, 0.01))

# -----------------------------------------------------------
# Both Legs
# -----------------------------------------------------------
problem.setStateInfoPattern("/jointset/.*/hip_adduction_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))
problem.setStateInfoPattern("/jointset/.*/mtp_angle_.*/value",
                            osim.MocoBounds(-0.01, 0.01),
                            osim.MocoInitialBounds(0))

# ------------------------------------------------------------
# Set Velocities to start and end at zero
problem.setStateInfoPattern("/jointset/.*/speed", [-50, 50], 0, 0)

# Set Activations
problem.setStateInfoPattern("/forceset/.*/activation",
                            osim.MocoBounds(0.01, 1),
                            osim.MocoInitialBounds(0.01))

# Set Goals
problem.addGoal(osim.MocoControlGoal('effort', 1))

# Solver Configurations
solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(30)
solver.set_verbosity(2)
solver.set_optim_solver('ipopt')
solver.set_optim_convergence_tolerance(1e-2)
solver.set_optim_constraint_tolerance(1e-2)
# solver.set_optim_max_iterations(2)

# Guess
solver.setGuessFile(project_path + "/Data_files/solution_2392.sto")

# Solve the problem.
solution = study.solve()
solution.unseal()
solution.write(results_path + "/predicted_solution.sto")
# study.visualize(solution)

# osim.STOFileAdapter.write(solution.exportToStatesTable(),
#                           results_path + "/states.sto")
# osim.STOFileAdapter.write(solution.exportToControlsTable(),
#                           results_path + "/controls.sto")
# fix_controls(results_path + "/controls.sto")

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
contact_l.append('foot_l_1')
contact_l.append('foot_l_2')
contact_l.append('foot_l_3')
contact_l.append('foot_l_4')


externalForcesTableFlat = osim.createExternalLoadsTableForGait(model,
                                                               solution,
                                                               contact_r,
                                                               contact_l)
osim.STOFileAdapter.write(externalForcesTableFlat, results_path + "/GRF.sto")

# Generate Report
report = osim.report.Report(model, results_path + "/predicted_solution.sto",
                            bilateral=True,
                            output=results_path + '/track_report.pdf')
report.generate()
