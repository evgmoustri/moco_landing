import os
import opensim as osim
import numpy as np
from utils import fix_controls, read_GRF_JR_file, index_containing_substring,\
    estimate_cop, fix_sto_2392
from matplotlib import pyplot as plt
import pandas as pd

# intervals = [5, 10, 25, 50, 100, 200]
# nodes = [11, 21, 51, 101 , 201, 401]

intervals = [150, 250, 300, 350, 400]
nodes = [301, 501, 601, 701 , 801]

iterations = []
time = []
cost = []

iterations = []
time = []
cost = []

project_path = os.getcwd()
at_dict = {
    1: {'scenario': 'h30', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
        'lumbar_ben_final': 0, 'height': 1.25},
}

for num_of_intervals in intervals:
    scenario = "case_" + str(num_of_intervals)
    model_path = os.path.abspath(project_path + "/Opensim_Models")
    model_file = os.path.abspath(model_path
                                 + "/Gait2392\gait2392_only_left.osim")
    modelProcessor = osim.ModelProcessor(model_file)
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    modelProcessor.append(osim.ModOpAddReserves(100))
    model = modelProcessor.process()
    #scenario = at_dict[1]['scenario']
    lumbar_ext_final = np.deg2rad(at_dict[1]['lumbar_ext_final'])
    lumbar_rot_final = np.deg2rad(at_dict[1]['lumbar_rot_final'])
    lumbar_ben_final = np.deg2rad(at_dict[1]['lumbar_ben_final'])
    pelvis_ty = at_dict[1]['height']

    results_path = os.path.abspath(project_path +
                                   "/Results/" +
                                   scenario)
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # -----------------------------------------------------------
    #  Moco Predict
    # -----------------------------------------------------------
    study = osim.MocoStudy()
    problem = study.updProblem()
    problem.setModel(model)
    problem.setTimeBounds(0, [0.1, 1.0])

    ################################################################################
    # Control Info for case_Gait2392_track_InAct0
    ################################################################################
    # Set Velocities to start and end at zero
    problem.setStateInfoPattern("/jointset/.*/speed", [-50, 50], 0, 0)

    # Set Activations
    problem.setStateInfoPattern("/forceset/.*/activation",
                                osim.MocoBounds(0, 1),
                                osim.MocoInitialBounds(0))
    # -----------------------------------------------------------
    # Pelvis
    # -----------------------------------------------------------
    problem.setStateInfo("/jointset/ground_pelvis/pelvis_ty/value",
                         osim.MocoBounds(0.5, pelvis_ty),
                         osim.MocoInitialBounds(pelvis_ty),
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
                         osim.MocoBounds(-0.01, 0.01),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/hip_r/hip_rotation_r/value",
                         osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/hip_r/hip_adduction_r/value",
                         osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/subtalar_r/subtalar_angle_r/value",
                         osim.MocoBounds(-0.01, 0.01),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/mtp_r/mtp_angle_r/value",
                         osim.MocoBounds(-0.01, 0.01))

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

    # problem.setStateInfo("/jointset/hip_l/hip_adduction_l/value",
    #                      osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
    #                      osim.MocoInitialBounds(0))
    problem.setStateInfo("/jointset/hip_l/hip_adduction_l/value",
                         osim.MocoBounds(-0.01, 0.01),
                         osim.MocoInitialBounds(0))

    # problem.setStateInfo("/jointset/hip_l/hip_rotation_l/value",
    #                      osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
    #                      osim.MocoInitialBounds(0))
    problem.setStateInfo("/jointset/hip_l/hip_rotation_l/value",
                         osim.MocoBounds(-0.01, 0.01),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/subtalar_l/subtalar_angle_l/value",
                         osim.MocoBounds(-0.05, 0.05))

    problem.setStateInfo("/jointset/mtp_l/mtp_angle_l/value",
                         osim.MocoBounds(-0.1, 0.1))

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

    # Set Goals
    problem.addGoal(osim.MocoControlGoal('effort', 0.001))


    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(num_of_intervals)
    solver.set_verbosity(2)
    solver.set_optim_solver("ipopt")
    solver.set_optim_convergence_tolerance(1e-2)
    solver.set_optim_constraint_tolerance(1e-2)
    # solver.set_optim_max_iterations(2)

    solver.setGuessFile(project_path + "/Data_files/solution_2392.sto")

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
    contact_l.append('foot_l_1')
    contact_l.append('foot_l_2')
    contact_l.append('foot_l_3')


    externalForcesTableFlat = osim.createExternalLoadsTableForGait(model,
                                                                   solution,
                                                                   contact_r,
                                                                   contact_l)
    osim.STOFileAdapter.write(externalForcesTableFlat,
                              results_path + '/GRF.sto')
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
        Torque_ground = osim.Vec3(jr[i, knee_l_idx[0]],
                                  jr[i, knee_l_idx[0] + 1],
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


    iterations.append(solution.getNumIterations())
    cost.append(round(solution.getObjective() * 10 ** 4, 3))
    time.append(round(solution.getSolverDuration() / 3600,3))
    print(scenario)

# -----------------------------------------------------------
#  Plot Results
# -----------------------------------------------------------
column_labels=["Intervals", "Nodes", "Iterations" , "Time(h)", "Cost(xe4)"]
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df = pd.DataFrame(data=list(zip(intervals, nodes, iterations, time, cost)),
                  columns=column_labels)
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(14)
fig.tight_layout()
plt.savefig(project_path + "/Results/GridConvergenceStudy_2.png")
plt.show()
print(table)

df.to_csv(project_path  + '/Results/GridConvergenceStudy_2.csv')

df.to_pickle(project_path  + "/Results/a_file_2.pkl")
output = pd.read_pickle(project_path  + "/Results/a_file_2.pkl")
print(output)