import os
import opensim as osim
import numpy as np
from utils import fix_controls, read_GRF_JR_file, index_containing_substring,\
    estimate_cop, fix_sto_2392

project_path = os.getcwd()


# # Flexion
# at_dict = {
#     1: {'scenario': 'ext_-30', 'lumbar_ext_final': -30, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     2: {'scenario': 'ext_-25', 'lumbar_ext_final': -25, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     3: {'scenario': 'ext_-20', 'lumbar_ext_final': -20, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     4: {'scenario': 'ext_-15', 'lumbar_ext_final': -15, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     5: {'scenario': 'ext_-10', 'lumbar_ext_final': -10, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     6: {'scenario': 'ext_-5', 'lumbar_ext_final': -5, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     7: {'scenario': 'ext_5', 'lumbar_ext_final': 5, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     8: {'scenario': 'ext_10', 'lumbar_ext_final': 10, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     9: {'scenario': 'ext_15', 'lumbar_ext_final': 15, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0},
#     10: {'scenario': 'ext_20', 'lumbar_ext_final': 20, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 0}
# }\

# at_dict = {
# #Rotation
#     15: {'scenario': 'rot_-30', 'lumbar_ext_final': 0, 'lumbar_rot_final': -30,
#         'lumbar_ben_final': 0},
#     16: {'scenario': 'rot_-25', 'lumbar_ext_final': 0, 'lumbar_rot_final': -25,
#         'lumbar_ben_final': 0},
#     17: {'scenario': 'rot_-20', 'lumbar_ext_final': 0, 'lumbar_rot_final': -20,
#         'lumbar_ben_final': 0},
#     18: {'scenario': 'rot_-15', 'lumbar_ext_final': 0, 'lumbar_rot_final': -15,
#         'lumbar_ben_final': 0},
#     19: {'scenario': 'rot_-10', 'lumbar_ext_final': 0, 'lumbar_rot_final': -10,
#         'lumbar_ben_final': 0},
#     20: {'scenario': 'rot_-5', 'lumbar_ext_final': 0, 'lumbar_rot_final': -5,
#         'lumbar_ben_final': 0},
#     21: {'scenario': 'rot_5', 'lumbar_ext_final': 0, 'lumbar_rot_final': 5,
#         'lumbar_ben_final': 0},
#     22: {'scenario': 'rot_10', 'lumbar_ext_final': 0, 'lumbar_rot_final': 10,
#         'lumbar_ben_final': 0},
#     23: {'scenario': 'rot_15', 'lumbar_ext_final': 0, 'lumbar_rot_final': 15,
#         'lumbar_ben_final': 0},
#     24: {'scenario': 'rot_20', 'lumbar_ext_final': 0, 'lumbar_rot_final': 20,
#         'lumbar_ben_final': 0},
#     25: {'scenario': 'rot_25', 'lumbar_ext_final': 0, 'lumbar_rot_final': 25,
#         'lumbar_ben_final': 0},
#     26: {'scenario': 'rot_30', 'lumbar_ext_final': 0, 'lumbar_rot_final': 30,
#         'lumbar_ben_final': 0}}
# Bending
# at_dict = {
#     1: {'scenario': 'ben_-30', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -30},
#     2: {'scenario': 'ben_-25', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -25},
#     3: {'scenario': 'ben_-20', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -20},
#     4: {'scenario': 'ben_-15', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -15},
#     5: {'scenario': 'ben_-10', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -10},
#     6: {'scenario': 'ben_-5', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': -5},
#     7: {'scenario': 'ben_5', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 5},
#     8: {'scenario': 'ben_10', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 10},
#     9: {'scenario': 'ben_15', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 15},
#     10: {'scenario': 'ben_20', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 20},
#     11: {'scenario': 'ben_25', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final': 25},
#     12: {'scenario': 'ben_30', 'lumbar_ext_final': 0, 'lumbar_rot_final': 0,
#         'lumbar_ben_final':30}
# }

at_dict = {
    1: {'scenario': 'ext_-20', 'lumbar_ext_final': -20, 'lumbar_rot_final': 0,
        'lumbar_ben_final': 0}}


for i in range(len(at_dict)):
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
    scenario = at_dict[i+1]['scenario']
    lumbar_ext_final = np.deg2rad(at_dict[i+1]['lumbar_ext_final'])
    lumbar_rot_final = np.deg2rad(at_dict[i+1]['lumbar_rot_final'])
    lumbar_ben_final = np.deg2rad(at_dict[i+1]['lumbar_ben_final'])

    results_path = os.path.abspath(project_path + "/Results/Predict/Trunk/" +
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
                                osim.MocoBounds(0.01, 1),
                                osim.MocoInitialBounds(0.01))
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

    problem.setStateInfo("/jointset/hip_l/hip_adduction_l/value",
                         osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
                         osim.MocoInitialBounds(0))

    problem.setStateInfo("/jointset/hip_l/hip_rotation_l/value",
                         osim.MocoBounds(np.deg2rad(-0.01), np.deg2rad(0.01)),
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
    problem.addGoal(osim.MocoControlGoal('effort', 1))


    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(30)
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

