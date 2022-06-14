# -*- coding: utf-8 -*-
"""
@author:  Evgenia Moustridi (evgmoustridi@gmail.com)
"""

import os
import opensim as osim
import numpy as np
from utils import fix_controls, read_GRF_JR_file, index_containing_substring,\
    estimate_cop, fix_sto_2392


project_path = os.getcwd()
# at_dict = {
#     1: {'scenario': 'weak_quad', 'model': 'gait2392_only_left_weak_quad.osim'},
#     2: {'scenario': 'strong_quad', 'model':
#         'gait2392_only_left_strong_quad.osim'},
#     3: {'scenario': 'weak_ham', 'model': 'gait2392_only_left_weak_ham.osim'},
#     4: {'scenario': 'strong_ham', 'model':
#         'gait2392_only_left_strong_ham.osim'}
# }
at_dict = {
    1: {'scenario': 'normal', 'model': 'gait2392_only_left.osim'}}

# at_dict = {
#     1: {'scenario': 'strong_quad_weak_ham', 'model':
#         'gait2392_only_left_strong_quad_weak_ham.osim'},
#     2: {'scenario': 'strong_quad_strong_ham', 'model':
#         'gait2392_only_left_strong_quad_strong_ham.osim'},
#     3: {'scenario': 'weak_quad_weak_ham', 'model':
#         'gait2392_only_left_weak_quad_weak_ham.osim'},
#     4: {'scenario': 'weak_quad_strong_ham', 'model':
#         'gait2392_only_left_weak_quad_strong_ham.osim'},
# }

for i in range(len(at_dict)):

    scenario = at_dict[i+1]['scenario']
    model_name = at_dict[i + 1]['model']

    model_path = os.path.abspath(project_path + "/Opensim_Models/Gait2392/")
    model_file = os.path.abspath(model_path + "/" + model_name)
    modelProcessor = osim.ModelProcessor(model_file)
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    modelProcessor.append(osim.ModOpAddReserves(100))
    model = modelProcessor.process()

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
                         osim.MocoBounds(0.7, 1.25),
                         osim.MocoInitialBounds(1.25),
                         osim.MocoFinalBounds(0.75, 0.85))
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
    problem.setStateInfo("/jointset/back/lumbar_extension/value",
                         osim.MocoBounds(-0.1, 0.1))
    problem.setStateInfo("/jointset/back/lumbar_bending/value",
                         osim.MocoBounds(-0.1, 0.1))
    problem.setStateInfo("/jointset/back/lumbar_rotation/value",
                         osim.MocoBounds(-0.1, 0.1))

    # Set Goals
    problem.addGoal(osim.MocoControlGoal('effort', 0.002))


    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(100)
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

