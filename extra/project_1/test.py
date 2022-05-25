import os
import opensim as osim
import numpy as np
from utils import fix_for_left_side,fix_states_from_scone, plot,\
    readExternalLoadsFile, plot_muscle_forces_Sritharan, estimate_cop, \
    double_isometric_force, fix_sto_2392, plot_muscle_forces_Mokhtazadeh

# project_path = os.getcwd()
# data_file = "/scone_solution.sto"
# data_path = os.path.abspath(project_path + "\Data_files")
# filename = os.path.abspath(data_path + data_file)
#
# fix_for_left_side(filename)
#

# scenario = 'case_Gait2392_inverse'
# project_path = os.getcwd()
# data_file = "/tendon_forces.sto"
# data_path = os.path.abspath(project_path + "/Results/" + scenario)
# filename = os.path.abspath(data_path + data_file)
# plot_muscle_forces(filename,scenario,data_path)


# scenario = "case_Gait2392_track_trunk_ext20"
# scenario = "case_Gait2392_track"
# project_path = os.getcwd()
# results_path = os.path.abspath(project_path + "/Results/" + scenario)
# estimate_cop(results_path)

# project_path = os.getcwd()
# model_path = os.path.abspath(project_path + "/Opensim_Models")
# model_file = os.path.abspath(model_path
#                              +"/Gait2392\gait2392_only_left_UPD.osim")
#
# double_isometric_force(model_file)


# data_file = "/solution.sto"
#
# data_path = os.path.abspath(project_path + "/Results/case_Gait2354_track/")
# stateRef = os.path.abspath(data_path + data_file)
# fix_sto_2392(stateRef)
#
#
# plot_muscle_forces_Sritharan(results_path + "/tendon_forces.sto", scenario, results_path)
# plot_muscle_forces_Mokhtazadeh(results_path + "/tendon_forces.sto", scenario, results_path)