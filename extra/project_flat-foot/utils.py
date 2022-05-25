import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET
from lxml import etree
import opensim as osim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# plot attribute(time) from file
def plot(attribute_label, attribute_array, time_array, title, results_path):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.plot(time_array,attribute_array)
    plt.xlabel('Time (sec)', fontsize=18)
    plt.ylabel('{}'.format(attribute_label), fontsize=18)
    plt.legend()
    plt.savefig(os.path.abspath(results_path + '/' + title + ".png"))
    plt.show()

def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def settings_reader(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    childs = []
    for child in root:
        childs.append(child)
    # print(childs)
    data = {}
    for child in root:
        data[child.tag] = child.text

    return data


def settings_writer(xml_file, dict):
    tree = etree.parse(xml_file)
    for i in dict:
        parent_tag = dict[i]['parent']
        element = dict[i]['element']
        tag = dict[i]['tag']
        parent = tree.find(parent_tag)
        childs = []
        for child in parent:
            childs.append(child)
        for child in parent:
            if child.tag == element:
                child.text = tag
    xml_format = etree.tostring(tree, encoding="unicode", pretty_print=True)
    with open(xml_file, 'wb') as f:
        f.write(xml_format.encode("utf-8"))

def read_GRF_JR_file(filename):
    if not os.path.exists(filename):
        print('file do not exists')

    with open(filename) as file_id:
        nr = sum(1 for line in file_id)

    file_id = open(filename, 'r')
    # read header
    next_line = file_id.readline()
    header = [next_line]
    counter = 2
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])
        counter = counter +1

        next_line = file_id.readline()
        header.append(next_line)
    nr = nr - counter

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()
    nc = len(labels)
    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data

def readExternalLoadsFile(filename):
    if not os.path.exists(filename):
        print('file do not exists')

    with open(filename) as file_id:
        nr = sum(1 for line in file_id)

    file_id = open(filename, 'r')
    # read header
    next_line = file_id.readline()
    header = [next_line]
    counter = 0
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])
        counter = counter +1

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()
    nc = len(labels)
    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        if len(d)!=0:
            data.append(d)

    file_id.close()

    return header, labels, data



def readMotionFile(filename):
    """Reads OpenSim .sto files.

    Parameters
    ----------
    filename: str absolute path to the .sto file

    Returns
    -------
    header: list of str the header of the .sto labels: list of str the labels of
        the columns data: list of lists an array of the data

    """

    if not os.path.exists(filename):
        print('file do not exists')

    with open(filename) as file_id:
        nr = sum(1 for line in file_id)

    file_id = open(filename, 'r')
    # read header
    next_line = file_id.readline()
    header = [next_line]
    counter = 2
    while not 'endheader' in next_line:
        if 'datacolumns' in next_line:
            nc = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'datarows' in next_line:
            nr = int(next_line[next_line.index(' ') + 1:len(next_line)])
        elif 'nColumns' in next_line:
            nc = int(next_line[next_line.index('=') + 1:len(next_line)])
        elif 'nRows' in next_line:
            nr = int(next_line[next_line.index('=') + 1:len(next_line)])
        counter = counter +1

        next_line = file_id.readline()
        header.append(next_line)

    # process column labels
    next_line = file_id.readline()
    if next_line.isspace() == True:
        next_line = file_id.readline()

    labels = next_line.split()
    nc = len(labels)
    nr = nr - counter

    # get data
    data = []
    for i in range(1, nr + 1):
        d = [float(x) for x in file_id.readline().split()]
        data.append(d)

    file_id.close()

    return header, labels, data


def flip_signs(filename, pattern, save_file_prefix):

    header, labels, data = readMotionFile(filename)
    data = np.asarray(data)

    for idx, label in enumerate(labels):
        if pattern in label:
            data[:, idx] *= -1

    first_line = header
    second_line = labels
    third_line = data
    dir = Path(filename).parent.as_posix()
    print(dir)
    with open(dir + '/{}_edited_solution.sto'.format(save_file_prefix),
              'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')


def fix_states_from_scone(filename, flip_angle_sign=False):
    header, labels, data = readMotionFile(filename)
    data = np.asarray(data)
    m, n = data.shape

    for idx, label in enumerate(labels):
        if label == "pelvis_tilt":
            labels[idx] = '/jointset/ground_pelvis/pelvis_tilt/value'
        elif label == "pelvis_tx":
            labels[idx] = '/jointset/ground_pelvis/pelvis_tx/value'
        elif label == "pelvis_ty":
            labels[idx] = '/jointset/ground_pelvis/pelvis_ty/value'
        elif label == "hip_flexion_r":
            labels[idx] = '/jointset/hip_r/hip_flexion_r/value'
        elif label == "knee_angle_r":
            labels[idx] = '/jointset/knee_r/knee_angle_r/value'
            if flip_angle_sign:
                data[:, idx] *= -1
        elif label == "ankle_angle_r":
            labels[idx] = '/jointset/ankle_r/ankle_angle_r/value'
        elif label == "hip_flexion_l":
            labels[idx] = '/jointset/hip_l/hip_flexion_l/value'
        elif label == "knee_angle_l":
            labels[idx] = '/jointset/knee_l/knee_angle_l/value'
            if flip_angle_sign:
                data[:, idx] *= -1
        elif label == "ankle_angle_l":
            labels[idx] = '/jointset/ankle_l/ankle_angle_l/value'
        elif label == "pelvis_tilt_u":
            labels[idx] = '/jointset/ground_pelvis/pelvis_tilt/speed'
        elif label == "pelvis_tx_u":
            labels[idx] = '/jointset/ground_pelvis/pelvis_tx/speed'
        elif label == "pelvis_ty_u":
            labels[idx] = '/jointset/ground_pelvis/pelvis_ty/speed'
        elif label == "hip_flexion_r_u":
            labels[idx] = '/jointset/hip_r/hip_flexion_r/speed'
        elif label == "knee_angle_r_u":
            labels[idx] = '/jointset/knee_r/knee_angle_r/speed'
        elif label == "ankle_angle_r_u":
            labels[idx] = '/jointset/ankle_r/ankle_angle_r/speed'
        elif label == "hip_flexion_l_u":
            labels[idx] = '/jointset/hip_l/hip_flexion_l/speed'
        elif label == "knee_angle_l_u":
            labels[idx] = '/jointset/knee_l/knee_angle_l/speed'
        elif label == "ankle_angle_l_u":
            labels[idx] = '/jointset/ankle_l/ankle_angle_l/speed'

    data = data[:, 0:19]
    labels = labels[0:19]
    # data = data[:, 0:47]
    # labels = labels[0:47]

    first_line = header
    second_line = labels
    third_line = data
    dir = Path(filename).parent.as_posix()
    print(dir)
    with open(dir + '/edited_solution.sto', 'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')


def fix_controls(filename):
    header, labels, data = readMotionFile(filename)
    data = np.asarray(data)
    m, n = data.shape
    i = 0
    for label in labels:
        if "/forceset/" in label:
            labels[i] = labels[i].replace("/forceset/","")
        i = i + 1
    first_line = header
    second_line = labels
    third_line = data
    with open(filename, 'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')


def delete_left_side_muscles(model_file):
    model_file = model_file
    model = osim.Model(model_file)
    num_of_muscles = model.getForceSet().getSize()
    for i in range(num_of_muscles):
        musc = model.getForceSet().get(i)
        musc_name = musc.getName()
        if "_l" in musc_name:
            musc.remove()


def index_containing_substring(list_str, pattern):
    """For a given list of strings finds the index of the element that
    contains the substring.

    Parameters
    ----------
    list_str: list of str

    pattern: str pattern


    Returns
    -------
    indices: list of int the indices where the pattern matches

    """
    indices = []
    for i, s in enumerate(list_str):
        if pattern in s:
            indices.append(i)

    return indices


def estimate_cop(results_dir):
    """

    Parameters
    ----------
    results_dir: directory to save new grf
    sto_file: grf file with cop at origin

    Returns
    -------

    """

    sto_file = os.path.abspath(results_dir + '/GRF.sto')
    attr_m_z = 'ground_torque_l_z'
    attr_m_x = 'ground_torque_l_x'
    attr_f_y = 'ground_force_l_vy'
    attr_z = 'ground_force_l_pz'
    attr_x = 'ground_force_l_px'

    header, labels, data = read_GRF_JR_file(sto_file)
    data = np.asarray(data)
    m, n = data.shape
    time_array = data[:, 0]

    # i = 0
    for i, label in enumerate(labels):
        if label == attr_m_z:
            m_z = data[:, i]
        elif label == attr_m_x:
            m_x = data[:, i]
        elif label == attr_f_y:
            f_y = data[:, i]
        # i = i+1

    i = 0
    for label in labels:
        if label == attr_x:
            x = np.nan_to_num(np.divide(-m_x, f_y))
            data[:, i] = -x
        elif label == attr_z:
            z = np.nan_to_num(np.divide(m_z, f_y))
            data[:, i] = -z
        i = i + 1

    new_sto_file = os.path.abspath(results_dir + '/GRF_CoP.sto')
    data = data[:, ]
    labels = labels[:]
    first_line = header
    second_line = labels
    third_line = data
    with open(new_sto_file, 'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')


def fix_for_left_side(filename):
    header, labels, data = readMotionFile(filename)
    data = np.asarray(data)
    m, n = data.shape
    i = 0
    r_list = index_containing_substring(labels, '_r')
    l_list = index_containing_substring(labels, '_l')

    for i in r_list:
        t = data[:,i].copy()
        data[:,i] = data[:,i+1]
        data[:,i+1] = t


    first_line = header
    second_line = labels
    third_line = data
    with open(filename, 'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')


def plot_muscle_forces_Sritharan(file_path,scenario,results_path):
    header, labels, data = readExternalLoadsFile(file_path)
    data = np.asarray(data)
    time_array = data[:,0]
    m,n = data.shape
    glut_med_force = np.zeros(m)

    # BW = 75.16 kg = 737.0678161524997 N
    bw =737.0678
    gas_force = []
    quad_force = []
    ham_force = []
    glut_max_force = []
    glut_med_force = []
    erc_force = []

    for idx, label in enumerate(labels):
        if label =='/forceset/soleus_l|tendon_force':
            sol_force = data[:,idx] / bw

        # gastroc
        elif label == '/forceset/med_gas_l|tendon_force':
            gas_force = data[:, idx] / bw
        elif label == '/forceset/lat_gas_l|tendon_force':
            gas_force = np.add(data[:, idx] / bw, gas_force)

        # quadriceps
        elif label == '/forceset/rect_fem_l|tendon_force':
            rect_fem_force = data[:, idx] / bw

        elif label == '/forceset/vas_med_l|tendon_force':
            vasti_force = data[:, idx] / bw
        elif label == '/forceset/vas_int_l|tendon_force':
            vasti_force = np.add(data[:, idx] / bw, vasti_force)
        elif label == '/forceset/vas_lat_l|tendon_force':
            vasti_force = np.add(data[:, idx] / bw, vasti_force)


        # hamstrings
        elif label == '/forceset/semimem_l|tendon_force':
            ham_force = data[:, idx] / bw
        elif label == '/forceset/semiten_l|tendon_force':
            ham_force = np.add(data[:, idx] / bw, ham_force)
        elif label == '/forceset/bifemlh_l|tendon_force':
            ham_force = np.add(data[:, idx] / bw, ham_force)
        elif label == '/forceset/bifemsh_l|tendon_force':
            ham_force = np.add(data[:, idx] / bw, ham_force)


        elif label == '/forceset/glut_max1_l|tendon_force':
            glut_max_force = data[:,idx]
        elif label == '/forceset/glut_max2_l|tendon_force':
            glut_max_force = np.add(data[:,idx],glut_max_force)
        elif label == '/forceset/glut_max3_l|tendon_force':
            glut_max_force = np.add(data[:,idx],glut_max_force) / bw
        elif label == '/forceset/glut_med1_l|tendon_force':
            glut_med_force = data[:,idx]
        elif label == '/forceset/glut_med2_l|tendon_force':
            glut_med_force = np.add(data[:,idx],glut_med_force)
        elif label == '/forceset/glut_med3_l|tendon_force':
            glut_med_force = np.add(data[:,idx],glut_med_force) / bw

        elif label == '/forceset/ercspn_l|tendon_force':
            erc_force = data[:,idx] / bw


    fig, axs = plt.subplots(3, 3,figsize=(15,15))
    fig.suptitle(scenario)
    axs[0, 0].plot(time_array, sol_force)
    axs[0, 0].set_title('Soleus')
    axs[0, 0].axvline(x=0.2, linestyle='--', color='red')
    axs[0, 1].plot(time_array, gas_force)
    axs[0, 1].set_title('Gastrocnemius')
    axs[0, 1].axvline(x=0.2, linestyle='--', color='red')
    axs[0, 2].plot(time_array, rect_fem_force)
    axs[0, 2].set_title('Rectus Femoris')
    axs[0, 2].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 0].plot(time_array, ham_force)
    axs[1, 0].set_title('Hamstrings')
    axs[1, 0].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 1].plot(time_array, glut_max_force)
    axs[1, 1].set_title('Gluteus maximus')
    axs[1, 1].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 2].plot(time_array, glut_med_force)
    axs[1, 2].set_title('Gluteus medius')
    axs[1, 2].axvline(x=0.2, linestyle='--', color='red')
    axs[2, 0].plot(time_array,erc_force)
    axs[2, 0].set_title('erc')
    axs[2, 0].axvline(x=0.2, linestyle='--', color='red')
    axs[2, 1].plot(time_array,vasti_force)
    axs[2, 1].set_title('Vasti')
    axs[2, 1].axvline(x=0.2, linestyle='--', color='red')

    plt.savefig(results_path + '/MuscleForces_Sritharan.png')
    # plt.show()

def plot_muscle_forces_Mokhtazadeh(file_path,scenario,results_path):
    header, labels, data = readExternalLoadsFile(file_path)
    data = np.asarray(data)
    time_array = data[:,0]
    m,n = data.shape
    glut_med_force = np.zeros(m)

    # BW = 75.16 kg = 737.0678161524997 N
    bw =737.0678

    project_path = os.getcwd()
    model_path = os.path.abspath(project_path + "/Opensim_Models")
    model_file = os.path.abspath(model_path
                                 + "/Gait2392\gait2392_only_left.osim")
    model = osim.Model(model_file)
    muscles = model.getMuscles()
    # for j in range(muscles.getSize()):
    #     mus = muscles.get(j)
    #     mus_force = mus.get_max_isometric_force()
    #     mus_name = mus.getName()
    #     if mus_name =='soleus_l':
    #         soleus_max = mus_force
    #     elif mus_name == 'med_gas_l':
    #         med_gas_max = mus_force
    #     elif mus_name == 'lat_gas_l':
    #         lat_gas_max = mus_force
    #     elif mus_name == 'rect_fem_l':
    #         rect_fem_max = mus_force
    #     elif mus_name == 'vas_med_l':
    #         vas_med_max = mus_force
    #     elif mus_name == 'vas_int_l':
    #         vas_int_max = mus_force
    #     elif mus_name == 'vas_lat_l':
    #         vas_lat_max = mus_force
    #     elif mus_name == 'semimem_l':
    #         semimem_max = mus_force
    #     elif mus_name == 'semiten_l':
    #         semiten_max = mus_force
    #     elif mus_name == 'bifemlh_l':
    #         bifemlh_max = mus_force
    #     elif mus_name == 'bifemsh_l':
    #         bifemsh_max = mus_force
    #     elif mus_name == 'glut_max1_l':
    #         glut_max1_max = mus_force
    #     elif mus_name == 'glut_max2_l':
    #         glut_max2_max = mus_force
    #     elif mus_name == 'glut_max3_l':
    #         glut_max3_max = mus_force
    #     elif mus_name == 'glut_med1_l':
    #         glut_med1_max = mus_force
    #     elif mus_name == 'glut_med2_l':
    #         glut_med2_max = mus_force
    #     elif mus_name == 'glut_med3_l':
    #         glut_med3_max = mus_force
    #     elif mus_name == 'ercspn_l':
    #         ercspn_max = mus_force


    for idx, label in enumerate(labels):
        if label =='/forceset/soleus_l|tendon_force':
            sol_force = data[:,idx] / np.max(data[:,idx])

        # gastroc
        elif label == '/forceset/med_gas_l|tendon_force':
            gas_force = data[:, idx] / np.max(data[:,idx])
        elif label == '/forceset/lat_gas_l|tendon_force':
            gas_force = np.add(data[:, idx] / np.max(data[:,idx]), gas_force)

        # quadriceps
        elif label == '/forceset/rect_fem_l|tendon_force':
            quad_force = data[:, idx] / np.max(data[:,idx])
        elif label == '/forceset/vas_med_l|tendon_force':
            quad_force = np.add(data[:, idx] / np.max(data[:,idx]), quad_force)
        elif label == '/forceset/vas_int_l|tendon_force':
            quad_force = np.add(data[:, idx] / np.max(data[:,idx]), quad_force)
        elif label == '/forceset/vas_lat_l|tendon_force':
            quad_force = np.add(data[:, idx] / np.max(data[:,idx]), quad_force)


        # hamstrings
        elif label == '/forceset/semimem_l|tendon_force':
            ham_force = data[:, idx] / np.max(data[:,idx])
        elif label == '/forceset/semiten_l|tendon_force':
            ham_force = np.add(data[:, idx] / np.max(data[:,idx]), ham_force)
        elif label == '/forceset/bifemlh_l|tendon_force':
            ham_force = np.add(data[:, idx] / np.max(data[:,idx]), ham_force)
        elif label == '/forceset/bifemsh_l|tendon_force':
            ham_force = np.add(data[:, idx] / np.max(data[:,idx]), ham_force)


        elif label == '/forceset/glut_max1_l|tendon_force':
            glut_max_force = data[:,idx] /np.max(data[:,idx])
        elif label == '/forceset/glut_max2_l|tendon_force':
            glut_max_force = np.add(data[:,idx] / np.max(data[:,idx]),glut_max_force)
        elif label == '/forceset/glut_max3_l|tendon_force':
            glut_max_force = np.add(data[:,idx] / np.max(data[:,idx]),glut_max_force)
        elif label == '/forceset/glut_med1_l|tendon_force':
            glut_med_force = data[:,idx] / np.max(data[:,idx])
        elif label == '/forceset/glut_med2_l|tendon_force':
            glut_med_force = np.add(data[:,idx] / np.max(data[:,idx]) ,glut_med_force)
        elif label == '/forceset/glut_med3_l|tendon_force':
            glut_med_force = np.add(data[:,idx] / np.max(data[:,idx]) ,glut_med_force)

        elif label == '/forceset/ercspn_l|tendon_force':
            erc_force = data[:,idx] / np.max(data[:,idx])

    gas_force = np.divide(gas_force , 2)
    quad_force = np.divide(quad_force , 4)
    ham_force = np.divide(ham_force , 4)
    glut_max_force = np.divide(glut_max_force , 3)
    glut_med_force = np.divide(glut_med_force , 3)


    fig, axs = plt.subplots(3, 3,figsize=(15,15))
    fig.suptitle(scenario)
    axs[0, 0].plot(time_array, sol_force)
    axs[0, 0].set_title('Soleus')
    axs[0, 0].axvline(x=0.2, linestyle='--', color='red')
    axs[0, 1].plot(time_array, gas_force)
    axs[0, 1].set_title('Gastrocnemius')
    axs[0, 1].axvline(x=0.2, linestyle='--', color='red')
    axs[0, 2].plot(time_array, quad_force)
    axs[0, 2].set_title('Quadriceps')
    axs[0, 2].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 0].plot(time_array, ham_force)
    axs[1, 0].set_title('Hamstrings')
    axs[1, 0].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 1].plot(time_array, glut_max_force)
    axs[1, 1].set_title('Gluteus maximus')
    axs[1, 1].axvline(x=0.2, linestyle='--', color='red')
    axs[1, 2].plot(time_array, glut_med_force)
    axs[1, 2].set_title('Gluteus medius')
    axs[1, 2].axvline(x=0.2, linestyle='--', color='red')
    axs[2, 0].plot(time_array,erc_force)
    axs[2, 0].set_title('erc')
    axs[2, 0].axvline(x=0.2, linestyle='--', color='red')

    plt.savefig(results_path + '/MuscleForces_Mokhtazadeh.png')
    # plt.show()


def double_isometric_force(model_file):
    model= osim.Model(model_file)
    muscles = model.getMuscles()
    for j in range(muscles.getSize()):
        mus = muscles.get(j)
        new_force = mus.get_max_isometric_force() * 2
        mus.set_max_isometric_force(new_force)
    model.finalizeConnections()
    model.printToXML(model_file)



def addCoordinateActuator(modelObject,coordinate,optForce,controlLevel,
                          appendStr):
    """

    Parameters
    ----------
    modelObject - Opensim model object to add actuator to 
    coordinate - string of coordinate name for actuator 
    optForce - value  for actuators optimalforce 
    controlLevel - [x, y] values for max and min control
    appendStr - string to append to coordinate name in setting actuator name
    
    """

    actu = osim.CoordinateActuator()
    actu.setName([coordinate, appendStr])
    actu.setCoordinate(modelObject.getCoordinateSet().get(coordinate))
    actu.setOptimalForce(optForce)
    actu.setMaxControl(controlLevel(1))
    actu.setMinControl(controlLevel(2))
    modelObject.addComponent(actu)



def fix_sto_2392(filename):
    header, labels, data = readMotionFile(filename)
    data = np.asarray(data)
    m, n = data.shape

    # from 2354
    data = data[:, 0:47]
    labels = labels[0:47]

    # from 2392
    data = data[:, 0:47]
    labels = labels[0:47]

    first_line = header
    second_line = labels
    third_line = data
    dir = Path(filename).parent.as_posix()
    print(dir)
    with open(dir + '/edited_solution.sto', 'w') as out:
        out.writelines("%s" % item for item in first_line)
        out.write('\t'.join(second_line) + '\n')
        np.savetxt(out, data, delimiter='\t')





