from csv import reader
with open('Results/Compare/predict_trunk_effort0.002_2'
          '/predict_trunk_flexion_output.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        print(*row,sep = " & ")

# from csv import reader
# with open('Results/Compare/predict_trunk_effort0.002_2'
#           '/predict_trunk_extension_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")

# from csv import reader
# with open('Results/Compare/predict_trunk_effort0.002_2'
#           '/predict_trunk_ben_right_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")

# from csv import reader
# with open('Results/Compare/predict_trunk_effort0.002_2'
#           '/predict_trunk_ben_left_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")

# from csv import reader
# with open('Results/Compare/predict_trunk_effort0.002_2'
#           '/predict_trunk_rot_int_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")

# from csv import reader
# with open('Results/Compare/predict_trunk_effort0.002_2'
#           '/predict_trunk_rot_ext_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")