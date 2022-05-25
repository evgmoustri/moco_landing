# from csv import reader
# with open('Results/Compare/predict_hip_effort0.001_2'
#           '/predict_hip_effort0.001_internal_output.csv', 'r') as read_obj:
#     csv_reader = reader(read_obj)
#     for row in csv_reader:
#         print(*row,sep = " & ")


from csv import reader
with open('Results/Compare/predict_hip_effort0.001_2'
          '/predict_hip_effort0.001_external_output.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        print(*row,sep = " & ")
