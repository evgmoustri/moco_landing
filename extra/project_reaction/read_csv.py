from csv import reader
with open('Results/Compare/'
          '/predict_effort_output.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for row in csv_reader:
        print(*row,sep = " & ")
