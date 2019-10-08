import csv

RESULTS_DIR = "results/"
AGGREGATE_PATH = RESULTS_DIR + "catboost.csv"

def gen_csv(file_path):
    with open(file_path, 'wb') as csvfile:
        global_f1_error = 1 - gen_global_catboost()
        local_f1_error = 1 - gen_local_catboost()
        node5_f1_error = 1 - gen_group_catboost()
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['1 - F1 Scores Percentage'])
        filewriter.writerow(['Model', 'Human Activity'])
        filewriter.writerow(['Global', str(global_f1_error * 100)])
        filewriter.writerow(['5 Node', str(node5_f1_error * 100)])
        filewriter.writerow(['Local', str(local_f1_error * 100)])

def avg_accuracy_catboost(catboost_files_list):
    sum = 0

    for accuracy_file in catboost_files_list:    
        with open(accuracy_file, 'r') as file:
            line = file.readline()
            index = line.index(':')
            sum += float(line[index+2:])

    return sum/len(catboost_files_list)

def gen_local_catboost():
    local_catboost_results = []

    for i in range(1,31):
        accuracy_file = RESULTS_DIR + "CatBoostAccuracy_" + str(i) + ".txt"
        local_catboost_results.append(accuracy_file)

    return avg_accuracy_catboost(local_catboost_results)

def gen_group_catboost():
    group_catboost_results = []

    for i in range(0,5):
        accuracy_file = RESULTS_DIR + "CatBoostAccuracy_" + str(i * 5 + 1) + "_" + str(i * 5 + 5) + ".txt"
        group_catboost_results.append(accuracy_file)

    return avg_accuracy_catboost(group_catboost_results)

def gen_global_catboost():
    central_catboost_results = []
    accuracy_file = RESULTS_DIR + "CatBoostAccuracy_all.txt"
    central_catboost_results.append(accuracy_file)
    
    return avg_accuracy_catboost(central_catboost_results)

gen_csv(AGGREGATE_PATH)