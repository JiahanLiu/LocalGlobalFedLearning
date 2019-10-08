import csv

RESULTS_DIR = "results/"
AGGREGATE_PATH = RESULTS_DIR + "catboost.csv"

def gen_csv(file_path):
    with open(file_path, 'w') as csvfile:
        global_f1, global_acc = gen_global_catboost()
        local_f1, local_acc = gen_local_catboost()
        node5_f1, node5_acc = gen_group_catboost()
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Average Prediction Error Percentage'])
        filewriter.writerow(['Model', 'Human Activity'])
        filewriter.writerow(['Global', str((1-global_acc) * 100)])
        filewriter.writerow(['5 Node', str((1-node5_acc) * 100)])
        filewriter.writerow(['Local', str((1-local_acc) * 100)])

        filewriter.writerow([''])

        filewriter.writerow(['1 - F1 Scores Percentage'])
        filewriter.writerow(['Model', 'Human Activity'])
        filewriter.writerow(['Global', str((1-global_f1) * 100)])
        filewriter.writerow(['5 Node', str((1-node5_f1) * 100)])
        filewriter.writerow(['Local', str((1-local_f1) * 100)])

def avg_accuracy_catboost(catboost_files_list):
    f1_sum = 0
    accuracy_sum = 0

    for accuracy_file in catboost_files_list:    
        with open(accuracy_file, 'r') as file:
            f1_line = file.readline()
            f1_index = f1_line.index(':')
            f1_sum += float(f1_line[f1_index+2:])
            accuracy_line = file.readline()
            accuracy_index = accuracy_line.index(':')
            accuracy_sum += float(accuracy_line[accuracy_index+2:])

    return f1_sum/len(catboost_files_list), accuracy_sum/len(catboost_files_list)

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