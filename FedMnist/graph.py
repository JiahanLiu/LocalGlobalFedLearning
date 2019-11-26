import csv
import json
import matplotlib.pyplot as plt 

def plot_multiple(file_list, labels_list, model_num, graph_title):
    balanced_lists = []
    accuracy_lists = []
    labels = []

    for i in range(len(file_list)):
        bal_list, acc_list = read_csv(file_list[i])
        balanced_lists.append(bal_list)
        accuracy_lists.append(acc_list)
        labels.append(labels_list[i] + " " + str(model_num))
    plot(balanced_lists, accuracy_lists, graph_title, labels)
    

def plot(balanced_percentage_lists, accuracy_lists, graph_title, labels):
    for i in range(len(balanced_percentage_lists)):
        plt.plot(balanced_percentage_lists[i], accuracy_lists[i], label=labels[i])
    plt.xlabel('Percentage of Balanced Data', fontsize=24)
    plt.ylabel('Test Set Accuracy', fontsize=24)
    plt.title(graph_title, fontsize=24)
    plt.legend(prop={'size': 24})
    plt.show()

def read_csv(file_path):
    balanced_percentage_list = []
    accuracy_list = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)
        print(file_path)
        for row in csv_reader:
            print(row)
            balanced_percentage_list.append(float(row[0]))
            accuracy_list.append(float(row[3]))

    return balanced_percentage_list, accuracy_list

def main():
    with open('config.json') as config_file:
        config = json.load(config_file)
        FED_NETFC1_TEST_UNBALANCED_FILE = config['results']['FED_NETFC1_TEST_UNBALANCED_FILE']
        FED_NETC2R3_TEST_UNBALANCED_FILE = config['results']['FED_NETC2R3_TEST_UNBALANCED_FILE']
        FED_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['FED_NETCR3R3_TEST_UNBALANCED_FILE']
        LOCAL_NETFC1_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETFC1_TEST_UNBALANCED_FILE']
        LOCAL_NETC2R3_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETC2R3_TEST_UNBALANCED_FILE']
        LOCAL_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['LOCAL_NETCR3R3_TEST_UNBALANCED_FILE']
        CENTRAL_NETFC1_TEST_BALANCED_FILE = config['results']['CENTRAL_NETFC1_TEST_BALANCED_FILE']
        CENTRAL_NETC2R3_TEST_BALANCED_FILE = config['results']['CENTRAL_NETC2R3_TEST_BALANCED_FILE']
        CENTRAL_NETCR3R3_TEST_BALANCED_FILE = config['results']['CENTRAL_NETCR3R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE']
        SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE']
        SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE = config['results']['SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE']

        FED_NETFC1_TEST_BALANCED_FILE = config['results']['FED_NETFC1_TEST_BALANCED_FILE']
        FED_NETC2R3_TEST_BALANCED_FILE = config['results']['FED_NETC2R3_TEST_BALANCED_FILE']
        FED_NETCR3R3_TEST_BALANCED_FILE = config['results']['FED_NETCR3R3_TEST_BALANCED_FILE']
        LOCAL_NETFC1_TEST_BALANCED_FILE = config['results']['LOCAL_NETFC1_TEST_BALANCED_FILE']
        LOCAL_NETC2R3_TEST_BALANCED_FILE = config['results']['LOCAL_NETC2R3_TEST_BALANCED_FILE']
        LOCAL_NETCR3R3_TEST_BALANCED_FILE = config['results']['LOCAL_NETCR3R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE']
        SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE = config['results']['SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE']

    # model_num = 1
    # file_list = [FED_NETFC1_TEST_UNBALANCED_FILE, LOCAL_NETFC1_TEST_UNBALANCED_FILE, 
    #     SELECTIVE_FED_NETFC1_TEST_UNBALANCED_FILE]
    # labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    # graph_title = "Single Layer ReLu Model with Unbalanced Test Set"
    # plot_multiple(file_list, labels_list, model_num, graph_title)

    # model_num = 1
    # file_list = [FED_NETFC1_TEST_BALANCED_FILE, LOCAL_NETFC1_TEST_BALANCED_FILE, 
    #     SELECTIVE_FED_NETFC1_TEST_BALANCED_FILE]
    # labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    # graph_title = "Single Layer ReLu Model with Balanced Test Set"
    # plot_multiple(file_list, labels_list, model_num, graph_title)

    # model_num = 2
    # file_list = [FED_NETC2R3_TEST_UNBALANCED_FILE, LOCAL_NETC2R3_TEST_UNBALANCED_FILE, 
    #     SELECTIVE_FED_NETC2R3_TEST_UNBALANCED_FILE]
    # labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    # graph_title = "Four Layer Convolutional and ReLu Model with Unbalanced Test Set"
    # plot_multiple(file_list, labels_list, model_num, graph_title)

    # model_num = 2
    # file_list = [FED_NETC2R3_TEST_BALANCED_FILE, LOCAL_NETC2R3_TEST_BALANCED_FILE, 
    #     SELECTIVE_FED_NETC2R3_TEST_BALANCED_FILE]
    # labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    # graph_title = "Four Layer Convolutional and ReLu Model with Balanced Test Set"
    # plot_multiple(file_list, labels_list, model_num, graph_title)

    # model_num = 3
    # file_list = [FED_NETCR3R3_TEST_UNBALANCED_FILE, LOCAL_NETCR3R3_TEST_UNBALANCED_FILE, 
    #     SELECTIVE_FED_NETCR3R3_TEST_UNBALANCED_FILE]
    # labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    # graph_title = "Six Layer Extra Wide Convolutional and ReLu Model with Unbalanced Test Set"
    # plot_multiple(file_list, labels_list, model_num, graph_title)

    model_num = 3
    file_list = [FED_NETCR3R3_TEST_BALANCED_FILE, LOCAL_NETCR3R3_TEST_BALANCED_FILE, 
        SELECTIVE_FED_NETCR3R3_TEST_BALANCED_FILE]
    labels_list = ["Federated Model", "Local Model", "Selective Federated Model"]
    graph_title = "Six Layer Extra Wide Convolutional and ReLu Model with Balanced Test Set"
    plot_multiple(file_list, labels_list, model_num, graph_title)

if __name__ == "__main__":
    main()