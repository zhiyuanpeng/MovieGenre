import numpy as np
import random


def x_write(list_name, list_to_file_name):
    """
    write a list of test value to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def y_write(list_name, list_to_file_name):
    """
    write the test result matrix to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(int(line_value[i]))
                    s += " "
                else:
                    s += str(int(line_value[i]))
                    s += "\n"
            f.write(s)


def y_write_float(list_name, list_to_file_name):
    """
    wirte the float test result matrix to file
    :param list_name:
    :param list_to_file_name:
    :return: no
    """
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            s = ""
            for i in range(len(line_value)):
                if i != len(line_value) - 1:
                    s += str(line_value[i])
                    s += " "
                else:
                    s += str(line_value[i])
                    s += "\n"
            f.write(s)


def load_data(node_list, edge_list):
    """
    load text
    :param node_list: a list of the nodes
    :param edge_list: a list of the edges
    :return:
    node_predict a list of matrix according to the node_list
    edge_predict a list of matrix according to the edge_list
    """
    # convert the input list to string list
    node_str = ["l" + str(i) for i in node_list]
    edge_str = ["l" + str(i[0]) + "_" + "l" + str(i[1]) for i in edge_list]
    node_predict = []
    node_predict_round = []
    edge_predict = []
    edge_predict_round = []
    edge_truth = []
    # for node
    for node in node_str:
        node_path = "data/store/" + node + "/" + node
        node_predict.append(np.loadtxt(node_path + "_predict.txt"))
        node_predict_round.append(np.loadtxt(node_path + "_predict_round.txt", dtype=int))
    # for edge
    for edge in edge_str:
        edge_path = "data/store/" + edge + "/" + edge
        edge_predict.append(np.loadtxt(edge_path + "_predict.txt"))
        edge_predict_round.append(np.loadtxt(edge_path + "_predict_round.txt", dtype=int))
        # for edge truth
        edge_truth_path = "data/split_test/" + edge + "_truth.txt"
        edge_truth.append(np.loadtxt(edge_truth_path, dtype=int))
    return node_predict, node_predict_round, edge_predict, edge_predict_round, edge_truth


def write_edge_truth(edge_list, y_test):
    """
    write the true edge info to split
    :param edge_list: a list of edge
    :param y_test: the result of the test
    :return: no
    """
    for edge in edge_list:
        edge_name = "l" + str(edge[0]) + "_" + "l" + str(edge[1])
        true_edge = [y_test[i, edge[0]] * y_test[i, edge[1]] for i in range(y_test.shape[0])]
        true_edge_path = "data/split_test/" + edge_name + "_truth.txt"
        with open(true_edge_path, "a+") as f:
            for value in true_edge:
                f.write(str(value) + "\n")


def split_data(node_list, edge_list, node_predict, node_predict_round, edge_predict, edge_predict_round, edge_truth,
               y_test, original_predict, original_predict_round):
    """
    split data to 1400 and 1478 randomly
    :param node_list:
    :param edge_list:
    :param node_predict:
    :param node_predict_round:
    :param edge_predict:
    :param edge_predict_round:
    :param edge_truth:
    :param y_test:
    :param original_predict:
    :param original_predict_round:
    :return: the split result
    """
    random.seed(0)
    sample_index = random.sample(range(2878), 1400)
    #
    node_str = ["l" + str(i) for i in node_list]
    edge_str = ["l" + str(i[0]) + "_" + "l" + str(i[1]) for i in edge_list]
    # for node
    for node_index in range(len(node_list)):
        node_predict_1400 = []
        node_predict_1400_round = []
        node_predict_1478 = []
        node_predict_1478_round = []
        for index in range(y_test.shape[0]):
            if index in sample_index:
                node_predict_1400.append(node_predict[node_index][index])
                node_predict_1400_round.append(node_predict_round[node_index][index])
            else:
                node_predict_1478.append(node_predict[node_index][index])
                node_predict_1478_round.append(node_predict_round[node_index][index])
        x_write(node_predict_1400, "data/split_test/" + node_str[node_index] + "_predict_1400.txt")
        x_write(node_predict_1400_round, "data/split_test/" + node_str[node_index] + "_predict_1400_round.txt")
        x_write(node_predict_1478, "data/split_test/" + node_str[node_index] + "_predict_1478.txt")
        x_write(node_predict_1478_round, "data/split_test/" + node_str[node_index] + "_predict_1478_round.txt")
    # for edge
    for edge_index in range(len(edge_list)):
        edge_predict_1400 = []
        edge_predict_1400_round = []
        edge_predict_1478 = []
        edge_predict_1478_round = []
        edge_truth_1400 = []
        edge_truth_1478 = []
        for index in range(y_test.shape[0]):
            if index in sample_index:
                edge_predict_1400.append(edge_predict[edge_index][index])
                edge_predict_1400_round.append(edge_predict_round[edge_index][index])
                edge_truth_1400.append(edge_truth[edge_index][index])
            else:
                edge_predict_1478.append(edge_predict[edge_index][index])
                edge_predict_1478_round.append(edge_predict_round[edge_index][index])
                edge_truth_1478.append(edge_truth[edge_index][index])
        x_write(edge_predict_1400, "data/split_test/" + edge_str[edge_index] + "_predict_1400.txt")
        x_write(edge_predict_1400_round, "data/split_test/" + edge_str[edge_index] + "_predict_1400_round.txt")
        x_write(edge_predict_1478, "data/split_test/" + edge_str[edge_index] + "_predict_1478.txt")
        x_write(edge_predict_1478_round, "data/split_test/" + edge_str[edge_index] + "_predict_1478_round.txt")
        x_write(edge_truth_1400, "data/split_test/" + edge_str[edge_index] + "_truth_1400.txt")
        x_write(edge_truth_1478, "data/split_test/" + edge_str[edge_index] + "_truth_1478.txt")
    # split y_test, original_predict, original_predict_round
    y_test_1400 = []
    original_predict_1400 = []
    original_predict_1400_round = []
    y_test_1478 = []
    original_predict_1478 = []
    original_predict_1478_round = []
    for index in range(y_test.shape[0]):
        if index in sample_index:
            y_test_1400.append(y_test[index])
            original_predict_1400.append(original_predict[index])
            original_predict_1400_round.append(original_predict_round[index])
        else:
            y_test_1478.append(y_test[index])
            original_predict_1478.append(original_predict[index])
            original_predict_1478_round.append(original_predict_round[index])
    y_write(y_test_1400, "data/split_test/y_test_1400.txt")
    y_write(y_test_1478, "data/split_test/y_test_1478.txt")
    y_write_float(original_predict_1400, "data/store/original/original_predict_1400.txt")
    y_write_float(original_predict_1478, "data/store/original/original_predict_1478.txt")
    y_write(original_predict_1400_round, "data/store/original/original_predict_1400_round.txt")
    y_write(original_predict_1478_round, "data/store/original/original_predict_1478_round.txt")


def main():
    node_list = [i for i in range(15)]
    edge_list = [(0, 3), (0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (0, 14), (1, 4), (1, 7), (1, 8), (1, 11), (1, 13),
                 (2, 3), (2, 4)]
    y_test = np.loadtxt("data/processed/y_test.txt", dtype=int)
    original_predict = np.loadtxt("data/store/original/original_predict.txt")
    original_predict_round = np.loadtxt("data/store/original/original_predict_round.txt")
    write_edge_truth(edge_list, y_test)
    node_predict, node_predict_round, edge_predict, edge_predict_round, edge_truth = load_data(node_list, edge_list)
    split_data(node_list, edge_list, node_predict, node_predict_round, edge_predict, edge_predict_round, edge_truth,
               y_test, original_predict, original_predict_round)


if __name__ == "__main__":
    main()
