import numpy as np


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def read_to_list_threshold(file_name, threshold):
    list_threshold = []
    with open("./data/split_test/" + file_name + "_predict_2000.txt", "r") as f:
        without_round = f.readlines()
        for value in without_round:
            if float(value) > threshold:
                list_threshold.append(1)
            else:
                list_threshold.append(0)
    return list_threshold


def round_with_threshold(file_name, threshold):
    round_list = read_to_list_threshold(file_name, threshold)
    x_write(round_list, "./data/split_test/" + file_name + "_predict_round_" + str(threshold) + "_2000.txt")


def adjust_threshold(file_dire, threshold):
    """
    if the predict value > threshold, value = 1
    :param file_dire:
    :param threshold:
    :return: the predict_round and right_sum
    """
    predict = np.loadtxt("./data/split_test/" + file_dire + "_predict_878.txt")
    label = np.loadtxt("./data/split_test/" + file_dire + "_truth_878.txt", dtype=int)
    right_sum = 0
    predict_round = []
    for i in range(predict.shape[0]):
        if predict[i, ] > threshold:
            after_round = 1
        else:
            after_round = 0
        predict_round.append(after_round)
        if after_round == label[i, ]:
            right_sum += 1
    return predict_round, right_sum/predict.shape[0]


def optimal_search(file_name):
    """
    find the maximum right_sum and the corresponding threshold, predict result
    :param file_name:
    :return:
    """
    threshold = 0.5
    predict_list = []
    accuracy_list = []
    threshold_list = []
    for index in range(50):
        threshold += 0.01
        predict_round, accuracy = adjust_threshold(file_name, threshold)
        predict_list.append(predict_round)
        accuracy_list.append(accuracy)
        threshold_list.append(threshold)
    max_index = accuracy_list.index(np.max(accuracy_list))
    return predict_list[max_index], accuracy_list[max_index], round(threshold_list[max_index], 2)


def main():
    edge_list = [(0, 3), (0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (0, 14), (1, 4), (1, 7), (1, 8), (1, 11), (1, 13),
                 (2, 3), (2, 4)]
    optimal_file = "data/optimal_result.txt"
    for edge in edge_list:
        # get the optimal
        edge_name = "l" + str(edge[0]) + "_" + "l" + str(edge[1])
        opt_predict, opt_accracy, opt_threshold = optimal_search(edge_name)
        # use the optimal threshold to deal the left 2000 data
        round_with_threshold(edge_name, opt_threshold)
        with open(optimal_file, "a+") as opt:
            opt.write(edge_name + " with optimal threshold " + str(opt_threshold) + ", the accuracy is " + str(opt_accracy) + "\n")
        x_write(opt_predict, "./data/store/" + edge_name + "/" + edge_name + "_predict_round_"
                + str(opt_threshold) + "_878.txt")


if __name__ == '__main__':
    main()


