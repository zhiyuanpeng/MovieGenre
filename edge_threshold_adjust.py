import numpy as np


def x_write(list_name, list_to_file_name):
    with open(list_to_file_name, 'a+') as f:
        for line_value in list_name:
            f.write(str(line_value) + "\n")


def read_to_list_threshold(file_name, threshold):
    list_threshold = []
    with open("./data/split_test/" + file_name + "_predict_1000.txt", "r") as f:
        without_round = f.readlines()
        for value in without_round:
            if float(value) > threshold:
                list_threshold.append(1)
            else:
                list_threshold.append(0)
    return list_threshold


def round_with_threshold(file_name, threshold):
    round_list = read_to_list_threshold(file_name, threshold)
    x_write(round_list, "./data/split_test/" + file_name + "_predict_round_" + str(round(threshold, 2)) + "_1000.txt")


def adjust_threshold(file_dire, threshold):
    predict = np.loadtxt("./data/split_test/" + file_dire + "_predict_575.txt")
    label = np.loadtxt("./data/split_test/" + file_dire + "_truth_575.txt", dtype=int)
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
    print("with threshold " + str(round(threshold, 2)) + ", the accuracy is " + str(right_sum/predict.shape[0]))
    x_write(predict_round, "./data/store/" + file_dire + "/" + file_dire + "_predict_round_" + str(round(threshold, 2)) + "_575.txt")


def optimal_search(file_name):
    threshold = 0.5
    for index in range(50):
        threshold += 0.01
        adjust_threshold(file_name, threshold)


def main():
    #optimal_search("l1_l2")
    #round_with_threshold("l0_l1", 0.92)
    round_with_threshold("l0_l3", 0.98)
    round_with_threshold("l0_l4", 0.54)
    round_with_threshold("l0_l5", 0.86)
    round_with_threshold("l1_l2", 0.91)


if __name__ == '__main__':
    main()


