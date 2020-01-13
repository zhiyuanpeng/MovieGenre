import numpy as np
import math


def bit_to_list(t, n):
    """
    convert an int to list
    @param t: the int
    @param n: the length of the bit
    @return: a lit
    """
    bit_list = [0 for i in range(n)]
    i = -1
    while t != 0:
        bit_list[i] = t % 2
        t = t >> 1
        i -= 1
    return bit_list


def get_candidate(list_length):
    """
    the length of the list
    @param list_length:
    @return: list contains all the possible
    """
    candidate = []
    for i in range(pow(2, list_length)):
        candidate.append(bit_to_list(i, list_length))
    return candidate


def transfer_label(trans_value,):
    """
    trans_value is the possibility of 1 1, we need to get
       0 1
     0 x x
     1 x x
    """
    label_trans_matrix = np.zeros((2, 2))
    label_trans_matrix[0, 0] = -1000
    label_trans_matrix[0, 1] = -1000
    label_trans_matrix[1, 0] = -1000
    label_trans_matrix[1, 1] = trans_value if trans_value != -1000 else -2000
    return label_trans_matrix


#  combine the result of each binary classifier
def get_node_matrix(node_list):
    # get the matrix size
    path = "./data/split_test/"
    example = list(np.loadtxt(path + node_list[0]))
    node_matrix = np.zeros((len(example), len(node_list)))
    for index in range(len(node_list)):
        node_column = list(np.loadtxt(path + node_list[index]))
        node_matrix[:, index] = node_column
    return node_matrix


def get_transfer_matrix(link_list):
    transfer_list = []
    # convert the file to matrix
    transfer_data = get_node_matrix(link_list)
    for line in transfer_data:
        transfer_matrix = np.zeros((len(line) + 1, len(line) + 1))
        transfer_matrix[0, 1] = line[0]
        transfer_matrix[0, 3] = line[1]
        transfer_matrix[0, 4] = line[2]
        transfer_matrix[0, 5] = line[3]
        transfer_matrix[1, 2] = line[4]
        #
        transfer_matrix[1, 0] = line[0]
        transfer_matrix[3, 0] = line[1]
        transfer_matrix[4, 0] = line[2]
        transfer_matrix[5, 0] = line[3]
        transfer_matrix[2, 1] = line[4]
        transfer_list.append(transfer_matrix.copy())
    return transfer_list

def overall_optimal(status_list, transfer_matrix, lambda_value):
    score_list = []
    all_possible = get_candidate(len(status_list))
    for one_possible in all_possible:
        score = 0
        for index in range(len(one_possible)):
            logpossibility = status_list[index] if one_possible[index] == 1 else np.log(1 - math.exp(status_list[index]))
            score += logpossibility
        # add the transfer possibility
        # for 0 1
        label_trans_matrix = transfer_label(transfer_matrix[0, 1])
        score += label_trans_matrix[one_possible[0], one_possible[1]]
        # for 1 2
        label_trans_matrix = transfer_label(transfer_matrix[1, 2])
        score += label_trans_matrix[one_possible[1], one_possible[2]]
        # for 0 3
        label_trans_matrix = transfer_label(transfer_matrix[0, 3])
        score += label_trans_matrix[one_possible[0], one_possible[3]]
        # for 0 4
        label_trans_matrix = transfer_label(transfer_matrix[0, 4])
        score += label_trans_matrix[one_possible[0], one_possible[4]]
        # for 0 5
        label_trans_matrix = transfer_label(transfer_matrix[0, 5])
        score += label_trans_matrix[one_possible[0], one_possible[5]]
        #
        penalty = lambda_value*np.sum(one_possible)
        score += lambda_value*np.sum(one_possible)
        score_list.append(score)
    max_score = max(score_list)
    max_index = score_list.index(max_score)
    max_label = all_possible[max_index]
    return max_score, max_label


def exhust_search(filename, lambda_value):
    node_path_list = ["l0_predict_575.txt", "l1_predict_575.txt", "l2_predict_575.txt", "l3_predict_575.txt",
                      "l4_predict_575.txt", "l5_predict_575.txt"]

    link_path_list = ["l0_l1_predict_round_0.92_1000.txt", "l0_l3_predict_round_0.98_1000.txt", "l0_l4_predict_round_0.54_1000.txt",
                      "l0_l5_predict_round_0.86_1000.txt", "l1_l2_predict_round_0.91_1000.txt"]
    # link_path_list = ["l0_l1_truth_575.txt", "l0_l3_truth_575.txt", "l0_l4_truth_575.txt",
    #                   "l0_l5_truth_575.txt", "l1_l2_truth_575.txt"]
    # node_matrix = v.get_node_matrix(node_path_list)
    node_matrix = np.loadtxt("data/store/original/original_predict_1000.txt")
    transfer_list = get_transfer_matrix(link_path_list)
    with open("data/enumeration/" + filename, "a+") as vf:
        iteration = 0
        for index in range(node_matrix.shape[0]):
            node_matrix_log = np.zeros_like(node_matrix[index, :])
            for i in range(len(node_matrix[index, :])):
                node_matrix_log[i, ] = np.log(node_matrix[index, i])
            transfer_list_log = np.zeros_like(transfer_list[index])
            for i in range(transfer_list[index].shape[0]):
                for j in range(transfer_list[index].shape[1]):
                    if transfer_list[index][i, j] == 1:
                        transfer_list_log[i, j] = 0
                    else:
                        transfer_list_log[i, j] = -1000
            max_score, best_label = overall_optimal(node_matrix_log, transfer_list_log, lambda_value)
            s = ""
            for i in range(len(best_label)):
                if i != len(best_label) - 1:
                    s += str(best_label[i])
                    s += " "
                else:
                    s += str(best_label[i])
            # if iteration < 2000:
            vf.write(s + "\n")
            # else:
            #     break
            # iteration += 1
        print(index)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("file_name", type=str, help="file name of the result file")
    # parser.add_argument("lambda_value", type=float, help="lambda * #of 1s")
    # args = parser.parse_args()
    # args.file_name = "exhaustion_print_575_0.3.txt"
    # args.lambda_value = 0.3
    # exhust_search(args.file_name, args.lambda_value)
    exhust_search("original_edge_predict_b_differ_0_1000.txt", 0)


if __name__ == '__main__':
    main()

