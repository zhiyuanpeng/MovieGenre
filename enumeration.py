import numpy as np
import math
from tqdm import tqdm



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


def transfer_label(trans_value):
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
def get_edge_matrix(link_list):
    # get the matrix size
    path = "./data/split_test/"
    example = list(np.loadtxt(path + link_list[0]))
    edge_matrix = np.zeros((len(example), len(link_list)))
    for index in range(len(link_list)):
        edge_column = list(np.loadtxt(path + link_list[index]))
        edge_matrix[:, index] = edge_column
    return edge_matrix


def get_transfer_matrix(link_list, edge_list):
    transfer_list = []
    # convert the file to matrix
    transfer_data = get_edge_matrix(link_list)
    for line in transfer_data:
        transfer_matrix = np.zeros((len(line) + 1, len(line) + 1))
        line_index = 0
        for edge in edge_list:
            transfer_matrix[edge[0], edge[1]] = line[line_index]
            transfer_matrix[edge[1], edge[0]] = line[line_index]
            line_index += 1
        transfer_list.append(transfer_matrix.copy())
    return transfer_list


def overall_optimal(status_list, transfer_matrix, edge_list):
    score_list = []
    all_possible = get_candidate(len(status_list))
    for one_possible in all_possible:
        score = 0
        for index in range(len(one_possible)):
            logpossibility = status_list[index] if one_possible[index] == 1 else np.log(1 - math.exp(status_list[index]))
            score += logpossibility
        # add the transfer possibility
        for edge in edge_list:
            label_trans_matrix = transfer_label(transfer_matrix[edge[0], edge[1]])
            score += label_trans_matrix[one_possible[edge[0]], one_possible[edge[1]]]
        score_list.append(score)
    max_score = max(score_list)
    max_index = score_list.index(max_score)
    max_label = all_possible[max_index]
    return max_score, max_label


def exhust_search(filename, edge_list):
    link_path_list = ["l0_l3_predict_round_0.75_2000.txt", "l0_l5_predict_round_0.86_2000.txt",
                      "l0_l6_predict_round_0.71_2000.txt", "l0_l9_predict_round_0.74_2000.txt",
                      "l0_l10_predict_round_0.97_2000.txt", "l0_l12_predict_round_0.96_2000.txt",
                      "l0_l14_predict_round_0.6_2000.txt", "l1_l4_predict_round_0.89_2000.txt",
                      "l1_l7_predict_round_0.89_2000.txt", "l1_l8_predict_round_0.82_2000.txt",
                      "l1_l11_predict_round_0.92_2000.txt", "l1_l13_predict_round_0.69_2000.txt",
                      "l2_l3_predict_round_0.85_2000.txt", "l2_l4_predict_round_0.94_2000.txt"]
    node_matrix = np.loadtxt("data/store/original/original_predict_2000.txt")
    transfer_list = get_transfer_matrix(link_path_list, edge_list)
    with open("data/enumeration/" + filename, "a+") as vf:
        iteration = 0
        for index in tqdm(range(node_matrix.shape[0])):
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
            max_score, best_label = overall_optimal(node_matrix_log, transfer_list_log, edge_list)
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
    edge_list = [(0, 3), (0, 5), (0, 6), (0, 9), (0, 10), (0, 12), (0, 14), (1, 4), (1, 7), (1, 8), (1, 11), (1, 13),
                 (2, 3), (2, 4)]
    exhust_search("original_edge_predict_b_differ_0_2000.txt", edge_list)


if __name__ == '__main__':
    main()

