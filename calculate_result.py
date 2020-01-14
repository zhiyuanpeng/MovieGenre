import numpy as np


# viterbi_result = np.array(np.loadtxt("./data/store/original/original_predict_2000_round.txt", dtype=int))
viterbi_result = np.array(np.loadtxt("./data/enumeration/original_edge_predict_b_differ_0_2000.txt", dtype=int))
label = np.loadtxt("./data/split_test/y_test_2000.txt", dtype=int)
# right_list = [0 for i in range(viterbi_result.shape[1])]
total_right = 0
row_right = 0
# l0_l1_num = 0
# l1_l2_num = 0
# l0_l3_num = 0
# l0_l4_num = 0
# l0_l5_num = 0
for i in range(viterbi_result.shape[0]):
    if (viterbi_result[i, :] == label[i, :]).all():
        row_right += 1
    for j in range(viterbi_result.shape[1]):
        if viterbi_result[i, j] == label[i, j]:
            # right_list[j] += 1
            total_right += 1
    # if viterbi_result[i, 0]*viterbi_result[i, 1] == label[i, 0]*label[i, 1]:
    #     l0_l1_num += 1
    # if viterbi_result[i, 1]*viterbi_result[i, 2] == label[i, 1]*label[i, 2]:
    #     l1_l2_num += 1
    # if viterbi_result[i, 0]*viterbi_result[i, 3] == label[i, 0]*label[i, 3]:
    #     l0_l3_num += 1
    # if viterbi_result[i, 0]*viterbi_result[i, 4] == label[i, 0]*label[i, 4]:
    #     l0_l4_num += 1
    # if viterbi_result[i, 0]*viterbi_result[i, 5] == label[i, 0]*label[i, 5]:
    #     l0_l5_num += 1
# for right_num in right_list:
#     print(right_num/label.shape[0])

print("total accuracy is: " + str(total_right/(label.shape[0]*label.shape[1])) + "\n")
print("row accuracy is: " + str(row_right/(label.shape[0])) + "\n")
# print("l0_l1 accuracy is: " + str(l0_l1_num/label.shape[0]) + "\n")
# print("l1_l2 accuracy is: " + str(l1_l2_num/label.shape[0]) + "\n")
# print("l0_l3 accuracy is: " + str(l0_l3_num/label.shape[0]) + "\n")
# print("l0_l4 accuracy is: " + str(l0_l4_num/label.shape[0]) + "\n")
# print("l0_l5 accuracy is: " + str(l0_l5_num/label.shape[0]) + "\n")

