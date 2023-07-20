# part of data preprocessing
# #----json2mat--------------------------------------------------------------------------------------------------
import json
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def data2mat(raw_data):
    n_user, n_item = len(raw_data.userID.unique()), len(raw_data.itemID.unique())
    train, valid = (
        raw_data[(raw_data.x_label == 0).values & (raw_data.rating >= 3.0).values],
        raw_data[raw_data.x_label == 1 & (raw_data.rating >= 3.0).values],
    )

    train_mat = csr_matrix(
        (train.rating.values, (train.userID.values, train.itemID.values)),
        shape=(n_user, n_item),
    )
    val_mat = csr_matrix(
        (valid.rating.values, (valid.userID.values, valid.itemID.values)),
        shape=(n_user, n_item),
    )
    return train_mat, val_mat


def mat2json(mat):
    total_array = mat.toarray()
    total_dict = {}

    for i in range(total_array.shape[0]):
        total_dict[str(i)] = list(map(int, np.where(total_array[i] > 0)[0]))
    json_str = json.dumps(total_dict)
    return json_str


if __name__ == "__main__":
    os.chdir("/home/inhamath/competition/MMSSL")

    raw_data = pd.read_csv("../MMRec/data/Inha/Inha-v4.inter", sep="\t")
    train_mat, val_mat = data2mat(raw_data)
    train_json = mat2json(train_mat)
    valid_json = mat2json(val_mat)

    pickle.dump(train_mat, open("./Inha/train_mat", "wb"))

    with open("./Inha/train.json", "w") as test_file:
        test_file.write(train_json)

    with open("./Inha/val.json", "w") as test_file:
        test_file.write(valid_json)
