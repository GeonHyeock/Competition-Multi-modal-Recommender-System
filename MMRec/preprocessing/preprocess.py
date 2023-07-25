import os, csv
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter


def splitting_1(df, version=0):
    df = df.sample(frac=1).reset_index(drop=True)
    df.sort_values(by=["userID"], inplace=True)

    uid_field, iid_field = "userID", "itemID"
    uid_freq = df.groupby(uid_field)[iid_field]
    u_i_dict = {}
    for u, u_ls in uid_freq:
        u_i_dict[u] = list(u_ls)

    u_ids_sorted = sorted(u_i_dict.keys())
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = [[] for _ in range(5)]
    item3 = [[0, 0, 1], [0, 1, 0], [1, 0, 0]] * 2
    item4 = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    for u in u_ids_sorted:
        items = u_i_dict[u]
        n_items = len(items)
        if n_items == 3:
            random.shuffle(item3)
            for i in range(5):
                fold[i].extend(item3[i])
        elif n_items == 4:
            random.shuffle(item4)
            for i in range(4):
                fold[i].extend(item4[i])
            fold[4].extend(random.choice(item4))
        else:
            for i, (_, valid_index) in enumerate(kf.split(items)):
                tmp_ls = np.zeros([n_items])
                tmp_ls[valid_index] = 1
                fold[i].extend(tmp_ls)
    for i in range(5):
        df[f"fold_{i}"] = fold[i]
    new_labeled_file = rslt_file[:-6] + "-v" + f"{version}" + ".inter"
    return df, new_labeled_file


def rating2inter_0(df, splitting=[0.8, 0.2]):
    learner_id, course_id = "userID", "itemID"
    df.dropna(subset=[learner_id, course_id], inplace=True)
    df.drop_duplicates(subset=[learner_id, course_id], inplace=True)
    print(f"After dropped data: {df.shape}")

    filter_by_k_core(df)
    print(f"k-core shape: {df.shape}")
    print(f"shape after k-core: {df.shape}")

    df.reset_index(drop=True, inplace=True)

    i_mapping_file = "i_id_mapping.csv"
    u_mapping_file = "u_id_mapping.csv"

    uid_field, iid_field = learner_id, course_id

    uni_users = pd.unique(df[uid_field])
    uni_items = pd.unique(df[iid_field])

    # start from 0
    u_id_map = {k: i for i, k in enumerate(uni_users)}
    i_id_map = {k: i for i, k in enumerate(uni_items)}

    df[uid_field] = df[uid_field].map(u_id_map)
    df[iid_field] = df[iid_field].map(i_id_map)
    df[uid_field] = df[uid_field].astype(int)
    df[iid_field] = df[iid_field].astype(int)

    # dump
    rslt_dir = "./MMRec/data/Inha/"
    u_df = pd.DataFrame(list(u_id_map.items()), columns=["user_id", "userID"])
    i_df = pd.DataFrame(list(i_id_map.items()), columns=["asin", "itemID"])

    u_df.to_csv(os.path.join(rslt_dir, u_mapping_file), sep="\t", index=False)
    i_df.to_csv(os.path.join(rslt_dir, i_mapping_file), sep="\t", index=False)
    print(f"mapping dumped...")

    print(f"splitting ...")
    tot_ratio = sum(splitting)
    # remove 0.0 in ratios
    ratios = [i for i in splitting if i > 0.0]
    ratios = [_ / tot_ratio for _ in ratios]
    split_ratios = np.cumsum(ratios)[:-1]
    print("split_ratios : ", split_ratios)

    # get df training dataset unique users/items
    df_train = df.sample(frac=splitting[0], random_state=42)
    df_val = df.drop(df_train.index)

    x_label, rslt_file = "x_label", "Inha.inter"
    df_train[x_label] = 0
    df_val[x_label] = 1

    temp_df = pd.concat([df_train, df_val])
    temp_df = temp_df[[learner_id, course_id, "rating", x_label]]
    print(f"columns: {temp_df.columns}")

    temp_df.columns = [learner_id, course_id, "rating", x_label]
    print(temp_df.x_label.value_counts())
    return temp_df, rslt_dir, rslt_file


def get_illegal_ids_by_inter_num(df, field, max_num=None, min_num=None):
    if field is None:
        return set()
    if max_num is None and min_num is None:
        return set()

    max_num = max_num or np.inf
    min_num = min_num or -1

    ids = df[field].values
    inter_num = Counter(ids)
    ids = {
        id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num
    }
    print(f"{len(ids)} illegal_ids_by_inter_num, field={field}")

    return ids


def filter_by_k_core(df, min_u_num=0, min_i_num=0):
    learner_id, course_id = "userID", "itemID"
    while True:
        ban_users = get_illegal_ids_by_inter_num(
            df, field=learner_id, max_num=None, min_num=min_u_num
        )
        ban_items = get_illegal_ids_by_inter_num(
            df, field=course_id, max_num=None, min_num=min_i_num
        )
        if len(ban_users) == 0 and len(ban_items) == 0:
            return

        dropped_inter = pd.Series(False, index=df.index)
        if learner_id:
            dropped_inter |= df[learner_id].isin(ban_users)
        if course_id:
            dropped_inter |= df[course_id].isin(ban_items)
        print(f"{len(dropped_inter)} dropped interactions")
        df.drop(df.index[dropped_inter], inplace=True)


def BM3():
    Path = "MMRec/data/Inha"
    rslt_file = "Inha.inter"
    df = pd.read_csv(os.path.join(Path, rslt_file), sep="\t")
    print(f"shape: {df.shape}")

    df, new_labeled_file = splitting_1(df)
    df.to_csv(os.path.join(Path, new_labeled_file), sep="\t", index=False)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    df = pd.read_csv(
        "./data/raw/train.csv",
        names=["userID", "itemID", "rating"],
        header=0,
    )
    print(f"raw data shape: {df.shape}")
    temp_df, rslt_dir, rslt_file = rating2inter_0(df)
    temp_df.to_csv(os.path.join(rslt_dir, rslt_file), sep="\t", index=False)

    BM3()
