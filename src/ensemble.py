import os
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter


def hard_voting(output_file_list, name="hard_voting"):
    """csv 파일을 기준으로 앙상블 하여 새로운 csv 저장

    Args:
        output_file_list (list[str]): 앙상블할 csv파일의 주소 리스트
        name (str): 저장할 csv파일의 이름
    """
    result = []
    for file in output_file_list:
        data = pd.read_csv(file)
        data = data.groupby("user_id").apply(lambda x: x.item_id.values).values
        result.append(np.array([*data]))

    result, user_id, item_id = np.array(result).swapaxes(0, 1), [], []
    for user in tqdm(range(result.shape[0])):
        user_id.extend(np.repeat(user, 50))
        unique, counts = np.unique(result[user], return_counts=True)
        item_id.extend(unique[np.argsort(counts)[::-1][:50]])

    result = pd.DataFrame({"user_id": user_id, "item_id": item_id})
    result.to_csv(f"./{name}.csv", index=False)


def weighted_voting(output_file_list, name="weighted_voting", item_num=63001):
    """csv 파일을 기준으로 앙상블 하여 새로운 csv 저장
    한 유저당 추천된 아이템의 Rank가 높을수록 가중치 반영,
    가중치 : idx = i일때 log2(2+i)

    Args:
        output_file_list (list[str]): 앙상블할 csv파일의 주소 리스트
        name (str): 저장할 csv파일의 이름
        item_num (int): output item의 수
    """
    result = []
    for file in output_file_list:
        data = pd.read_csv(file)
        data = data.groupby("user_id").apply(lambda x: x.item_id.values).values
        result.append(np.array([*data]))

    result, user_id, item_id = np.array(result).swapaxes(0, 1), [], []
    for user in tqdm(range(result.shape[0])):
        data, item = result[user, :, :], np.zeros(item_num)
        for i in range(50):
            for idx in data[:, i]:
                item[idx] += 1 + 1 / np.log2(i + 2)

        user_id.extend(np.repeat(user, 50))
        item_id.extend(np.argsort(item)[::-1][:50])

    result = pd.DataFrame({"user_id": user_id, "item_id": item_id})
    result.to_csv(f"../submission/{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="weighted_voting",
        choices=["weighted_voting", "hard_voting"],
        help="name of ensemble",
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=str,
        default="my_best",
        help="ensemble folder name",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default="best",
        help="ensemble result csv name",
    )
    args, _ = parser.parse_known_args()

    os.chdir("submission")
    np.random.seed(42)
    output_file_list = [f"{args.folder}/" + f for f in os.listdir(f"{args.folder}")]

    ensemble = eval(args.type)
    ensemble(output_file_list, f"{args.name}")
