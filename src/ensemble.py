import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter
import os


def hard_voting(output_file_list, name="result"):
    """csv 파일을 기준으로 앙상블 하여 새로운 csv 저장

    Args:
        output_file_list (list[str]): 앙상블할 csv파일의 주소 리스트
        name (str): 저장할 csv파일의 이름
    """
    data = pd.concat([pd.read_csv(file) for file in output_file_list])
    user_id, item_id = [], []

    for user in tqdm(data.user_id.unique()):
        user_id.extend(np.repeat(user, 50))
        item_id.extend(
            map(
                lambda x: x[0],
                sorted(
                    Counter(data[data.user_id == user].item_id).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:50],
            )
        )
    result = pd.DataFrame({"user_id": user_id, "item_id": item_id})
    result.to_csv(f"./{name}.csv", index=False)


if __name__ == "__main__":
    os.chdir("submission")
    output_file_list = glob("./BM3/*.csv")
    output_file_list = list(filter(lambda x: "inter" in x, output_file_list))
    hard_voting(output_file_list, "fold_ensemble")