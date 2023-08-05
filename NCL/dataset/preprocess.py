import pandas as pd
import numpy as np


def main(data_path):
    raw_data = pd.read_csv(data_path)

    unique_user = np.sort(raw_data.user_id.unique())
    unique_item = np.sort(raw_data.item_id.unique())

    u_id_map = {k: i for i, k in enumerate(unique_user)}
    i_id_map = {k: i for i, k in enumerate(unique_item)}

    raw_data.user_id.map(lambda x: u_id_map[x])
    raw_data.item_id.map(lambda x: i_id_map[x])

    raw_data.rename(
        columns={
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating": "rating:float",
        },
        inplace=True,
    )

    raw_data.sort_values(["user_id:token", "item_id:token"], inplace=True)

    raw_data.to_csv("dataset/Inha/Inha.inter", index=False, sep="\t")

    u = pd.DataFrame({"raw": u_id_map.keys(), "processed": u_id_map.values()})
    i = pd.DataFrame({"raw": i_id_map.keys(), "processed": i_id_map.values()})

    u.to_csv("dataset/Inha/u_id_map.csv", index=False)
    i.to_csv("dataset/Inha/i_id_map.csv", index=False)


if __name__ == "__main__":
    data_path = (
        "/home/inhamath/Challenge-Multi-modal-Recommender-System/data/raw/train.csv"
    )
    main(data_path)
