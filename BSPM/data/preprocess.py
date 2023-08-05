import pandas as pd
import random
import os


def processing(data):
    user_list = {v: k for k, v in enumerate(data.user_id.unique())}
    item_list = {v: k for k, v in enumerate(data.item_id.unique())}

    data.user_id = data.user_id.map(user_list).astype(int)
    data.item_id = data.item_id.map(item_list).astype(int)

    u = pd.DataFrame(user_list.items(), columns=["org_id", "remap_id"])
    i = pd.DataFrame(item_list.items(), columns=["org_id", "remap_id"])

    u.to_csv("./data/Inha/user_list.txt", index=False, sep=" ")
    i.to_csv("./data/Inha/item_list.txt", index=False, sep=" ")

    train, test = "", ""
    for user in sorted(data.user_id.unique()):
        item = data[data.user_id == user].item_id.values
        N = int(len(item) * 0.8)
        train += f"{user} {' '.join(map(str, item))}\n"
        test += f"{user} {' '.join(map(str, item))}\n"
        # else:
        #     random.shuffle(item)
        #     train += f"{user} {' '.join(map(str, item[:N]))}\n"
        #     test += f"{user} {' '.join(map(str, item[N:]))}\n"

    for type in ["train", "test"]:
        with open(f"data/Inha/{type}.txt", "w") as f:
            f.write(train)
    print("DONE")


if __name__ == "__main__":
    os.chdir("BSPM")
    random.seed(42)
    data = pd.read_csv("../data/raw/train.csv")
    data = data[data.rating >= 3.0]
    item = data.item_id.value_counts() >= 6
    data = data[data.item_id.isin(item[item.values == True].index)]
    processing(data)
