import re
import pandas as pd


def my_time(s):
    h, s = s // 3600, s % 3600
    m, s = s // 60, s % 60
    return f"{int(h)}h {int(m)}m {s:.2f}s"


def wandb_train_time_log(log_path):
    data = pd.read_csv(log_path)
    col = ["n_layers", "embedding_size", "feat_embed_dim"]
    for c in col:
        data.loc[:, c] = data.loc[:, c].astype(str)
    data = pd.DataFrame(data.groupby(col).Runtime.mean())
    data = pd.DataFrame(data.Runtime.apply(lambda x: my_time(x)))
    data.rename(columns={"Runtime": "training_time_avg"}, inplace=True)
    return data.training_time_avg


def log_summary(log_paths):
    """log 정보를 취합

    Args:
        log_paths list[str]: log들의 주소를 리스트로 주세요!
    """
    logs = []
    for log_path in log_paths:
        start = "============All Over====================="
        end = "█████████████ BEST ████████████████"
        text = "".join(open(log_path, "r"))
        result = re.search(f"{start}*((\n|.)*?){end}", text, re.DOTALL)

        if result:
            logs.extend(result.group(1).replace("\n\n", "").split("\n")[1:-1])
        else:
            print(f"해당 로그를 확인해주세요. : {log_path}")

    p = re.compile("\(.*\)")
    q = re.compile("\d+.\d+")

    A = []
    for idx in range(len(logs) // 2):
        a, b = idx * 2, (idx * 2) + 1
        a, b = logs[a], logs[b]
        A.extend([i.strip() for i in p.findall(a)[0][1:-1].split(",")])
        A.extend(q.findall(b))

    return (
        pd.DataFrame(
            {
                "n_layers": A[0::9],
                # "dropout": A[1::9],
                "embedding_size": A[2::9],
                "feat_embed_dim": A[3::9],
                #     "inter_splitting_label": A[4::9],
                "ndcg@50": map(float, A[6::9]),
                "precision@50": map(float, A[7::9]),
                "recall@50": map(float, A[5::9]),
                "map@50": map(float, A[8::9]),
            }
        )
        .groupby(["n_layers", "embedding_size", "feat_embed_dim"])
        .mean()
        .sort_values(by=["ndcg@50"], ascending=False)
    )


if __name__ == "__main__":
    log_paths = [
        "MMRec/log/BM3-Inha-Jul-21-2023-18-03-14.log",
        "MMRec/log/BM3-Inha-Jul-21-2023-18-04-45.log",
        "MMRec/log/BM3-Inha-Jul-22-2023-23-40-59.log",
        "MMRec/log/BM3-Inha-Jul-23-2023-18-01-50.log",
        "MMRec/log/BM3-Inha-Jul-28-2023-01-15-42.log",
        "MMRec/log/BM3-Inha-Jul-28-2023-01-16-03.log",
        "MMRec/log/BM3-Inha-Jul-30-2023-01-13-31.log",
        "MMRec/log/BM3-Inha-Jul-30-2023-01-13-43.log",
    ]
    wandb_log = "logs/wandb_export_2023-08-02T11_03_01.181+09_00.csv"

    infer_time = pd.read_csv("logs/infer_time.csv")
    # infer_time = infer_time[infer_time.n_layer != 1].reset_index(drop=True)
    infer_time["n_layers"] = infer_time.n_layer.astype(str)
    infer_time.embedding_size = infer_time.embedding_size.astype(str)
    infer_time.feat_embed_dim = infer_time.feat_embed_dim.astype(str)
    infer_time = (
        infer_time.groupby(["n_layers", "embedding_size", "feat_embed_dim"])
        .mean()
        .inference_time.round(2)
    )

    log = log_summary(log_paths)
    wandb_log = wandb_train_time_log(wandb_log)
    my_log = pd.merge(log, wandb_log, left_index=True, right_index=True, how="inner")
    my_log = pd.merge(
        my_log, infer_time, left_index=True, right_index=True, how="inner"
    )
    my_log.inference_time = my_log.inference_time.astype(str) + "s"
    my_log.to_csv("logs/log_summary.csv", float_format="%.6f")
