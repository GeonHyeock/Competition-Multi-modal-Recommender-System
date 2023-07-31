import re
import pandas as pd


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
        A.extend(p.findall(a)[0][1:-1].split(","))
        A.extend(q.findall(b))

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
    ).groupby(["n_layers", "embedding_size", "feat_embed_dim"]).mean().sort_values(
        by=["ndcg@50"], ascending=False
    ).to_csv(
        "submission/log_summary.csv", float_format="%.6f"
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

    log_summary(log_paths)
