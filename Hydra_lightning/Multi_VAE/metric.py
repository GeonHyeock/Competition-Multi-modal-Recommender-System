import bottleneck as bn
import numpy as np


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=50):
    """
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    X_pred, heldout_batch = X_pred.cpu().numpy(), heldout_batch.cpu().numpy()
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1.0 / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(
        axis=1
    )
    IDCG = np.array(
        [(tp[: min(n, k)]).sum() for n in heldout_batch.sum(axis=1).astype(np.int)]
    )
    return np.nan_to_num(DCG / IDCG)


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    X_pred, heldout_batch = X_pred.cpu().numpy(), heldout_batch.cpu().numpy()
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = heldout_batch > 0
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return np.nan_to_num(recall)
