# briefme_retrieval/evaluation.py

import numpy as np

def calculate_mrr(retrieved_docs, ground_truth):
    """Calculates Mean Reciprocal Rank."""
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth:
            return 1.0 / (i + 1)
    return 0.0

# ... (the rest of the file is identical to the previous version)
def calculate_ndcg(retrieved_docs, ground_truth, k):
    """Calculates Normalized Discounted Cumulative Gain."""
    dcg = 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def calculate_recall_at_k(retrieved_docs, ground_truth, k_values):
    """Calculates Recall@k for multiple k values."""
    recall = {k: 0 for k in k_values}
    ground_truth_set = set(ground_truth)
    if not ground_truth_set:
        return recall

    for k in k_values:
        retrieved_set = set(retrieved_docs[:k])
        intersection = retrieved_set.intersection(ground_truth_set)
        recall[k] = len(intersection) / len(ground_truth_set)
    return recall

def evaluate(results, ground_truths, k_values_recall):
    """
    Evaluates the retrieval results against the ground truths.
    """
    mrr_scores, ndcg_scores = [], []
    recall_scores = {k: [] for k in k_values_recall}

    for query_id, retrieved in results.items():
        ground_truth = ground_truths.get(query_id, [])
        if not ground_truth:
            continue

        mrr_scores.append(calculate_mrr(retrieved, ground_truth))
        ndcg_scores.append(calculate_ndcg(retrieved, ground_truth, k=10))
        recall_at_k = calculate_recall_at_k(retrieved, ground_truth, k_values_recall)
        for k in k_values_recall:
            recall_scores[k].append(recall_at_k[k])

    return {
        "MRR": np.mean(mrr_scores) if mrr_scores else 0,
        "nDCG@10": np.mean(ndcg_scores) if ndcg_scores else 0,
        "Recall": {k: np.mean(scores) if scores else 0 for k, scores in recall_scores.items()}
    }
