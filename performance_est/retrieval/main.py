# briefme_retrieval/main.py

import argparse
from tqdm import tqdm
from data_loader import load_briefme_dataset
from retrieval_models.bm25_model import BM25Model
from retrieval_models.colbert_model import ColbertModel
from evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description="BriefMe Case Retrieval")
    parser.add_argument("--method", type=str, choices=["bm25", "colbert"], default="bm25", help="Retrieval method to use.")
    parser.add_argument("--split", type=str, default="test", help="Dataset query split to use.")
    args = parser.parse_args()

    print(f"Running retrieval with method: {args.method} on query split: {args.split}")

    # 1. Load data
    queries, corpus_dataset = load_briefme_dataset(args.split)
    if not queries or not corpus_dataset:
        return

    # 2. Prepare corpus texts and IDs for the retrieval model
    print("Preparing corpus texts and IDs...")
    corpus_texts = [doc['cited_text'] for doc in corpus_dataset]
    corpus_ids = [doc['id'] for doc in corpus_dataset]

    # 3. Initialize the retrieval model
    if args.method == "bm25":
        model = BM25Model(corpus_texts, corpus_ids)
    elif args.method == "colbert":
        model = ColbertModel(corpus_texts, corpus_ids)
    else:
        raise ValueError("Invalid method specified.")

    # 4. Prepare ground truths for evaluation using IDs
    # The ground truth for each query is its 'citation_value_orig'
    ground_truths = {i: [q['citation_value_orig']] for i, q in enumerate(queries)}

    # 5. Perform retrieval for each query
    results = {}
    print(f"Performing retrieval for {len(queries)} queries...")
    cnt = 0
    for i, query_info in enumerate(tqdm(queries, desc="Processing queries")):
        query_text = query_info['context']
        # The model now returns a ranked list of document IDs
        retrieved_ids = model.retrieve(query_text, k=100)
        results[i] = retrieved_ids
        cnt += 1
        if cnt == 10:
            break
    # 6. Evaluate the results (retrieved IDs vs. ground truth ID)
    k_values_recall = [1, 5, 10, 50, 100]
    metrics = evaluate(results, ground_truths, k_values_recall)

    # 7. Report the metrics
    print("\n--- Evaluation Results ---")
    print(f"Method: {args.method.upper()}")
    print(f"Query Split: {args.split}")
    print("-" * 26)
    print(f"Mean Reciprocal Rank (MRR): {metrics['MRR']:.4f}")
    print(f"Normalized DCG @10 (nDCG@10): {metrics['nDCG@10']:.4f}")
    for k in k_values_recall:
        print(f"Recall@{k}: {metrics['Recall'][k]:.4f}")
    print("-" * 26 + "\n")

if __name__ == "__main__":
    main()
