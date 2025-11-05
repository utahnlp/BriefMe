# briefme_retrieval/main.py

import argparse
from tqdm import tqdm
from data_loader import load_briefme_data
from retrieval_models.bm25_model import BM25Model
from retrieval_models.colbert_model import ColbertModel
from evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description="BriefMe Case Retrieval")
    parser.add_argument(
        "--method",
        type=str,
        choices=["bm25", "colbert"],
        default="bm25",
        help="Retrieval method to use."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset query split to use."
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=10,
        help="Maximum number of queries to process. Processes all by default."
    )
    args = parser.parse_args()

    print(f"Running retrieval with method: {args.method} on query split: {args.split}")

    # 1. Load data
    queries, corpus_dataset = load_briefme_data(args.split)
    if not queries or not corpus_dataset:
        return

    # 2. Limit the number of queries if specified
    if args.max_queries and args.max_queries > 0:
        if args.max_queries < len(queries):
            print(f"Limiting to the first {args.max_queries} queries.")
            queries = queries.select(range(args.max_queries))
        else:
            print(f"Requested {args.max_queries} queries, but split only has {len(queries)}. Using all available queries.")


    # 3. Prepare corpus texts and IDs for the retrieval model
    print("Preparing corpus texts and IDs...")
    corpus_texts = [doc['cited_text'] for doc in corpus_dataset]
    corpus_ids = [doc['id'] for doc in corpus_dataset]

    # 4. Initialize the retrieval model
    if args.method == "bm25":
        model = BM25Model(corpus_texts, corpus_ids)
    elif args.method == "colbert":
        model = ColbertModel(corpus_texts, corpus_ids)
    else:
        raise ValueError("Invalid method specified.")

    # 5. Prepare ground truths for evaluation using IDs
    ground_truths = {i: [q['citation_value_orig']] for i, q in enumerate(queries)}

    # 6. Perform retrieval for each query
    results = {}
    print(f"Performing retrieval for {len(queries)} queries...")
    for i, query_info in enumerate(tqdm(queries, desc="Processing queries")):
        query_text = query_info['context']
        retrieved_ids = model.retrieve(query_text, k=100)
        results[i] = retrieved_ids

    # 7. Evaluate the results
    k_values_recall = [1, 5, 10, 50, 100]
    metrics = evaluate(results, ground_truths, k_values_recall)

    # 8. Report the metrics
    print("\n--- Evaluation Results ---")
    print(f"Method: {args.method.upper()}")
    print(f"Query Split: {args.split}")
    if args.max_queries:
        print(f"Queries Processed: {len(queries)}")
    print("-" * 26)
    print(f"Mean Reciprocal Rank (MRR): {metrics['MRR']:.4f}")
    print(f"Normalized DCG @10 (nDCG@10): {metrics['nDCG@10']:.4f}")
    for k in k_values_recall:
        print(f"Recall@{k}: {metrics['Recall'][k]:.4f}")
    print("-" * 26 + "\n")

if __name__ == "__main__":
    main()
