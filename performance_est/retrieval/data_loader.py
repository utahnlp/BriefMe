
from datasets import load_dataset

def load_briefme_data(split='test'):
    """
    Loads a specific split of the BriefMe 'case_retrieval' subset.

    Args:
        split (str): The dataset split to load (e.g., 'train', 'validation', 'test').

    Returns:
        tuple: A tuple containing the queries and the retrieval corpus.
    """
    try:
        # Load the "case_retrieval" subset and the specified split
        dataset = load_dataset("jw4202/BriefMe", name="case_retrieval", split=split)
        
        # The structure is slightly different for this subset.
        # The queries and corpus are directly in the dataset features.
        # We need to extract the text content.
        queries = dataset
        print("Queries are loaded\n-----------------------")

        # corpus_dataset = load_dataset("jw4202/BriefMe", name="retrieval_corpus", split='train')
        # corpus_dataset = load_dataset('json', data_files='downloaded_dataset/retrieval_dataset_fixed.jsonl', split='train')
        corpus_dataset = load_dataset('json', data_files='downloaded_dataset/small_corpus.jsonl', split='train')
        # corpus = [doc['cited_text'] for doc in corpus_dataset]
        corpus= corpus_dataset
        return queries, corpus
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

if __name__ == '__main__':
    # Example usage
    test_queries, test_corpus = load_briefme_data('test')
    if test_queries and test_corpus:
        print(f"Loaded {len(test_queries)} queries from the 'test' split.")
        print(f"Loaded {len(test_corpus)} documents in the corpus.")
        print("\nExample Query:")
        # Queries are now the full dataset object for the split
        print(test_queries[0])
        print("\nExample Corpus Document:")
        print(test_corpus[0][:500] + "...") # Print first 500 chars
