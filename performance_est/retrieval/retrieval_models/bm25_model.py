from rank_bm25 import BM25Okapi
from tqdm import tqdm

class BM25Model:
    def __init__(self, corpus_texts, corpus_ids):
        """
        Initializes the BM25 model.

        Args:
            corpus_texts (list[str]): A list of the document texts for indexing.
            corpus_ids (list[str]): A list of corresponding document IDs.
        """
        print("Initializing BM25... This may take a moment.")
        if len(corpus_texts) != len(corpus_ids):
            raise ValueError("corpus_texts and corpus_ids must have the same length.")

        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        
        tokenized_corpus = [doc.split(" ") for doc in tqdm(self.corpus_texts, desc="Tokenizing corpus")]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 initialized successfully.")

    def retrieve(self, query, k=100):
        """
        Retrieves the top-k most relevant document IDs for a given query.

        Args:
            query (str): The input query text.
            k (int): The number of document IDs to retrieve.

        Returns:
            list[str]: A ranked list of the top-k retrieved document IDs.
        """
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get the corpus indices of the top-k documents
        top_k_indices = doc_scores.argsort()[::-1][:k]
        
        # Return the IDs corresponding to those indices
        return [self.corpus_ids[i] for i in top_k_indices]
