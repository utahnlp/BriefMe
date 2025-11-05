# briefme_retrieval/retrieval_models/colbert.py

import os
import shutil
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

class ColBERTModel:
    def __init__(self, corpus_texts, corpus_ids):
        """
        Initializes the ColBERT model. It will build an index if one doesn't exist,
        otherwise it will load the existing index.

        Args:
            corpus_texts (list[str]): List of document texts.
            corpus_ids (list[str]): List of corresponding document IDs.
        """
        self.checkpoint = 'colbertv2.0'
        self.index_root = 'colbert_indexes'
        self.index_name = f'briefme.corpus.nbits-2' # Standard ColBERT naming convention
        self.full_index_path = os.path.join(self.index_root, self.index_name)
        
        self.corpus_ids = corpus_ids

        # --- Build or Load Index ---
        if not os.path.exists(self.full_index_path):
            self._build_index(corpus_texts)
        else:
            print(f"Loading existing ColBERT index from {self.full_index_path}")

        # --- Initialize Searcher ---
        print("Initializing ColBERT Searcher...")
        # RunConfig is used to manage execution settings, like GPU usage
        with Run().context(RunConfig(nranks=1, experiment="briefme")): # nranks=1 for single-GPU/CPU
            self.searcher = Searcher(index=self.index_name, index_root=self.index_root)
        print("ColBERT initialized successfully.")

    def _build_index(self, corpus_texts):
        """
        Builds the ColBERT index from the corpus texts.
        WARNING: This is a slow, resource-intensive process.
        """
        print("---" * 20)
        print(f"WARNING: Building new ColBERT index at '{self.full_index_path}'.")
        print("This is a one-time process and can be very slow (hours) and requires >20GB of disk space.")
        print("---" * 20)

        # Clean up any potentially failed previous indexing attempts
        shutil.rmtree(self.full_index_path, ignore_errors=True)
        
        # Configure ColBERT indexing
        # nbits=2 is a standard setting for good compression and quality
        config = ColBERTConfig(nbits=2)
        
        # RunConfig controls the execution environment (e.g., number of GPUs)
        # We'll use nranks=1 for a single GPU or CPU. If you have more GPUs, you can increase this.
        with Run().context(RunConfig(nranks=1, experiment="briefme")):
            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(name=self.index_name, collection=corpus_texts, overwrite=True)

        print(f"Index built successfully at {self.full_index_path}")

    def retrieve(self, query, k=100):
        """
        Searches for a query and returns the top-k document IDs.

        Args:
            query (str): The input query text.
            k (int): The number of document IDs to retrieve.

        Returns:
            list[str]: A ranked list of the top-k retrieved document IDs.
        """
        # The searcher returns a tuple of (passage_ids, ranks, scores)
        # The passage_ids are the integer indices of the documents in the original collection.
        results_pids = self.searcher.search(query, k=k)[0]

        # Map the internal integer IDs back to our original citation_value IDs
        retrieved_ids = [self.corpus_ids[pid] for pid in results_pids]
        return retrieved_ids
