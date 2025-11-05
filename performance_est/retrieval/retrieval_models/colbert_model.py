import torch
import os
import tempfile
import shutil
from tqdm import tqdm
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries


class ColBERTModel:
    def __init__(self, corpus_texts, corpus_ids, batch_size=16, checkpoint="colbert-ir/colbertv2.0"):
        """
        Initializes the ColBERT model and builds the index.

        Args:
            corpus_texts (list[str]): List of document texts.
            corpus_ids (list[str]): List of corresponding document IDs.
            batch_size (int): Batch size for indexing the corpus.
            checkpoint (str): ColBERT checkpoint to use.
        """
        print("Initializing ColBERT-v2... This will download models and may take some time.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        
        # Create temporary directory for ColBERT files
        self.temp_dir = tempfile.mkdtemp(prefix="colbert_")
        self.collection_path = os.path.join(self.temp_dir, "collection.tsv")
        self.index_name = "corpus_index"
        
        # Create mapping from integer PIDs to corpus IDs
        self.pid_to_corpus_id = {i: corpus_id for i, corpus_id in enumerate(corpus_ids)}

        # --- Prepare data and build index ---
        self._prepare_collection()
        self._build_index()
        print("ColBERT initialized successfully.")

    def _prepare_collection(self):
        """Prepares the collection in TSV format required by ColBERT."""
        print("Preparing collection in TSV format...")
        with open(self.collection_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(tqdm(self.corpus_texts, desc="Writing collection")):
                # Clean the text: remove newlines and tabs, strip whitespace
                cleaned_text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                
                # Skip empty documents
                if not cleaned_text:
                    cleaned_text = "[EMPTY]"
                
                # ColBERT expects: pid \t passage_text
                # Use integer PIDs (0, 1, 2, ...) which we'll map back later
                f.write(f"{i}\t{cleaned_text}\n")
        print(f"Collection saved to {self.collection_path}")

    def _build_index(self):
        """Builds a ColBERT index from the collection."""
        print("Building ColBERT index...")
        
        # Determine number of GPUs
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        with Run().context(RunConfig(nranks=n_gpus, experiment="colbert_retrieval")):
            config = ColBERTConfig(
                nbits=2,  # Compression level (2 or 8 typically)
                doc_maxlen=512,  # Maximum document length
                root=self.temp_dir,
                index_bsize=self.batch_size,
            )
            
            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            indexer.index(
                name=self.index_name,
                collection=self.collection_path,
                overwrite=True
            )
        
        print(f"ColBERT index built with {len(self.corpus_texts)} documents.")
        
        # Initialize searcher
        with Run().context(RunConfig(nranks=1, experiment="colbert_retrieval")):
            config = ColBERTConfig(
                root=self.temp_dir,
            )
            self.searcher = Searcher(index=self.index_name, config=config)

    def retrieve(self, query, k=100):
        """
        Encodes a query and retrieves the top-k document IDs from the ColBERT index.

        Args:
            query (str): The input query text.
            k (int): The number of document IDs to retrieve.

        Returns:
            list[str]: A ranked list of the top-k retrieved document IDs.
        """
        with Run().context(RunConfig(nranks=1, experiment="colbert_retrieval")):
            # Search using ColBERT
            # searcher.search returns a list of tuples: (passage_id, rank, score)
            results = self.searcher.search(query, k=k)
            
            # Map integer PIDs back to original corpus IDs
            retrieved_ids = [self.pid_to_corpus_id[pid] for pid, _, _ in results]
            
            return retrieved_ids

    def retrieve_with_scores(self, query, k=100):
        """
        Encodes a query and retrieves the top-k document IDs with their scores.

        Args:
            query (str): The input query text.
            k (int): The number of documents to retrieve.

        Returns:
            list[tuple]: A list of tuples (document_id, score) ranked by score.
        """
        with Run().context(RunConfig(nranks=1, experiment="colbert_retrieval")):
            results = self.searcher.search(query, k=k)
            
            # Map PIDs to corpus IDs and include scores
            retrieved_with_scores = [
                (self.pid_to_corpus_id[pid], score) 
                for pid, _, score in results
            ]
            
            return retrieved_with_scores

    def batch_retrieve(self, queries, k=100):
        """
        Retrieves top-k documents for multiple queries efficiently.

        Args:
            queries (list[str]): List of query texts.
            k (int): The number of documents to retrieve per query.

        Returns:
            list[list[str]]: A list where each element is a ranked list of document IDs for a query.
        """
        # Create temporary queries file
        queries_path = os.path.join(self.temp_dir, "queries_temp.tsv")
        with open(queries_path, 'w', encoding='utf-8') as f:
            for i, query in enumerate(queries):
                f.write(f"{i}\t{query}\n")
        
        with Run().context(RunConfig(nranks=1, experiment="colbert_retrieval")):
            # Load queries
            queries_obj = Queries(queries_path)
            
            # Search all queries
            ranking = self.searcher.search_all(queries_obj, k=k)
            
            # Parse results
            results = []
            for qid in range(len(queries)):
                query_results = ranking.data.get(qid, [])
                retrieved_ids = [self.pid_to_corpus_id[pid] for pid, _, _ in query_results]
                results.append(retrieved_ids)
        
        # Clean up temporary queries file
        if os.path.exists(queries_path):
            os.remove(queries_path)
        
        return results

    def cleanup(self):
        """Removes temporary files and directories created by ColBERT."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass
