
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer
)

class DPRModel:
    def __init__(self, corpus_texts, corpus_ids, batch_size=16):
        """
        Initializes the DPR model, encoders, and builds the FAISS index.

        Args:
            corpus_texts (list[str]): List of document texts.
            corpus_ids (list[str]): List of corresponding document IDs.
            batch_size (int): Batch size for encoding the corpus.
        """
        print("Initializing DPR... This will download models and may take some time.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Load Models and Tokenizers ---
        ctx_model_name = "facebook/dpr-ctx_encoder-single-nq-base"
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model_name)
        self.ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model_name).to(self.device)

        q_model_name = "facebook/dpr-question_encoder-single-nq-base"
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(q_model_name).to(self.device)

        self.corpus_ids = corpus_ids
        self.batch_size = batch_size

        # --- Build FAISS Index ---
        self._build_index(corpus_texts)
        print("DPR initialized successfully.")

    def _build_index(self, corpus_texts):
        """Encodes all corpus texts and builds a FAISS index."""
        print("Encoding corpus and building FAISS index...")
        all_embeddings = []
        self.ctx_encoder.eval() # Set model to evaluation mode
        with torch.no_grad():
            for i in tqdm(range(0, len(corpus_texts), self.batch_size), desc="Encoding corpus"):
                batch_texts = corpus_texts[i:i + self.batch_size]
                inputs = self.ctx_tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(self.device)
                embeddings = self.ctx_encoder(**inputs).pooler_output
                all_embeddings.append(embeddings.cpu().numpy())

        corpus_embeddings = np.vstack(all_embeddings).astype('float32')
        embedding_dim = corpus_embeddings.shape[1]

        # DPR uses dot product, so we use IndexFlatIP
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.index.add(corpus_embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve(self, query, k=100):
        """
        Encodes a query and retrieves the top-k document IDs from the FAISS index.

        Args:
            query (str): The input query text.
            k (int): The number of document IDs to retrieve.

        Returns:
            list[str]: A ranked list of the top-k retrieved document IDs.
        """
        self.question_encoder.eval()
        with torch.no_grad():
            inputs = self.question_tokenizer(
                query, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            query_embedding = self.question_encoder(**inputs).pooler_output.cpu().numpy().astype('float32')

        # FAISS search returns distances (scores) and indices
        _scores, top_k_indices = self.index.search(query_embedding, k)

        # Map indices back to the original document IDs
        retrieved_ids = [self.corpus_ids[idx] for idx in top_k_indices[0]]
        return retrieved_ids
