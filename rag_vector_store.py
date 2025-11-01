"""
RAG Vector Store Builder for Project Samarth
Creates FAISS vector database from RAG corpus using sentence-transformers
"""

import json
import pickle
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

print("=" * 60)
print("🔍 RAG Vector Store Builder - Project Samarth")
print("=" * 60)

# Configuration
CORPUS_FILE = Path("rag_files") / "rag_corpus.jsonl"
OUTPUT_DIR = Path("rag_store")
OUTPUT_DIR.mkdir(exist_ok=True)
FAISS_INDEX_DIR = OUTPUT_DIR / "faiss_index"
FAISS_INDEX_DIR.mkdir(exist_ok=True)

# Model configuration
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Check if corpus exists
if not CORPUS_FILE.exists():
    print(f"❌ Error: {CORPUS_FILE} not found!")
    print("Please run rag_corpus_builder.py first to generate the RAG corpus.")
    exit(1)

print(f"\n📖 Loading RAG corpus from: {CORPUS_FILE}")

# Load all documents from corpus
documents = []
texts = []
print("  Reading documents...")
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line.strip())
        documents.append(doc)
        texts.append(doc["text"])

print(f"✅ Loaded {len(documents):,} documents")

# Load sentence transformer model
print(f"\n🤖 Loading embedding model: {MODEL_NAME}")
print("  (This may take a few minutes on first run as the model downloads)")
model = SentenceTransformer(MODEL_NAME)
print(f"✅ Model loaded successfully")
print(f"  Embedding dimension: {EMBEDDING_DIM}")

# Generate embeddings
print(f"\n🔄 Generating embeddings for {len(texts):,} documents...")
print("  This may take several minutes...")

# Process in batches for efficiency
batch_size = 256
embeddings_list = []

for i in tqdm(range(0, len(texts), batch_size), desc="  Embedding batches"):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = model.encode(
        batch_texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
        convert_to_numpy=True
    )
    embeddings_list.append(batch_embeddings)

# Concatenate all embeddings
embeddings = np.vstack(embeddings_list).astype("float32")
print(f"✅ Generated embeddings: shape {embeddings.shape}")

# Create FAISS index
print(f"\n🔨 Creating FAISS vector index...")

# Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
# or IndexFlatL2 for L2 distance
index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine similarity for normalized vectors

# Add vectors to index
index.add(embeddings)
print(f"✅ FAISS index created with {index.ntotal:,} vectors")

# Save FAISS index
faiss_index_file = FAISS_INDEX_DIR / "index.faiss"
faiss.write_index(index, str(faiss_index_file))
print(f"✅ Saved FAISS index to: {faiss_index_file}")

# Save metadata (documents) for retrieval
metadata_file = FAISS_INDEX_DIR / "metadata.pkl"
with open(metadata_file, "wb") as f:
    pickle.dump(documents, f)
print(f"✅ Saved metadata to: {metadata_file}")

# Save embedding model name for reference
config_file = FAISS_INDEX_DIR / "config.json"
config = {
    "model_name": MODEL_NAME,
    "embedding_dim": EMBEDDING_DIM,
    "num_vectors": len(documents),
    "index_type": "IndexFlatIP"
}
with open(config_file, "w") as f:
    json.dump(config, f, indent=2)
print(f"✅ Saved configuration to: {config_file}")

# Test the index
print(f"\n🧪 Testing the vector index...")
test_query = "What is the rice production in Tamil Nadu?"
test_embedding = model.encode([test_query], normalize_embeddings=True).astype("float32")
k = 5
distances, indices = index.search(test_embedding, k)

print(f"  Test query: '{test_query}'")
print(f"  Top {k} similar documents:")
for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
    doc = documents[idx]
    print(f"    {i}. [Similarity: {dist:.3f}] {doc['text'][:100]}...")

print("=" * 60)
print("✅ Vector Store Creation Complete!")
print("=" * 60)
print(f"✅ FAISS index: {faiss_index_file}")
print(f"✅ Metadata: {metadata_file}")
print(f"✅ Configuration: {config_file}")
print(f"\n📊 Statistics:")
print(f"  Total vectors: {len(documents):,}")
print(f"  Embedding dimension: {EMBEDDING_DIM}")
try:
    index_size = os.path.getsize(faiss_index_file) / (1024*1024)
    print(f"  Index size: {index_size:.2f} MB")
except:
    print(f"  Index saved: {faiss_index_file}")
print("\n✅ Ready for RAG query processing!")

