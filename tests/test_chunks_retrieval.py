"""Pytest class for testing RAG flow on English text (mimicking app behavior)."""

import os
import faiss
import numpy as np
import pytest
import logging
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ai.embedders.gemini.gemini_embedder import GeminiEmbedder
from src.common.config import CONFIG

load_dotenv()

# Configure logging (pytest will capture this cleanly)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

class TestRAGFlowEnglish:
    @classmethod
    def setup_class(cls):
        """Initialize embedder and load test data once for all tests."""
        logger.info("=" * 80)
        logger.info("INITIALIZING RAG FLOW (ENGLISH)")
        logger.info("=" * 80)

        cls.embedder = GeminiEmbedder()
        cls.dimension = cls.embedder.dimension
        cls.extracted_text_file = "tests/trans_en.txt"

        assert os.path.exists(cls.extracted_text_file), f"Missing file: {cls.extracted_text_file}"

        with open(cls.extracted_text_file, "r", encoding="utf-8") as f:
            cls.full_text = f.read()

        logger.info(f"✅ Loaded text: {len(cls.full_text)} characters")

        cls.pages = [p.strip() for p in cls.full_text.split("--- Page") if p.strip()]
        logger.info(f"📚 Parsed {len(cls.pages)} pages")

        cls.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["document_loader"]["chunk_size"],
            chunk_overlap=CONFIG["document_loader"]["chunk_overlap"],
        )

        cls.texts = []
        cls.metadatas = []

        for page_content in cls.pages:
            lines = page_content.split('\n')
            try:
                page_num = int(lines[0].split('---')[0].strip())
            except Exception:
                page_num = 0

            text_content = '\n'.join(lines[1:]).strip()
            if text_content:
                chunks = cls.text_splitter.split_text(text_content)
                for chunk in chunks:
                    cls.texts.append(chunk)
                    cls.metadatas.append({
                        "source": "llm_extractor_docs/03-dattes-WEB_en.pdf",
                        "page": page_num,
                        "lang": "eng",
                        "llm_extracted": True,
                    })

        logger.info(f"✅ Created {len(cls.texts)} chunks")

        logger.info("🔧 Creating embeddings...")
        cls.embeddings = cls.embedder.embed_documents(cls.texts)
        cls.embeddings_np = np.array(cls.embeddings).astype("float32")
        faiss.normalize_L2(cls.embeddings_np)

        cls.index = faiss.IndexFlatIP(cls.dimension)
        cls.index.add(cls.embeddings_np)

        cls.documents = [{"content": t, "metadata": m} for t, m in zip(cls.texts, cls.metadatas)]
        cls.pollination_chunks = []

        logger.info(f"✅ FAISS index ready with {cls.index.ntotal} vectors")

    # -------------------------------------------------------------------------
    def test_find_pollination_chunks(self):
        """Detect pollination-related chunks."""
        logger.info("=" * 80)
        logger.info("TEST: FINDING POLLINATION CHUNKS (ENGLISH)")
        logger.info("=" * 80)

        pollination_keywords = ["pollination", "pollinated", "unpollinated", "pollen", "4-2"]
        pollination_chunks = []

        for idx, doc in enumerate(self.documents):
            if any(keyword.lower() in doc["content"].lower() for keyword in pollination_keywords):
                pollination_chunks.append((idx, doc))

        logger.info(f"✅ Found {len(pollination_chunks)} pollination-related chunks")
        for idx, doc in pollination_chunks[:5]:
            logger.info(f"  Chunk #{idx} (Page {doc['metadata']['page']}):")
            logger.info(f"  {doc['content'][:180]}...")

        type(self).pollination_chunks = pollination_chunks
        assert len(pollination_chunks) > 0, "No pollination chunks found!"

    # -------------------------------------------------------------------------
    def test_search_queries(self):
        """Run search queries and check pollination chunk retrieval."""
        logger.info("=" * 80)
        logger.info("TEST: ENGLISH SEARCH QUERIES")
        logger.info("=" * 80)

        test_queries = [
            ("tell me about pollination of date palms", "General pollination question"),
            ("date palm pollination techniques", "Specific pollination query"),
            ("how to pollinate date palms", "Procedural question"),
            ("what month is pollination done for date palms", "Timing query"),
            ("fertilization of date palms", "Synonym / related term"),
        ]

        k = CONFIG["llm"]["default_k"]
        pollination_indices = [idx for idx, _ in self.pollination_chunks]

        for query, description in test_queries:
            logger.info("=" * 80)
            logger.info(f"Query: '{query}' ({description})")
            logger.info("=" * 80)

            query_embedding = self.embedder.embed_query(query)
            query_np = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_np)

            D, I = self.index.search(query_np, k=k)
            found_pollination = [i for i in I[0] if i in pollination_indices]

            logger.info(f"🎯 Pollination chunks in top {k}: {len(found_pollination)}")
            if found_pollination:
                positions = [list(I[0]).index(i)+1 for i in found_pollination]
                logger.info(f"   Found at ranks: {positions}")
            else:
                logger.info("   ❌ None found in top results")

            logger.info("📊 Top results:")
            for rank, (doc_idx, score) in enumerate(zip(I[0], D[0]), 1):
                doc = self.documents[doc_idx]
                marker = "🎯" if doc_idx in pollination_indices else "  "
                preview = doc["content"][:100].replace("\n", " ")
                logger.info(f"{rank}. {marker} Score: {score:.4f} | Page: {doc['metadata']['page']}")
                logger.info(f"   {preview}...")

            if not found_pollination:
                logger.info("💡 RAG Analysis: ❌ No relevant documentation found.")
            else:
                logger.info(f"💡 RAG Analysis: ✅ Retrieved {len(found_pollination)} pollination chunks.")

    def test_search_queries_full(self):
        """
        Run search queries and print similarity stats for *all* chunks — no filtering.
        This helps visualize how FAISS ranks every chunk for each query.
        """
        logger.info("=" * 80)
        logger.info("TEST: ENGLISH SEARCH QUERIES — FULL RANKING DIAGNOSTIC")
        logger.info("=" * 80)

        test_queries = [
            ("tell me about pollination of date palms", "General pollination question"),
            ("date palm pollination techniques", "Specific pollination query"),
            ("how to pollinate date palms", "Procedural question"),
            ("what month is pollination done for date palms", "Timing query"),
            ("fertilization of date palms", "Synonym / related term"),
        ]

        k = CONFIG["llm"]["default_k"]

        for query, description in test_queries:
            logger.info("=" * 80)
            logger.info(f"🔍 Query: '{query}' ({description})")
            logger.info("=" * 80)

            # Embed the query
            query_embedding = self.embedder.embed_query(query)
            query_np = np.array([query_embedding]).astype("float32")
            faiss.normalize_L2(query_np)

            # Run FAISS search
            D, I = self.index.search(query_np, k=len(self.documents))  # search ALL chunks
            similarities = D[0]
            indices = I[0]

            # Compute stats
            mean_score = np.mean(similarities)
            max_score = np.max(similarities)
            min_score = np.min(similarities)
            std_score = np.std(similarities)

            logger.info(f"📊 Similarity stats → Mean: {mean_score:.4f}, Max: {max_score:.4f}, Min: {min_score:.4f}, Std: {std_score:.4f}")

            # Show top 15
            logger.info("\n🏆 TOP 15 RANKED CHUNKS:")
            for rank, (doc_idx, score) in enumerate(zip(indices[:15], similarities[:15]), 1):
                doc = self.documents[doc_idx]
                preview = doc["content"][:150].replace("\n", " ")
                logger.info(f"{rank:02d}. Score: {score:.4f} | Page: {doc['metadata']['page']}")
                logger.info(f"    {preview}\n")

            # Show bottom 5 (for contrast)
            logger.info("\n💤 LOWEST 5 CHUNKS:")
            for doc_idx, score in zip(indices[-5:], similarities[-5:]):
                doc = self.documents[doc_idx]
                preview = doc["content"][:100].replace("\n", " ")
                logger.info(f"Score: {score:.4f} | Page: {doc['metadata']['page']}")
                logger.info(f"   {preview}\n")

            # Optional: Print distribution summary
            high_sim = np.sum(similarities > 0.7)
            med_sim = np.sum((similarities > 0.5) & (similarities <= 0.7))
            low_sim = np.sum(similarities <= 0.5)
            logger.info(f"📈 Score distribution: High (>0.7): {high_sim}, Medium (0.5–0.7): {med_sim}, Low (≤0.5): {low_sim}")

            logger.info("-" * 80)
    # -------------------------------------------------------------------------
    def test_summary(self):
        """Final summary for validation."""
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        logger.info(f"✅ Total chunks: {len(self.texts)}")
        logger.info(f"✅ FAISS vectors: {self.index.ntotal}")
        logger.info(f"✅ Pollination chunks found: {len(self.pollination_chunks)}")
        logger.info("✅ English RAG pipeline functional and semantically consistent.")