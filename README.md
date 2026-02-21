# RAG Evaluation Pipeline

A production-grade **Retrieval Augmented Generation (RAG)** system with a comprehensive **RAGAS-based evaluation harness** for systematic benchmarking and quality measurement.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG EVALUATION PIPELINE                      │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐  │
│  │  Raw     │   │ Chunking │   │Embedding │   │  Vector Store    │  │
│  │  Docs    │──▶│  (size,  │──▶│  Model   │──▶│  (FAISS/Chroma)│  │
│  │  (.txt,  │   │  overlap)│   │  (HF /   │   │  + Disk Cache    │  │
│  │  .md,    │   │          │   │  OpenAI) │   │                  │  │
│  │  .pdf)   │   │          │   │          │   │                  │  │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘  │
│                                                          │          │
│                    ┌─────────────────────────────────────┘           │
│                    │ top-k chunks                                     │
│                    ▼                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  User    │   │Retriever │   │Generator │   │   RAG Answer     │ │
│  │  Query   │──▶│ (sim /   │──▶│  (OpenAI/│──▶│  (grounded to    │ │
│  │          │   │  MMR)    │   │ Anthropic│   │   context)       │ │
│  └──────────┘   └──────────┘   └──────────┘   └────────┬─────────┘ │
│                                                          │           │
│  ┌──────────────────────────────────────────────────────┘           │
│  │                    RAGAS Evaluation                               │
│  │                                                                   │
│  │  Question + Answer + Contexts + Ground Truth                      │
│  │                    │                                              │
│  │         ┌──────────┼──────────┬──────────┐                       │
│  │         ▼          ▼          ▼          ▼                       │
│  │   Faithfulness  Answer    Context    Context                      │
│  │                Relevancy  Precision  Recall                       │
│  │                                                                   │
│  │  → results/ragas_eval_results.csv                                 │
│  │  → results/plots/*.png                                            │
│  └───────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── data/
│   ├── raw/              ← Place your source documents here (.txt, .md, .pdf)
│   ├── processed/        ← Serialised chunks (auto-generated)
│   ├── eval/
│   │   └── eval_dataset.json  ← Evaluation Q&A pairs
│   └── embedding_cache/  ← Embedding cache (auto-generated)
│
├── src/
│   ├── config.py         ← Central typed configuration (all parameters)
│   ├── ingestion.py      ← Document loading: PDF / TXT / Markdown
│   ├── chunking.py       ← Recursive text splitting + JSON persistence
│   ├── embeddings.py     ← Embedding factory + disk-based caching
│   ├── vectorstore.py    ← FAISS / Chroma build & load
│   ├── retriever.py      ← Top-k similarity / MMR retrieval
│   ├── generator.py      ← Grounded LLM generation (OpenAI / Anthropic)
│   ├── rag_pipeline.py   ← Unified pipeline orchestrator
│   ├── evaluator.py      ← RAGAS evaluation harness
│   └── benchmark.py      ← Multi-experiment comparison & visualisation
│
├── experiments/          ← Store experiment config snapshots here
├── results/
│   ├── *.csv             ← RAGAS evaluation results
│   └── plots/            ← Generated comparison charts
│
├── tests/
│   ├── conftest.py       ← Shared pytest fixtures (no API keys needed)
│   ├── test_chunking.py
│   ├── test_retriever.py
│   └── test_evaluator.py
│
├── main.py               ← CLI entry point
├── requirements.txt
└── .env.example
```

---

## Installation

### 1. Clone / set up environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `faiss-cpu` requires a compatible BLAS library. On Windows, install it via pip normally. On Linux, you may need `apt install libopenblas-dev` first.

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

---

## Quick Start

### Run the full pipeline end-to-end

```bash
python main.py
```

This will:
1. Ingest all documents from `data/raw/`
2. Chunk, embed, and index them
3. Run RAG on the evaluation dataset
4. Compute RAGAS metrics
5. Save results to `results/`

### Using CLI sub-commands

```bash
# Step 1: Ingest documents
python main.py ingest

# Step 2: Run evaluation
python main.py evaluate --name my_experiment

# Step 3: Ask a question interactively
python main.py query "What is Retrieval Augmented Generation?"

# Run benchmarks comparing chunk sizes
python main.py benchmark --experiment chunk_size --chunk-sizes 256 512 1024

# Run benchmarks comparing retrieval k values
python main.py benchmark --experiment k_values --k-values 2 4 6 8
```

### CLI reference

| Command | Option | Description |
|---------|--------|-------------|
| `ingest` | `--data-dir PATH` | Override raw document directory |
| `ingest` | `--chunk-size N` | Override chunk size |
| `ingest` | `--rebuild` | Force vector store rebuild |
| `evaluate` | `--eval-dataset PATH` | Path to evaluation JSON |
| `evaluate` | `--chunk-size N` | Override chunk size |
| `evaluate` | `--k N` | Override retrieval top-k |
| `evaluate` | `--name LABEL` | Experiment name for output files |
| `benchmark` | `--experiment TYPE` | `chunk_size`, `k_values`, or `all` |
| `benchmark` | `--chunk-sizes N N …` | Chunk sizes to compare |
| `benchmark` | `--k-values N N …` | k values to compare |
| `query` | `question` | Question string to answer |

---

## Configuration

All parameters live in `src/config.py` as typed dataclasses:

```python
from src.config import RAGConfig, ChunkingConfig, LLMConfig

# Default config
config = RAGConfig()

# Custom config (benchmark-friendly)
config = RAGConfig(
    chunking=ChunkingConfig(chunk_size=256, chunk_overlap=32),
    llm=LLMConfig(provider="openai", model_name="gpt-4o"),
)
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunking.chunk_size` | 512 | Max chars per chunk |
| `chunking.chunk_overlap` | 64 | Overlap between chunks |
| `embedding.provider` | `huggingface` | `huggingface` or `openai` |
| `embedding.model_name` | `all-MiniLM-L6-v2` | Embedding model |
| `vectorstore.backend` | `faiss` | `faiss` or `chroma` |
| `retriever.k` | 4 | Number of chunks to retrieve |
| `retriever.search_type` | `similarity` | `similarity` or `mmr` |
| `llm.provider` | `openai` | `openai` or `anthropic` |
| `llm.model_name` | `gpt-4o-mini` | LLM model name |
| `llm.temperature` | 0.0 | Generation temperature |

---

## Evaluation Dataset Format

The evaluation dataset is a JSON array of objects:

```json
[
  {
    "question": "What is Retrieval Augmented Generation?",
    "ground_truth": "RAG is an AI architecture that enhances LLM responses by retrieving relevant information from an external knowledge base at inference time."
  },
  {
    "question": "What does faithfulness measure?",
    "ground_truth": "Faithfulness measures what fraction of claims in the generated answer are supported by the retrieved context."
  }
]
```

Place your dataset at `data/eval/eval_dataset.json` or specify a custom path with `--eval-dataset`.

### Dataset best practices

- **Minimum 20 questions** for statistically meaningful scores
- Include questions of varying difficulty: lookup, multi-hop, definition, comparison
- Ground truth answers should be **specific and traceable** to source documents
- Include a few questions whose answers are **not** in the corpus (to test negative rejection)

---

## RAGAS Metrics Explained

### Faithfulness (most important)

**What it measures**: Are all claims in the generated answer supported by the retrieved context?

**How it's computed**:
1. An LLM decomposes the answer into atomic factual claims
2. Each claim is verified against the retrieved context
3. Score = (supported claims) / (total claims)

**Range**: 0–1. Higher is better.

**Why it matters**: Low faithfulness = hallucination. The model is making claims that aren't in the source documents.

**How to improve**: Strengthen the system prompt to enforce grounded answers. Use a more instruction-following LLM. Reduce `temperature` to 0.0.

---

### Answer Relevancy

**What it measures**: Does the generated answer actually address the question that was asked?

**How it's computed**:
1. An LLM generates several hypothetical questions from the answer
2. Embeddings of those questions are compared to the original question
3. Score = mean cosine similarity

**Range**: 0–1. Higher is better.

**Why it matters**: A model can be faithful (all claims in context) but still give an evasive or off-topic answer.

**How to improve**: Improve the prompt template to require direct answers. Use a stronger LLM. Reduce retrieved context noise.

---

### Context Precision

**What it measures**: Are the **most relevant** retrieved chunks ranked **first**?

**How it's computed**:
1. An LLM judges whether each retrieved chunk is relevant to answering the question
2. Precision@k is computed at each rank position where a relevant chunk appears
3. Score = mean precision over all relevant positions

**Range**: 0–1. Higher is better.

**Why it matters**: LLMs suffer from "lost in the middle" — they use information near the beginning and end of context more than the middle. Ranking errors waste the LLM's attention.

**How to improve**: Use a better embedding model. Reduce chunk size so chunks are more topically focused. Try MMR for more diverse retrieval. Add a cross-encoder re-ranker.

---

### Context Recall

**What it measures**: Does the retrieved context **contain all the information** needed to answer the question?

**How it's computed**:
1. An LLM decomposes the ground-truth answer into individual claims
2. Each claim is checked against the retrieved context
3. Score = (claims found in context) / (total claims in ground truth)

**Range**: 0–1. Higher is better.

**Why it matters**: Low recall means relevant documents weren't retrieved. The generator cannot produce a complete answer if the information isn't in the context.

**How to improve**: Increase `retriever.k`. Try a better embedding model. Ensure documents covering the topic are in the corpus. Check if chunk size is too large (making chunks too broad to match specific queries).

---

## Running Experiments

### Chunk size comparison

```bash
python main.py benchmark --experiment chunk_size --chunk-sizes 128 256 512 1024
```

Generates `results/plots/comparison_chunk_size.png` showing how each RAGAS metric varies with chunk size.

### Retrieval k comparison

```bash
python main.py benchmark --experiment k_values --k-values 1 2 4 6 8 10
```

Shows the trade-off between context recall (increases with k) and context precision (often decreases with k).

### Full benchmark suite

```bash
python main.py benchmark --experiment all
```

Runs both suites and generates a `results/plots/benchmark_heatmap.png` overview.

### Programmatic experiments

```python
from src.config import RAGConfig, ChunkingConfig
from src.benchmark import BenchmarkRunner
from src.ingestion import ingest_directory

config = RAGConfig()
documents = ingest_directory(config.raw_data_dir)

runner = BenchmarkRunner(config)

# Custom embedding model comparison
runner.run_embedding_comparison(
    embedding_configs=[
        {
            "name": "minilm",
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        },
        {
            "name": "mpnet",
            "provider": "huggingface",
            "model_name": "sentence-transformers/all-mpnet-base-v2",
        },
    ],
    raw_documents=documents,
)

runner.save_benchmark_results()
runner.generate_plots()
runner.generate_heatmap()
```

---

## Interpreting Results

### Output files

| File | Contents |
|------|----------|
| `results/ragas_eval_<name>_<timestamp>.csv` | Per-question RAGAS scores |
| `results/experiment_<timestamp>.csv` | Benchmark summary across all experiments |
| `results/plots/comparison_<param>.png` | Bar charts per metric vs parameter |
| `results/plots/benchmark_heatmap.png` | Colour-coded score matrix |

### Score interpretation

| Score range | Meaning |
|-------------|---------|
| 0.85–1.00 | Excellent |
| 0.70–0.85 | Good — production-ready |
| 0.55–0.70 | Acceptable — room for improvement |
| 0.40–0.55 | Poor — significant issues |
| 0.00–0.40 | Very poor — diagnose root cause |

### Diagnostic flowchart

```
Low faithfulness?
  → Strengthen grounding prompt
  → Use better LLM (gpt-4o instead of gpt-4o-mini)
  → Set temperature=0.0

Low answer relevancy?
  → Prompt asks for direct answers
  → Reduce k (less noise in context)
  → Better LLM

Low context precision?
  → Better embedding model
  → Smaller chunk size
  → Add cross-encoder re-ranker
  → Try MMR retrieval

Low context recall?
  → Increase k
  → Larger chunk size
  → Better embedding model
  → Check corpus coverage
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

Tests do **not** require API keys — they use mocked LLMs and in-memory data.

---

## Adding Your Own Documents

1. Place `.txt`, `.md`, or `.pdf` files in `data/raw/`
2. Update `data/eval/eval_dataset.json` with questions and ground-truth answers about your documents
3. Run `python main.py ingest --rebuild` to re-index
4. Run `python main.py evaluate` to see RAGAS scores

---

## Switching LLM Providers

**Use Anthropic Claude instead of OpenAI:**

```bash
# Set in .env:
ANTHROPIC_API_KEY=sk-ant-...

# Run with Anthropic:
python main.py evaluate --llm-provider anthropic
```

Or configure in code:

```python
from src.config import RAGConfig, LLMConfig

config = RAGConfig(
    llm=LLMConfig(
        provider="anthropic",
        model_name="claude-3-5-haiku-20241022",
    )
)
```

---

## Using OpenAI Embeddings

Higher-quality embeddings typically improve context precision and recall:

```python
from src.config import RAGConfig, EmbeddingConfig

config = RAGConfig(
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small",
    )
)
```

---

## Advanced: LangSmith Experiment Tracking

Set these in your `.env` to log all LLM calls to LangSmith:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=rag-eval-pipeline
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Pipeline orchestration | LangChain 0.3+ |
| Embeddings (local) | sentence-transformers |
| Embeddings (cloud) | OpenAI text-embedding-3 |
| Vector store | FAISS (default) / Chroma |
| LLM (cloud) | OpenAI GPT / Anthropic Claude |
| Evaluation | RAGAS 0.2+ |
| Analysis | pandas, numpy |
| Visualisation | matplotlib, seaborn |

---
