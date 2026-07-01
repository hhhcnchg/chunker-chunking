# Rigid Chunker 🧩

**Rigid paragraph boundary + flexible semantic intra-paragraph splitting** — a novel text chunking strategy for RAG.

Treats natural paragraphs as **high-voltage fences** that are never crossed. Splits are made only *within* a paragraph. Designed to solve **semantic fragmentation** and **cross-paragraph debris** plaguing traditional chunking methods.

## Core Idea

```
Paragraph boundary ── NEVER CROSS ── ── ── ── ── ── ── 
                  ┌──────────────────────┐
  Sub-chunk A     │  Complete semantic    │  ← used for retrieval
                  │  unit A               │
                  └──────────────────────┘
                  ┌──────────────────────┐
  Sub-chunk B     │  Complete semantic    │
                  │  unit B               │
                  └──────────────────────┘
                  ┌──────────────────────┐
  Parent para     │  Full text (A+B)      │  ← used for generation
                  └──────────────────────┘
```

- **Sub-chunks** for fine-grained matching
- **Parent paragraph** replaces the sub-chunk at generation time — never lose context

## Key Features

| Feature | Description |
|---------|-------------|
| **Paragraph locking** | Natural paragraphs are absolute boundaries; never split across them |
| **Sub-structure detection** | Auto-detects Chinese numbered items `（一）（二）（三）`, each becomes its own chunk |
| **Parent paragraph storage** | Every sub-chunk carries its full parent text; auto-swapped at retrieval |
| **Model detection chain** | Auto-detect LLM → Embedding → Rule fallback, degrades gracefully |
| **Smart merging** | Short paragraphs auto-merged; headings/titles intelligently attached |
| **Dynamic threshold** | Adaptively computed from document paragraph-length distribution |
| **No LLM required** | Works with just an embedding model; LLM is optional |

## Three-Tier Model Detection

```
Tier 1: LLM   → Coarse semantic splitting + Embedding fine-splitting
Tier 2: Embed → Cosine similarity between sentences, split at minima
Tier 3: Rule  → Sentence-break fallback (。！？)
```

If no LLM is detected, the tool **asks whether you'd like to configure one** — it never silently degrades.

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Embedding Model (Optional)

By default the tool auto-downloads `BAAI/bge-small-zh-v1.5` from HuggingFace.
Users in China can use a mirror:

```bash
python download_model.py
```

### Configure LLM (Optional)

Three options:

1. **Environment variables** (recommended):
   ```bash
   export DEEPSEEK_API_KEY="sk-xxx"
   export DEEPSEEK_MODEL="deepseek-v4-flash"
   ```
2. **Interactive setup**: the tool asks at first run
3. **Local runtime**: auto-detects Ollama, LM Studio, llama.cpp, vLLM

### Run

```bash
# Chunk a document
python rigid_chunker.py your-document.txt

# Optional: create and edit custom config
cp .config.example .config.py
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL_PATH` | `""` | Local model path; leave empty for auto-download |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-zh-v1.5` | HuggingFace model name |
| `MAX_SUB_CHUNKS` | `6` | Max sub-chunks per paragraph |
| `MIN_CHUNK_CHARS` | `50` | Minimum sub-chunk length |
| `SHORT_PARA_LIMIT` | `50` | Short paragraph threshold |
| `BASE_THRESHOLD` | `500` | Dynamic threshold anchor |
| `COLLECTION_NAME` | `rigid_chunker_demo` | Chroma collection name |

## How It Works

### Phase 1: Paragraph Anchoring

1. Split raw text by blank lines
2. Detect and smart-merge short paragraphs (LLM decides strategy if available)
3. Scan for sub-structure markers (`（一）...（十）`)

### Phase 2: Intra-Paragraph Splitting

1. Sub-structure detected → split sequentially, each item becomes an independent chunk
2. Short paragraphs (< dynamic threshold) → kept whole
3. Long paragraphs:
   - **With LLM**: LLM coarsely splits → Embedding finely splits each section
   - **With Embedding**: cosine similarity minima as split points
   - **Fallback**: sentence-boundary split (。！？)

### Phase 3: Storage & Retrieval

- Chroma vector DB for storage
- Sub-chunks for retrieval matching
- Parent paragraph text replaces sub-chunk at generation
- Deduplication by parent paragraph

## Comparison with LumberChunker / Meta-Chunking

| Aspect | Rigid Chunker | LumberChunker | Meta-Chunking |
|--------|---------------|---------------|---------------|
| **Paragraph locking** | ✅ Rigid boundary | ❌ May cross | ❌ May cross |
| **LLM required** | Optional (auto-degrade) | Required | Required |
| **Sub-structure detection** | ✅ Automatic | ❌ None | ❌ None |
| **Parent paragraph storage** | ✅ Yes | ❌ No | ❌ No |
| **Dynamic threshold** | ✅ Automatic | Fixed | Fixed |
| **Model cost** | Low (Embedding-driven) | High (LLM per segment) | High (LLM per boundary) |
| **Determinism** | High (paragraph-locked) | Medium (LLM variance) | Medium (LLM variance) |
| **Chinese-optimized** | ✅ Yes | ❌ English-first | ❌ English-first |

## Project Structure

```
chunker-chunking/
├── rigid_chunker.py      # Core chunker
├── download_model.py     # Model download helper
├── .config.example       # Configuration template
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
├── README.md             # Chinese docs
└── README.en.md          # English docs
```

## License

MIT
