# AI Spectra: Semantic Mutation Engine

A sophisticated system for discovering hidden patterns and chains of meaning through semantic graph exploration.

## 🚀 What's New

### 1. **Interesting Facts Discovery** - `explore_facts.py`
Discovers unexpected connections and insights across knowledge domains without political bias.

**Features:**
- Cross-domain hypothesis generation (e.g., Indian History + Aerospace)
- Semantic similarity-based walking through concept networks
- Automatic pattern recognition and chain discovery

**Usage:**
```bash
python explore_facts.py --sources india_history.txt aerospace.txt --iterations 10 --visualize
```

**Included Domains:**
- `india_history.txt` - Scientific & cultural contributions from Indian history
- `aerospace.txt` - Aerospace engineering principles and concepts

**Example Output:**
```
Score: 0.7812
  Affect may connect confidence and courage through a shared latent pattern.
  Path: w3(aerospace) → w5(india_history) → w2(aerospace)
```

### 2. **Web Extraction & Walking** - `fetch_and_walk.py`
Fetch content from external URLs and automatically discover semantic chains.

**Features:**
- Automatic HTML parsing and text extraction
- Sentence-level content processing
- Direct integration with mutation engine

**Usage:**
```bash
python fetch_and_walk.py \
  --urls https://example.com/page1 https://example.com/page2 \
  --iterations 8 \
  --threshold 0.45 \
  --visualize
```

**What it does:**
1. Fetches content from each URL
2. Extracts meaningful sentences
3. Generates embeddings
4. Builds semantic graph
5. Runs random walks and generates hypotheses
6. Merges findings and creates visualizations

### 3. **Graph Visualization** - `src/visualizer.py`
Creates clear, publication-ready visualizations of the semantic graph and discovery chains.

**Features:**
- Main graph visualization with node types distinguished
- Individual walk path highlighting
- Edge weight transparency showing connection strength
- Batch reporting with top discoveries

**Visualizations Created:**
- `graph.png` - Overall semantic network with original and generated nodes
- `walk_01.png` through `walk_05.png` - Top 5 discovered chains

**Usage (integrated in all scripts):**
```bash
python explore_facts.py --visualize
python fetch_and_walk.py --visualize
python main.py  # Also generates visualizations
```

## 📊 Architecture

```
Raw Text Input
    ↓
Sentence Splitting
    ↓
Semantic Embeddings (Sentence Transformers)
    ↓
Graph Building (similarity edges)
    ↓
Weighted Random Walks
    ↓
Hypothesis Generation (keyword extraction)
    ↓
Scoring & Filtering
    ↓
Visualizations & Reports
```

## 🔧 New Modules

### `src/fetcher.py`
```python
fetch_and_extract(url) → str  # Fetch & extract text from URL
extract_sentences(text) → list[str]  # Convert to sentences
```

### `src/visualizer.py`
```python
visualize_graph(graph, title, output_file)  # Draw main graph
visualize_walk_chain(graph, path, generated_text, output_file)  # Draw walk path
generate_report(graph, generated_records, output_dir)  # Create full report
```

### Enhanced `src/ingest.py`
```python
load_multiple_texts(file_names) → dict  # Load multiple source files
create_nodes_from_sources(sources) → list  # Create nodes from multiple sources
```

## 📁 Updated Project Structure

```
stupidity/
├── main.py                          # Original pipeline (now with visualization)
├── explore_facts.py                 # New: Domain-specific discovery
├── fetch_and_walk.py                # New: Web extraction + walking
├── requirements.txt                 # Updated with requests, beautifulsoup4, matplotlib
├── data/
│   ├── raw/
│   │   ├── input.txt                # Original sample
│   │   ├── india_history.txt        # New: Indian history facts
│   │   └── aerospace.txt            # New: Aerospace concepts
│   ├── processed/
│   │   ├── nodes.json
│   │   ├── edges.json
│   │   └── generated.json
│   └── graph/
│       └── graph.gpickle
├── visualizations/                  # New: Output directory for visualizations
│   ├── graph.png
│   ├── walk_01.png
│   └── ...
└── src/
    ├── fetcher.py                   # New: Web extraction
    ├── visualizer.py                # New: Graph visualization
    ├── ingest.py                    # Enhanced: Multi-source loading
    ├── embed.py
    ├── graph_builder.py
    ├── loop.py
    ├── mutator.py
    ├── utils.py
    └── walker.py
```

## 🎯 Example Workflows

### Discover Indian History & Aerospace Connections
```bash
python explore_facts.py --sources india_history.txt aerospace.txt --iterations 10
```

### Extract & Analyze Wikipedia Content
```bash
python fetch_and_walk.py \
  --urls https://wikipedia.org/wiki/Mathematics \
           https://wikipedia.org/wiki/Physics \
  --iterations 5 \
  --visualize
```

### Run Full Pipeline with Visualization
```bash
python main.py --iterations 5 --walk-steps 4 --threshold 0.45
```

## 🔍 How It Works

1. **Graph Construction**: Builds semantic similarity graph using embeddings
2. **Weighted Walking**: Traverses high-similarity paths (not random)
3. **Hypothesis Generation**: Extracts keywords from walks to create new hypotheses
4. **Iterative Refinement**: Generated nodes feed back into the graph
5. **Pattern Discovery**: Emergent chains reveal hidden connections

## 📦 Dependencies

```
sentence-transformers  # Semantic embeddings
networkx              # Graph construction
numpy                 # Numerical operations
scikit-learn          # Similarity metrics
transformers          # NLP models
torch                 # Deep learning backend
requests              # HTTP fetching
beautifulsoup4        # HTML parsing
matplotlib            # Visualization
```

## 💡 Interesting Facts to Discover

The system can find connections like:
- How ancient mathematical concepts relate to modern aerospace precision
- Connections between historical trade networks and current supply chains
- Patterns linking consciousness philosophy to neuroscience discoveries
- Relationships between architectural principles and structural engineering

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Explore Indian history & aerospace connections
python explore_facts.py --visualize

# Or extract & analyze web content
python fetch_and_walk.py --urls <your-urls> --visualize

# Or run with custom data
python main.py --iterations 8 --threshold 0.42
```

## 📈 Output

Each run generates:
- **Terminal output**: Top hypotheses with scores
- **JSON records**: `generated.json` with all discovered chains
- **Visualizations**: Network graphs and path diagrams (if `--visualize`)
- **Statistics**: Node counts, edge counts, average scores

## 🎓 Research Applications

- **Cross-domain discovery**: Finding unexpected connections between fields
- **Knowledge synthesis**: Combining disparate information sources
- **Pattern recognition**: Identifying emergent structures in text
- **Hypothesis generation**: Automated suggestion of research directions

---

Built with curiosity and semantic chaos. Not just random, but *meaningful* randomness.
