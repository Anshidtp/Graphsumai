# Graphsum AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![React](https://img.shields.io/badge/React-18.2+-61DAFB.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_4-FF6B6B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A production-ready Retrieval-Augmented Generation (RAG) system powered by Knowledge Graphs and Groq Cloud**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Installation](#-installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements a state-of-the-art **Graph RAG (Retrieval-Augmented Generation)** system using the **FB15k-237 knowledge graph dataset**. It combines:

- 🗄️ **Neo4j AuraDB** for graph database storage
- 🔍 **Sentence Transformers** for semantic vector embeddings
- 🚀 **Groq Cloud (Llama 4)** for ultra-fast natural language generation
- 🎨 **React + Vite** for a beautiful, modern UI
- 🌐 **FastAPI** for high-performance REST API

### What is Graph RAG?

Graph RAG enhances traditional RAG by using **knowledge graphs** instead of simple vector databases. This provides:

- ✅ **Structured relationships** between entities
- ✅ **Multi-hop reasoning** across connected facts
- ✅ **Explainable results** with entity citations
- ✅ **Richer context** for LLM generation

---

## ✨ Features

### 🎯 Core Features

- **Wikidata Entity Resolution** - Automatically resolves Freebase IDs to human-readable names
- **Triplet Deduplication** - Intelligent removal of duplicate knowledge triplets
- **Vector Similarity Search** - Semantic search using sentence embeddings (384-dim)
- **Hybrid Retrieval** - Combines keyword search + vector similarity
- **Multi-hop Traversal** - Explores entity relationships up to N hops
- **LLM Graph Enrichment** - Uses Groq Llama 4 to extract additional relationships
- **Conversational Answers** - Natural language responses powered by Groq's ultra-fast inference

### 🎨 Frontend Features

- **Glassmorphic Design** - Modern blur effects and transparency
- **Animated Background** - Floating gradient blobs
- **Real-time Stats** - Live entity and triplet counts
- **Entity Tags** - Visual representation of related entities
- **Smart Suggestions** - Quick-start question buttons
- **Responsive Design** - Works on desktop, tablet, and mobile
- **Dark Theme** - Easy on the eyes with purple-blue gradients

### 🔧 Technical Features

- **Production-Ready** - Comprehensive error handling and logging
- **Scalable Architecture** - Modular components for easy extension
- **Batch Processing** - Efficient handling of large datasets (272K+ triplets)
- **Caching System** - Wikidata API response caching
- **REST API** - Clean, documented endpoints with Swagger UI
- **Ultra-Fast Inference** - Groq Cloud delivers answers in <2 seconds

---

## 🎬 Demo

### Query Example

**User:** "Who is Barack Obama?"

**System Response:**
> Barack Obama is an American author and professor of Irish American heritage. He received his education at Columbia University, earning a Juris Doctor degree. In his professional life, Obama has worked both in academia as a professor and as an author. He's known to have connections with several prominent figures, including former Vice President Al Gore and actress Charlize Theron.

**Performance:**
- 🔍 Retrieved: 10 relevant triplets
- 🏷️ Entities: 5 related entities
- 📊 Context: 2-hop neighborhood expansion
- ⚡ Response time: 1.2 seconds (powered by Groq)

### Screenshots

```
┌─────────────────────────────────────────────┐
│  🧠 Knowledge Graph Explorer 💾             │
│  📚 14,541 entities  ⚡ 272,115 facts       │
├─────────────────────────────────────────────┤
│                                             │
│  💬 Chat Interface                          │
│     ┌─────────────────────────────────┐    │
│     │ 🤖 AI Assistant                 │    │
│     │ Barack Obama is an American...  │    │
│     │                                 │    │
│     │ Related: Barack Obama, United   │    │
│     │ States, Columbia University     │    │
│     │ 💾 10 facts retrieved            │    │
│     └─────────────────────────────────┘    │
│                                             │
│  💡 Smart Suggestions                       │
│     "Who is Barack Obama?"                  │
│     "What professions are in the database?" │
│                                             │
│  ✍️ Input Box                               │
│     [Ask me anything...] [🚀 Send]          │
│                                             │
└─────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │      │   FastAPI    │      │   Neo4j     │
│  Frontend   │─────▶│   Backend    │─────▶│   AuraDB    │
│  (Vite)     │ HTTP │   REST API   │ Bolt │   Graph DB  │
└─────────────┘      └──────────────┘      └─────────────┘
       │                     │                     │
       │                     ▼                     │
       │            ┌──────────────┐              │
       │            │  Groq Cloud  │              │
       │            │  Llama 4 API │              │
       │            └──────────────┘              │
       │                     │                     │
       ▼                     ▼                     ▼
  Tailwind CSS      Sentence-BERT          Vector Index
   Lucide Icons      (Embeddings)         Cypher Queries
```

### Data Flow

```
1. Data Loading & Processing
   └─▶ FB15k-237 Dataset (272,115 triplets)
       └─▶ Entity Resolution (Wikidata SPARQL API)
           └─▶ Deduplication (Remove duplicates)
               └─▶ Triplet Embedding (Sentence-BERT 384-dim)
                   └─▶ Graph Construction (Neo4j Cypher)
                       └─▶ Vector Index Creation (Cosine similarity)

2. Query Processing (Real-time)
   └─▶ User Query (Frontend)
       └─▶ Query Embedding (Sentence-BERT)
           └─▶ Vector Search (Top-K triplets)
               └─▶ Entity Extraction (Parse triplets)
                   └─▶ Multi-hop Expansion (2-3 hops)
                       └─▶ Context Formatting (Group by entity)
                           └─▶ LLM Generation (Groq Llama 4)
                               └─▶ Response (JSON + Display)
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Graph Database** | Neo4j AuraDB | Store knowledge graph with 272K triplets |
| **Vector Embeddings** | Sentence-BERT (all-MiniLM-L6-v2) | Semantic similarity (384-dim) |
| **LLM** | Groq Cloud (Llama 4) | Ultra-fast answer generation (500+ tok/s) |
| **API Framework** | FastAPI | REST API with automatic docs |
| **Frontend** | React 18 + Vite | Modern SPA with HMR |
| **Styling** | Tailwind CSS | Utility-first CSS framework |
| **Icons** | Lucide React | 1000+ beautiful icons |
| **Entity Resolution** | Wikidata SPARQL API | Real entity names from Freebase IDs |
| **Graph Enrichment** | Groq (Llama 3.1-70B) | Extract implicit relationships |

---

## 🚀 Installation

### Prerequisites

Before you begin, ensure you have:

- ✅ **Python 3.9+** ([Download](https://www.python.org/downloads/))
- ✅ **Node.js 18+** ([Download](https://nodejs.org/))
- ✅ **Neo4j AuraDB** account ([Sign up](https://neo4j.com/cloud/aura/))
- ✅ **Groq API Key** ([Get free key](https://console.groq.com/))

### Backend Setup

#### Step 1: Clone Repository

```bash
git clone https://github.com/Anshidtp/Graphsumai
cd Graphsumai/backend
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Download FB15k-237 Dataset

```bash
# Create data directory
mkdir -p data/FB15k-237

# Download dataset
wget https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip

# Extract
unzip FB15K-237.2.zip -d data/FB15k-237/
```

Or download manually from [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52312).


#### Step 5: Configure Environment

Create `.env` file in the root directory:

```bash
# Neo4j AuraDB Configuration
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_auradb_password

# Groq Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Data Paths
DATA_DIR=./data/FB15k-237
RESOLVED_DIR=./data/resolved
```

#### Step 7: Run Pipeline

**Option 1: Run All Steps At Once** (Recommended)

```bash
# Complete pipeline (takes 2-3 hours for first run)
python main.py --all --enrich 100
```

**Option 2: Run Step-by-Step**

```bash
# Step 1: Resolve entities and deduplicate (1-2 hours)
python main.py --step 1

# Step 2: Build knowledge graph (30-60 minutes)
python main.py --step 2 

# Step 3: Initialize RAG system
python main.py --step 3

# Step 4: Test query
python main.py --step 4 --query "Who is Barack Obama?"

# Step 5: Start API server
python main.py --step 5
```

```
🌐 API Server running at http://localhost:8000
📖 Docs at http://localhost:8000/docs
```

### Frontend Setup

#### Step 1: Create React Project

```bash

# Navigate to frontend
cd frontend
```

#### Step 2: Install Dependencies

```bash
# Install base dependencies
npm install

# Install Tailwind CSS
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Install Lucide React icons
npm install lucide-react
```

#### Step 3: Start Development Server

```bash
npm run dev
```

Open browser at `http://localhost:5173/`

---

## 💻 Usage

### Quick Start

```bash
# Terminal 1: Start backend API
cd backend
python main.py --step 5

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Then open:
- **Frontend**: http://localhost:5173/
- **API Docs**: http://localhost:8000/docs

### Example Queries

Try these questions to explore the knowledge graph:

```
✅ "Who is Barack Obama?"
✅ "Who won the 52nd Grammy Award?"
✅ "What movies did actors work on?"
✅ "What universities are mentioned?"
✅ "Tell me about the United States"
```

## 📡 API Documentation

### Base URL

```
http://localhost:8000/api/v1
```

### Endpoints

#### 1. Query Knowledge Graph

**POST** `/query`

Query the knowledge graph with natural language and get AI-generated answers.

**Request Body:**
```json
{
  "query": "Who is Barack Obama?",
  "top_k": 15
}
```

**Response:**
```json
{
  "query": "Who is Barack Obama?",
  "answer": "Barack Obama is an American author and professor of Irish American heritage. He received his education at Columbia University, earning a Juris Doctor degree...",
  "entities_found": [
    "Barack Obama",
    "United States",
    "Columbia University",
    "Juris Doctor",
    "Al Gore"
  ],
  "num_triplets": 10,
  "status": "success"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"Who is Barack Obama?","top_k":15}'
```

#### 2. Search Entities

**POST** `/search`

Search for entities by name or partial match.

**Request Body:**
```json
{
  "search_term": "obama",
  "limit": 10
}
```

**Response:**
```json
{
  "entities": [
    {
      "name": "Barack Obama",
      "degree": 145
    },
    {
      "name": "Michelle Obama",
      "degree": 67
    }
  ]
}
```

#### 3. Get Graph Statistics

**GET** `/stats`

Get comprehensive statistics about the knowledge graph.

**Response:**
```json
{
  "entities": 14541,
  "triplets": 272115,
  "relationships": 272115,
  "avg_degree": 37.42,
  "max_degree": 1891,
  "min_degree": 1,
  "top_entities": [
    {
      "name": "United States",
      "degree": 1891
    },
    {
      "name": "New York City",
      "degree": 876
    }
  ]
}
```

#### 4. Health Check

**GET** `/health`

Check if all system components are running.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "query_engine": true,
    "retriever": true,
    "generator": true
  }
}
```

### Interactive API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI documentation where you can:

- 📖 View all endpoints
- 🧪 Test API calls directly
- 📝 See request/response schemas
- 💻 Generate code snippets

---


## 🐛 Troubleshooting

### Common Issues

#### 1. **Neo4j Connection Error**

**Error:** `Failed to connect to Neo4j`

**Solutions:**
```bash
# Check Neo4j credentials
# Verify in .env:
NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io  # Must include neo4j+s://
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('your_uri', auth=('neo4j', 'password')); print('Connected!')"
```

**Additional steps:**
- Whitelist your IP in AuraDB console
- Verify instance is running (not paused)
- Check firewall settings

#### 2. **Groq API Error**

**Error:** `Groq API key invalid` or `Rate limit exceeded`

**Solutions:**
```bash
# Verify API key
echo $GROQ_API_KEY  # Should start with gsk_

# Check free tier limits
# 14,400 requests/day
# 30 requests/minute

# If rate limited, wait 60 seconds or upgrade plan
```

**Get new API key:**
1. Go to https://console.groq.com/keys2. Delete old key
3. Create new key
4. Update `.env`


#### 3. **CORS Error**

**Error:** `Access to fetch blocked by CORS policy`

**Solutions:**
```python
# Verify in main.py that CORS is enabled:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check both servers are running:
# Backend: http://localhost:8000
# Frontend: http://localhost:5173
```

#### 4. **Wikidata API Rate Limit**

**Error:** `Too many requests to Wikidata`

**Solutions:**
```bash
# Wait 10-15 minutes
# Wikidata has rate limits: 60 requests/minute

# Reduce batch size in config/settings.py:
WIKIDATA_BATCH_SIZE = 10  # Default: 50

# Use cached results
# Cache file: data/entity_cache.json
```

#### 5. **Sentence Transformers Download Error**

**Error:** `Failed to download model`

**Solutions:**
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Or use different model in .env:
EMBEDDING_MODEL=paraphrase-MiniLM-L6-v2
```

#### 6. **Memory Error During Graph Building**

**Error:** `MemoryError` or system freezes

**Solutions:**
```python
# Reduce batch size in config/settings.py:
BATCH_SIZE = 500  # Default: 1000

# Or process in smaller chunks:
python main.py --step 2  # Will use less memory
```

---


## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Anshid T P]
```

---

## 🙏 Acknowledgments

- **FB15k-237 Dataset** - [Microsoft Research](https://www.microsoft.com/en-us/download/details.aspx?id=52312)
- **Neo4j** - Graph database platform ([neo4j.com](https://neo4j.com))
- **Groq Cloud** - Ultra-fast LLM inference ([groq.com](https://groq.com))
- **Sentence-BERT** - Semantic embeddings ([SBERT.net](https://www.sbert.net))
- **Wikidata** - Entity knowledge base ([wikidata.org](https://www.wikidata.org))
- **Tailwind CSS** - Utility-first CSS ([tailwindcss.com](https://tailwindcss.com))
- **Lucide Icons** - Beautiful icons ([lucide.dev](https://lucide.dev))
- **FastAPI** - Modern Python API framework ([fastapi.tiangolo.com](https://fastapi.tiangolo.com))
- **React** - JavaScript UI library ([react.dev](https://react.dev))
- **Vite** - Next-generation frontend tooling ([vitejs.dev](https://vitejs.dev))

---


## 📊 Performance Benchmarks

### System Performance

| Metric | Value |
|--------|-------|
| **Graph Size** | 14,541 entities, 272,115 triplets |
| **Vector Index** | 384-dim embeddings, cosine similarity |
| **Query Latency** | <2 seconds (avg) |
| **Groq Inference** | 500+ tokens/second |
| **Memory Usage** | ~4GB RAM (during queries) |
| **Disk Space** | ~2GB (graph + embeddings) |

### Query Performance Examples

| Query Type | Triplets Retrieved | Response Time |
|------------|-------------------|---------------|
| Simple entity lookup | 10 | 0.8s |
| Multi-hop reasoning | 25 | 1.5s |
| Complex relationships | 40 | 2.1s |

---

## 🎓 Learn More

### Documentation

- 📖 [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- 📖 [Groq Documentation](https://console.groq.com/docs)
- 📖 [Sentence-BERT Guide](https://www.sbert.net/docs/)
- 📖 [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- 📖 [React Documentation](https://react.dev/)

### Related Papers

- **FB15k-237**: "Observed versus latent features for knowledge base and text inference" (2015)
- **Graph RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- **Sentence-BERT**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)

### Tutorials

- [Building Knowledge Graphs](https://neo4j.com/developer/graph-database/)
- [RAG System Design](https://www.deeplearning.ai/short-courses/building-applications-vector-databases/)
- [Groq Quickstart](https://console.groq.com/docs/quickstart)

---

## ❓ FAQ

### General Questions

**Q: What is FB15k-237?**
A: FB15k-237 is a subset of Freebase with 14,541 entities and 237 relation types, commonly used for knowledge graph research.

**Q: Why use Groq instead of OpenAI?**
A: Groq is 10x faster (500+ tok/s vs 50 tok/s), has a generous free tier, and delivers sub-second response times.

**Q: Can I use a different dataset?**
A: Yes! Modify `src/data/loader.py` to load your custom triplets in the format: `subject\trelation\tobject`

**Q: Does this work offline?**
A: Partially. The graph and embeddings work offline, but Groq requires internet for LLM inference.

### Technical Questions

**Q: How do I add more entities?**
A: Add new triplets to your dataset, then run `python main.py --step 2` to rebuild the graph.

**Q: Can I use a local LLM instead of Groq?**
A: Yes! Replace `src/rag/generator.py` with Ollama or any other LLM. Groq is just faster.

**Q: How do I improve answer quality?**
A: Increase `top_k` parameter, use LLM enrichment (`--step 1`), or fine-tune the prompt in `generator.py`.

**Q: What's the cost?**
A: Free! Groq free tier (14,400 req/day) is enough for development. Neo4j AuraDB free tier includes 200K nodes.

---

## 🎯 Use Cases

This system can be adapted for:

- 🏥 **Medical Knowledge Graphs** - Drug interactions, disease relationships
- 📚 **Educational Platforms** - Student questions on course materials
- 🏢 **Enterprise Knowledge Management** - Company wikis, documentation
- 🔬 **Research Assistants** - Scientific paper Q&A
- 🛒 **E-commerce** - Product recommendations based on knowledge graphs
- 🎬 **Entertainment** - Movie/music recommendations
- 🌍 **General Q&A** - Any domain with structured knowledge

---


