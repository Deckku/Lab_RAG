# Lab_RAG
# Retrieval-Augmented Generation (RAG) System Lab

## Introduction

Retrieval-Augmented Generation (RAG) represents a paradigm shift in how language models access and utilize information. By combining the generative capabilities of large language models with external knowledge retrieval systems, RAG architectures overcome the limitations of static training data, enabling models to provide accurate, up-to-date, and contextually relevant responses grounded in specific knowledge bases.

## Lab Overview

This lab provides hands-on experience with building a complete RAG system from scratch. Through practical implementation, we explore how retrieval mechanisms can be integrated with generative models to create an intelligent question-answering system. The lab demonstrates the entire RAG pipeline, from document processing to semantic search and answer generation.

### Document Context

The document provided with this lab is **my internship report**, which serves as the knowledge base for our RAG system. By applying RAG techniques to this report, we can efficiently query and extract information about my internship experience, projects completed, technologies learned, and insights gained. This practical application demonstrates how RAG systems can transform static documents into interactive, queryable knowledge sources.

### What We'll Build

In this laboratory session, we implement **2-3 distinct RAG architectures** with varying retrieval and generation strategies to understand:

- **Document Processing Pipeline**: Advanced PDF extraction with structure preservation, text cleaning, and semantic chunking
- **Retrieval System**: FAISS-based vector search with semantic embeddings and intelligent re-ranking mechanisms
- **Generation Component**: Answer synthesis using Microsoft Phi-2 model with context-aware prompting

Each iteration explores different aspects of the RAG pipeline:

1. **Basic RAG**: Simple retrieval with direct answer generation
2. **Enhanced RAG**: Semantic chunking and re-ranking for improved relevance
3. **Production RAG**: Interactive Streamlit interface with quality metrics and history tracking

Through these implementations, we gain deep insight into how RAG systems balance retrieval precision with generation quality, and how architectural choices impact the overall system performance.

## Key Features

- ✅ **Semantic Chunking**: Preserves document structure and context boundaries
- ✅ **Dual-Stage Retrieval**: Initial FAISS search followed by cross-encoder re-ranking
- ✅ **Microsoft Phi-2 LLM**: Advanced language model for coherent answer generation
- ✅ **Interactive UI**: Streamlit-based interface with real-time processing
- ✅ **Quality Metrics**: Chunk analysis and relevance scoring
- ✅ **Conversation History**: Track questions and answers across sessions

## Architecture

```
┌─────────────┐
│  PDF Upload │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Text Extraction │ ← PDFPlumber with layout preservation
└──────┬──────────┘
       │
       ▼
┌──────────────────┐
│ Semantic Chunking│ ← Sentence-aware splitting with overlap
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│  Embeddings  │ ← SentenceTransformer (MiniLM)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ FAISS Index  │ ← Vector similarity search
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Re-ranking  │ ← Cosine similarity refinement
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Phi-2 LLM   │ ← Context-aware answer generation
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Answer    │
└──────────────┘
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-internship-lab


```

### Requirements

```txt
streamlit>=1.28.0
transformers>=4.35.0
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
pdfplumber>=0.10.0
torch>=2.0.0
numpy>=1.24.0
```

### Running the Application

```bash
# Launch the Streamlit interface
streamlit run RAG.py
```

The application will open in your default browser at `http://localhost:8501`

### Usage Workflow

1. **Upload PDF**: Click "Browse files" and upload your internship report PDF
2. **Wait for Processing**: The system will:
   - Extract text from PDF pages
   - Create semantic chunks
   - Build vector search index
   - Load the Phi-2 language model
3. **Ask Questions**: Type questions about the internship report
4. **Review Answers**: See generated answers with source chunk references
5. **Explore Context**: Expand source chunks to verify answer accuracy

## Configuration

Adjust retrieval and generation parameters in the sidebar:

- **Initial Retrieval (3-10)**: Number of chunks to retrieve initially
- **Final Chunks (1-5)**: Number of chunks to use for answer generation
- **Chunk Size (300-800)**: Characters per chunk
- **Overlap (50-200)**: Character overlap between chunks

## Technical Details

### PDF Processing
- Uses `pdfplumber` for layout-aware extraction
- Deep cleaning removes headers, page numbers, and artifacts
- Preserves paragraph structure and semantic boundaries

### Semantic Chunking
- Sentence-boundary aware splitting
- Configurable chunk size with intelligent overlap
- Maintains context continuity across chunks

### Retrieval System
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- FAISS IndexFlatIP for efficient similarity search
- Two-stage retrieval: initial search + re-ranking

### Answer Generation
- Microsoft Phi-2 (2.7B parameters)
- Context-aware prompting
- Temperature and top-p sampling for coherent outputs
- Repetition penalty to avoid redundancy

## Example Questions

Try asking your RAG system:

- "What were the main objectives of the internship?"
- "Which technologies were used during the internship?"
- "What challenges were encountered and how were they resolved?"
- "What were the key learnings from this internship experience?"
- "Describe the projects completed during the internship"

## Performance Considerations

- **GPU Recommended**: Phi-2 model runs faster on CUDA-enabled GPUs
- **Memory Usage**: ~6GB RAM for model loading and inference
- **Processing Time**: 30-60 seconds for initial PDF processing
- **Query Response**: 5-15 seconds per question

## Project Structure

```
├── RAG.py                 # Main application script
├── README.md              # This file
└── asset.pdf  # Sample document
```

---

**Author:** AZAMI HASSANI ADNANE  
**Supervisor:** Prof. MASROUR TAWFIK
