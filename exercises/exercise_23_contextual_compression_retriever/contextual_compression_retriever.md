# 🧠 Context Compression Retrieval

## 📖 Definition

**Contextual Compression Retrieval** is a retrieval technique that **reduces the number of tokens in the context window** by using an LLM to **summarize or compress the retrieved documents** before passing them to a downstream language model. The goal is to fit more relevant information into the model’s limited context window without losing key insights.

---

## 💡 Motivation

Large Language Models (LLMs) like GPT-4 have a limited **context window** (e.g., 8K–128K tokens). Traditional retrieval methods fetch relevant documents and stuff them into this window. But as more documents are retrieved, they can:
- Exceed the context limit
- Include redundancy or irrelevant filler text

**Context compression** helps by **summarizing** or **distilling** the content into a concise version while preserving the essential information.

---

## ⚙️ How It Works (Step-by-Step)

1. **User Query**  
   The user provides a natural language query.

2. **Initial Retrieval**  
   A vector store (e.g., FAISS, Chroma, Pinecone) retrieves `k` relevant documents using similarity search.

3. **Compression Phase**  
   An LLM is used to **compress** or **summarize** the retrieved documents. Common methods:
   - Summarize each document individually
   - Summarize them as a batch
   - Extract only query-relevant parts (selective abstraction)

4. **Final Context Construction**  
   The compressed summary is passed as input to the final LLM for answer generation.

---

## 📌 Benefits

- ✅ **Fits more knowledge** into the context window
- ✅ **Reduces token usage**
- ✅ **Eliminates redundancy**
- ✅ **Improves response quality** by focusing on essential content

---

## 🧪 Example Use Case

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

# Underlying retriever
base_retriever = vectorstore.as_retriever()

# Compressor: extracts relevant parts of documents using an LLM
compressor = LLMChainExtractor.from_llm(ChatOpenAI())

# Wrap with context compression
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# Use for retrieval
docs = compression_retriever.get_relevant_documents("Explain quantum entanglement")
```

---

## 🧠 Who Proposed It?

**Context Compression Retrieval** was proposed and popularized by the **LangChain team** as part of their modular retrieval strategies. It is part of the `langchain.retrievers` module and represents a key technique for managing long document retrieval and summarization efficiently.

It builds on foundational ideas of:
- **Summarization-based retrieval**
- **Re-ranking and filtering**
- **Context-aware document compression**

---

## 🔗 Related Concepts

- **Multi-Query Retrieval**: Increases coverage using query variations
- **Reranking**: Prioritizes more relevant documents post-retrieval
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to aid retrieval
- **Self-query Retriever**: Converts questions into structured filters before retrieving

---

# 🧩 LLMChainExtractor vs LLMChainFilter

## 📖 Overview

Both `LLMChainExtractor` and `LLMChainFilter` are **document compressors** provided by LangChain to support **context compression retrieval**. They use an LLM to **either extract** the most relevant parts from documents or **filter out** irrelevant documents before they are passed to a final model.

They are typically used inside `ContextualCompressionRetriever`.

---

## 🔍 LLMChainExtractor

### 📖 What It Does

`LLMChainExtractor` uses an LLM to **extract only the most relevant parts** of a document with respect to the user query. Instead of passing the entire document to the final LLM, it compresses each document into a shorter, query-focused form.

### ⚙️ How It Works

1. The user submits a query.
2. A retriever pulls top-k documents from a vector store.
3. Each document is passed to the `LLMChainExtractor` with the query.
4. The extractor summarizes or selects relevant portions.
5. The compressed results are returned as context.

### 🧪 Example

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI

compressor = LLMChainExtractor.from_llm(ChatOpenAI())
```

### ✅ Use When

- You want to **retain most documents** but **shrink** them to fit the context window.
- Documents are **long but partially relevant**.

---

## 🔍 LLMChainFilter

### 📖 What It Does

`LLMChainFilter` uses an LLM to **decide whether to include or exclude an entire document** based on its relevance to the query. Instead of extracting snippets, it keeps or discards whole documents.

### ⚙️ How It Works

1. The user submits a query.
2. A retriever pulls top-k documents.
3. Each document is paired with the query and evaluated by the filter LLM.
4. Only documents deemed relevant are passed to the final stage.

### 🧪 Example

```python
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.chat_models import ChatOpenAI

compressor = LLMChainFilter.from_llm(ChatOpenAI())
```

### ✅ Use When

- You want to **reduce the number of documents**, not just their length.
- Some retrieved documents may be **entirely irrelevant**.
- You prefer precision over recall.

---

## 🔁 Comparison Table

| Feature             | `LLMChainExtractor`                            | `LLMChainFilter`                         |
|---------------------|------------------------------------------------|------------------------------------------|
| Output              | Compressed version of each document            | Subset of full documents (included/excluded) |
| Granularity         | Works **within** a document                    | Works **on** the document level          |
| Token Saving        | ✅ High (by compressing)                       | ✅ High (by filtering)                   |
| Best Use Case       | Long docs with partial relevance               | Many docs, only a few are fully relevant |
| Example Tool        | Summarizer / Content Extractor                 | Relevance Classifier                     |

---

## 🧠 Where to Use Them

Wrap either inside a `ContextualCompressionRetriever` like this:

```python
from langchain.retrievers import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_compressor=LLMChainExtractor.from_llm(ChatOpenAI()),
    base_retriever=vectorstore.as_retriever()
)
```

---

## 🔗 Related Concepts

- **ContextualCompressionRetriever**: Wrapper to apply these compressors
- **MultiQueryRetriever**: Expands query coverage
- **HyDE**: Generates hypothetical docs to help retrieval
- **LLM-based Rerankers**: Prioritize documents by scoring
