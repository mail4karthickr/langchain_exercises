# 🔍 Multi-Query Retrieval

## 📖 Definition

**Multi-query retrieval** is a retrieval strategy where **multiple variations of a single query are generated and executed independently**, and the results are **merged and aggregated** to improve the **diversity and relevance** of the retrieved documents.

---

## ⚙️ How It Works (Step-by-Step)

### 1. Input Query

Start with a user query.  
_Example_:

> "How does photosynthesis work?"

---

### 2. Query Expansion / Reformulation

Use an LLM to generate semantically similar variations:

```text
Original: "How does photosynthesis work?"

Generated:
- "Explain the process of photosynthesis in plants."
- "What are the stages involved in photosynthesis?"
- "How do plants convert sunlight into energy?"
```

This step is done by an **LLM** using techniques such as:

- Prompt-based paraphrasing  
- Semantic variation generation

---

### 3. Parallel Retrieval

Each generated query is independently sent to a retriever (e.g., vector database like FAISS, Chroma, Pinecone).  
For each query, the top `k` relevant documents are retrieved.

---

### 4. Merging / Aggregation

This step **is not typically done by the LLM**, but instead handled by the framework or your orchestration logic.

| Task                      | Done By             | Description                                           |
|---------------------------|---------------------|-------------------------------------------------------|
| Merge retrieved documents | ✅ Framework/Code    | Combine results from all queries                     |
| Deduplicate documents     | ✅ Framework/Code    | Remove duplicates or near-duplicates                 |
| Rerank (optional)         | ✅ Framework / LLM   | Rank results based on relevance or similarity        |
| Final top-k selection     | ✅ Framework/Code    | Select top `k` most relevant documents from the pool |

---

## 🧠 When is the LLM Involved?

- ✅ **Used for generating multiple queries** (Step 2)  
- ❌ **Not used for merging or aggregation** (Step 4)  
- 🧠 **May be used for reranking** if explicitly implemented (e.g., using LLM to score documents)

---

## ✅ Why Use Multi-Query Retrieval?

- **Improves recall**: Captures more aspects of the user's intent  
- **Reduces retrieval bias**: Different phrasings can hit different documents  
- **Boosts relevance**: Enhances coverage for downstream LLM generation

---

## 🧪 Example in LangChain

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI()
)
```

This:
- Uses the LLM to generate alternate queries  
- Performs retrieval for each query  
- Merges the results for final consumption
