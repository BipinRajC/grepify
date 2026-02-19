# **Advanced Architectures for Grounded Intelligence: A Deep Research Report on the Optimization of Retrieval-Augmented Generation for Social Data Synthesis** 
The paradigm of grounded intelligence has witnessed a significant shift from simple retrieval mechanisms toward sophisticated, multi-stage architectures that prioritize semantic precision, structural awareness, and agentic reasoning. As the volume of unstructured social data, particularly from platforms like Reddit, continues to proliferate, the challenge of extracting actionable insights while maintaining factual integrity has intensified. Traditional retrieval-augmented generation (RAG) models often encounter a "nearest neighbor" fallacy, where simple semantic proximity is mistaken for relevance or accuracy. This report provides a technical analysis of advanced RAG systems, exploring the innovations in chunking strategies, hybrid retrieval, and the specific "source grounding" philosophy exemplified by Google’s NotebookLM. The objective is to delineate a blueprint for highly optimized systems capable of transforming heterogeneous social discourse into structured, verifiable reports. 
## **The Architecture of Source Grounding: Deconstructing Google’s NotebookLM** 
Google’s NotebookLM represents a departure from general-purpose large language model interfaces, prioritizing a philosophy known as source grounding. This architectural decision, initiated in early 2022 and publicly realized through Project Tailwind, serves to constrain a model's probabilistic generation within a closed-loop system defined by user-provided documents.1 Unlike standard RAG implementations that might prioritize a broad web search, NotebookLM focuses on the "overloaded reader" problem, addressing the synthesis of specific corpora such as research papers, lecture notes, or corporate documentation.2 

The core technical engine of NotebookLM is the Gemini 1.5 Pro model, which utilizes a sparse mixture-of-experts (MoE) Transformer architecture.1 The MoE design allows the system to 

scale its total parameter count while maintaining a constant number of activated parameters during inference, ensuring computational efficiency even when dealing with the massive context windows required for source-heavy analysis.3 Gemini 1.5 Pro facilitates a near-perfect "needle-in-a-haystack" recall of over 99% across context lengths reaching up to 10 million tokens in text modality.3 This breakthrough allows the model to treat the context window as a form of "short-term memory," loading entire research sets or long-form documentation directly 

into the active reasoning space.1 

The system facilitates a rigorous source attribution model where every factual assertion in the text interface includes inline citations linked directly to the original source passages.1 This 

serves the dual purpose of enabling user verification and providing the foundational grounding for multimodal outputs, such as the Audio Overview feature. This feature employs advanced voice synthesis models developed by DeepMind to generate podcast-style conversations, incorporating intentional disfluencies—such as interjections and filler words—to enhance the perceived humanity and listenability of the AI hosts.1 
### **Comparison of Gemini Model Capabilities for Grounding** 


|**Feature** |**Gemini 1.0 Pro** |**Gemini 1.0 Ultra** |**Gemini 1.5 Pro** |**Gemini 1.5 Flash** |
| - | :- | :- | - | :- |
|**Context Window (Tokens)** |32,000 |32,000 |Up to 10,000,000 |1,000,000+ |
|**Architecture** |Transformer |Transformer |Sparse MoE |Sparse MoE |
|**Retrieval Accuracy (1M Tokens)** |N/A |N/A |>99% |>99% |
|**Multimodal Support** |Limited |Yes |Full (Text, Audio, Video) |Full (Speed-optimi zed) |
|**Standard Inference Cost** |Low |High |Moderate |Very Low |

3 

The agentic pivot of NotebookLM, particularly its "Deep Research" function, utilizes an autonomous workflow that decomposes complex user queries into sub-questions. It executes parallel searches across both the web and the private corpus, identifies information gaps, and iteratively generates new queries to fill those gaps before producing a structured briefing document.2 This transition from a "retrieval tool" to an "agentic researcher" represents the 

future trajectory of RAG-based productivity platforms.2 
## **Semantic Unit Segmentation: Advanced Chunking for Threaded Discussions** 
The efficacy of a RAG system is fundamentally rooted in its chunking strategy. While fixed-size chunking (e.g., splitting every 512 tokens) is computationally efficient, it frequently severs the logical flow of information, particularly in the context of Reddit's threaded discussions.8 To achieve uniform and accurate retrieval, a system must move toward structure-aware and semantic-based segmentation. 
### **Recursive Semantic Chunking (RSC)** 
Recursive Semantic Chunking (RSC) addresses the limitations of character-based splitting by maintaining semantic coherence across segments.8 The RSC framework applies an adaptive 

approach where documents are initially segmented at linguistic boundaries. For chunks that exceed a defined threshold—typically 1,500 characters—the semantic chunker is recursively applied, gradually reducing the breakpoint threshold until the segments are both manageable and semantically intact.8 

One of the more sophisticated aspects of RSC is the strategic merging of short chunks. Segments that fall below a minimum threshold, such as 350 characters, are evaluated against their neighbors. The system merges the short chunk with the adjacent segment that shares the highest cosine similarity, thereby preventing information loss and ensuring the retriever does 

not encounter "starved" context.8 This is particularly relevant for social media data, where short, fragmented comments often require the context of a parent post or a preceding reply to be meaningful. 
### **Hierarchical Indexing and Reddit Nesting** 
When processing Reddit data, the system must navigate deep comment hierarchies. Evidence suggests that "metadata enrichment at indexing time" is a superior strategy for preserving structural integrity.10 By prepending the full folder or thread path as metadata to each chunk—for instance, including the subreddit, post title, and parent comment identifiers—the embeddings capture the relational context natively.10 
#### **Two-Layer Retrieval Patterns** 
A hierarchical indexing scheme typically employs two levels of chunks: 

1. **Parent Chunks:** These represent higher-level summaries of entire threads or 

   sub-sections, serving as a navigation map.10 

2. **Child Chunks:** These are granular pieces of actual content (individual comments) that act 

   as retrieval anchors for precise similarity matching.10 

Once a relevant child chunk is identified through vector search, the system retrieves its larger parent text to provide the LLM with the full context of the discussion. This mitigates the "lost in the middle" effect and semantic drift that often plague flat RAG systems.12 
### **Chunking Performance and Accuracy Metrics** 


|**Strategy** |**Contextual Relevancy** |**Contextual Precision** |**Retrieval Time (Avg)** |**Computationa l Overhead** |
| - | :- | :- | :- | :- |
|**Fixed-Size (512 tokens)** |Low |Moderate |Fast |Low |
|**Recursive Character** |Moderate |Moderate |Moderate |Moderate |
|**Semantic Chunking** |High |High |Slow |High |
|**RSC (Adaptive)** |Very High |Very High |Moderate |Moderate |

8 
## **Retrieval Beyond the Nearest Neighbor: Hybrid and Diverse Search** 
A significant challenge in optimized RAG systems is moving beyond the simple "nearest source" retrieval, which often results in redundant or overly localized information. To achieve uniform and accurate information gathering, advanced systems employ hybrid search and diversity-focused reranking. 
### **Hybrid Retrieval (Dense and Sparse Fusion)** 
Dense vector search is powerful for capturing semantic intent but often fails with exact keyword matches, such as product IDs, technical acronyms, or proper nouns—elements that are pervasive in Reddit's technical subreddits.14 Hybrid retrieval combines: 

- **Lexical Search (BM25):** Utilizes keyword matching and term frequency-inverse document frequency (TF-IDF) principles to ensure precision for specific tokens.14 
- **Semantic Search (Dense Embeddings):** Captures the conceptual meaning and intent of the query.14 

The results from both streams are fused using techniques like Reciprocal Rank Fusion (RRF), which calculates a unified score based on the reciprocal rank of a document in both search lists.14 This fusion ensures the system catches both exact terminologies and broad semantic relationships, which is essential for accurate advice and reporting. 
### **Maximal Marginal Relevance (MMR) for Diversity** 
To ensure retrieval is "uniform" and not just concentrated on the single most similar source, Maximal Marginal Relevance (MMR) is employed.16 MMR iteratively builds a retrieval list by 

maximizing a weighted combination of query relevance and document novelty relative to already selected items.17 This prevents the LLM from receiving five identical chunks that 

provide the same viewpoint, instead forcing the retrieval of diverse perspectives across a Reddit thread.16 

The MMR scoring formula is defined as: 

![](Aspose.Words.6000fe1b-344a-4c7f-b0e7-a0797d3d923c.001.png)

where  ![](Aspose.Words.6000fe1b-344a-4c7f-b0e7-a0797d3d923c.002.png) is a parameter that balances relevance and diversity.17 
### **Reranking and Context Distillation** 
Initial retrieval passes, while fast, often lack the precision needed for complex report 

generation. Post-retrieval reranking employs cross-encoders—Transformer models that evaluate the relevancy of a query-document pair in a single inference step—to provide a more accurate evaluation than precomputed vector similarity.14 Although computationally more expensive, reranking can improve recall by 6-12% by ensuring that the most qualitatively useful documents are prioritized for the LLM.20 

Furthermore, context distillation or compression steps (such as LLMLingua) can remove low-information tokens from the retrieved context.22 LongLLMLingua, for instance, can reduce 

prompts from thousands of tokens to a few hundred while increasing accuracy by as much as 21.4% by repositioning the most relevant information to the "sides" of the prompt—areas where models have higher recall.22 
## **Agentic RAG: Transforming Retrieval into Decision-Ready Reports** 
The evolution from a passive "search and summarize" system to an active "agentic" architecture 

is critical for generating advice and reports. Agentic RAG systems use LLMs not just for generation, but as planners and observers that manage the research lifecycle.24 
### **Core Agentic Design Patterns** 
1. **Reflection Pattern:** The agent evaluates its own decisions and outputs, identifying errors and missing data points before finalizing a report.24 
1. **Planning Pattern:** The agent decomposes a high-level goal into a structured sequence of tasks, such as specific query generation for different sub-topics.24 
1. **Tool Use Pattern:** The agent interacts with external APIs, databases, or specialized models to gather evidence beyond its initial training set.25 
1. **Multi-Agent Collaboration:** Different specialized agents—for example, a "Reddit Scraper Agent," a "Financial Analyst Agent," and a "Compliance Reviewer"—work in a hierarchical or sequential flow to synthesize multi-modal or multi-domain data.25 
### **Comparison of Agentic Frameworks (2025)** 


|**Framework** |**Orchestration Pattern** |**Best Use Case** |**Key Strength** |
| - | :- | - | - |
|**LangGraph** |Cyclic Graphs |Stateful, iterative research |Controllability & Debugging |
|**CrewAI** |Multi-Agent "Crews" |Role-based task execution |High-level abstraction |
|**Haystack** |Modular Pipelines |Enterprise RAG & Search |Scalability & Stability |
|**LlamaIndex** |Knowledge Agents |Data-heavy retrieval |Advanced memory/context |
|**Semantic Kernel** |Connectors & Planners |Enterprise.NET integration |Professional/IT focus |

29 

The integration of these frameworks allows for the creation of "self-correcting" RAG systems. For example, using LangGraph, a system can grade the relevance of retrieved Reddit documents and automatically rewrite queries if the initial results are insufficient or noisy.34 This iterative "thought-action-observation" (ReAct) cycle ensures that the final report is not just a 

summary of what was found, but a reasoned conclusion based on verified evidence.25 
## **Infrastructure and Performance Optimization** 
Efficiency in a RAG system is a multi-dimensional metric involving latency, throughput, memory footprint, and cost. For a project at the scale of Reddit data, the choice of vector database and indexing algorithm is paramount. 
### **Indexing Algorithms: HNSW vs. DiskANN** 
- **HNSW (Hierarchical Navigable Small Worlds):** The de facto standard for in-memory vector search, HNSW delivers sub-millisecond latencies and high recall for datasets that fit within RAM (typically under 100 million vectors).36 
- **DiskANN:** Optimized for NVMe disk storage, DiskANN is essential for billion-scale datasets where memory costs are prohibitive. It allows for high-throughput searching with minimal memory overhead, though with slightly higher latency than HNSW.18 
### **Vector Database Performance Benchmarks (May 2025)** 


|**Database** |**Throughput (QPS @ 99% Recall)** |**Latency (P99)** |**Indexing Speed (v/s)** |**Scale Optimized For** |
| - | - | - | :- | :- |
|**Qdrant** |41 |15-25ms |8,000-12,000 |10M - 100M |
|**Milvus** |~60 (HNSW) |20-35ms |10,000-15,000 |100M+ |
|**pgvectorscale** |471 |Variable |Moderate |Enterprise/SQL |
|**Weaviate** |~35 |25-40ms |6,000-9,000 |Knowledge Graphs |

36 

The data suggests that for high-throughput applications, purpose-built systems like Milvus or optimized extensions like pgvectorscale offer significant advantages over general-purpose solutions when scaling beyond 50 million vectors.36 
### **Context Caching and Economics** 
Cost optimization is a major concern for "Grepify," especially given the high volume of Reddit data. Google's context caching on Gemini models provides a 75-90% discount on cached 

tokens compared to standard input tokens.39 For repetitive analysis of large document sets—such as the same Reddit thread being queried by multiple users—explicit caching ensures that the system does not re-process common prefixes, substantially reducing API bills and latency.39 
#### **Caching Tiers for Gemini Models (Paid Tier)** 


|**Metric** |**prompts <= 200k tokens** |**prompts > 200k tokens** |
| - | - | - |
|**Gemini 2.5 Pro Caching** |$0.125 / 1M tokens |$0.25 / 1M tokens |
|**Standard Input (Non-cached)** |$1.25 / 1M tokens |$2.50 / 1M tokens |
|**Storage Fee (per hour)** |$4.50 / 1M tokens |$4.50 / 1M tokens |

40 
## **State-of-the-Art Embedding and Reranking Models** 
The choice of embedding model determines the initial retrieval quality. As of 2025, the MTEB (Massive Text Embedding Benchmark) leaderboard reveals a convergence among top-tier models, where the difference between the rank 1 and rank 10 models is often less than 3% in accuracy.42 
### **Leading Open-Weight Models (2025)** 
1. **Qwen3-Embedding-8B:** Demonstrates superior performance on multi-lingual benchmarks and long-text understanding. It is particularly effective for retrieval, classification, and semantic similarity tasks.42 
1. **BAAI/bge-base-en-v1.5:** Fine-tuned with hard negative mining, it is highly effective at distinguishing between correct documents and semantically similar but irrelevant ones.44 
1. **intfloat/e5-base-v2:** Trained on CCPairs (270 million high-quality pairs from Reddit, Wikipedia, and StackExchange), this model is naturally aligned with the linguistic patterns of social discourse.44 
1. **Stella\_en\_1.5B\_v5:** A compact model that offers high English-only retrieval performance, suitable for deployment on limited GPU resources or CPU environments.42 
### **The Role of Rerankers** 
Because embedding models compress information into a single vector, some nuances are lost. Rerankers like **bge-reranker-large** or **ms-marco-TinyBERT** should be used on the top-20 

retrieved candidates to ensure the final context provided to the LLM is of the highest quality.22 This multi-stage approach—Recall via embeddings followed by Precision via rerankers—is the industry best practice for production RAG systems.18 
## **Blueprint for Grepify: Synthesis of Research Findings** 
Building an efficient and optimized system for Reddit data requires the integration of multiple advanced strategies to overcome the inherent noise and structural complexity of the data source. 
### **Data Engineering and Ingestion** 
The system should not treat Reddit comments as flat text documents. Ingestion must include recursive transformation of threads into nested structures, with authority-based weighting (e.g., upvote count) used as metadata to prioritize high-value content.46 Applying Recursive Semantic Chunking will preserve the logical integrity of discussions, ensuring that arguments are not split across retrieval units.8 
### **Retrieval and Search Optimization** 
To move beyond "nearest source" retrieval, Grepify should implement a hybrid search pipeline using pgvector or Milvus, combining dense embeddings (E5 or BGE) with BM25 lexical search.14

Maximal Marginal Relevance (MMR) must be applied to the retrieved candidates to ensure a diversity of perspectives from different users in a thread, followed by a reranking step using a 

cross-encoder to select the final  ![](Aspose.Words.6000fe1b-344a-4c7f-b0e7-a0797d3d923c.003.png) chunks for the generation prompt.14 
### **Agentic Synthesis and Reporting** 
The final "Advice and Reporting" layer should be built on an agentic framework like LangGraph. This allows the system to plan a research path (e.g., "Search for consensus," "Find counter-arguments," "Identify factual outliers"), reflect on the retrieved evidence, and generate reports with precise source grounding and inline citations, similar to the NotebookLM model.1 
### **Infrastructure and Scaling** 
For production-grade scalability, the system should leverage context caching for popular threads to reduce inference costs and latency.39 Horizontal scaling of the vector database using 

HNSW indices will ensure sub-100ms response times even as the indexed Reddit corpus grows to millions of posts.36 

By adhering to this multi-stage architectural funnel—from semantic-aware chunking and hybrid retrieval to agentic reflection and source-grounded generation—Grepify can achieve the precision and accuracy required to serve as a high-performance research assistant for the 
#### complex world of social data. **Works cited** 
1. Google / NotebookLLM: Source-Grounded LLM Assistant with Multi-Modal Output Capabilities - ZenML LLMOps Database, accessed on February 20, 2026, [https://www.zenml.io/llmops-database/source-grounded-llm-assistant-with-mult i-modal-output-capabilities](https://www.zenml.io/llmops-database/source-grounded-llm-assistant-with-multi-modal-output-capabilities) 
1. The Cognitive Engine: A Comprehensive Analysis of NotebookLM's Evolution (2023–2026), accessed on February 20, 2026, [https://medium.com/@jimmisound/the-cognitive-engine-a-comprehensive-analy sis-of-notebooklms-evolution-2023-2026-90b7a7c2df36](https://medium.com/@jimmisound/the-cognitive-engine-a-comprehensive-analysis-of-notebooklms-evolution-2023-2026-90b7a7c2df36) 
1. Gemini 1.5 Pro - Prompt Engineering Guide, accessed on February 20, 2026, <https://www.promptingguide.ai/models/gemini-pro> 
1. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context - Googleapis.com, accessed on February 20, 2026, <https://storage.googleapis.com/deepmind-media/gemini/gemini_v1_5_report.pdf> 
1. Long context | Gemini API - Google AI for Developers, accessed on February 20, 2026, <https://ai.google.dev/gemini-api/docs/long-context> 
1. NotebookLM: Document-Grounded AI by Google - Emergent Mind, accessed on February 20, 2026, <https://www.emergentmind.com/topics/notebooklm> 
1. Introducing Gemini 1.5, Google's next-generation AI model - The Keyword, accessed on February 20, 2026, [https://blog.google/innovation-and-ai/products/google-gemini-next-generation- model-february-2024/](https://blog.google/innovation-and-ai/products/google-gemini-next-generation-model-february-2024/) 
1. The Chunking Paradigm: Recursive Semantic for ... - ACL Anthology, accessed on February 20, 2026, <https://aclanthology.org/2025.icnlsp-1.15.pdf> 
1. Semantic chunking + metadata filtering actually fixes RAG hallucinations - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/Rag/comments/1r3mmxy/semantic_chunking_metadata _filtering_actually/](https://www.reddit.com/r/Rag/comments/1r3mmxy/semantic_chunking_metadata_filtering_actually/) 
1. How to give rag understanding of folder structure? - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/Rag/comments/1r3z1qn/how_to_give_rag_understandin g_of_folder_structure/](https://www.reddit.com/r/Rag/comments/1r3z1qn/how_to_give_rag_understanding_of_folder_structure/) 
1. Adding and utilising metadata to improve RAG? : r/LocalLLaMA - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1euvuiy/adding_and_utilising_m etadata_to_improve_rag/](https://www.reddit.com/r/LocalLLaMA/comments/1euvuiy/adding_and_utilising_metadata_to_improve_rag/) 
1. From RAG to Context - A 2025 year-end review of RAG - RAGFlow, accessed on February 20, 2026, <https://ragflow.io/blog/rag-review-2025-from-rag-to-context> 
1. Hierarchical Agentic RAG: What are your thoughts? : r/LocalLLaMA - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1odystc/hierarchical_agentic_ra g_what_are_your_thoughts/](https://www.reddit.com/r/LocalLLaMA/comments/1odystc/hierarchical_agentic_rag_what_are_your_thoughts/) 
14. Advanced RAG Techniques for High-Performance LLM Applications - Graph Database & Analytics - Neo4j, accessed on February 20, 2026, <https://neo4j.com/blog/genai/advanced-rag-techniques/> 
14. Multi-Vector Retrieval Techniques That Improve RAG Recall - Indium Software, accessed on February 20, 2026, <https://www.indium.tech/blog/multi-vector-retrieval-rag-recall-optimization/> 
14. Advanced Retrieval Strategies for RAG, accessed on February 20, 2026, <https://app.ailog.fr/en/blog/guides/retrieval-strategies> 
14. Next-Level Retrieval in RAG: Retrieval Strategies (Part 2) | by Mahima Arora - Medium, accessed on February 20, 2026, [https://medium.com/@mahimaarora025/next-level-retrieval-in-rag-retrieval-strat egies-part-2-9297604d15fb](https://medium.com/@mahimaarora025/next-level-retrieval-in-rag-retrieval-strategies-part-2-9297604d15fb) 
14. Graph-Augmented Hybrid Retrieval and Multi-Stage Re-ranking: A Framework for High-Fidelity Chunk Retrieval in RAG Systems - DEV Community, accessed on February 20, 2026, [https://dev.to/lucash_ribeiro_dev/graph-augmented-hybrid-retrieval-and-multi-st age-re-ranking-a-framework-for-high-fidelity-chunk-50ca](https://dev.to/lucash_ribeiro_dev/graph-augmented-hybrid-retrieval-and-multi-stage-re-ranking-a-framework-for-high-fidelity-chunk-50ca) 
14. Re-Ranking Algorithms in Vector Databases: An In-Depth Analysis | by Bishal Bose, accessed on February 20, 2026, [https://bishalbose294.medium.com/re-ranking-algorithms-in-vector-databases-i n-depth-analysis-b3560b1ebd6f](https://bishalbose294.medium.com/re-ranking-algorithms-in-vector-databases-in-depth-analysis-b3560b1ebd6f) 
14. Improving RAG Performance: WTF are Re-Ranking Techniques? - Fuzzy Labs, accessed on February 20, 2026, <https://www.fuzzylabs.ai/blog-post/improving-rag-performance-re-ranking> 
14. RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs, accessed on February 20, 2026, [https://openreview.net/forum?id=S1fc92uemC&referrer=%5Bthe%20profile%20of %20Jiaxuan%20You%5D(%2Fprofile%3Fid%3D~Jiaxuan_You2)](https://openreview.net/forum?id=S1fc92uemC&referrer=%5Bthe+profile+of+Jiaxuan+You%5D\(/profile?id%3D~Jiaxuan_You2\)) 
14. LongLLMLingua Prompt Compression Guide | LlamaIndex, accessed on February 20, 2026, [https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save- on-your-rag-costs-via-prompt-compression-54b559b9ddf7](https://www.llamaindex.ai/blog/longllmlingua-bye-bye-to-middle-loss-and-save-on-your-rag-costs-via-prompt-compression-54b559b9ddf7) 
14. LLMLingua Series | Effectively Deliver Information to LLMs via ..., accessed on February 20, 2026, <https://llmlingua.com/> 
14. Agent Factory: The new era of agentic AI—common use cases and ..., accessed on February 20, 2026, [https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai -common-use-cases-and-design-patterns/](https://azure.microsoft.com/en-us/blog/agent-factory-the-new-era-of-agentic-ai-common-use-cases-and-design-patterns/) 
14. 7 Must-Know Agentic AI Design Patterns - MachineLearningMastery.com, accessed on February 20, 2026, <https://machinelearningmastery.com/7-must-know-agentic-ai-design-patterns/> 
14. Developing “Agentic Workflows” Using Top Frameworks | by Megha Verma | Predict | Dec, 2025, accessed on February 20, 2026, [https://medium.com/predict/developing-agentic-workflows-using-top-framewor ks-6fa14b5dd57e](https://medium.com/predict/developing-agentic-workflows-using-top-frameworks-6fa14b5dd57e) 
27. asinghcsu/AgenticRAG-Survey: Agentic-RAG explores advanced Retrieval-Augmented Generation systems enhanced with AI LLM agents. - GitHub, accessed on February 20, 2026, <https://github.com/asinghcsu/AgenticRAG-Survey> 
27. What is LangGraph? - IBM, accessed on February 20, 2026, <https://www.ibm.com/think/topics/langgraph> 
27. Agentic AI Frameworks: Top 8 Options in 2026 - NetApp Instaclustr, accessed on February 20, 2026, [https://www.instaclustr.com/education/agentic-ai/agentic-ai-frameworks-top-8- options-in-2026/](https://www.instaclustr.com/education/agentic-ai/agentic-ai-frameworks-top-8-options-in-2026/) 
27. Best Agentic AI Frameworks Compared: Top Picks & Feature Breakdown - Riseup Labs, accessed on February 20, 2026, <https://riseuplabs.com/best-agentic-ai-frameworks-compared/> 
27. Best Agentic AI Frameworks in 2025: Comparing LangChain, LangGraph, CrewAI, and Haystack - Blogs - Trixly AI Solutions, accessed on February 20, 2026, [https://www.trixlyai.com/blog/technical-14/best-agentic-ai-frameworks-in-2025- comparing-langchain-langgraph-crewai-and-haystack-74](https://www.trixlyai.com/blog/technical-14/best-agentic-ai-frameworks-in-2025-comparing-langchain-langgraph-crewai-and-haystack-74) 
27. A Detailed Comparison of Top 6 AI Agent Frameworks in 2026 - Turing, accessed on February 20, 2026, <https://www.turing.com/resources/ai-agent-frameworks> 
27. Top Open Source Agentic Frameworks : CrewAI vs AutoGen vs LangGraph vs Lyzr, accessed on February 20, 2026, <https://www.lyzr.ai/blog/top-open-source-agentic-frameworks/> 
27. Build an Agentic RAG System with LangGraph, accessed on February 20, 2026, <https://www.youtube.com/watch?v=moaWbFTVOEo> 
27. Build a Multi-Agent System with LangGraph and Mistral on AWS | Artificial Intelligence, accessed on February 20, 2026, [https://aws.amazon.com/blogs/machine-learning/build-a-multi-agent-system-wit h-langgraph-and-mistral-on-aws/](https://aws.amazon.com/blogs/machine-learning/build-a-multi-agent-system-with-langgraph-and-mistral-on-aws/) 
27. Vector Database Showdown 2025: Qdrant vs Milvus vs Weaviate Performance Benchmarks for RAG on VPS - Onidel, accessed on February 20, 2026, <https://onidel.com/blog/vector-database-benchmarks-vps> 
27. HNSW vs DiskANN: comparing the leading ANN algorithms - Vectroid Resources, accessed on February 20, 2026, [https://www.vectroid.com/resources/HNSW-vs-DiskANN-comparing-the-leading -ANN-algorithm](https://www.vectroid.com/resources/HNSW-vs-DiskANN-comparing-the-leading-ANN-algorithm) 
27. Best Vector Databases in 2025: A Complete Comparison Guide - Firecrawl, accessed on February 20, 2026, <https://www.firecrawl.dev/blog/best-vector-databases-2025> 
27. Context caching overview | Generative AI on Vertex AI - Google Cloud Documentation, accessed on February 20, 2026, [https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/contex t-cache-overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview) 
27. Google Gemini API Pricing 2026: Complete Cost Guide per 1M Tokens - MetaCTO, accessed on February 20, 2026, [https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api](https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api-pricing-and-integration)

    [-pricing-and-integration](https://www.metacto.com/blogs/the-true-cost-of-google-gemini-a-guide-to-api-pricing-and-integration) 

41. Context caching | Gemini API | Google AI for Developers, accessed on February 20, 2026, <https://ai.google.dev/gemini-api/docs/caching> 
41. Top embedding models on the MTEB leaderboard - Modal, accessed on February 20, 2026, <https://modal.com/blog/mteb-leaderboard-article> 
41. Embedding models have converged : r/LocalLLaMA - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1ozf9al/embedding_models_ha ve_converged/](https://www.reddit.com/r/LocalLLaMA/comments/1ozf9al/embedding_models_have_converged/) 
41. Best Open-Source Embedding Models Benchmarked and Ranked - Supermemory, accessed on February 20, 2026, [https://supermemory.ai/blog/best-open-source-embedding-models-benchmark ed-and-ranked/](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/) 
41. RAG (Retrieval-augmented generation) - Reddit, accessed on February 20, 2026, <https://www.reddit.com/r/Rag/rising/> 
41. The Anatomy of Reddit-Style Comments — A Weekend Engineering ..., accessed on February 20, 2026, [https://vikkrraant.medium.com/the-anatomy-of-reddit-style-comments-a-weeke nd-engineering-dive-6b52cd80139d](https://vikkrraant.medium.com/the-anatomy-of-reddit-style-comments-a-weekend-engineering-dive-6b52cd80139d) 
41. How to improve RAG with metadata - Reddit, accessed on February 20, 2026, [https://www.reddit.com/r/Rag/comments/1mevy9i/how_to_improve_rag_with_met adata/](https://www.reddit.com/r/Rag/comments/1mevy9i/how_to_improve_rag_with_metadata/) 
41. RAG Retrieval Beyond Semantic Search: Day 5- Metadata Filtering | by Vansh Kharidia, accessed on February 20, 2026, [https://medium.com/@vanshkharidia7/rag-retrieval-beyond-semantic-search-day -5-metadata-filtering-4cf22eb6d016](https://medium.com/@vanshkharidia7/rag-retrieval-beyond-semantic-search-day-5-metadata-filtering-4cf22eb6d016) 
