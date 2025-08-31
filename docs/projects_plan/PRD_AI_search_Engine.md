# Project Requirements Document: AI-Based Search Engine for Supreme Court

## 1. Project Overview

### 1.1 Purpose
The AI-Based Search Engine for Supreme Court Decisions is a core feature of the CourtSight platform designed to democratize access to legal information in Southeast Asia. The system transforms dense, unstructured legal documents into searchable, actionable insights, making justice more transparent and inclusive.

### 1.2 Scope
This document specifically addresses the requirements for the AI-based search engine component of CourtSight, which enables semantic searching of public Supreme Court decisions across Southeast Asian jurisdictions.

### 1.3 Target Users
- Law Firms
- Independent Lawyers
- Law Students
- Investigative Journalists
- Researchers & Academics

## 2. System Architecture

### 2.1 High-Level Architecture
The search engine will implement a Retrieval-Augmented Generation (RAG) architecture with these core components:

- **Frontend:** Next.js web application providing search interface
- **Backend API:** FastAPI server handling search requests
- **Embedding Service:** Transforms text into vector representations
- **Vector Database:** Stores and retrieves document chunks based on semantic similarity
- **Document Store:** Maintains full context documents
- **LLM Service:** Processes search results and generates insights
- **Validation Module:** Ensures factual accuracy of generated responses

### 2.2 Parent-Child Chunking Strategy
To balance precision and contextual relevance, the system will implement a parent-child document chunking strategy:

- **Child Chunks:** Small segments (400 characters) optimized for precise semantic search
- **Parent Documents:** Larger segments (2000 characters) providing full context for LLM processing
- **Retrieval Flow:** Search identifies matching child chunks, then retrieves associated parent documents

## 3. Functional Requirements

### 3.1 Document Ingestion & Processing
- **F1.1.1:** System must ingest PDF documents from the Supreme Court repository
- **F1.1.2:** System must convert, clean, and standardize document text
- **F1.1.3:** System must implement parent-child chunking strategy for document storage
- **F1.1.4:** System must generate vector embeddings for child chunks

### 3.2 Search Functionality
- **F1.2.1:** System must accept natural language search queries from users
- **F1.2.2:** System must convert search queries to vector embeddings
- **F1.2.3:** System must perform semantic similarity search to find relevant document chunks
- **F1.2.4:** System must retrieve full context from parent documents based on matched child chunks
- **F1.2.5:** System must support filtering by jurisdiction, date range, and case type

### 3.3 Response Generation
- **F1.3.1:** System must format retrieved context and query for LLM processing
- **F1.3.2:** System must generate concise summaries of relevant case information
- **F1.3.3:** System must extract key legal points from documents
- **F1.3.4:** System must include proper citations to source documents

### 3.4 Response Validation
- **F1.4.1:** System must extract factual claims from generated responses
- **F1.4.2:** System must verify each claim against source documents
- **F1.4.3:** System must categorize claims as "Supported", "Partially Supported", "Unsupported", or "Uncertain"
- **F1.4.4:** System must filter out unsupported claims from final responses
- **F1.4.5:** System must include source citations for verification


## 4. Non-Functional Requirements

### 4.1 Performance
- **NF1.1:** Initial search results must be displayed within 3 seconds for 95% of queries
- **NF1.2:** Comprehensive results with summaries must be returned within 10 seconds
- **NF1.3:** System must handle at least 100 concurrent search queries

### 4.2 Security & Privacy
- **NF2.1:** All data must be encrypted in transit (TLS 1.3) and at rest (AES-256)
- **NF2.2:** System must implement robust authentication and role-based access control
- **NF2.3:** System must maintain clear separation between public court data and private user data

### 4.3 Scalability
- **NF3.1:** Vector database must scale to handle tens of millions of document chunks
- **NF3.2:** System must support horizontal scaling to accommodate user growth
- **NF3.3:** Document ingestion pipeline must efficiently process bulk document additions

### 4.4 Reliability
- **NF4.1:** Search service must maintain 99.5% uptime
- **NF4.2:** System must implement regular database backups
- **NF4.3:** System must provide fallback mechanisms when LLM services are unavailable

### 4.5 Usability
- **NF5.1:** Interface must be accessible across desktop and mobile devices
- **NF5.2:** Search interface must minimize learning curve for non-technical users
- **NF5.3:** System must provide clear error messages and feedback

## 5. Technical Specifications

### 5.1 Technology Stack
- **Backend API:** FastAPI
- **Vector Database:** PostgreSQL with pgvector extension
- **Document Store:** PostgreSQL or Redis
- **Embedding Service:** text-embedding-3-small (OpenAI) or equivalent multilingual model
- **LLM Service:** SeaLionLLM base on vertexAI or equivalent model with guardrails
- **Guardrails Service:** [Guardrails AI](https://www.guardrailsai.com/docs) Python library for input/output validation and safety
- **Infrastructure:** Google Cloud Platform

### 5.2 API Endpoints
- **5.2.1 Search Endpoint**
```json
POST /api/search
Request:
{
  "query": string,       // Natural language search query
  "filters": {           // Optional filters
    "jurisdiction": string,
    "date_range": {
      "start": string,
      "end": string
    },
    "case_type": string
  }
}

Response:
{
  "results": [
    {
      "summary": string,
      "key_points": string[],
      "source_documents": [
        {
          "title": string,
          "case_number": string,
          "date": string,
          "url": string,
          "excerpt": string
        }
      ],
      "validation_status": string
    }
  ]
}
```
## 6. Implementation Requirements

### 6.1 RAG Pipeline Implementation
- **IR1.1:** Implement document loading and preprocessing pipeline
- **IR1.2:** Configure parent-child chunking with RecursiveCharacterTextSplitter
- **IR1.3:** Implement vector database with pgvector for efficient similarity search
- **IR1.4:** Build LangChain Expression Language (LCEL) pipeline for RAG workflow
- **IR1.5:** Implement guardrails validation flow for claim verification using Guardrails AI library as a service layer

### 6.2 Development Environment
- **IR2.1:** Python 3.9+ environment with virtual environment management
- **IR2.2:** Dependencies managed via requirements.txt and version control
- **IR2.3:** Environment variables for API keys and configuration parameters
- **IR2.4:** PostgreSQL database with pgvector extension

### 6.3 Core Dependencies

## 7. Evaluation Protocol

### 7.1 Component-Level Evaluation
Using RAGAS framework to evaluate RAG workflow effectiveness:

- **EV1.1:** Faithfulness - measure factual grounding of answers in source content
- **EV1.2:** Answer Relevancy - assess relevance of answers to user queries
- **EV1.3:** Context Precision - evaluate signal-to-noise ratio in retrieved contexts
- **EV1.4:** Context Recall - verify retriever captures all necessary information

### 7.2 Domain-Specific Evaluation
- **EV2.1:** Create "golden dataset" of legal questions with expert-verified answers
- **EV2.2:** Measure semantic similarity using BERTScore between generated and reference answers
- **EV2.3:** Conduct expert legal review classifying answers using the "Supported/Partially Supported/Unsupported" scheme

## 8. Implementation Phases

### 8.1 Phase 1 (MVP)
- Basic document ingestion pipeline
- Simple semantic search implementation
- Basic result display with document links

### 8.2 Phase 2
- Enhanced result presentation with summaries and key points
- LLM-based answer generation
- Initial implementation of validation guardrails

### 8.3 Phase 3
- Advanced claim validation and fact-checking
- Citation verification against source documents
- Performance optimization for scale

## 9. Risks and Mitigation Strategies

### 9.1 Technical Risks

- **Risk:** LLM hallucinations producing inaccurate legal information  
    **Mitigation:** Implement robust guardrail validation system for factual verification

- **Risk:** Vector database performance degradation at scale  
    **Mitigation:** Implement database indexing, sharding, and caching strategies

- **Risk:** Poor quality OCR in source documents affecting search accuracy  
    **Mitigation:** Implement document quality preprocessing and filtering

### 9.2 Legal and Ethical Risks

- **Risk:** Privacy concerns related to case data  
    **Mitigation:** Ensure only public data is used and implement strong privacy controls

- **Risk:** Misinterpretation of legal concepts by the AI  
    **Mitigation:** Clear disclaimer about system limitations and human verification requirement

## 10. References

This requirements document is based on the sequence diagram `AIBasedSearchEngineforSupremeCourt.jpg` and the detailed analysis provided in the project documentation.