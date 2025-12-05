# Intergrax — Project Structure Overview

This document was generated automatically by the Intergrax Project Structure Document Generator.

## Purpose
- Provide a clean overview of the current codebase structure.
- Enable new developers to understand architectural roles quickly.
- Serve as context for LLM agents (e.g., ChatGPT, Intergrax agents) to reason about and improve the project.

## File Index

- `api\__init__.py`
- `api\chat\__init__.py`
- `api\chat\main.py`
- `api\chat\tools\__init__.py`
- `api\chat\tools\chroma_utils.py`
- `api\chat\tools\db_utils.py`
- `api\chat\tools\pydantic_models.py`
- `api\chat\tools\rag_pipeline.py`
- `applications\chat_streamlit\api_utils.py`
- `applications\chat_streamlit\chat_interface.py`
- `applications\chat_streamlit\sidebar.py`
- `applications\chat_streamlit\streamlit_app.py`
- `applications\company_profile\__init__.py`
- `applications\figma_integration\__init__.py`
- `applications\ux_audit_agent\__init__.py`
- `applications\ux_audit_agent\components\__init__.py`
- `applications\ux_audit_agent\components\compliance_checker.py`
- `applications\ux_audit_agent\components\cost_estimator.py`
- `applications\ux_audit_agent\components\final_summary.py`
- `applications\ux_audit_agent\components\financial_audit.py`
- `applications\ux_audit_agent\components\general_knowledge.py`
- `applications\ux_audit_agent\components\project_manager.py`
- `applications\ux_audit_agent\components\ux_audit.py`
- `applications\ux_audit_agent\UXAuditTest.ipynb`
- `generate_project_overview.py`
- `intergrax\__init__.py`
- `intergrax\chains\__init__.py`
- `intergrax\chains\langchain_qa_chain.py`
- `intergrax\chat_agent.py`
- `intergrax\llm\__init__.py`
- `intergrax\llm\llm_adapters_legacy.py`
- `intergrax\llm\messages.py`
- `intergrax\llm_adapters\__init__.py`
- `intergrax\llm_adapters\base.py`
- `intergrax\llm_adapters\gemini_adapter.py`
- `intergrax\llm_adapters\ollama_adapter.py`
- `intergrax\llm_adapters\openai_responses_adapter.py`
- `intergrax\logging.py`
- `intergrax\memory\__init__.py`
- `intergrax\memory\conversational_memory.py`
- `intergrax\memory\conversational_store.py`
- `intergrax\memory\organization_profile_manager.py`
- `intergrax\memory\organization_profile_memory.py`
- `intergrax\memory\organization_profile_store.py`
- `intergrax\memory\stores\__init__.py`
- `intergrax\memory\stores\in_memory_conversational_store.py`
- `intergrax\memory\stores\in_memory_organization_profile_store.py`
- `intergrax\memory\stores\in_memory_user_profile_store.py`
- `intergrax\memory\user_profile_manager.py`
- `intergrax\memory\user_profile_memory.py`
- `intergrax\memory\user_profile_store.py`
- `intergrax\multimedia\__init__.py`
- `intergrax\multimedia\audio_loader.py`
- `intergrax\multimedia\images_loader.py`
- `intergrax\multimedia\ipynb_display.py`
- `intergrax\multimedia\video_loader.py`
- `intergrax\openai\__init__.py`
- `intergrax\openai\rag\__init__.py`
- `intergrax\openai\rag\rag_openai.py`
- `intergrax\rag\__init__.py`
- `intergrax\rag\documents_loader.py`
- `intergrax\rag\documents_splitter.py`
- `intergrax\rag\dual_index_builder.py`
- `intergrax\rag\dual_retriever.py`
- `intergrax\rag\embedding_manager.py`
- `intergrax\rag\rag_answerer.py`
- `intergrax\rag\rag_retriever.py`
- `intergrax\rag\re_ranker.py`
- `intergrax\rag\vectorstore_manager.py`
- `intergrax\rag\windowed_answerer.py`
- `intergrax\runtime\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\attachments.py`
- `intergrax\runtime\drop_in_knowledge_mode\config.py`
- `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine_history_layer.py`
- `intergrax\runtime\drop_in_knowledge_mode\history_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`
- `intergrax\runtime\drop_in_knowledge_mode\rag_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`
- `intergrax\runtime\drop_in_knowledge_mode\runtime_state.py`
- `intergrax\runtime\drop_in_knowledge_mode\session_store.py`
- `intergrax\runtime\drop_in_knowledge_mode\websearch_prompt_builder.py`
- `intergrax\supervisor\__init__.py`
- `intergrax\supervisor\supervisor.py`
- `intergrax\supervisor\supervisor_components.py`
- `intergrax\supervisor\supervisor_prompts.py`
- `intergrax\supervisor\supervisor_to_state_graph.py`
- `intergrax\system_prompts.py`
- `intergrax\tools\__init__.py`
- `intergrax\tools\tools_agent.py`
- `intergrax\tools\tools_base.py`
- `intergrax\websearch\__init__.py`
- `intergrax\websearch\cache\__init__.py`
- `intergrax\websearch\context\__init__.py`
- `intergrax\websearch\context\websearch_context_builder.py`
- `intergrax\websearch\fetcher\__init__.py`
- `intergrax\websearch\fetcher\extractor.py`
- `intergrax\websearch\fetcher\http_fetcher.py`
- `intergrax\websearch\integration\__init__.py`
- `intergrax\websearch\integration\langgraph_nodes.py`
- `intergrax\websearch\pipeline\__init__.py`
- `intergrax\websearch\pipeline\search_and_read.py`
- `intergrax\websearch\providers\__init__.py`
- `intergrax\websearch\providers\base.py`
- `intergrax\websearch\providers\bing_provider.py`
- `intergrax\websearch\providers\google_cse_provider.py`
- `intergrax\websearch\providers\google_places_provider.py`
- `intergrax\websearch\providers\reddit_search_provider.py`
- `intergrax\websearch\schemas\__init__.py`
- `intergrax\websearch\schemas\page_content.py`
- `intergrax\websearch\schemas\query_spec.py`
- `intergrax\websearch\schemas\search_hit.py`
- `intergrax\websearch\schemas\web_document.py`
- `intergrax\websearch\service\__init__.py`
- `intergrax\websearch\service\websearch_answerer.py`
- `intergrax\websearch\service\websearch_executor.py`
- `intergrax\websearch\utils\__init__.py`
- `intergrax\websearch\utils\dedupe.py`
- `intergrax\websearch\utils\rate_limit.py`
- `main.py`
- `mcp\__init__.py`
- `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`
- `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`
- `notebooks\langgraph\simple_llm_langgraph.ipynb`
- `notebooks\langgraph\simple_web_research_langgraph.ipynb`
- `notebooks\openai\rag_openai_presentation.ipynb`
- `notebooks\rag\chat_agent_presentation.ipynb`
- `notebooks\rag\output_structure_presentation.ipynb`
- `notebooks\rag\rag_custom_presentation.ipynb`
- `notebooks\rag\rag_multimodal_presentation.ipynb`
- `notebooks\rag\rag_video_audio_presentation.ipynb`
- `notebooks\rag\tool_agent_presentation.ipynb`
- `notebooks\supervisor\supervisor_test.ipynb`
- `notebooks\websearch\websearch_presentation.ipynb`

## Detailed File Documentation

### `api\__init__.py`

DESCRIPTION: The `api` package initializes and sets up the API framework for Intergrax, defining entry points and exposing functionality to clients.

DOMAIN: API setup

KEY RESPONSIBILITIES:
- Initializes the API application
- Registers routes and endpoints
- Configures middleware and authentication mechanisms

### `api\chat\__init__.py`

Description: This module serves as the initialization point for the chat API, responsible for importing and configuring various components.

Domain: API Infrastructure

Key Responsibilities:
* Imports necessary modules for the chat API
* Sets up configuration and dependencies
* Initializes API endpoints and routes

### `api\chat\main.py`

**Description:** This is the main API router for chat functionality in the Integrax framework. It handles user queries, manages session history, and interacts with the underlying database and indexing utilities.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Handles incoming user queries through the `/chat` endpoint
* Manages session history using the `_history_pairs_from_db` function
* Interacts with answerers (RAG models) to generate responses
* Inserts application logs and chat history into the database
* Provides endpoints for uploading, indexing, listing, and deleting documents

**Note:** This file appears to be a critical component of the Integrax framework's chat functionality. It has been thoroughly documented and is likely in active use.

### `api\chat\tools\__init__.py`

DESCRIPTION: This file serves as an initialization point for the chat API tools, defining the entry points and imports necessary for other modules to function correctly.

DOMAIN: Chat API Utilities

KEY RESPONSIBILITIES:
- Initializes tool modules
- Defines entry points for tool usage
- Provides importable utility functions for chat-related operations

### `api\chat\tools\chroma_utils.py`

Description: This module provides utility functions for interacting with the Chroma vector store, allowing for loading and splitting documents, indexing documents, and deleting documents by file ID.

Domain: Vector Store Utilities

Key Responsibilities:
- Load and split documents from a given file path
- Index a document to the Chroma vector store with provided file ID
- Delete a document from the Chroma vector store by its file ID

### `api\chat\tools\db_utils.py`

**Description:** This module provides database utilities for managing chat sessions and associated data in the Integrax framework. It handles schema creation, migration from legacy applications logs, and offers a public API for inserting and retrieving messages, as well as document records.

**Domain:** Database Utilities (for Chat Sessions)

**Key Responsibilities:**

* Provides low-level helpers for establishing database connections
* Offers functions for creating and migrating the database schema
* Public API for:
	+ Ensuring chat sessions are created or updated
	+ Inserting new messages with their associated metadata
	+ Retrieving lists of messages for a given session, optionally limited by a specific number
	+ Obtaining recent conversation history pairs (user text, assistant text) for a given session
* Handles document records:
	+ Insertion and deletion of documents with associated metadata
	+ Retrieval of all documents in the store

Note: The code appears to be comprehensive and well-structured. There are no obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines Pydantic models for handling chat queries, responses, and document information within the Integrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**
- Defines a `QueryInput` model to structure user questions with optional session ID and default model selection.
- Defines a `QueryResponse` model to represent answers from chosen models along with session IDs and selected models.
- Introduces an enumeration `ModelName` for listing available model names.
- Defines a `DocumentInfo` model to store document metadata, including ID, filename, and upload timestamp.
- Specifies a `DeleteFileRequest` model for deleting files by their IDs.

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides utilities for building and managing components of the Rag pipeline, including vector store management, embedding, retriever, reranker, and LLM adapters.

**Domain:** RAG (Reptile) Logic / Embeddings

**Key Responsibilities:**

* Manages vector stores and embeddings using `VectorstoreManager` and `EmbeddingManager`
* Provides access to retrievers, rerankers, and LLM adapters through lazy initialization
* Defines default prompts for user interaction
* Exposes functions for building answerers with specific models or configurations
* Uses environment variables to configure settings such as vector store directories and model names

### `applications\chat_streamlit\api_utils.py`

Description: This module provides utility functions for interacting with the Integrax framework's API, including making requests to chat and document endpoints.

Domain: API utilities

Key Responsibilities:
- Provides a function to get an API endpoint URL based on the given endpoint name.
- Offers functions to send POST requests to various API endpoints (chat, upload-doc, delete-doc), handling JSON data and error cases.
- Includes functionality for uploading documents via file uploads and listing existing documents through GET requests.

### `applications\chat_streamlit\chat_interface.py`

Description: This module provides a Streamlit-based user interface for interacting with the chat interface, allowing users to send queries and display responses.

Domain: LLM adapters

Key Responsibilities:
- Displays a chat interface using Streamlit components (chat_input, chat_message)
- Retrieves user input from the chat_input component
- Sends the user input to the API for response generation
- Displays the response from the API in the chat interface
- Provides additional details about the generated answer and model used in an expander section

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides a user interface for interacting with uploaded documents, including model selection, uploading files, listing and deleting documents.

**Domain:** Chat Streamlit Application Components

**Key Responsibilities:**
- Display model selection component in the sidebar
- Handle file upload through the sidebar's uploader
- List and refresh uploaded documents in the sidebar
- Allow users to delete selected documents from the sidebar

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module serves as the main entry point for the chatbot application, utilizing Streamlit to create a user interface with sidebar and chat functionality.

**Domain:** Configuration/UI Logic

**Key Responsibilities:**

* Initializes Streamlit app with title "intergrax RAG Chatbot"
* Sets up initial state variables for messages and session ID
* Displays sidebar and chat interface using external modules

### `applications\company_profile\__init__.py`

**Description:** This is the entry point for the company profile application, responsible for initializing and configuring its components.

**Domain:** Application Initialization

**Key Responsibilities:**
- Initializes the company profile application
- Configures application-specific settings and dependencies

### `applications\figma_integration\__init__.py`

Description: This file serves as the entry point for Figma integration within Intergrax, enabling seamless interaction between the platform and our framework.

Domain: Integration Modules

Key Responsibilities:
* Initializes the Figma API connection
* Defines configuration settings for Figma integration
* Registers necessary plugins and functionality

### `applications\ux_audit_agent\__init__.py`

DESCRIPTION: The `__init__.py` file initializes and exports the UX Audit Agent application.

DOMAIN: Agents

KEY RESPONSIBILITIES:
• Initializes the UX Audit Agent application
• Exports the agent's entry point and configuration options

### `applications\ux_audit_agent\components\__init__.py`

DESCRIPTION: This file serves as the entry point for the UX audit agent's components, responsible for initializing and configuring other modules.

DOMAIN: RAG (Reasoning And Generation) logic

KEY RESPONSIBILITIES:
- Initializes and imports necessary components for the UX audit agent
- Sets up configuration and dependencies for downstream modules

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** The compliance_checker module is responsible for evaluating the proposed changes against the organization's privacy policy and regulatory rules, simulating a validation process.

**Domain:** RAG logic (Regulatory Approval Guidance)

**Key Responsibilities:**
- Evaluates proposed changes against organizational policies and regulations.
- Returns simulated compliance results with probability of being compliant (80%).
- Identifies non-compliance issues and provides corrective actions.
- Can stop pipeline execution if non-compliant findings are detected.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a component for cost estimation of UX-related changes based on an audit report. It uses a mock pricing model to calculate the estimated cost.

**Domain:** RAG logic

**Key Responsibilities:**
- Provides a "Cost Estimation Agent" component
- Estimates the cost of UX updates derived from an audit using a mock pricing model
- Returns a cost estimate and related metadata in the pipeline state

### `applications\ux_audit_agent\components\final_summary.py`

Description: This module provides a final summary of the execution pipeline using collected artifacts, always executed at the final stage.

Domain: UX Audit Agent components

Key Responsibilities:
- Generates a complete summary of the entire execution pipeline.
- Collects and processes various artifacts (e.g., project manager decision, notes, UX report, financial report, citations).
- Returns a ComponentResult with the final report.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module defines a financial audit agent component for the Integrax framework, providing a test data generator for financial reports and VAT calculations.

**Domain:** RAG logic (Reasoning and Acting)

**Key Responsibilities:**
- Generates mock financial reports with test data.
- Calculates VAT amounts based on provided rates.
- Provides example use cases for financial computations and budget constraints.
- Exposes the component as part of the Integrax framework.

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a component that answers general questions about the Intergrax system, including its structure, features, and configuration.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Defines a component named "General" for answering general questions
* Provides mock knowledge responses to simulate expert-level understanding of the Intergrax system
* Returns relevant citations (mock documents) in response to user queries
* Uses Intergrax's supervisor components and pipeline state to generate responses

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a project manager component for the UX audit agent, responsible for reviewing and making mock decisions on UX reports.

**Domain:** RAG logic (Risk Assessment and Governance)

**Key Responsibilities:**
- Reviews UX reports and makes mock decisions based on a random approval/rejection model.
- Produces decision and notes as output.
- Stops pipeline execution if proposal is rejected.

### `applications\ux_audit_agent\components\ux_audit.py`

**Description:** This module provides a UX auditing component for the Integrax framework, allowing users to analyze UI/UX based on Figma mockups and generate sample reports with recommendations.

**Domain:** RAG logic (Reporting, Analysis, and Generation)

**Key Responsibilities:**
- Performs UX audit based on Figma mockups
- Generates sample report with recommendations
- Returns report with issues, summary, and estimated cost

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This Jupyter Notebook file defines a workflow for UX audit and compliance verification, leveraging various components of the Integrax framework.

Domain: RAG logic

Key Responsibilities:
- Perform UX audit on FIGMA mockups
- Verify changes comply with company policy
- Prepare summary report for Project Manager
- Evaluate financial impact of changes
- Project Manager decision and project continuation
- Final report preparation and synthesis

### `generate_project_overview.py`

**Description:** This module generates a Markdown file that provides an overview of the Intergrax framework's project structure. It uses Large Language Models (LLMs) to summarize each file's purpose, domain, and key responsibilities.

**Domain:** Project Structure Documentation

**Key Responsibilities:**

* Recursively scans the project directory
* Collects all relevant source files (Python, Jupyter Notebooks, configurable)
* Generates a structured summary for each file using an LLM:
	+ Provides a high-level description of the file's purpose
	+ Identifies the domain (e.g., "LLM adapters", "RAG logic", "data ingestion")
	+ Lists key responsibilities/main functionality in bullet points
* Builds and writes a Markdown file containing the summaries

Note: This module appears to be production-ready, with clear documentation and a straightforward implementation. However, it relies on external dependencies (e.g., `langchain_ollama`) that may need to be installed separately.

### `intergrax\__init__.py`

DESCRIPTION: This module serves as the entry point for the Intergrax framework, initializing and setting up essential components.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes the core components of the Intergrax framework.
• Sets up necessary configuration and environment variables.
• Registers available modules and adapters.

### `intergrax\chains\__init__.py`

DESCRIPTION: The __init__.py file is the entry point for Intergrax's chain functionality, responsible for defining and initializing chains in the framework.

DOMAIN: Chain Management

KEY RESPONSIBILITIES:
- Initializes chain instances and configuration
- Defines chain-related functions and utilities
- Sets up chain dependencies and relationships

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module defines a LangChain-style QA chain (RAG → [rerank] → prompt → LLM) with hooks modifying data at stages. It provides a flexible way to build and execute question-answering chains.

**Domain:** RAG (Retrieval-Augmented Generation)

**Key Responsibilities:**

* Builds a QA chain using LangChain's Runnable pipeline
* Provides hooks for modifying data at different stages of the chain
* Supports retrieval, reranking, context building, prompt construction, and LLM execution
* Returns a dictionary with answer, sources, prompt, raw hits, and used hits as output

### `intergrax\chat_agent.py`

**Description:** This module provides a chat agent implementation using the Integrax framework, allowing for routing decisions based on LLM outputs and configuration.

**Domain:** Chat Agents, LLM Routing, Conversational Memory Management

**Key Responsibilities:**

* Initializes the chat agent with an LLM adapter and optional conversational memory
* Handles routing decisions via the LLM router, choosing between RAG, TOOLS, or GENERAL routes
* Executes route-specific logic for RAG, TOOLS, or GENERAL routes
* Manages conversational memory and streams data as needed
* Returns a stable result object containing answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component information

**Note:** This file appears to be the main implementation of the chat agent, and its functionality is central to the Integrax framework.

### `intergrax\llm\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the LLM adapters in the Intergrax framework.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
* Registers LLM adapters
* Initializes adapters on import
* Allows for easy swapping between different LLM models or implementations.

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This file defines a set of adapters and utilities for interacting with large language models (LLMs) via the Intergrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an interface for generating messages from LLMs using various protocols (e.g., OpenAI Chat Completions)
* Offers tools for structured output, including validation and conversion of JSON data
* Supports interaction with different LLM models through specific adapter classes
* Enables integration with external tools and tools schema

Note: This file appears to be a part of the Intergrax framework's core functionality. The code is well-structured, and there are no obvious signs of incompleteness or experimental nature.

### `intergrax\llm\messages.py`

**Description:** This module provides classes and functions for representing chat messages in the Integrax framework. It includes utility classes `AttachmentRef` and `ChatMessage`, as well as a custom reducer function `append_chat_messages`.

**Domain:** LLM adapters (due to its interaction with OpenAI Responses API and tool calls)

**Key Responsibilities:**
- Define lightweight reference class `AttachmentRef` for attachments associated with messages or sessions.
- Provide the `ChatMessage` class, which is an extended version of the OpenAI Responses API's chat message format, supporting fields like `tool_call_id`, `tool_calls`, and metadata.
- Implement methods on the `ChatMessage` class:
  - `to_dict`: convert object to a dict compatible with OpenAI Responses API / ChatCompletions
  - `__repr__`: provide a string representation of the object
- Offer custom reducer function `append_chat_messages` for merging new chat messages into existing ones.

**Status:** This file appears to be fully featured and well-documented, suggesting it is part of the main framework functionality rather than experimental or auxiliary.

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point for LLM adapters in Intergrax, providing a registry and registration mechanism for various language model interfaces.

Domain: LLM adapters

Key Responsibilities:
- Registers LLM adapters with the `LLMAdapterRegistry`
- Exposes adapters from the `.base` and adapter-specific modules (e.g., `openai_responses_adapter`, `gemini_adapter`, `ollama_adapter`)
- Provides a default registration for common LLM interfaces (OpenAI, Gemini, LangChain Ollama)

### `intergrax\llm_adapters\base.py`

**Description:** This module provides the base class and protocol for interacting with Large Language Models (LLMs) in the Intergrax framework, including utilities for token counting and structured output.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a `BaseLLMAdapter` class that offers shared utilities such as token counting.
* Defines an `LLMAdapter` protocol for interacting with LLMs.
* Includes methods for generating and streaming messages from LLMs.
* Offers optional tools for using specific models or tools, including structured output generation.

This file appears to be a core component of the Intergrax framework's LLM interaction mechanism.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a minimal adapter for integrating the Gemini large language model into the Integrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides estimates for context window tokens based on Gemini model names
* Implements basic chat functionality using Gemini's `start_chat` method
* Supports generating and streaming messages with adjustable temperature and maximum token count
* Does not currently support tools (e.g., entity recognition, summarization)

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides an adapter class (`LangChainOllamaAdapter`) for using Ollama models with the Intergrax framework, specifically via LangChain's `ChatModel` interface.

**Domain:** LLM adapters

**Key Responsibilities:**

* Adapts Ollama models to be used in the Intergrax framework
* Provides context window estimation for Ollama models based on their names
* Caches maximum context windows for configured Ollama models
* Converts internal `ChatMessage` lists into LangChain message objects
* Injects tool results as contextual system messages
* Generates output using Ollama with optional temperature and max tokens settings
* Streams output from Ollama with optional temperature and max tokens settings
* Supports structured output via prompt + validation

**Note:** The file is a part of the Intergrax framework's LLM adapters, which suggests that it is not experimental or auxiliary. However, some parts of the code (e.g., the "planner" pattern for tool calls) seem to be more ad-hoc and might benefit from refactoring or further documentation.

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides a class that acts as an adapter for the OpenAI Responses API, allowing Intergrax to interact with it in a way that's compatible with its own internal interface.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a way to estimate the context window size for OpenAI models
* Converts messages from Intergrax's format to the Responses API input format
* Handles streaming and single-shot completion requests to the Responses API
* Extracts assistant output text from Responses API results
* Supports tools functionality, allowing tool calls in responses

### `intergrax\logging.py`

Here is the documentation for the file:

**Description:** This module configures and standardizes logging behavior within the Integrax framework.
**Domain:** Logging configuration
**Key Responsibilities:**
- Configures logging level to show INFO messages and higher (including DEBUG)
- Sets logging format to include timestamp, log level, and message
- Forces new logging configurations, overriding any previous settings

### `intergrax\memory\__init__.py`

Description: The `__init__.py` file in the `memory` module serves as an entry point for importing and initializing memory-related functionality within Intergrax.

Domain: Memory management

Key Responsibilities:
- Initializes memory modules
- Exposes core memory functions to other components
- Manages memory dependencies and imports

### `intergrax\memory\conversational_memory.py`

**Description:** This module provides a universal in-memory conversation history component for storing and managing chat messages.

**Domain:** Conversation Management

**Key Responsibilities:**
- Keep messages in RAM
- Provide API to add, extend, read, and clear messages
- Enforce max messages limit (optional)
- Prepare messages for different model backends

### `intergrax\memory\conversational_store.py`

**Description:** This module defines an abstract interface (`ConversationalMemoryStore`) for persistent storage of conversational memory, allowing for various backend implementations while keeping runtime logic consistent.

**Domain:** Conversational Memory Storage

**Key Responsibilities:**
* Load full conversational history for a given session
* Save the entire state of conversational memory for a session
* Append single messages to persistent storage and update in-memory instance
* Permanently remove stored history for a given session

Note: This file appears to be a well-designed, abstract interface for conversation memory storage, with clear guidelines for implementations. The module is likely part of the Intergrax framework's core functionality.

### `intergrax\memory\organization_profile_manager.py`

**Description:** This module, `organization_profile_manager`, provides a high-level facade for working with organization profiles in the Intergrax framework.

**Domain:** Memory management

**Key Responsibilities:**
- Load and persist organization profiles using an underlying store.
- Provide convenient methods to load, save, delete, and manage system instructions for organizations.
- Hide direct interaction with the underlying store.
- Offer a deterministic method to build system instructions from an organization profile.
- Update system instructions by setting them directly in the profile.

### `intergrax\memory\organization_profile_memory.py`

**Description:** This module provides data classes and utilities for managing organization profiles, including identification data, preferences, and long-term memory entries.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Define stable identification data for organizations (OrganizationIdentity)
* Define organization-level preferences that influence runtime behavior (OrganizationPreferences)
* Represent long-term organization memory entries as unit-of-work (OrganizationProfileMemoryEntry)
* Define the single source of truth for an organization's long-term profile, including identity, preferences, system instructions, and memory entries (OrganizationProfile)

### `intergrax\memory\organization_profile_store.py`

**intergrax\memory\organization_profile_store.py**

Description: This module defines a protocol for persistent storage of organization profiles, abstracting away backend-specific concerns and providing a standardized interface for loading, saving, and deleting profiles.

Domain: Memory Storage Abstraction

Key Responsibilities:
- Loading organization profiles by ID
- Saving organization profiles to persistent storage
- Deleting organization profiles from storage
- Providing default profiles for new organizations
- Hiding backend-specific implementation details

### `intergrax\memory\stores\__init__.py`

DESCRIPTION: This module initializes the memory stores for the Intergrax framework, providing a standardized interface for storing and retrieving data.

DOMAIN: Memory Stores Management

KEY RESPONSIBILITIES:
* Initializes memory stores
* Provides a unified API for accessing and modifying stored data
* Sets up default store configurations
* Registers store instances with the framework

### `intergrax\memory\stores\in_memory_conversational_store.py`

**Description:** 
This module provides an in-memory conversational store implementation for the Intergrax framework, suitable for local development, prototyping, and testing purposes.

**Domain:** Memory Management (in-memory conversational store)

**Key Responsibilities:**

* Provides an in-memory storage for conversation history
* Supports loading and saving conversation data from/to memory
* Allows appending messages to existing conversations
* Offers deletion of persisted session data
* Includes optional diagnostics helper for listing active sessions

### `intergrax\memory\stores\in_memory_organization_profile_store.py`

**Description:** This module provides an in-memory implementation of the `OrganizationProfileStore` interface for storing and managing organization profiles within the Integrax framework.

**Domain:** Memory Stores (in-memory data management)

**Key Responsibilities:**

* Provides a lightweight, volatile storage solution for organization profiles
* Supports basic CRUD operations (create, read, update, delete) on organization profiles
* Offers a default profile creation mechanism when an unknown organization ID is queried
* Optional list method for debugging or testing purposes

### `intergrax\memory\stores\in_memory_user_profile_store.py`

**Description:** This module provides an in-memory implementation of the `UserProfileStore` interface for storing and retrieving user profiles.

**Domain:** Memory stores

**Key Responsibilities:**
- Provides a simple, in-memory store for user profiles
- Allows getting existing or default profiles by user ID
- Enables saving updated profiles to memory
- Supports deleting stored profiles
- Optionally stores default profiles for faster subsequent access

### `intergrax\memory\user_profile_manager.py`

**Description:** 
This module provides a high-level facade for working with user profiles, abstracting away direct interaction with the underlying UserProfileStore. It offers methods to load or create user profiles, persist profile changes, manage long-term user memory entries, and derive system-level instructions from the profile.

**Domain:** User Profile Management

**Key Responsibilities:**
- Load or create a UserProfile for a given user_id
- Persist profile changes
- Manage long-term user memory entries
- Derive system-level instructions from the profile
- Hide direct interaction with the underlying UserProfileStore

### `intergrax\memory\user_profile_memory.py`

Description: This module defines core domain models for user and organization profiles, as well as prompt bundles, decoupled from storage or engine logic.

Domain: User Profile Management / Identity

Key Responsibilities:
- Define UserProfileMemoryEntry to store long-term facts and notes about users.
- Introduce UserIdentity to describe high-level information about users (name, role, expertise).
- Define UserPreferences for user's preferred language, answer length, tone, formatting rules, etc.
- Create UserProfile aggregate class containing identity, preferences, system instructions, and memory entries.

### `intergrax\memory\user_profile_store.py`

**Description:** This module defines a protocol for persistent storage of user profiles within the Integrax framework. It provides interfaces for loading, saving, and deleting user profiles while abstracting away backend-specific concerns.

**Domain:** Memory Storage (UserProfile)

**Key Responsibilities:**

* Load user profile for a given user ID
* Save user profile aggregate for associated user ID
* Delete stored profile data for a given user ID

### `intergrax\multimedia\__init__.py`

Description: This is the entry point for the multimedia module in Intergrax, responsible for initializing and setting up various multimedia-related components.

Domain: Multimedia Utilities

Key Responsibilities:
* Initializes multimedia engines and managers
* Sets up data pipelines for media processing
* Configures multimedia formats and codecs 
* Exposes interfaces for integrating with other modules

### `intergrax\multimedia\audio_loader.py`

Description: This module provides functionality for downloading and translating audio files from YouTube URLs.

Domain: Multimedia

Key Responsibilities:
- Downloads audio from a provided YouTube URL in the specified format.
- Translates the downloaded audio into a target language using Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module provides an image transcription service, utilizing the ollama library to generate text descriptions from images.
Domain: LLM adapters

Key Responsibilities:
- Transcribes images into text using a provided model (defaulting to "llava-llama3:latest")
- Utilizes ollama's chat functionality with image support for generating descriptive text
- Allows for custom model selection via the `model` parameter

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, such as audio, images, and videos, within the Intergrax framework.

**Domain:** Multimedia Display Utilities

**Key Responsibilities:**

* Displays audio files with optional start time and autoplay
* Displays image files using IPython display
* Serves video files by copying them to a temporary directory and returning a URL for playback
* Plays video content with customizable start time, poster frame, autoplay, and playback rate

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides functionality for working with multimedia content, specifically video loading and processing.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Loading videos from YouTube URLs using `yt-dlp` library
* Transcribing videos to VTT files using Whisper model
* Extracting frames from videos at regular intervals or based on transcript segments
* Saving extracted frames and metadata in specified directories
* Resizing images to maintain aspect ratio

### `intergrax\openai\__init__.py`

Description: The __init__.py file serves as an entry point for the Intergrax OpenAI module, allowing for easy import of its contents.

Domain: LLM adapters

Key Responsibilities:
- Exposes the OpenAI adapter class and other related functionality to the outside world.
- Provides a centralized access point for interacting with the OpenAI API.

### `intergrax\openai\rag\__init__.py`

**Description:** 
This module serves as the entry point for RAG (Retrieval-Augmented Generation) logic in Intergrax, handling initialization and setup for related components.

**Domain:** RAG Logic

**Key Responsibilities:**
- Initializes RAG modules and their dependencies
- Sets up RAG-specific configurations

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module, `rag_openai.py`, provides a class-based implementation of the Retrieval Augmented Generator (RAG) model using OpenAI's vector store and client APIs. It enables integration with the Integrax framework.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Initializes the RAG model with an OpenAI client and vector store ID
* Generates a prompt for the RAG model based on the provided configuration
* Retrieves the vector store by its ID
* Clears the vector store and storage (deletes all files loaded into the vector store)
* Uploads a folder to the vector store, allowing multiple file patterns to be specified

**Notes:**

The file appears to be a functional implementation of the RAG model using OpenAI's APIs. The code is well-structured, with clear responsibility assignments between methods. However, some minor improvements can be made for better readability and maintainability (e.g., using type hints consistently). Overall, this module seems complete and production-ready.

### `intergrax\rag\__init__.py`

Description: This is the entry point for the RAG (Retrieval-Augmented Generation) logic in Intergrax, responsible for setting up and configuring RAG components.

Domain: RAG logic

Key Responsibilities:
- Initializes RAG-related modules and services
- Defines RAG component configurations and parameters
- Sets up data pipelines for RAG training and inference

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader that can handle various file formats, including text documents, images, PDFs, Excel files, and more. It allows for customization of loading settings through the use of adapters.

**Domain:** RAG (Retrieval-Augmented Generation) logic / Document Loading

**Key Responsibilities:**

* Loads documents from a variety of file types (e.g., text, images, PDFs, Excel)
* Supports OCR (Optical Character Recognition) for image and PDF files
* Allows customization of loading settings through adapters
* Provides metadata injection and safety guards to ensure robustness
* Includes support for captioning images using LLM adapters

**Note:** This module appears to be a comprehensive and well-maintained part of the Intergrax framework, with clear documentation and a wide range of features.

### `intergrax\rag\documents_splitter.py`

**Description:** The DocumentsSplitter class provides a high-quality text splitter for RAG pipelines, capable of generating stable chunk ids and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**
- Provides a sophisticated text splitting algorithm
- Generates stable chunk ids using semantic anchors (para_ix/row_ix/page_index) when available
- Extracts parent_id, source_name, and source_path from document metadata
- Merges tiny tail chunks with previous ones per document
- Applies optional hard cap on the number of chunks per document
- Adds metadata such as chunk index, total chunks, page index (if present), and stable chunk id to each chunk

**Status:** Not experimental or auxiliary; appears to be a critical component of the RAG pipeline.

### `intergrax\rag\dual_index_builder.py`

**Description:** This module is responsible for building and maintaining two vector indexes: a primary index (CHUNKS) and an auxiliary index (TOC), utilizing embeddings from documents in the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Build two vector indexes: CHUNKS and TOC
* CHUNKS index contains all chunks/documents after splitting
* TOC index contains only DOCX headings within specified levels
* Embedding manager is used to compute embeddings for documents
* Documents are split into batches for efficient insertion into the indexes
* Indexes can be enabled or disabled based on configuration options

Note: The file appears to be fully functional and does not exhibit any characteristics of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\dual_retriever.py`

**Description:** This module implements a Dual Retriever class for fetching relevant chunks of text from a vector store based on a query. It first searches the Table of Contents (TOC) to identify sections relevant to the query, and then retrieves local chunks from those sections.

**Domain:** RAG logic

**Key Responsibilities:**

*   **Initialization**: The `DualRetriever` class is initialized with a `VectorstoreManager` for chunk retrieval and an optional `EmbeddingManager` for text embeddings.
*   **Querying the TOC**: The class queries the TOC to identify relevant sections based on the query. This involves searching using similarity metrics.
*   **Local Chunk Retrieval**: For each matched section, local chunks are retrieved from the chunk vector store.
*   **Merging and Sorting Results**: The results from both steps are merged, deduplicated, and sorted by similarity score.

**Status:** Not experimental, auxiliary, legacy or incomplete.

### `intergrax\rag\embedding_manager.py`

**Description:** This module, `embedding_manager.py`, is responsible for managing and providing unified access to various text embeddings from different sources (Hugging Face, Ollama, and OpenAI). It handles model loading, parameter configuration, embedding computation, and provides utility functions for handling embeddings.

**Domain:** RAG logic

**Key Responsibilities:**

*   Manage providers for text embeddings (Hugging Face, Ollama, OpenAI)
*   Load models from specified providers with configurable parameters
*   Compute embeddings for given texts using selected provider's model
*   Provide utilities for handling embeddings (normalization, cosine similarity)
*   Handle model loading errors and retries

### `intergrax\rag\rag_answerer.py`

**Description:** This file defines the RAG Answerer class, which is responsible for generating answers to user questions using a combination of retrieval and ranking algorithms.

**Domain:** Retrieval-Augmented Generation (RAG) logic

**Key Responsibilities:**

*   Retrieve relevant context fragments from a conversational memory
*   Re-rank retrieved hits based on similarity scores
*   Build context by selecting the most relevant hits and concatenating their text
*   Generate messages for the LLM, including system and user instructions
*   Send messages to the LLM and retrieve generated answers
*   Optionally generate structured output using a provided Pydantic model

**Notes:** The code is well-organized, readable, and follows standard Python conventions. The class has several optional parameters, allowing for customization of its behavior based on specific requirements. However, some parts of the code appear to be experimental or auxiliary (e.g., `stream` parameter in `run` method). Overall, this implementation seems solid and reliable.

### `intergrax\rag\rag_retriever.py`

**Description:** This module provides a scalable, provider-agnostic RAG retriever for the Intergrax framework. It offers features such as normalization of filters and query vectors, unified similarity scoring, deduplication, and batch retrieval.

**Domain:** RAG logic

**Key Responsibilities:**

* Normalizes `where` filters for Chroma (flat dict → $and/$eq)
* Normalizes query vector shape (1D/2D → [[D]])
* Unified similarity scoring:
	+ Chroma → converts distance to similarity = 1 - distance
	+ Others → uses raw similarity as returned
* Deduplication by ID + per-parent result limiting (diversification)
* Optional MMR diversification when embeddings are returned
* Batch retrieval for multiple queries
* Optional reranker hook (e.g., cross-encoder, re-ranking model)

Note: The module appears to be complete and production-ready.

### `intergrax\rag\re_ranker.py`

**Description:** The `re_ranker.py` file is part of the Integrax framework and provides a cosine re-ranker for candidate chunks. It accepts hits from the `intergraxRagRetriever` or raw LangChain Documents, embeds texts in batches using an `intergraxEmbeddingManager`, and optionally fuses scores with the original retriever similarity.

**Domain:** RAG logic

**Key Responsibilities:**

* Embed query and documents using a caching mechanism for improved efficiency
* Compute cosine similarities between query and document embeddings
* Rank candidates based on cosine similarities, preserving their schema
* Optional score fusion with original retriever similarity
* Lightweight in-memory cache for query embeddings to reduce redundant computation

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module provides a unified vector store manager for Intergrax, supporting ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**
- Initializes target vector store (ChromaDB, Qdrant, or Pinecone) based on provided configuration.
- Upserts documents and embeddings with batching support.
- Queries top-K similar vectors by cosine/dot/euclidean similarity.
- Counts vectors in the store.
- Deletes vectors by their IDs.

**Note:** The file appears to be a primary implementation module for vector store management within the Intergrax framework. It is well-documented and does not show any obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a Windowed Answerer class, which is a layer on top of the base Answerer in the Intergrax framework. It provides functionality to process answers in a windowed manner, allowing for more efficient and effective handling of large amounts of context.

**Domain:** RAG (Reactive Agents) logic

**Key Responsibilities:**

* Provides a Windowed Answerer class that extends the base Answerer
* Implements methods for building context and messages with memory-awareness
* Allows for windowing of answers, enabling more efficient processing of large contexts
* Supports optional summarization of partial answers per window
* Deduplicates sources and appends final answer (and optional summary) to the memory store

### `intergrax\runtime\__init__.py`

Description: The `__init__.py` file is the entry point for the Intergrax runtime, responsible for setting up the framework and making its components available for import.

Domain: Framework Core

Key Responsibilities:
- Initializes the Intergrax runtime environment
- Defines the root package and makes it available for import
- Sets up the package structure and namespace

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: This module initializes and sets up the knowledge graph for Intergrax's drop-in knowledge mode.

Domain: RAG logic

Key Responsibilities:
• Initializes the knowledge graph
• Sets up necessary components for knowledge retrieval and manipulation
• Exposes APIs for interacting with the knowledge graph in drop-in knowledge mode 

(Note: No indication of experimental, auxiliary, legacy, or incomplete code)

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides attachment resolution utilities for Intergrax's Drop-In Knowledge Mode, decoupling the storage of attachments from their consumption in the RAG pipeline.

**Domain:** RAG (Reasoning And Generation) logic / Data ingestion

**Key Responsibilities:**
- Defines an `AttachmentResolver` protocol that knows how to turn an `AttachmentRef` into a local file path
- Provides a minimal implementation, `FileSystemAttachmentResolver`, for resolving local filesystem-based URIs
- Allows for the extension of attachment resolvers for various storage types (e.g., object storage, databases)

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This configuration file defines the settings for the Drop-In Knowledge Runtime in Intergrax. It controls various aspects of runtime behavior, including LLM adapters, RAG and web search configurations, tool usage, and multi-tenancy.

**Domain:** Configuration

**Key Responsibilities:**

* Defining primary LLM adapter and its label
* Configuring RAG (retrieval-augmented generation) settings:
	+ Enablement flag
	+ Vectorstore manager instance
	+ Embedding manager instance
	+ Retrieval parameters (max docs per query, max tokens)
	+ Semantic score threshold
* Configuring web search settings:
	+ Executor instance or null for disabling
* Defining tool usage policy and context scope
* Managing multi-tenancy settings (tenant ID, workspace ID)
* Enabling/disabling various memory components (user profile, long-term memory)

**Note:** This file appears to be a crucial part of the Intergrax framework's configuration mechanism.

### `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`

**Description:** This module provides a context builder for Drop-In Knowledge Mode in the Intergrax framework. It decides when to use Retrieval-Augmented Generation (RAG) for a given request, retrieves relevant document chunks from the vector store, and provides a RAG-specific system prompt.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Decide whether to use RAG for a given request
* Retrieve relevant document chunks from the vector store using session/user/tenant/workspace metadata
* Provide a RAG-specific system prompt
* Compose a BuiltContext object with:
	+ System prompt
	+ Reduced history messages
	+ Retrieved chunks
	+ Structured RAG debug info

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

Here's the documentation for the provided file:

**Description:** The `engine.py` file defines a high-level conversational runtime for the Intergrax framework, enabling Drop-In Knowledge Mode.

**Domain:** LLM adapters and conversational runtime logic.

**Key Responsibilities:**

* Defines the `DropInKnowledgeRuntime` class, which loads or creates chat sessions and builds a conversation history for the LLM.
* Provides a stateful pipeline for augmenting context with RAG, web search, and tools results.
* Produces a `RuntimeAnswer` object as a high-level response to user queries.

Note: The code appears to be well-structured and complete, without any obvious signs of being experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\engine_history_layer.py`

**Description:** This module provides a HistoryLayer class that encapsulates logic for loading conversation history, counting tokens, applying history compression strategies, and updating the RuntimeState with preprocessed history.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Load raw conversation history from SessionStore
* Compute token usage for the raw history
* Apply per-request history compression strategy to trim the history
* Update RuntimeState with preprocessed conversation history and debug information
* Handle degenerate cases where token budget is extremely small or misconfigured

### `intergrax\runtime\drop_in_knowledge_mode\history_prompt_builder.py`

**Description:** This module provides a history prompt builder for Drop-In Knowledge Mode, allowing users to summarize older conversation turns into an information-dense summary.

**Domain:** LLM adapters (specifically, Drop-In Knowledge Mode)

**Key Responsibilities:**

* Provides a `HistorySummaryPromptBuilder` interface for building the history-summary-related part of the prompt
* Offers a default implementation (`DefaultHistorySummaryPromptBuilder`) that provides a generic system prompt for summarization
* Allows users to customize the prompt text using request and config fields (e.g., domain, language, user preferences)
* Supports summarizing older conversation turns into an information-dense summary

Note: This file appears to be part of the main Intergrax framework codebase and is not marked as experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

**Description:** This module provides a high-level service for ingesting attachments in the context of Drop-In Knowledge Mode, reusing existing Intergrax RAG building blocks.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Resolve AttachmentRef objects into filesystem Paths using AttachmentResolver
* Load documents using IntergraxDocumentsLoader.load_document(...)
* Split documents into chunks using IntergraxDocumentsSplitter.split_documents(...)
* Embed chunks via IntergraxEmbeddingManager
* Store vectors in a vector database via IntergraxVectorstoreManager
* Return structured IngestionResult per attachment

The service does not manage ChatSession objects, perform retrieval or answering. It is intended to be called from orchestration layers when new attachments are added to a session.

**Note:** The module appears to be well-structured and complete, with clear documentation and responsibilities outlined.

### `intergrax\runtime\drop_in_knowledge_mode\rag_prompt_builder.py`

**Description:** 
This module provides functionality for building prompts related to Retrieval-Augmented Generation (RAG) in Intergrax's Drop-In Knowledge Mode.

**Domain:** RAG logic

**Key Responsibilities:**

* Defines the `RagPromptBundle` data class, which contains a system prompt and additional context messages.
* Introduces the `RagPromptBuilder` protocol for customizing RAG-related prompt building.
* Provides the `DefaultRagPromptBuilder` implementation, responsible for:
	+ Using the built context's system prompt as-is.
	+ Formatting retrieved chunks into natural, model-friendly text blocks for additional context messages.

**Status:** This module appears to be part of the main Intergrax framework and is used in Drop-In Knowledge Mode. It does not seem experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

**Description:** This module defines dataclasses and utilities for handling requests and responses in the Drop-In Knowledge Mode runtime of the Intergrax framework.

**Domain:** RAG logic / Data Ingestion

**Key Responsibilities:**

* Define high-level request and response structures (dataclasses) for communication between applications and the DropInKnowledgeRuntime.
* Provide data models for citations, routing information, tool calls, and basic statistics.
* Enumerate history compression strategies for compressing conversation history before sending it to the LLM.
* Facilitate creation of requests with optional tenant/workspace scoping, UI/app metadata, and user-provided instructions.

### `intergrax\runtime\drop_in_knowledge_mode\runtime_state.py`

Description: This module defines the RuntimeState class, which serves as a container for aggregating and passing mutable state throughout the Intergrax runtime pipeline.

Domain: RAG logic

Key Responsibilities:
- Aggregates request and session metadata.
- Manages ingestion results and conversation history.
- Stores flags indicating subsystem usage (RAG, websearch, tools, memory).
- Keeps track of tools traces and agent answers.
- Maintains a full debug_trace for observability & diagnostics.

### `intergrax\runtime\drop_in_knowledge_mode\session_store.py`

**Description:** This module provides a centralized memory component for the Intergrax Drop-In Knowledge Runtime, responsible for managing chat sessions, storing and retrieving conversational history, exposing user and organization profiles, and producing LLM-ready message context.

**Domain:** Memory Management

**Key Responsibilities:**

* Create and manage chat sessions
* Maintain conversational message history per session
* Expose user/organization profile bundles and long-term memory context
* Return an LLM-ready ordered list of messages representing the session context
* Manage session lifecycle, including creating, saving, and retrieving sessions
* Append chat messages to a session's conversational history
* List sessions owned by a user, sorted by recent activity

Note: The implementation is currently in-memory only, with plans for future expansion to include persistent storage and deeper integration with long-term semantic memory.

### `intergrax\runtime\drop_in_knowledge_mode\websearch_prompt_builder.py`

**Description:** This module is responsible for building the web search part of the prompt in Drop-In Knowledge Mode, providing a way to construct system-level messages and debug information from web documents.

**Domain:** LLM adapters / Web Search Integration

**Key Responsibilities:**

* Building a `WebSearchPromptBundle` object containing system-level messages and debug information
* Providing a strategy interface (`WebSearchPromptBuilder`) for custom implementations
* Default prompt builder (`DefaultWebSearchPromptBuilder`) that takes web documents, summarizes them, and constructs a system message with titles, URLs, and snippets.

### `intergrax\supervisor\__init__.py`

DESCRIPTION: The supervisor module serves as the entry point for the Intergrax framework, handling setup and initialization tasks.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
- Sets up core modules and dependencies
- Initializes logging and configuration systems
- Defines main application loop or runner

### `intergrax\supervisor\supervisor.py`

**Description:** This module is responsible for providing the core functionality of the Intergrax framework's supervisor, which manages and orchestrates interactions with Language Models (LLMs).

**Domain:** LLM supervisors / RAG logic

**Key Responsibilities:**

*   Manages interactions with LLM adapters
*   Provides methods for planning and executing tasks with LLMs
*   Includes functionality for decomposing queries into smaller steps, assigning components to each step, and executing the plan
*   Handles fallback strategies when necessary (e.g., heuristic-based planning or minimal fallback plans)
*   Offers support for two-stage planning, which involves decomposing a query into individual steps and then assigning components to each step with the help of LLMs.

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides a framework for building and managing components, which are self-contained units of logic that can be plugged into the Integrax pipeline. Components define their own behavior and can access shared resources.

**Domain:** Supervisor Components

**Key Responsibilities:**

* Defines the `Component` class, which represents a self-contained unit of logic in the pipeline.
* Provides the `run` method for executing components with a given state and context.
* Offers a decorator (`component`) for registering new components quickly.
* Includes utility classes like `PipelineState`, `ComponentResult`, and `ComponentContext`.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides default prompt templates for the Intergrax framework's Supervisor, outlining the structure and rules for planning and execution.

**Domain:** RAG (Reasoning And Gathering) logic

**Key Responsibilities:**

* Defines the universal prompts for the Supervisor-Planner in the Intergrax framework
* Specifies the decomposition-first mandate, primary principles, and component selection policy
* Provides default plan system and user templates as dataclass instances
* Exposes these templates through a `SupervisorPromptPack` class

**Note:** This file appears to be a well-documented, production-ready part of the Intergrax framework.

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module implements the state graph construction and node factory for the LangGraph pipeline in the Intergrax framework. It manages global state traveling through the pipeline, resolves input values for each plan step, persists outputs, and constructs a graph topology based on dependencies between steps.

**Domain:** Supervision/Plan Execution

**Key Responsibilities:**

*   Manages global state traveling through the pipeline
*   Resolves input values for each plan step using conventions for artifact lookup
*   Persists node results into artifacts
*   Constructs readable node names from step titles
*   Creates LangGraph nodes as functions built by `make_node_fn` based on plan steps and components
*   Builds a graph topology in a stable topological order based on dependencies between steps

The module appears to be fully functional, well-documented, and follows best practices for structure and naming conventions. It seems to be an integral part of the Intergrax framework's LangGraph pipeline implementation.

### `intergrax\system_prompts.py`

Description: This module defines a strict RAG system instruction for knowledge assistants to provide accurate and precise answers based on document content.

Domain: RAG (Retrieve And Generate) logic

Key Responsibilities:
* Provide clear guidelines for knowledge assistants
* Define the role of the assistant as a retriever and generator of information
* Specify the sources of information (documents in the vector store)
* Outline the procedure for answering questions, including searching, verifying consistency, and providing references
* Emphasize the importance of precision, accuracy, and referencing sources
* Provide examples of how to format responses and citations.

### `intergrax\tools\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the Intergrax framework, importing and initializing necessary modules.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes core components of the Intergrax framework.
• Imports essential modules.
• Sets up framework configuration.

### `intergrax\tools\tools_agent.py`

**Description:** The `ToolsAgent` module is responsible for orchestrating interactions between the user and various tools within the Intergrax framework. It provides a high-level API for running tools in a sequence, handling communication with the LLM, and generating output based on tool results.

**Domain:** LLM Adapters / Tools Orchestration

**Key Responsibilities:**

* Initialize the `ToolsAgent` instance with an LLM adapter, tool registry, memory, and configuration
* Prune messages to comply with OpenAI's requirements for native tools
* Build output structure by prioritizing full tool results or extracted JSON from answer text
* Run the tools orchestration loop based on input data (either a list of chat messages or a string)
* Handle streaming mode and tool choice options
* Return a dictionary containing the final answer, tool traces, and messages used in the tools loop

### `intergrax\tools\tools_base.py`

**Description:** This module provides core functionality for tool registration and management within the Integrax framework.

**Domain:** Tooling/Utilities

**Key Responsibilities:**

* Provides a `ToolBase` class that serves as a base class for all tools, offering essential attributes and methods such as `name`, `description`, `schema_model`, and `run`.
* Offers a `ToolRegistry` class to store and manage registered tool instances, supporting registration, retrieval by name, listing of available tools, and exporting tools in OpenAI-compatible format.
* Includes the `_limit_tool_output` function for safely truncating long tool outputs to prevent context overflow issues.

### `intergrax\websearch\__init__.py`

Description: The websearch package serves as an entry point for integrating external search functionalities within the Intergrax framework.

Domain: Web Search Integration

Key Responsibilities:
• Registers a blueprint for handling search queries
• Configures external search API interactions
• Initializes any required dependencies or services

### `intergrax\websearch\cache\__init__.py`

**Description:** This module provides an in-memory query cache for web search results with optional time-to-live (TTL) and maximum size.

**Domain:** Web Search Cache

**Key Responsibilities:**

* Provides a simple in-memory cache to store web search results
* Supports caching of already serialized web documents as a list of dictionaries
* Allows setting a TTL for cache entries, after which they will be automatically removed
* Has an optional maximum number of entries stored in memory, with the oldest entry being evicted when this limit is reached
* Includes utility classes for creating unique cache keys and storing cached results

### `intergrax\websearch\context\__init__.py`

**File:** `intergrax\websearch\context\__init__.py`

**Description:** Initializes the web search context and provides a basic implementation for searching within the Intergrax framework.

**Domain:** Web Search Context Initialization

**Key Responsibilities:**
* Initializes the web search context
* Provides a basic implementation for searching

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides utilities for building LLM-ready textual context and chat messages from web search results. It offers two primary functions: building a context string from web documents or serialized dicts, as well as creating system and user prompts for chat-style LLMs.

**Domain:** Web Search & LLM Integration

**Key Responsibilities:**

* Building LLM-ready textual context strings from web documents
* Creating system and user prompts for chat-style LLMs
* Handling serialized web documents (dicts produced by WebSearchExecutor)
* Configurable parameters for building context and prompts (e.g., max documents, characters per document, including snippets and URLs)

Note: The code appears to be well-structured, and there are no obvious issues or warnings. It is likely a production-ready module within the Intergrax framework.

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: This module initializes the web search fetcher, responsible for retrieving relevant documents from external sources.

DOMAIN: Web Search Fetcher

KEY RESPONSIBILITIES:
* Initializes the fetcher instance
* Sets up connection to external search engines
* Defines API endpoints and request parameters

### `intergrax\websearch\fetcher\extractor.py`

**Description:** 
This module provides functionality for extracting metadata and content from web pages, with two main methods: `extract_basic` for lightweight HTML extraction and `extract_advanced` for more thorough readability-based extraction.

**Domain:** Web Search

**Key Responsibilities:**

* Extract basic metadata (title, meta description, language, Open Graph tags)
	+ extract_basic function
* Perform advanced readability-based extraction
	+ extract_advanced function
* Remove boilerplate elements and normalize whitespace
* Handle exceptions and fallback to manual HTML cleanup and extraction when necessary
* Attach extraction metadata for observability and debugging

The code appears well-structured and follows best practices. However, it's worth noting that the `extract_basic` function is intentionally conservative, whereas the `extract_advanced` function performs more thorough extraction with potential performance implications. Additionally, there are some try-except blocks to handle exceptions, but it would be beneficial to provide more detailed error messages for debugging purposes.

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module provides a high-level interface for fetching web pages over HTTP, encapsulating common concerns like header management and redirect following.

**Domain:** Web Search (HTTP Fetcher)

**Key Responsibilities:**

* Performs an asynchronous HTTP GET request with sane defaults
* Captures final URL, status code, raw HTML, and body size
* Keeps higher-level concerns (robots, throttling, extraction) outside of the fetch logic
* Handles transport-level failures and returns None on error

### `intergrax\websearch\integration\__init__.py`

DESCRIPTION: This module initializes the web search integration, setting up necessary components for external API connections.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
• Initializes web search integrations
• Sets up API connection parameters
• Defines default integration settings 
• Enables or disables specific integration features

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** 
This module provides a web search functionality by encapsulating configuration and operations within a LangGraph-compatible node wrapper. The node delegates to an externally configured or internally created WebSearchExecutor instance.

**Domain:** LLM adapters 

**Key Responsibilities:**
- Provides a LangGraph-compatible web search node wrapper (`WebSearchNode`)
- Exposes synchronous and asynchronous node methods for operating on the `WebSearchState` contract
- Encapsulates configuration of the underlying WebSearchExecutor instance
- Supports lazy construction of a default WebSearchNode instance for convenience and backward compatibility

### `intergrax\websearch\pipeline\__init__.py`

DESCRIPTION: This module initializes the web search pipeline by defining its core components and setting up the necessary configuration.

DOMAIN: Pipeline Configuration

KEY RESPONSIBILITIES:
* Initializes the pipeline with default settings
* Registers pipeline components and services
* Configures pipeline data processing and output settings

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module orchestrates a multi-provider web search pipeline for fetching and extracting relevant documents from the internet.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Orchestrates search across multiple providers using the `search_all` method
* Fetches and extracts relevant documents from the top-N hits using the `fetch_and_extract` method
* Deduplicates documents based on a simple dedupe key
* Provides quality scoring for fetched documents
* Offers synchronous convenience wrapper around the async pipeline execution via the `run_sync` method

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: This module serves as the entry point for web search providers in Intergrax, enabling external integrations with various search engines.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Registers available web search provider classes
- Allows easy addition of new providers through the registry mechanism

### `intergrax\websearch\providers\base.py`

**Description:** This module provides a base interface for web search providers in the Integrax framework.

**Domain:** Web Search Providers

**Key Responsibilities:**
- Provides an abstract base class `WebSearchProvider` with a stable interface for all web search providers
- Exposes minimal capabilities for feature negotiation (language, freshness)
- Requires subclasses to implement `search`, `capabilities`, and optional `close` methods

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module implements a provider for Bing Web Search (v7), enabling the framework to interact with the Bing search API. It handles tasks such as authentication, parameter construction, and data retrieval.

**Domain:** LLM adapters

**Key Responsibilities:**
- Authenticates with the Bing API using an environment variable or provided API key
- Constructs query parameters for filtering by language, region, freshness, and safe search settings
- Performs GET requests to the Bing API endpoint with constructed headers and parameters
- Parses JSON responses to extract relevant data (web pages)
- Maps extracted data into SearchHit objects for further processing

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module provides a Google Custom Search (CSE) provider for the Intergrax framework, enabling web search capabilities through the CSE REST API.

**Domain:** Web Search Providers

**Key Responsibilities:**

* Initializes the Google CSE provider with required environment variables and settings
* Builds query parameters for the CSE API request
* Handles language filtering using 'lr' (content language) and 'hl' (UI language)
* Ignores freshness parameter as it's not natively supported by CSE
* Maps search results from the CSE API to SearchHit objects
* Searches the web using the built query parameters and maps results to SearchHit objects

Note: The file appears to be a standard part of the Intergrax framework, with proper structure and documentation. It does not appear to be experimental, auxiliary, or legacy.

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a Google Places provider for the Intergrax framework, allowing it to fetch and process data from Google's Places API.

**Domain:** Web Search Providers

**Key Responsibilities:**

* Provides a Google Places provider class (`GooglePlacesProvider`) that extends `WebSearchProvider`
* Implements functionality to perform text search and details lookup using the Google Places API
* Handles environment variables, such as `GOOGLE_PLACES_API_KEY`, for API key authentication
* Supports features like language and region filtering, and freshness (although not applicable in this case)
* Includes methods for building query parameters, fetching place details, and mapping results to SearchHit objects

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module implements a Reddit search provider for the Intergrax framework, utilizing the official OAuth2 API.

**Domain:** Websearch providers (specifically, Reddit-based web search)

**Key Responsibilities:**

* Provides full-featured Reddit search capabilities
* Authenticates using client_credentials OAuth2 flow
* Supports rich post metadata and optional comment fetching
* Handles query parameterization and parsing of search results
* Integrates with Intergrax's websearch infrastructure

Note: This module appears to be a fully-fledged, production-ready component.

### `intergrax\websearch\schemas\__init__.py`

Description: This module defines the initial schema configuration for web search functionality in Intergrax.

Domain: Web Search Schemas

Key Responsibilities:
• Defines the base schema for web search functionality.
• Configures default fields and data types for search queries.
• Initializes schema-related constants and metadata. 

Note: The provided content is minimal, suggesting this module might be a basic entry point or an initialization file.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass `PageContent` to represent the fetched and extracted content of a web page, including metadata and derived information.

**Domain:** Web search schema

**Key Responsibilities:**

* Represents the content of a web page with various attributes (e.g., URL, HTTP status code, HTML, text, metadata)
* Provides methods for filtering empty or failed fetches (`has_content`)
* Allows truncated summaries of the content for logging and debugging purposes (`short_summary`)
* Calculates an approximate size of the content in kilobytes (`content_length_kb`)

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a dataclass for canonical search query specifications used by web search providers.

**Domain:** Query schema

**Key Responsibilities:**
* Defines the `QuerySpec` dataclass with fields for raw user query, top results per provider, locale, region, language, freshness, site filter, and safe search.
* Provides methods to normalize the query string (`normalized_query`) and cap the number of top results (`capped_top_k`).

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a dataclass `SearchHit` for representing search result metadata in an interoperable and provider-agnostic manner.

**Domain:** Search metadata schemas

**Key Responsibilities:**

* Defines a frozen dataclass `SearchHit` with fields for provider ID, query string, rank, title, URL, snippet, display link, publication date, source type, and extra fields.
* Enforces minimal safety checks in the `__post_init__` method to ensure valid URLs and ranks.
* Provides methods for extracting domain from the URL (`domain`) and converting search hits to a minimal dictionary format (`to_minimal_dict`).

### `intergrax\websearch\schemas\web_document.py`

Description: This module provides a unified data structure for representing web documents, integrating metadata from search hits with extracted content and analysis results.

Domain: Web Search Schema

Key Responsibilities:
- Represents a fetched and processed web document as a dataclass.
- Integrates original search hit metadata with extracted page content and analysis results.
- Provides methods for validating the document's validity, merging textual content, and generating summary lines.

### `intergrax\websearch\service\__init__.py`

DESCRIPTION: This module initializes the web search service by setting up necessary components and dependencies.

DOMAIN: Web Search Service

KEY RESPONSIBILITIES:
• Sets up the web search service instance
• Defines service-specific configuration
• Registers required dependencies for the web search service
• Initializes the service's internal state

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** 
This module provides a high-level helper class, `WebSearchAnswerer`, which conducts web searches via the `WebSearchExecutor` and builds messages from search results for input into LLM (Large Language Model) adapters.

**Domain:** RAG logic (Retrieval-Augmented Generation)

**Key Responsibilities:**
- Conducts web search via `WebSearchExecutor`
- Builds messages from search results using a context builder
- Calls an LLM adapter to generate a final answer
- Provides both asynchronous and synchronous interfaces for answering questions

### `intergrax\websearch\service\websearch_executor.py`

Description: This module provides a high-level web search executor that can be configured to use various web search providers and cache results.

Domain: Web Search Executor

Key Responsibilities:
- Construct QuerySpec from a raw query and configuration.
- Execute SearchAndReadPipeline with chosen providers.
- Convert WebDocument objects into LLM-friendly dicts.
- Build a simple, deterministic signature of the provider configuration for caching purposes.
- Serialize web documents into dicts suitable for LLM prompts and logging.

### `intergrax\websearch\utils\__init__.py`

Description: This utility module initializes and configures the web search functionality within the Intergrax framework.

Domain: Utility Modules

Key Responsibilities:
- Initializes web search utilities
- Configures search engine settings
- Provides basic search function implementation

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module contains utility functions for deduplication in the web search pipeline, including normalizing text and generating stable SHA-256 based keys.

**Domain:** Dedupe Utilities

**Key Responsibilities:**

* Normalizes text by converting it to lower case, stripping whitespace, and collapsing internal whitespace sequences
* Generates a stable SHA-256 based deduplication key for the given text using the normalized text as input
* Intentionally simple and fast, with heavy normalization (e.g. stemming, punctuation removal) intended to be done elsewhere if needed

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a simple asyncio-compatible token bucket rate limiter, designed to control the rate of concurrent operations.

**Domain:** Rate limiting utilities

**Key Responsibilities:**
* Provides a `TokenBucket` class for rate limiting with adjustable refill rate and capacity
* Offers two main methods:
	+ `acquire(tokens=1)`: Waits until at least `tokens` are available and consumes them, ensuring the average request rate does not exceed the specified limit.
	+ `try_acquire(tokens=1)`: Non-blocking attempt to consume `tokens`, returning whether tokens were available or not.

### `main.py`

Description: This module serves as the entry point for executing the Intergrax framework.

Domain: Application Launcher

Key Responsibilities:
• Provides a simple command-line interface to initiate the framework's execution.
• Prints a greeting message indicating the framework's startup.

### `mcp\__init__.py`

DESCRIPTION: This is the top-level initialization file for the MCP package, responsible for importing and setting up core modules.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
- Initializes core module imports
- Sets up package structure and dependencies

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

**Description:** This is a Jupyter notebook that serves as a basic sanity-check for the Drop-In Knowledge Mode runtime in the Intergrax framework. It demonstrates how to create and load sessions, append user and assistant messages, build conversation history from SessionStore, and return a RuntimeAnswer object.

**Domain:** RAG logic

**Key Responsibilities:**

* Verify the functionality of the DropInKnowledgeRuntime class
* Create or load a session using InMemorySessionStore for simplicity
* Append user and assistant messages to the session
* Build conversation history from SessionStore
* Return a RuntimeAnswer object

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:**
This notebook demonstrates the usage of Intergrax's Drop-In Knowledge Mode runtime, specifically how to work with sessions, attachments, and ingestion using Ollama + LangChain adapters.

**Domain:** RAG logic, LLM adapters

**Key Responsibilities:**

* Initialize an in-memory session store for notebook testing
* Set up an LLM adapter using Ollama + LangChain
* Configure embedding manager (Ollama embeddings) and vector store manager (Chroma as a vector store)
* Create a RuntimeConfig instance
* Instantiate the DropInKnowledgeRuntime with the specified configuration
* Prepare an AttachmentRef for a local project document to simulate attachment ingestion

**Note:** This notebook appears to be a demonstration or example code, intended to showcase specific features of the Intergrax framework.

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of the `ContextBuilder` component within the Intergrax framework's Drop-In Knowledge Mode runtime. It provides a step-by-step guide on how to integrate the RAG (Retrieval-Augmented Generation) context builder into the existing runtime.

**Domain:** LLM adapters, RAG logic

**Key Responsibilities:**

* Initializes components required for testing `ContextBuilder`
	+ In-memory session store
	+ LLM adapter (Ollama-based)
	+ Embedding manager (same model as ingestion pipeline)
	+ Vector store manager (Chroma, same collection as before)
	+ Runtime config (RAG will be enabled in the next cell)
* Demonstrates how to wire `ContextBuilder` into Drop-In Knowledge Mode runtime using minimal components
* Explains configuration knobs that affect RAG (e.g., max_docs_per_query, max_history_messages, optional score threshold)
* Builds context for a single user question using `ContextBuilder.build_context(session, request)`
* Inspects the result of context building (no LLM call yet)

**Note:** This notebook is part of an ongoing experiment and should be treated as auxiliary documentation until further integration with the main runtime engine is completed.

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

**Description:** This notebook demonstrates how to use the DropInKnowledgeRuntime with session-based chat, optional RAG (attachments ingested into a vector store), and live web search via WebSearchExecutor.

**Domain:** Drop-in knowledge runtime, chat engine, RAG logic, web search integration

**Key Responsibilities:**

* Initializes core runtime configuration (LLM + embeddings + vector store + web search)
	+ Sets up session store for in-memory storage of chat messages and metadata
	+ Configures LLM adapter using Ollama through LangChain adapter
	+ Initializes embedding manager with specific model and dimensions
	+ Configures vector store with Chroma provider and collection name
	+ Wraps one or more web search providers (Google CSE, Bing) in WebSearchExecutor
* Creates a fresh chat session for the web search demo
	+ Sets up runtime configuration with specified options (LLM label, embedding label, vector store label)
	+ Enables RAG and web search features
	+ Initializes DropInKnowledgeRuntime with configured runtime, session store, and providers

**Notes:**

This notebook appears to be a demonstration of the Intergrax framework's capabilities in drop-in knowledge mode. It showcases how to integrate various components (LLM adapter, embeddings, vector store, web search) into a single chat engine. The code is well-structured, and comments provide helpful explanations for each section. Overall, this notebook serves as a valuable resource for understanding the Intergrax framework's architecture and its potential applications in building conversational AI systems.

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

**Description:** This Jupyter notebook demonstrates the usage of the Intergrax framework's Drop-In Knowledge Runtime with a tools orchestration layer, showcasing how to integrate tools into a ChatGPT-like flow.

**Domain:** LLM adapters and RAG logic with data ingestion and tools integration.

**Key Responsibilities:**

* Configures Python path for `intergrax` package import
* Loads environment variables (API keys, etc.)
* Initializes Drop-In Knowledge Runtime components:
	+ Session store for chat message & metadata storage
	+ Ollama-based LLM adapter
	+ Runtime configuration with compact setup
* Demonstrates tools integration using Intergrax tools framework:
	+ Registers demo tools (`WeatherTool`, `CalcTool`) in a `ToolRegistry`
	+ Creates an `IntergraxToolsAgent` instance using an Ollama-based LLM
	+ Attaches the agent to `RuntimeConfig.tools_agent`

**Note:** This file appears to be a demonstration notebook, showcasing how to integrate tools into the Intergrax framework's Drop-In Knowledge Runtime.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates the use of Intergrax and LangGraph for building a hybrid multi-source RAG (Recurrent Attention Graph) pipeline, which combines local files and web search results into a single in-memory vector index.

**Domain:** RAG logic

**Key Responsibilities:**

* Ingest content from multiple sources:
	+ Local PDF files
	+ Local DOCX/Word files
	+ Live web results using the Intergrax `WebSearchExecutor`
* Build a unified RAG corpus:
	+ Normalize documents into a common internal format
	+ Attach basic metadata about origin (pdf / docx / web)
	+ Split documents into chunks suitable for embedding
* Create an in-memory vector index:
	+ Use an Intergrax embedding manager (e.g. OpenAI / Ollama)
	+ Store embeddings in an **in-memory Chroma** collection via Intergrax vectorstore manager
	+ Keep everything ephemeral (no persistence, perfect for “ad-hoc research” scenarios)
* Answer user questions with a RAG agent:
	+ The user provides a natural language question
	+ LangGraph orchestrates the flow: load → merge → index → retrieve → answer
	+ An Intergrax `RagAnswerer` (or `WindowedAnswerer`) generates a single, structured report

**Note:** This notebook appears to be a demonstration or proof-of-concept code and may not be part of the mainline Intergrax framework. It showcases how to combine local files + web search in a single RAG pipeline and plugs Intergrax components into a LangGraph `StateGraph`.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This notebook provides a minimal integration between Intergrax and LangGraph, demonstrating how to use an Intergrax LLM adapter as a node inside a LangGraph graph.

Domain: LLM adapters

Key Responsibilities:
- Initialize the OpenAI API client.
- Define a simple `State` class that holds messages and an answer.
- Implement a `llm_answer_node` function that calls the Intergrax LLM adapter to generate answers.
- Build a sample graph with a single node, demonstrating how to use the `llm_answer_node` function.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

Description: This notebook demonstrates a practical web research agent built from Intergrax components and LangGraph. It orchestrates the flow as a multi-step graph, normalizing user questions, running web search, building context, and generating final answers.

Domain: Web Research Agent

Key Responsibilities:
- Normalizes user questions
- Runs web search using Intergrax WebSearchExecutor
- Builds context from search results
- Generates final answers with sources

### `notebooks\openai\rag_openai_presentation.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of the Intergrax framework for loading a local folder into an OpenAI Vector Store and testing RAG (Relevant Answer Generation) queries.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

- Loads a local folder containing text documents into an OpenAI Vector Store
- Creates an instance of `IntergraxRagOpenAI` to interact with the Vector Store
- Ensures the existence and clears the Vector Store and storage as needed
- Uploads the local folder contents to the Vector Store
- Tests RAG queries using sample questions and prints answers

**Note:** This file appears to be a demonstration or testing notebook, rather than a production-ready implementation. It includes code for loading environment variables from a `.env` file and assumes specific configuration settings are set.

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This Jupyter Notebook defines a high-level hybrid chat agent that integrates RAG (Retrieval-Augmented Generation) logic, LLM adapters, conversational memory, and tooling for a multi-modal conversation management system.

**Domain:** Agents / Hybrid Agents

**Key Responsibilities:**

* Initializes the Intergrax framework components:
	+ LLM adapter registry
	+ Conversational memory
	+ Vector store manager
	+ Embedding manager
	+ Reranker
	+ Retriever
	+ Answerer
* Configures and registers available tools, such as a demo weather tool
* Defines RAG routing logic for document-based queries
* Creates a high-level hybrid agent (RAG + tools + LLM chat) with specified components and configuration
* Demonstrates test questions and responses using the configured agent

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the use of the Intergrax framework in a simple conversational scenario where an agent answers weather-related questions. It showcases how to define structured output models using Pydantic and leverage various Intergrax components such as the tools agent, LLM adapters, RAG (Retrieval-Augmented Generation) logic, and vector stores.

**Domain:** RAG logic

**Key Responsibilities:**

- Defines a structured output model `WeatherAnswer` for storing weather query results.
- Implements a basic `WeatherTool` class that returns demo data in the format of the `WeatherAnswer` schema.
- Demonstrates how to use the tools agent to orchestrate LLM reasoning, tool selection and invocation, and structured response generation.
- Provides an example usage of the RAG system with Pydantic models for structured output.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This notebook demonstrates a workflow for loading documents from a directory, splitting them into smaller chunks, creating embeddings using Ollama, and storing these embeddings in a Chroma vectorstore. It covers loading metadata, chunking documents, embedding creation, and vector store presence checks.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Loading documents from a directory into a unified structured format
* Splitting loaded documents into smaller chunks for optimal embedding granularity
* Creating vector embeddings for each document chunk using Ollama
* Storing these embeddings in a Chroma vector store

### `notebooks\rag\rag_multimodal_presentation.ipynb`

Description: This Jupyter notebook demonstrates the integration of multimodal documents into the Intergrax framework, showcasing document loading, splitting, embedding, and vector store management.

Domain: RAG logic (Retriever-Augmented Generator)

Key Responsibilities:
- Load multimodal documents (video, audio, image) using DocumentsLoader
- Split and embed documents using DocumentsSplitter and EmbeddingManager
- Manage vector store using VectorstoreManager
- Perform retriever test using RagRetriever

Note: This notebook appears to be a demonstration of the Intergrax framework's capabilities rather than an experimental or auxiliary file.

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This notebook demonstrates the usage of various multimedia processing utilities within the Intergrax framework, including video and audio downloading, transcription, frame extraction, and image description using AI models.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloads videos from YouTube using `yt_download_video` function
* Transcribes video to VTT format using `transcribe_to_vtt` function
* Extracts frames and metadata from a video using `extract_frames_and_metadata` function
* Translates audio using `translate_audio` function
* Describes images using Ollama model in `transcribe_image` function
* Extracts frames from a video and transcribes images using `yt_download_video` and `transcribe_image` functions

**Note:** This notebook appears to be a demonstration or example code for the Intergrax framework's multimedia processing capabilities. It does not seem to be an experimental, auxiliary, legacy, or incomplete file.

### `notebooks\rag\tool_agent_presentation.ipynb`

Description: This notebook contains a demonstration of the Integrax framework's tools agent, showcasing its ability to interact with various external tools and provide relevant outputs.

Domain: RAG logic (Reasoning Augmented Generation)

Key Responsibilities:
- Demonstrates the integration of the tools agent with conversational memory.
- Implements a weather tool (`WeatherTool`) and a calculator tool (`CalcTool`).
- Registers these tools in the `ToolRegistry`.
- Configures an LLM (LLama) for reasoning.
- Creates a ToolsAgent instance, which orchestrates the interaction between the LLM and the registered tools.
- Runs two tests: one using the weather tool to fetch information about Warsaw's current weather, and another using the calculator tool to evaluate a simple arithmetic expression.

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This notebook contains a collection of Python functions that implement various components for the Integrax framework, including compliance checking, cost estimation, final summary generation, and financial auditing. These components are designed to be executed within a pipeline workflow.

**Domain:** LLM adapters / RAG logic (supervisor components)

**Key Responsibilities:**

* Compliance Checker:
	+ Verifies whether proposed changes comply with privacy policies and terms of service
	+ Returns findings on compliance status and any detected policy violations
* Cost Estimator:
	+ Estimates the cost of changes based on UX audit reports
	+ Returns a mock formula-based estimate (base + per-issue * number_of_issues)
* Final Summary:
	+ Generates a consolidated summary report using all collected artifacts
	+ Includes status pipeline, terminated by, terminate reason, PM decision, and other relevant information
* Financial Audit:
	+ Generates a mock financial report with VAT calculation (test data)
	+ Returns a dictionary containing net value, VAT rate, VAT amount, gross value, currency, and budget last quarter

Note: This notebook appears to be a collection of functional components rather than experimental or auxiliary code. However, the use of "mock" in some functions suggests that these components may not be fully implemented or may require additional development for real-world usage.

### `notebooks\websearch\websearch_presentation.ipynb`

Description: This Jupyter notebook provides an example implementation of web search using the Google Custom Search API, showcasing how to retrieve and display search results.

Domain: Web Search

Key Responsibilities:
- Configures environment variables for Google Custom Search API keys.
- Defines a query specification for searching LangGraph and LangChain.
- Uses the GoogleCSEProvider to perform a search and retrieve top-ranked hits.
- Prints out details of each search hit, including title, URL, snippet, provider, rank, and domain.
