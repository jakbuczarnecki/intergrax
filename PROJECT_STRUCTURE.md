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
- `intergrax\memory\long_term_memory.py`
- `intergrax\memory\long_term_memory_manager.py`
- `intergrax\memory\long_term_memory_store.py`
- `intergrax\memory\organization_profile_manager.py`
- `intergrax\memory\organization_profile_memory.py`
- `intergrax\memory\organization_profile_store.py`
- `intergrax\memory\stores\__init__.py`
- `intergrax\memory\stores\in_memory_conversational_store.py`
- `intergrax\memory\stores\in_memory_long_term_memory_store.py`
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
- `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`
- `intergrax\runtime\drop_in_knowledge_mode\rag_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`
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

**FILE: api/__init__.py**

**Description:** Initializes and sets up the API framework for Intergrax, registering necessary components and routes.

**Domain:** API Framework Initialization

**Key Responsibilities:**
* Registers API blueprints and routes
* Sets up global configuration and dependencies
* Initializes logging and error handling mechanisms

### `api\chat\__init__.py`

Description: The __init__.py file in the api/chat directory initializes and sets up the chat API.

Domain: LLM adapters

Key Responsibilities:
* Initializes the chat API module.
* Sets up imports for the chat API functionality.

### `api\chat\main.py`

Description: This module provides the main entry point for the Integrax chat API, handling incoming queries and document uploads.

Domain: Chat API

Key Responsibilities:

* Handling incoming query requests through the `/chat` endpoint
	+ Extracting user input and session data
	+ Retrieving and setting up AI model based on user query
	+ Processing user query and retrieving answer from AI model
* Handling document uploads through the `/upload-doc` endpoint
	+ Checking file type and size restrictions
	+ Saving uploaded files to temporary storage and indexing them in Chroma
* Providing endpoints for listing and deleting documents
	+ Retrieving list of all documents via the `/list-docs` endpoint
	+ Deleting documents by ID via the `/delete-doc` endpoint

Note: The file appears to be a main entry point for the chat API, and its functionality is well-implemented. However, there are some error handling mechanisms in place that could potentially be improved upon (e.g., the generic 500 error messages). Overall, this file seems to be a central component of the Integrax framework's chat API functionality.

### `api\chat\tools\__init__.py`

**Description:** This is the entry point for the chat API tools package, responsible for initializing and exporting various utility functions.

**Domain:** Chat API utilities

**Key Responsibilities:**
- Initializes and exports chat API tools
- Provides interface for importing and using utility functions 
- Acts as a container for related modules

### `api\chat\tools\chroma_utils.py`

Description: This module provides utility functions for interacting with the Chroma vector store, enabling operations such as loading and splitting documents, indexing documents, and deleting documents.

Domain: RAG logic

Key Responsibilities:
- Load and split documents from a file path
- Index documents to Chroma vector store with provided file ID
- Delete documents from Chroma vector store by file ID

### `api\chat\tools\db_utils.py`

**Description:** This module provides database utilities for the Integrax framework, including connection management, schema creation and migration, and public API functions for message and document storage.

**Domain:** Database Utilities

**Key Responsibilities:**

* Connects to a SQLite database using `sqlite3`
* Creates and populates database tables with initial data
* Migrates from legacy application logs to the new messages table
* Provides public API functions for:
	+ Ensuring sessions exist in the database
	+ Inserting messages into the database
	+ Retrieving messages from the database
	+ Getting history pairs (user-text, assistant-text) from the database
	+ Managing document storage and retrieval
* Includes backward-compat entry points for legacy application logs integration

Note: The code appears to be well-structured and organized, with clear docstrings and comments explaining its functionality. There are no signs of experimental or auxiliary code that would indicate it as incomplete or legacy.

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines Pydantic models for API requests and responses, enabling robust data validation and serialization in the Integrax framework.

**Domain:** LLM adapters / API utilities

**Key Responsibilities:**

* Defines enumerations for model names
* Models for query inputs (e.g., question, session ID, model selection)
* Models for query responses (e.g., answer, session ID, used model)
* Model for document information (e.g., file ID, filename, upload timestamp)
* Model for delete file requests (e.g., file ID)

### `api\chat\tools\rag_pipeline.py`

Description: This module provides a pipeline for building and managing components of the RAG (Retrieval-Augmented Generation) model, including vector store management, embedding, retriever, reranker, and LLM adapters.

Domain: RAG logic

Key Responsibilities:
- Manages vector store and embedder instances using singleton design pattern.
- Creates retriever and reranker instances with configurable settings.
- Builds LLM adapters for different models.
- Provides default user and system prompts for the chat interface.

### `applications\chat_streamlit\api_utils.py`

Description: This module provides API utility functions for interacting with the Integrax framework's backend, including making requests to various endpoints and handling file uploads and deletions.

Domain: API Utilities (specifically for chat-related functionality)

Key Responsibilities:
* Making POST requests to the "chat" endpoint to retrieve API responses
* Uploading files using a POST request to the "upload-doc" endpoint
* Listing documents using a GET request to the "list-docs" endpoint
* Deleting documents by making a POST request to the "delete-doc" endpoint with a file ID

### `applications\chat_streamlit\chat_interface.py`

**Description:** This module provides a user interface for interacting with the chat-based conversational AI, utilizing Streamlit and integrating with the Intergrax framework's API utilities.

**Domain:** Chat Interface

**Key Responsibilities:**
- Displaying user and assistant messages in a chat format
- Handling user input via a chat input field
- Sending user queries to the API and displaying responses
- Displaying response details, including generated answer, model used, and session ID

### `applications\chat_streamlit\sidebar.py`

Description: This module provides a Streamlit sidebar for the chat application, enabling users to select models, upload documents, manage uploaded documents, and delete selected files.

Domain: Chat Application Interface

Key Responsibilities:
- Provides model selection component in the sidebar
- Enables uploading of documents with metadata display
- Allows listing and refreshing of uploaded documents
- Includes deletion functionality for individual documents

### `applications\chat_streamlit\streamlit_app.py`

**streamlit_app.py**

Description: This module serves as the entry point for the Intergrax framework's Streamlit-based chat application, integrating a sidebar and chat interface.

Domain: Chat Application

Key Responsibilities:
- Initializes Streamlit session state with message storage and session ID tracking
- Displays sidebar using `display_sidebar` function from `sidebar` module
- Displays chat interface using `display_chat_interface` function from `chat_interface` module

### `applications\company_profile\__init__.py`

**applications/company_profile/__init__.py**

Description: Initializes the company profile application, providing a foundation for its functionality and dependencies.

Domain: Application Initialization

Key Responsibilities:
- Registers application routes.
- Imports and initializes dependent modules (e.g., data providers).
- Sets up necessary configurations.
- Exposes APIs for external access (if applicable).

### `applications\figma_integration\__init__.py`

Description: This is the entry point for Figma integration into the Intergrax framework, enabling seamless collaboration between design and development teams.

Domain: Integration

Key Responsibilities:
- Initializes Figma API connections
- Defines interfaces for data exchange between Figma and Intergrax
- Sets up event listeners for real-time updates from Figma

### `applications\ux_audit_agent\__init__.py`

**applications\ux_audit_agent\__init__.py**

Description: Initializes and configures the UX audit agent.

Domain: Agents

Key Responsibilities:
* Registers the UX audit agent with the main application
* Sets up event listeners for UX-related activities
* Exposes configuration options for customizing audit behavior

### `applications\ux_audit_agent\components\__init__.py`

Description: The ux_audit_agent component's initialization script, responsible for setting up the necessary components and configurations.

Domain: Agents

Key Responsibilities:
* Initializes the UX audit agent component
* Sets up dependencies and configuration for the component
* Defines exportable functions or classes for the component

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module provides a compliance checker component for the Integrax framework, which evaluates proposed changes against privacy policies and regulations.

**Domain:** Compliance Checker (RAG logic)

**Key Responsibilities:**
- Evaluates proposed UX changes against privacy policy and regulatory rules.
- Simulates compliance result with an 80% chance of being compliant.
- Returns findings on compliance status, policy violations, and notes for correction or review.
- Optionally stops execution if non-compliant to require corrections.

### `applications\ux_audit_agent\components\cost_estimator.py`

Description: This module defines a component for estimating the cost of UX-related changes based on an audit report, using a mock pricing model.

Domain: RAG logic

Key Responsibilities:
- Provides a cost estimation function as a component.
- Uses the audit report to calculate estimated costs.
- Applies a basic pricing model (base + per-issue).
- Returns the estimate, including currency and method.

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module defines a pipeline component responsible for generating a comprehensive final report of the entire execution pipeline.

**Domain:** UX Audit Agent Components

**Key Responsibilities:**

* Generates a complete summary of the entire execution pipeline using all collected artifacts.
* Produces a final report as output, containing key metrics and statuses.
* Triggers always at the final stage (finally block) of the pipeline.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module defines a component for generating mock financial reports and VAT calculations within the Integrax framework.

**Domain:** Financial Reporting Agents

**Key Responsibilities:**

* Provides a "Financial Agent" component for testing financial computations
* Generates test data for mock financial reports and VAT calculations
* Offers example use cases for budget constraints, cost reports, and gross value calculations
* Returns a ComponentResult with the generated report and logs

### `applications\ux_audit_agent\components\general_knowledge.py`

Description: This module defines a component for answering general questions about the Intergrax system, providing information on its structure, features, and configuration.

Domain: LLM adapters

Key Responsibilities:
- Provides pre-defined responses to general user queries.
- Returns mock data for testing purposes (e.g., fake documents).
- Integrates with the Intergrax framework through the `@component` decorator.

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module implements a project management component for the UX audit pipeline, providing a mock decision-making process based on random selection.

**Domain:** RAG logic (Risk, Action, and Recommendation)

**Key Responsibilities:**

* Provides a project manager decision-making model that randomly approves or rejects proposals
* Generates decision notes based on the outcome
* Allows for stopping the pipeline execution if rejected
* Returns a ComponentResult object with relevant data and logs

### `applications\ux_audit_agent\components\ux_audit.py`

**applications\ux_audit_agent\components\ux_audit.py**

Description: This module provides a UX audit component that analyzes UI/UX based on Figma mockups and generates a sample report with recommendations.

Domain: LLM adapters

Key Responsibilities:
- Performs UX audit based on Figma mockups.
- Generates a sample report with recommendations.
- Returns a ComponentResult object containing the report.

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This Jupyter notebook, UXAuditTest.ipynb, contains a workflow that demonstrates the integration of various components in the Integrax framework. It performs UX audit on FIGMA mockups, verifies changes comply with company policy, prepares summary reports for Project Managers, evaluates financial impact of changes, and enables Project Manager decision-making.

Domain: RAG logic

Key Responsibilities:
- Performs UX audit on FIGMA mockups
- Verifies changes comply with company policy
- Prepares summary reports for Project Managers
- Evaluates financial impact of changes
- Enables Project Manager decision-making

### `generate_project_overview.py`

**Description:** This module generates an automatically structured overview of the Intergrax framework's project layout by collecting relevant source files, generating summaries via LLM adapters, and creating a Markdown file for documentation.

**Domain:** Project Structure Documentation Generator

**Key Responsibilities:**

- Recursively scans the project directory to collect all relevant source files.
- Generates summaries for each file using an LLM adapter (LangChain Ollama).
- Creates a structured Markdown file (`PROJECT_STRUCTURE.md`) detailing the project layout, including purpose, domain, and key responsibilities.

The code appears to be production-ready with proper documentation and testing in place. The use of `dataclasses` and type hints suggests a high degree of maintainability and scalability.

### `intergrax\__init__.py`

DESCRIPTION: The `__init__.py` file is the entry point for the Intergrax framework, responsible for initializing and setting up dependencies.

DOMAIN: Framework initialization

KEY RESPONSIBILITIES:

* Initializes the Intergrax framework
* Sets up dependencies for the framework's components
* Defines the entry point for the application

### `intergrax\chains\__init__.py`

Description: This module initializes and configures the chain architecture within Intergrax.

Domain: Chain management

Key Responsibilities:
* Initializes chain modules and their respective dependencies.
* Configures chain execution flow and data processing pipelines.
* Exposes API for registering and managing chains.

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This file defines a LangChain-based QA chain implementation, providing a flexible pipeline for question answering using a Retrieval-Augmented Generator (RAG) architecture. The chain consists of several stages: retrieval, reranking (optional), context building, prompt construction with hooks, LLM processing, and result parsing.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Initializes the QA chain with a retriever, LLM, and optional reranker
* Defines a sequence of runnables for each stage in the pipeline (retrieval, reranking, context building, prompt construction, LLM processing, result parsing)
* Provides hooks for modifying data at specific stages (before/after build prompt, after LLM answer)
* Offers methods for invoking the QA chain synchronously or asynchronously
* Handles input/output formatting and validation

### `intergrax\chat_agent.py`

**Description:** This module implements a chat agent with LLM routing and support for RAG, tools, and general routes.

**Domain:** LLM adapters, RAG logic, agents

**Key Responsibilities:**

* Provides a unified API for interacting with various models and routes (RAG, tools, general)
* Implements LLM-based routing to determine the best route for a given question
* Supports memory storage and streaming
* Returns a stable result structure containing answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component

**Note:** The file appears to be well-structured and complete, with clear documentation and a good balance of code and comments.

### `intergrax\llm\__init__.py`

DESCRIPTION:
This module serves as the entry point for LLM adapters in the Intergrax framework, handling initialization and setup.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
• Initializes and sets up LLM adapters
• Provides a centralized interface for adapter configuration and management

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** 
This module contains LLM adapters for interacting with various language models, providing a unified interface to different APIs and models.

**Domain:** LLM adapters

**Key Responsibilities:**

- Provides a protocol for LLM adapters (LLMAdapter)
- Offers specific implementations for OpenAI Chat Completions (OpenAIChatCompletionsAdapter)
- Defines tools-related functionality for supported adapters
- Supports structured output generation with Pydantic v2/v1 and fallback mechanisms
- Includes utility functions for message mapping, JSON schema extraction, and validation

### `intergrax\llm\messages.py`

**Description:** This module defines dataclasses for representing chat messages and attachments within the Integrax framework.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Defines `AttachmentRef` dataclass to represent lightweight references to message or session attachments
* Defines `ChatMessage` dataclass to represent universal chat messages compatible with OpenAI Responses API, including fields for tool calls and metadata
* Provides `to_dict()` method for converting `ChatMessage` objects to dictionaries compatible with OpenAI Responses API / ChatCompletions
* Implements a custom reducer function `append_chat_messages()` for merging state updates in LangGraph

**Note:** This module appears to be stable and production-ready, without any obvious signs of being experimental, auxiliary, or legacy.

### `intergrax\llm_adapters\__init__.py`

Description: This module initializes and exposes all available LLM adapters, providing a unified interface for interacting with various large language models.

Domain: LLM Adapters

Key Responsibilities:
- Exposes base classes for LLM adapters (LLMAdapter, LLMAdapterRegistry) and models (BaseModel)
- Provides concrete adapter implementations for OpenAI, Gemini, and Ollama
- Registers default adapters with the registry upon initialization

### `intergrax\llm_adapters\base.py`

Description: This module provides utilities and adapters for integrating with large language models (LLMs) within the Intergrax framework. It includes tools for handling structured output, validating model inputs, and mapping internal chat messages to OpenAI-compatible formats.

Domain: LLM adapters

Key Responsibilities:
- Providing a protocol for LLM adapters to generate and stream text based on input messages
- Offering methods for adapting internal chat messages to match OpenAI's message format
- Defining a registry for registering and creating instances of different LLM adapter factories
- Implementing tools for handling structured output, such as model validation and JSON schema generation

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module implements a minimal chat interface for the Gemini large language model.

**Domain:** LLM adapters

**Key Responsibilities:**
- Initializes a Gemini chat adapter with a provided model and optional defaults
- Splits system messages from conversation messages
- Generates responses to a sequence of chat messages using the model
- Streams generated responses as an iterable

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides a LangChain adapter for Ollama models, enabling their integration with the Integrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**
- Adapter for Ollama models used via LangChain's ChatModel interface
- No native tool-calling support; planner-style pattern recommended instead
- Conversion of internal ChatMessage list to LangChain message objects
- Generation and streaming of messages using Ollama options
- Structured output generation with JSON schema validation

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an adapter for OpenAI's Responses API, allowing it to be used with the Integrax framework. The adapter offers various methods for generating and streaming responses.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a public interface compatible with previous Chat Completions adapter
* Supports tools and tool calls
* Offers single-shot completion (non-streaming) using Responses API
* Enables streaming completion using Responses API
* Generates structured JSON output using Responses API + JSON Schema
* Validates user input against model schema

### `intergrax\logging.py`

Description: This module sets up and configures the global logger for the Integrax framework.
Domain: Logging configuration

Key Responsibilities:
- Sets the logging level to INFO, displaying INFO and higher-level messages.
- Configures the log format to include timestamp, log level, and message.
- Forces the configuration over any previous settings.

### `intergrax\memory\__init__.py`

Description: Initializes the memory module, providing a framework for data storage and retrieval within Intergrax.

Domain: Memory management

Key Responsibilities:
* Initializes the memory space
* Defines interfaces for data access and manipulation
* Sets up caching mechanisms (if applicable)

### `intergrax\memory\conversational_memory.py`

**Description:** This module provides an in-memory conversation history component, allowing for efficient storage and retrieval of chat messages.

**Domain:** RAG logic / Conversation History Management

**Key Responsibilities:**
* Keep messages in RAM
* Provide a simple API to add/extend/read/clear messages
* Optionally enforce a max_messages limit
* Prepare messages for different model backends (get_for_model)
* Do not persist data (no files, no SQLite, no external storage)

### `intergrax\memory\conversational_store.py`

**Description:** This module provides an abstract interface and base classes for managing conversational memory, including loading, saving, appending messages, and deleting sessions. It ensures deterministic persistence, idempotent write operations, and safe interaction in async environments.

**Domain:** Conversational Memory Management

**Key Responsibilities:**

* Provides an abstract interface (`ConversationalMemoryStore`) for persistent storage of conversational memory.
* Requires implementations to guarantee deterministic persistence, idempotent write operations, and safe interaction in async environments.
* Defines methods for loading, saving, appending messages, and deleting sessions.
	+ `load_memory`: Loads the full conversational history for a given session.
	+ `save_memory`: Persists the entire state of the conversational memory for a session.
	+ `append_message`: Appends a single message to persistent storage AND updates the in-memory instance accordingly.
	+ `delete_session`: Permanently removes stored history for a given session.

### `intergrax\memory\long_term_memory.py`

**Description:** This module provides data classes for representing long-term memory items and their owners, enabling efficient storage and management of semantic knowledge.

**Domain:** Long-Term Memory

**Key Responsibilities:**

* Representing long-term memory items with a globally unique identifier
* Defining ownership scope through the `MemoryOwnerRef` class
* Capturing core semantic content in the `LongTermMemoryItem` data class
* Enabling tagging and filtering for routing and classification purposes
* Providing methods for updating access timestamps and text metadata

### `intergrax\memory\long_term_memory_manager.py`

**Description:** This module serves as the high-level interface for managing long-term memory, encapsulating interactions with a memory store and providing methods for saving and retrieving relevant information.

**Domain:** Memory Management

**Key Responsibilities:**

* Saving explicit "remember this" notes
* Saving conversation summaries and distilled facts from documents
* Searching for context before answering user queries
* Hiding low-level details of `LongTermMemoryItem` construction
* Providing convenient, semantically meaningful methods for interacting with long-term memory

### `intergrax\memory\long_term_memory_store.py`

**intergrax\memory\long_term_memory_store.py**

Description: This module provides a protocol for persistent long-term memory storage, defining methods for storing, retrieving, and searching memory items.

Domain: Data Ingestion / Storage

Key Responsibilities:
- Persistent storage interface for long-term memory items
- Store and retrieve `LongTermMemoryItem` aggregates
- Provide basic search API scoped by owner
- Implementations MUST NOT call LLMs directly or implement prompt construction/RAG orchestration

### `intergrax\memory\organization_profile_manager.py`

**Description:** This module provides a high-level facade for working with organization profiles, encapsulating interactions with the underlying store and providing convenient methods to load, persist, and retrieve prompt-ready bundles.

**Domain:** Memory (organization profiling)

**Key Responsibilities:**

* Load or create an OrganizationProfile for a given organization_id
* Persist profile changes using the underlying OrganizationProfileStore
* Build a prompt-ready bundle for the LLM/runtime

### `intergrax\memory\organization_profile_memory.py`

**Description:** 
This module defines data structures and utility functions for managing organization profiles in the Integrax framework.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Define organization identity with stable identification data.
* Establish organization preferences influencing system behavior.
* Compress high-level organizational knowledge into a summary and metadata.
* Build compact, deterministic prompt bundles from organization profiles.
* Provide utility functions for creating organization profile prompt bundles.

### `intergrax\memory\organization_profile_store.py`

**Description:** This module defines a protocol for persisting and loading organization profiles in the Integrax framework. It provides an abstraction layer for storing and retrieving organization data, decoupling it from specific backend storage systems.

**Domain:** Memory Store (Organization Profiles)

**Key Responsibilities:**

* Loading and saving organization profiles
* Providing default values for new organizations
* Hiding backend-specific implementation details
* Persisting and removing profile aggregates

Note: This file appears to be a well-designed, high-level abstraction for storing organization data. The protocol defined here is likely used throughout the framework to interact with different storage systems (e.g., JSON files or SQL databases).

### `intergrax\memory\stores\__init__.py`

Description: The __init__.py file initializes and configures the memory stores within the Intergrax framework.

Domain: Memory Management

Key Responsibilities:
• Initializes the memory store modules
• Configures the store instances for use in the application
• Exposes methods for registering and retrieving stores 
• Defines dependencies between store modules

### `intergrax\memory\stores\in_memory_conversational_store.py`

Description: This module provides an in-memory implementation of the ConversationalMemoryStore interface.

Domain: Memory Management

Key Responsibilities:
- Stores conversation history for each session in memory
- Provides methods to load, save, and append messages to the conversational memory
- Offers an optional helper function to list active persisted session IDs

### `intergrax\memory\stores\in_memory_long_term_memory_store.py`

**Description:** This module implements an in-memory long-term memory store, suitable for unit testing, local development, or small-scale experiments.

**Domain:** Memory Management (in-memory storage)

**Key Responsibilities:**
- Stores and retrieves `LongTermMemoryItem` instances
- Supports upsert, get, delete operations on individual items
- List items for a specific owner with optional filtering and pagination
- Simple keyword-based search functionality

### `intergrax\memory\stores\in_memory_organization_profile_store.py`

**Description:** This module provides an in-memory implementation of the OrganizationProfileStore, used for unit testing, local development, and experimental purposes.

**Domain:** In-Memory Data Store

**Key Responsibilities:**

* Provides a simple in-memory storage solution for organization profiles
* Supports get, save, delete operations on organization profiles
* Returns default profiles if not present or unknown organization IDs are provided
* Offers an optional helper method to list stored organization IDs (for debugging/testing purposes)

### `intergrax\memory\stores\in_memory_user_profile_store.py`

Description: This module implements an in-memory user profile store, allowing for unit testing and local development.

Domain: User Profile Management

Key Responsibilities:
- Stores user profiles in memory as dictionaries.
- Provides methods to retrieve, save, and delete user profiles asynchronously.
- Creates default profiles if none exist and optionally stores them for subsequent calls.

### `intergrax\memory\user_profile_manager.py`

**Description:** This module provides a high-level facade for working with user profiles, abstracting away the underlying store and providing convenient methods for loading, persisting, and building prompt-ready bundles.

**Domain:** Memory Management

**Key Responsibilities:**

* Provide methods to load or create a UserProfile for a given user ID
* Persist profile changes
* Build a prompt-ready bundle for the LLM/runtime
* Hide direct interaction with the underlying UserProfileStore

### `intergrax\memory\user_profile_memory.py`

**Description:** This module provides core domain models for user and organization profiles, as well as prompt bundles derived from these profiles. It includes utilities for building compact, system-ready prompts.

**Domain:** User Profile Models

**Key Responsibilities:**

* Define data classes for `UserIdentity`, `UserPreferences`, `UserProfile`, and `UserProfilePromptBundle` to represent user profiles and their associated preferences.
* Provide methods for building prompt bundles from user profiles using `build_profile_prompt_bundle`.
* Include helper functions such as `_build_fallback_summary` to construct deterministic fallback summaries when explicit summary instructions are not available.

**Note:** This module appears to be a central component of the Integrax framework, responsible for managing user and organization profiles in a structured manner. The code is well-organized and clearly documents its purpose, making it easy to understand and extend.

### `intergrax\memory\user_profile_store.py`

**Description:** This module provides a protocol for persistent storage of user profiles, decoupling backend-specific implementation details from the rest of the framework. It defines an interface for loading and saving user data, as well as default values for new users.

**Domain:** User Profile Management

**Key Responsibilities:**
- Provide a standardized interface for storing and retrieving user profiles.
- Allow implementations to hide backend-specific concerns (e.g., JSON files, SQL databases).
- Ensure that profile data is persisted asynchronously.

### `intergrax\multimedia\__init__.py`

Description: Initializes the multimedia module, providing an entry point for multimedia-related functionality within Intergrax.
Domain: Multimedia Management

Key Responsibilities:
* Registers multimedia-related components and services
* Sets up default configurations for multimedia processing and ingestion
* Provides access to core multimedia utilities and functions

### `intergrax\multimedia\audio_loader.py`

**Description:** This module provides utilities for downloading and processing audio from YouTube URLs using the yt_dlp library and the Whisper speech recognition API.

**Domain:** Multimedia

**Key Responsibilities:**
- Downloads audio files from YouTube URLs in various formats (default is MP3).
- Extracts video ID from YouTube URL.
- Creates a directory to store downloaded audio files if it doesn't exist.
- Translates audio using the Whisper model (supports translation from audio file).

Note: This module appears to be fully functional and intended for production use, with proper documentation and a clear structure.

### `intergrax\multimedia\images_loader.py`

Description: This module loads and transcribes images using the ollama API.

Domain: Multimedia / Image Processing

Key Responsibilities:
- Loads an image from a specified path
- Uses the ollama API to generate text based on the image
- Allows user to specify model used for transcription (default is "llava-llama3:latest")

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utility functions for displaying multimedia content, including audio, images, and videos, within Jupyter Notebook environments.

**Domain:** Multimedia Display

**Key Responsibilities:**

* `display_audio_at_data`: Displays an audio file at a specified position with optional autoplay.
* `_is_image_ext`: Checks if a file has a valid image extension.
* `display_image`: Attempts to display an image using IPython's `Image` function or falls back to displaying the file path if it exists.
* `_serve_path`: Serves a local file by copying it to a temporary directory and returning a relative path for serving.
* `display_video_jump`: Displays a video with optional controls, autoplay, and poster frame.

Note: The code appears to be complete and well-maintained.

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides functionality for downloading YouTube videos, transcribing audio to text using the Whisper model, and extracting frames from video files with accompanying metadata.

**Domain:** Video Processing & Analysis

**Key Responsibilities:**

* Downloading YouTube videos using `yt_download_video`
* Transcribing audio to text using `transcribe_to_vtt` (Whisper model)
* Extracting frames from video files with accompanying metadata using `extract_frames_and_metadata` and `extract_frames_from_video`

No indications of experimental, auxiliary, legacy, or incomplete code. The module appears well-structured and functional for its intended purpose.

### `intergrax\openai\__init__.py`

DESCRIPTION: This is the entry point for the OpenAI adapters in Intergrax, responsible for importing and configuring adapter classes.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
* Imports and registers all available OpenAI adapters
* Exposes a unified interface for interacting with OpenAI models
* Handles adapter-specific configuration and initialization

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: This is the main entry point for the OpenAI RAG (Reactor And Generator) component within the Intergrax framework, responsible for initializing and configuring RAG models.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes RAG models
- Configures model parameters
- Exposes interface for downstream components to interact with RAG models

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides RAG (Retrieval-Augmented Generation) functionality using the OpenAI API, specifically designed for the Integrax framework. It enables the retrieval and processing of vector store data, file upload, and vector store management.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Initializes an IntergraxRagOpenAI instance with a client and vector store ID
* Generates a prompt for RAG, adhering to strict rules and guidelines
* Ensures the existence of a vector store by its ID
* Clears all files loaded into the vector store
* Uploads a folder to the vector store, handling file upload and status checking

Note: The module appears to be complete and well-documented. No signs of being experimental, auxiliary, legacy, or incomplete are evident.

### `intergrax\rag\__init__.py`

DESCRIPTION: This module initializes the RAG logic component of the Intergrax framework, providing entry points and setup for Rag objects.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
* Defines the main entry point for RAG initialization
* Sets up Rag object instances
* Provides access to RAG configuration settings

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader with metadata injection and safety guards. It allows loading documents from various formats, including PDF, DOCX, Excel, CSV, and image files.

**Domain:** RAG (Relevant Action Generator) logic / Document Loading Utilities

**Key Responsibilities:**

* Loads documents from various file formats
* Supports metadata injection and validation
* Provides options for customizing document loading behavior (e.g., OCR settings, text extraction modes)
* Includes adapters for LLM (Large Language Model) integration
* Handles image captioning and video processing (new functionality via framework adapter)

**Note:** This module appears to be a comprehensive and actively maintained part of the Integrax framework.

### `intergrax\rag\documents_splitter.py`

**Description:** 
This module implements a high-quality text splitter for RAG pipelines, which generates stable chunk ids and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a text splitting function with customizable parameters
* Infers page indices from common loader keys
* Ensures source fields are present in documents
* Decides if a document should be treated as an indivisible semantic atom
* Builds stable, human-readable chunk ids using available anchors (para_ix/row_ix/page_index)
* Finalizes chunks by adding chunk index, total, parent id, source name, and source path; optionally merges tiny tails and applies max caps

### `intergrax\rag\dual_index_builder.py`

**Description:** This module is responsible for building two vector indexes: a primary index (CHUNKS) and an auxiliary index (TOC), from a list of documents.

**Domain:** RAG logic

**Key Responsibilities:**

* Builds two vector indexes:
	+ CHUNKS: all chunks/documents after splitting
	+ TOC: only DOCX headings within levels [toc_min_level, toc_max_level]
* Computes embeddings for each document using an embedding manager
* Adds documents to the primary index (CHUNKS) and auxiliary index (TOC) using vector store managers
* Supports batch processing and logging

**Notes:** The module appears to be a core component of the Integrax framework's RAG logic, responsible for creating vector indexes from documents. The code is well-structured and follows best practices, with clear comments and logging statements.

### `intergrax\rag\dual_retriever.py`

**Description:** This module provides a dual retriever class that fetches relevant information from a vector store based on user input. It first queries the Table of Contents (TOC) to identify sections relevant to the query, then searches within those identified sections for more precise matches.

**Domain:** RAG logic

**Key Responsibilities:**

* Initializes VectorstoreManagers and EmbeddingManager instances
* Performs TOC-based retrieval to identify relevant sections
* Searches within retrieved sections using a vector store query function
* Merges results from both steps, removes duplicates, and sorts by similarity score
* Returns the top k matches based on user input

### `intergrax\rag\embedding_manager.py`

**Description:** 
This module, `embedding_manager.py`, is a unified embedding manager for HuggingFace (SentenceTransformer), Ollama, or OpenAI embeddings within the Integrax framework. It provides various features such as provider switchability, reasonable defaults if model name is None, batch/single text embedding with optional L2 normalization, and cosine similarity utilities.

**Domain:** 
RAG logic

**Key Responsibilities:**

- Unified embedding manager for HuggingFace (SentenceTransformer), Ollama, or OpenAI embeddings
- Provider switchability: "hg", "ollama", "openai"
- Reasonable defaults if model name is None
- Batch/single text embedding with optional L2 normalization
- Embedding for LangChain Documents (returns np.ndarray + aligned docs)
- Cosine similarity utilities and top-K retrieval
- Robust logging, shape validation, light retry for transient errors

Note: This file appears to be well-maintained and complete.

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides an implementation of the Retrieval-Augmented Generator (RAG) answerer, which integrates a retriever and a language model to provide answers to user questions.

**Domain:** RAG logic

**Key Responsibilities:**

*   Integrates a retriever to find relevant context fragments
*   Applies local similarity threshold filter if configured
*   Optionally re-ranks the retrieved hits using a reranker
*   Builds the context by aggregating the used hits and truncating based on a character limit
*   Creates citations for the sources of the context
*   Builds system and user messages, including the context and user instruction
*   Sends these messages to the language model to generate an answer
*   Optionally generates structured output next to text

### `intergrax\rag\rag_retriever.py`

**Description:** This module provides a scalable, provider-agnostic RAG retriever for intergrax, offering key features such as normalization of filters and embeddings, unified similarity scoring, deduplication by ID, and optional reranking.

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

Note: The file appears to be well-maintained and production-ready.

### `intergrax\rag\re_ranker.py`

**Description:** The `re_ranker.py` module provides a Re-Ranker class for fast and scalable cosine re-ranking of candidate chunks. It accepts hits from the intergrax RAG Retriever or raw LangChain Documents, embeds texts in batches using an embedding manager, and optionally fuses scores with original retriever similarity.

**Domain:** LLM adapters

**Key Responsibilities:**

* Embed query and documents using an embedding manager
* Compute cosine similarities between query and document vectors
* Re-rank candidate chunks based on cosine similarities
* Optionally fuse scores with original retriever similarity
* Return re-ranked hits with additional metadata (rerank score, fusion score, rank reranked)

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module provides a unified interface for managing vector stores, supporting ChromaDB, Qdrant, and Pinecone. It enables initializing target stores, upserting documents with embeddings, querying top-K by similarity, counting vectors, and deleting by IDs.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Initializes vector store based on provider configuration
* Creates collection or index in target store (lazy for Qdrant/Pinecone)
* Upserts documents with embeddings (with batching)
* Queries top-K by cosine/dot/euclidean similarity
* Counts vectors
* Deletes by IDs

Note that this file is a core part of the Intergrax framework, which suggests it is not experimental or auxiliary.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a Windowed Answerer, a component that extends the base Answerer with memory-awareness and context summarization capabilities.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Initializes a Windowed Answerer instance with an Answerer, retriever, and optional verbosity flag
* Builds messages with memory-awareness for context construction
* Performs broad retrieval and windowing to process candidate answers
* Synthesizes final answer from partials using LLM (Large Language Model)
* Deduplicates sources and appends final answer (and summary) to memory store (if available)

### `intergrax\runtime\__init__.py`

Description: This file serves as the top-level entry point for the Intergrax runtime, responsible for initializing and setting up the environment.

Domain: Runtime setup

Key Responsibilities:
- Initializes the Intergrax runtime
- Sets up the application context
- Configures logging and other core services

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: This module serves as the entry point for enabling Drop-In Knowledge mode within the Intergrax runtime.

Domain: RAG logic

Key Responsibilities:
- Initializes and configures the drop-in knowledge functionality
- Exposes necessary interfaces for integrating external knowledge sources

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides utilities for resolving attachments in Drop-In Knowledge Mode, allowing the framework to decouple attachment storage and consumption.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**
- Defines the `AttachmentResolver` protocol for resolving attachments into local file paths
- Implements a minimal `FileSystemAttachmentResolver` for handling local filesystem-based URIs
- Provides utility functions for resolving attachments and handling various URI schemes

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This file defines the global configuration object for the Drop-In Knowledge Runtime in Intergrax. It outlines settings for various components such as LLM adapters, RAG embedding and vectorstore managers, tools agents, and web search executors.

**Domain:** Configuration, Runtime Settings

**Key Responsibilities:**

* Defines primary LLM adapter used for chat-style generation
* Specifies configuration for RAG and vectorstore-based retrieval
* Enables/Disables features like Retrieval-Augmented Generation (RAG), real-time web search, long-term memory, and user profile memory
* Sets up tenant ID, workspace ID, and chat history limits
* Configures RAG and web search settings such as maximum documents per query, token budgets, and semantic score thresholds
* Specifies tools agent configuration including mode, context scope, and optional tools executor

**Status:** This file appears to be fully fleshed out and is not marked as experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`

**Description:** This module provides the Context Builder functionality for Intergrax's Drop-In Knowledge Mode. It decides when to use Retrieval-Augmented Generation (RAG) for a given session and request, retrieves relevant document chunks from the vector store, and composes a RAG-specific system prompt.

**Domain:** RAG logic

**Key Responsibilities:**

* Decide whether to use RAG for a given session and request
* Retrieve relevant document chunks from the vector store using session/user/tenant/workspace metadata
* Compose a RAG-specific system prompt
* Return BuiltContext with:
	+ system_prompt
	+ reduced history_messages
	+ retrieved_chunks
	+ structured RAG debug info

The Context Builder class is designed to be ignorant of LLM adapter details, how messages are serialized for OpenAI/Gemini/Claude, and how RouteInfo is built. It relies on the SessionStore to manage conversation history and only passes through the list it receives from the engine.

Note: This file appears to be well-structured and complete, with clear responsibilities and functionality. There is no indication of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

**Description:** This module implements the core runtime engine for Drop-In Knowledge Mode, providing a stateful pipeline to handle conversational requests and generate responses using Intergrax components.

**Domain:** LLM adapters, RAG logic, data ingestion, agents

**Key Responsibilities:**

* Load or create chat sessions via SessionStore
* Append user messages to the session
* Build an LLM-ready context with system prompts, conversation history, retrieved chunks from documents (RAG), web search context (if enabled), and tools results
* Call the main LLM adapter once with the fully enriched context to produce the final answer
* Return a RuntimeAnswer with the final answer text and metadata

**Notes:** This module appears to be production-ready, providing a comprehensive solution for conversational requests. However, some parts of the code (e.g., web search layer) seem optional or configurable, indicating potential flexibility in usage scenarios.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

**Description:** This module provides a service for ingesting attachments in the context of Drop-In Knowledge Mode, reusing existing Intergrax RAG building blocks.

**Domain:** Data ingestion

**Key Responsibilities:**

* Resolve AttachmentRef objects into filesystem Paths using an AttachmentResolver.
* Load documents using IntergraxDocumentsLoader and split them into chunks with IntergraxDocumentsSplitter.
* Embed chunks via IntergraxEmbeddingManager and store vectors via IntergraxVectorstoreManager.
* Return a structured IngestionResult per attachment.

**Note:** This file appears to be a core component of the Intergrax framework, providing a clean API for ingestion in Drop-In Knowledge Mode. It is likely not experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\rag_prompt_builder.py`

**Description:** This module provides a prompt builder for the RAG (Retrieval-Augmented Generation) logic in Drop-In Knowledge Mode, allowing customization of the system prompt and context messages injected into the model.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a `RagPromptBundle` class to containerize prompt elements related to RAG
* Defines the `RagPromptBuilder` protocol for building the RAG-related part of the prompt, allowing customization through implementation of this interface
* Offers a default implementation (`DefaultRagPromptBuilder`) that constructs the system prompt and injects retrieved chunks as context messages

**Notes:** This module appears to be part of the main Intergrax framework codebase, indicating it is not experimental or auxiliary. The documentation suggests a focus on customizability and adaptability for specific use cases.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

**Description:** This module defines dataclasses for request and response structures used by the Drop-In Knowledge Mode runtime in the Intergrax framework.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Define data classes for request and response structures
	+ `RuntimeRequest`: High-level request structure for the Drop-In Knowledge runtime
	+ `RuntimeAnswer`: High-level response structure returned by the Drop-In Knowledge runtime
* Include fields for citations, routing information, tool calls, and basic statistics in both request and response structures
* Use type hints and annotations to ensure clarity and compatibility with Intergrax framework requirements

### `intergrax\runtime\drop_in_knowledge_mode\session_store.py`

**Description:** This module serves as the primary memory backbone for the Intergrax Drop-In Knowledge Runtime, responsible for managing session lifecycle, storing and retrieving conversational history, exposing user/organization profile bundles, and producing LLM-ready message context.

**Domain:** RAG logic (Runtime Agent)

**Key Responsibilities:**

* Create and manage chat sessions
* Maintain conversational message history per session
* Expose user/organization profile bundles and long-term memory context
* Return an LLM-ready ordered list of messages representing the session context

Note: The file appears to be fully functional, with a clear design principle of consolidating all memory layers internally. However, it mentions future expansion points for persistent storage, deeper integration with long-term semantic memory, and tighter coupling with ContextBuilder and RAG components.

### `intergrax\runtime\drop_in_knowledge_mode\websearch_prompt_builder.py`

**Description:** This module provides a strategy interface and default implementation for building web search prompts in the Intergrax framework, allowing customization of how web documents are summarized and presented to the user.

**Domain:** RAG logic

**Key Responsibilities:**
* Provides a `WebSearchPromptBuilder` protocol for customizing web search prompt construction
* Includes a default implementation (`DefaultWebSearchPromptBuilder`) that takes a list of web documents and builds a system-level message listing titles, URLs, and snippets
* Allows configuration through the `RuntimeConfig` object to control aspects such as maximum number of documents per query

### `intergrax\supervisor\__init__.py`

Description: This is the entry point for the Intergrax supervisor, responsible for initializing and configuring the overall framework.

Domain: Supervisor initialization

Key Responsibilities:
• Initializes the Intergrax supervisor with default settings
• Configures logging and error handling
• Sets up event listeners for framework-wide events

### `intergrax\supervisor\supervisor.py`

**Description:** This module defines the core architecture of the Intergrax framework's supervisor component, responsible for planning and executing tasks based on user input.

**Domain:** Supervisor/Planning Logic

**Key Responsibilities:**

* Plan execution with support for two-stage planning (decomposition followed by step-wise assignment)
* LLM-based planning with fallback to heuristic and minimal plans
* Component management and registration
* Context construction and management
* Public API for planning, analysis, and component interaction
* Private methods for LLM text extraction, JSON parsing, and plan decomposition

Note: This file appears to be the central module of the Intergrax supervisor's logic, integrating various components and functionality.

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module defines the Supervisor's component management system, which enables registration, execution, and tracking of various pipeline components. It provides a framework for creating, registering, and running components, along with error handling and logging capabilities.

**Domain:** Supervisor Components

**Key Responsibilities:**

* Defines `Component` class, which represents a single pipeline step
	+ Has properties for name, description, availability, and function to execute
	+ Provides `run` method for executing the component's function with given state and context
* Introduces `component` decorator for registering components in a concise manner
* Defines data classes for `ComponentResult`, which stores output and metadata from component execution
* Includes `PipelineState` definition, which captures pipeline execution state

This module appears to be a core part of the Integrax framework's supervisor functionality.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides default prompt templates for the Intergrax unified Supervisor, including system and user prompts for planning tasks. The prompts define the structure and rules for generating plans in a hybrid RAG + Tools + General Reasoning engine.

**Domain:** Unified Planning Logic (RAG + Tools + General Reasoning)

**Key Responsibilities:**

* Provide default prompt templates for the Intergrax unified Supervisor
* Define system prompts that explain the planning process and rules
* Define user prompts that outline the structure of a plan, including decomposition, assignment, and synthesis steps
* Enforce strict rules for plan validation, such as resource flags, DAG acyclicity, gate conditions, and reliability checks
* Specify output format requirements, including JSON schema and data types
* Offer guidance on component selection, method assignment, and output contracts

**Note:** The file appears to be a complete, production-ready module with clear documentation and well-defined structure.

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** 
This module provides utilities for constructing and executing LangGraph pipelines from supervisor plans. It handles state management, node creation, and graph construction.

**Domain:** Supervisor to State Graph Integration

**Key Responsibilities:**

*   Manages global pipeline state through the `PipelineState` data structure
*   Provides utility functions for ensuring state defaults, appending logs, and resolving inputs
*   Creates LangGraph nodes from plan steps using the `make_node_fn` factory function
*   Builds a stable topological graph order from plan steps using the `topo_order` function
*   Compiles a supervisor plan into a runnable LangGraph pipeline with the `build_langgraph_from_plan` function

**Notes:** 
The provided code appears to be production-ready, handling various aspects of LangGraph pipeline construction and execution. However, it might benefit from additional documentation or comments to improve readability for users unfamiliar with the internal workings.

### `intergrax\system_prompts.py`

Description: This module contains a function that generates the default system instruction for a RAG (Restricted Answer Generation) system.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Provides a strict set of rules and guidelines for the RAG system
- Defines the format and structure of responses generated by the RAG system
- Specifies how to cite sources and provide references in the responses
- Outlines the procedures for handling uncertain or incomplete information
- Defines the style and formatting requirements for the final answers

### `intergrax\tools\__init__.py`

DESCRIPTION:
This module serves as the entry point and package initializer for the Intergrax tools.

DOMAIN: Utility modules

KEY RESPONSIBILITIES:
• Packages all tool-related functionality and classes
• Defines the main entry points for tool execution and initialization 
• Exposes tools to the application for usage.

### `intergrax\tools\tools_agent.py`

**Description:** 
This module provides a high-level interface for tools orchestration within the Intergrax framework. It enables the integration and management of various tools, providing a structured approach to tool invocation, output processing, and conversation flow.

**Domain:** LLM adapters / Tools Management

**Key Responsibilities:**

* Provides a configuration class (`ToolsAgentConfig`) to customize agent behavior.
* Offers a `ToolsAgent` class that initializes tools orchestration, including LLMA support detection.
* Implements various helper functions for message pruning, output structure building, and tool trace management.
* Exposes a public API (`run`) for high-level tools orchestration, supporting different input data formats (list of messages or string).
* Handles system instructions, context injection, and optional streaming capabilities.

### `intergrax\tools\tools_base.py`

**Description:** This module provides a base structure and utility functions for building and managing integrax tools.

**Domain:** Tools/Utilities

**Key Responsibilities:**

* Defines `ToolBase` class for tool implementation
	+ Provides schema model for parameters
	+ Enables validation of arguments with Pydantic
	+ Allows override of the `run` method
* Offers utility functions:
	+ `_limit_tool_output`: Truncates long tool output to avoid overflowing LLM context
	+ `ToolRegistry`: Manages ToolBase instances and exports them to a format compatible with OpenAI Responses API
* Includes support for Pydantic and a lightweight stub implementation

Note: The code appears to be well-structured, complete, and production-ready. There are no indications of being experimental, auxiliary, or legacy.

### `intergrax\websearch\__init__.py`

Description: Initializes the web search functionality and sets up necessary dependencies for web search-related modules.

Domain: Web Search Utilities

Key Responsibilities:
- Initializes the web search engine configuration
- Sets up URL routing for web search endpoints
- Registers necessary handlers for web search operations

### `intergrax\websearch\cache\__init__.py`

**Description:** 
This module implements an in-memory query cache for web search results.

**Domain:** Data Ingestion / Caching

**Key Responsibilities:**

* Stores web search query configurations as immutable `QueryCacheKey` objects
* Stores cached web documents as `QueryCacheEntry` objects, which include a TTL and creation timestamp
* Provides an `InMemoryQueryCache` class with methods for getting (`get`) and setting (`set`) cached results
* Supports optional TTL (time-to-live) and max size configuration
* Includes simple eviction mechanism to prevent cache overflow

### `intergrax\websearch\context\__init__.py`

DESCRIPTION:
This module initializes the web search context within the Intergrax framework.

DOMAIN: Web Search Context Initialization

KEY RESPONSIBILITIES:
• Initializes global context for web search functionality
• Sets up default values and configurations for web search operations
• Establishes connections to external services (e.g., APIs, databases) as needed
• Defines interfaces and hooks for integration with other components

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides a utility class for building LLM-ready textual context and chat messages from web search results.

**Domain:** Web Search Context Builder

**Key Responsibilities:**

* Builds a textual context string from WebDocument objects or serialized dicts.
* Supports customization of context formatting, including maximum number of documents, character limit per document, and inclusion of snippets and URLs.
* Provides methods for building system and user prompts for chat-style LLMs in strict "sources-only" mode.
* Includes functionality for handling user questions, web search results, and answer language.

**Note:** The file appears to be a well-documented and implemented module within the Intergrax framework.

### `intergrax\websearch\fetcher\__init__.py`

**File Path:** intergrax\websearch\fetcher\__init__.py

**Description:** The web search fetcher module is responsible for initializing and configuring the web search functionality within Intergrax.

**Domain:** Web Search Fetching

**Key Responsibilities:**
* Initializes the web search fetcher components
* Configures search parameters and settings
* Sets up connections to relevant external services (e.g., APIs)
* Provides a standardized interface for fetching search results

### `intergrax\websearch\fetcher\extractor.py`

**Description:** 
This module is responsible for extracting metadata and text content from web pages, providing both lightweight (extract_basic) and advanced (extract_advanced) extraction methods.

**Domain:** Web Search, Data Ingestion

**Key Responsibilities:**
- extract_basic:
  - Extract title
  - Extract meta description
  - Extract HTML language attribute
  - Extract Open Graph metadata tags
  - Produce a plain-text version of the page
- extract_advanced:
  - Remove obvious boilerplate elements (scripts, styles, iFrames, navigation)
  - Perform readability-based extraction using trafilatura (when available) or BeautifulSoup fallback
  - Normalize whitespace and reduce noise
  - Optionally overwrite existing text if it already exists

This file appears to be production-ready.

### `intergrax\websearch\fetcher\http_fetcher.py`

**intergrax\websearch\fetcher\http_fetcher.py**

**Description:** This module provides an asynchronous HTTP fetcher for web pages, responsible for sending GET requests and returning structured page content.

**Domain:** Web Search Fetching

**Key Responsibilities:**
- Perform asynchronous HTTP GET requests with customizable headers.
- Handle redirects, timeouts, and network errors.
- Return a `PageContent` instance containing URL metadata, HTML content, and extracted information.

### `intergrax\websearch\integration\__init__.py`

Description: This is the main entry point for the web search integration module, responsible for initializing and managing external search engine APIs.

Domain: LLM adapters

Key Responsibilities:
- Initializes search engines and their respective API clients
- Defines configuration and settings for web search integrations
- Provides a centralized interface for searching across multiple platforms

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a LangGraph-compatible web search node wrapper for integrating with external search services. It encapsulates configuration and delegates to a provided WebSearchExecutor instance.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a LangGraph-compatible web search node class (WebSearchNode) that:
	+ Encapsulates configuration of the WebSearchExecutor
	+ Implements sync and async node methods for operating on WebSearchState
* Offers a default, module-level node instance for convenience and backward compatibility
* Includes functional wrappers for synchronous and asynchronous usage

Note: This file appears to be well-maintained and complete. No flags for experimental, auxiliary, legacy, or incomplete status are applicable.

### `intergrax\websearch\pipeline\__init__.py`

Description: Initializes the web search pipeline, defining its configuration and setup.

Domain: Web Search Pipeline Configuration

Key Responsibilities:
- Defines the default pipeline configuration.
- Initializes necessary modules for the pipeline's operation.

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** 
This module provides a pipeline for multi-provider web search, fetching, extraction, deduplication, and quality scoring.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Orchestrates searches across multiple providers
* Fetches and extracts HTML pages into WebDocument objects
* Performs simple deduplication via text-based key
* Provides quality scores for extracted documents
* Allows asynchronous or synchronous execution of the pipeline

### `intergrax\websearch\providers\__init__.py`

Description: Initializes and configures web search providers for the Intergrax framework.

Domain: Web Search Providers

Key Responsibilities:
- Registers available web search provider classes.
- Sets default configuration settings for web searches.
- Provides a hook for customizing or extending web search behavior.

### `intergrax\websearch\providers\base.py`

Description: This module defines the base interface for web search providers, including the Google, Bing, DuckDuckGo, Reddit, News, etc.

Domain: Web Search Providers

Key Responsibilities:
- Accept a provider-agnostic QuerySpec
- Return a ranked list of SearchHit items
- Expose minimal capabilities for feature negotiation (language, freshness)
- Execute a single search request with sanitization and validation of URLs
- Provide a static capability map for feature negotiation
- Optional resource cleanup for providers that own resources such as HTTP sessions or clients.

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module implements a Bing Web Search provider for the Intergrax framework, utilizing the v7 REST API to fetch search results.

**Domain:** LLM adapters

**Key Responsibilities:**
- Initializes the Bing Web Search provider with optional API key and session parameters.
- Defines capabilities of the provider (language and freshness filtering).
- Builds HTTP headers and query parameters for the search request.
- Parses JSON response from Bing's v7 API to extract search hits.
- Formats search hits according to Intergrax's schema.

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module implements a provider for Google Custom Search (CSE) REST API in the Intergrax framework.

**Domain:** LLM adapters > Websearch providers > Google CSE provider

**Key Responsibilities:**

* Establishes a connection to the Google CSE REST API
* Supports query specification and execution with parameters like language, freshness, and page size
* Extracts relevant information from search results (e.g., title, snippet, URL)
* Handles pagination and error handling for failed requests
* Provides metadata about supported capabilities and configuration requirements

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module implements a Google Places / Google Business provider for the Intergrax framework, enabling text search and details retrieval for businesses.

**Domain:** RAG logic (Reverse Address Geocoding & Geospatial Search)

**Key Responsibilities:**

* Provides capabilities such as language support and freshness
* Builds parameters for text search and details retrieval endpoints
* Fetches place details using the Google Places API
* Maps results to SearchHit objects
* Handles URL construction for Google Maps links

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a full-featured Reddit search provider using the official OAuth2 API, allowing for searching and retrieving metadata of Reddit posts.

**Domain:** LLM adapters

**Key Responsibilities:**

* Establishes an authenticated session with the Reddit API using client credentials
* Supports language-independent searching and freshness filtering
* Fetches post metadata (score, num_comments, upvote_ratio, nsfw, etc.) from the Reddit API
* Optionally fetches top-level comments for each post
* Handles OAuth2 token refresh and expiration

Note: The file appears to be a well-maintained and functional part of the Intergrax framework.

### `intergrax\websearch\schemas\__init__.py`

DESCRIPTION: 
This module initializes and exports schema definitions for web search functionality within the Intergrax framework.

DOMAIN: Web Search Schemas

KEY RESPONSIBILITIES:
- Initializes schema definitions
- Exports schema definitions for web search functionality 
- Provides entry point for schema-related operations in websearch module

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass to represent the content of a web page, encapsulating raw HTML and derived metadata.

**Domain:** Web search and scraping

**Key Responsibilities:**

* Represents the fetched and optionally extracted content of a web page
* Encapsulates both raw HTML and derived metadata for post-processing stages
* Provides fields for various metadata, such as final URL, HTTP status code, text content, title, description, language, Open Graph tags, schema.org data, and more
* Includes methods to check if the page has non-empty content, generate a short summary of the text, and estimate the content size in kilobytes

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a dataclass representing a canonical search query specification for web search providers.

**Domain:** Query Schema Definition

**Key Responsibilities:**

* Defines the `QuerySpec` dataclass with a set of required and optional fields.
* Provides methods to normalize the query string (`normalized_query`) and cap the number of top results (`capped_top_k`).
* Keeps the model minimal, provider-agnostic, and stable as per design constraints. 

**Note:** The file appears to be part of the main framework codebase and does not show any signs of being experimental, auxiliary, or legacy.

### `intergrax\websearch\schemas\search_hit.py`

Description: This module defines a dataclass `SearchHit` representing a single search result entry, encapsulating metadata from various providers.

Domain: Search schemas

Key Responsibilities:
- Encapsulates provider-agnostic metadata for a search result.
- Validates and sanitizes input to ensure consistency (e.g., rank >= 1, valid URL scheme).
- Provides utility methods like `domain()` and `to_minimal_dict()` for easy access to relevant information.

### `intergrax\websearch\schemas\web_document.py`

**Description:** This module defines a dataclass for representing web documents, which combines metadata from search hits with extracted page content and analysis results.

**Domain:** Web Search

**Key Responsibilities:**

* Provides a unified structure (`WebDocument`) to represent fetched and processed web documents.
* Connects original search hit metadata with extracted page content and analysis results (e.g., deduplication and quality scores).
* Offers methods for document validation, text merging, and summary generation.

### `intergrax\websearch\service\__init__.py`

Description: This module initializes the web search service for the Intergrax framework, setting up essential components and configurations.

Domain: Web Search Service Configuration

Key Responsibilities:
• Initializes the web search service instance
• Defines default configuration parameters
• Sets up service-specific dependencies and imports
• Establishes connections to external services (e.g., databases, APIs)

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level helper class, `WebSearchAnswerer`, for answering user questions via web search and Large Language Model (LLM) adapters.

**Domain:** Web Search & LLM Integration

**Key Responsibilities:**

* Runs web search using `WebSearchExecutor` to retrieve relevant documents
* Builds LLM-ready context/messages from web documents using `WebSearchContextBuilder`
* Calls any `LLMAdapter` implementation to generate a final answer
* Provides asynchronous and synchronous API for answering user questions

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This file defines a high-level web search executor, responsible for constructing query specifications, executing the search pipeline, and converting results into LLM-friendly dictionaries.

**Domain:** Web Search

**Key Responsibilities:**

* Constructing QuerySpec from raw queries and configuration
* Executing the search pipeline with chosen providers (e.g., Google CSE, Bing Web)
* Converting WebDocument objects into serialized dicts for LLM prompts and logging
* Providing methods for building query specifications and executing asynchronous web searches

### `intergrax\websearch\utils\__init__.py`

**Description:** This module initializes the web search utility functionality within Intergrax.

**Domain:** Utilities (Web Search)

**Key Responsibilities:**

* Initializes the web search utility components
* Provides entry points for web search-related functionality
* May contain placeholder or stub code for future development.

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides utilities for deduplication in the web search pipeline, specifically for normalizing text and generating stable keys for near-identical document detection.

**Domain:** Web Search Utilities

**Key Responsibilities:**

* Normalizes input text for deduplication using a simple algorithm
	+ Treats None as empty string
	+ Strips leading and trailing whitespace
	+ Converts to lower case
	+ Collapses internal whitespace sequences to a single space
* Generates stable SHA-256 based deduplication keys from normalized text
* Returns hex-encoded digest of the normalized text

Note: This module appears to be part of the main Intergrax framework functionality, and its purpose seems clear. There are no indications that this file is experimental, auxiliary, legacy, or incomplete.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a simple asyncio-compatible token bucket rate limiter, designed to prevent excessive concurrent requests.

**Domain:** RAG logic (Rate and Adversarial Guarding)

**Key Responsibilities:**
- Provides a token bucket rate limiter for limiting the rate of concurrent requests
- Allows for adjustable refill rates and capacity
- Supports both blocking (`acquire`) and non-blocking (`try_acquire`) consumption of tokens
- Intended for use in concurrent coroutines (single process)

### `main.py`

**FILE PATH:** main.py

**Description:** This file serves as the entry point for the Intergrax framework, responsible for executing the core functionality and displaying a greeting message.

**Domain:** Framework Initialization/Entrypoint

**Key Responsibilities:**
- Serves as the primary entry point for the Intergrax framework.
- Contains the main execution loop that calls the core functionality.

### `mcp\__init__.py`

Description: This is the main entry point for the MCP package, responsible for initializing and setting up the framework components.

Domain: Configuration

Key Responsibilities:
* Initializes the MCP package and its dependencies
* Sets up the framework's configuration and logging
* Defines the entry points for other modules to interact with the MCP core

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

Description: This notebook serves as a basic sanity-check demonstration for the Drop-In Knowledge Mode runtime environment in the Intergrax framework.

Domain: RAG (Retrieval-Augmented Generation) logic / Runtime Environment

Key Responsibilities:
- Verifies the functionality of DropInKnowledgeRuntime
- Creates or loads a session
- Appends user and assistant messages
- Builds conversation history from SessionStore
- Returns a RuntimeAnswer object

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the functionality of Intergrax's Drop-In Knowledge Mode runtime, specifically focusing on attachments and ingestion.

**Domain:** LLM adapters, RAG logic, data ingestion

**Key Responsibilities:**

* Demonstrates how to initialize the Drop-In Knowledge Mode runtime
* Shows how to configure the runtime with an in-memory session store, Ollama + LangChain LLM adapter, Intergrax EmbeddingManager, and VectorstoreManager
* Illustrates how to prepare an AttachmentRef for a local project document and let the runtime handle ingestion of this attachment
* Verifies that attachments are correctly stored in the session, ingestion runs without errors, chunks are stored in the vector store, and ingestion details are visible in `debug_trace`

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

Description: This Jupyter Notebook demonstrates the usage of the `ContextBuilder` in Intergrax's Drop-In Knowledge Mode runtime, providing a practical end-to-end demonstration of how to use session history, RAG retrieval from the vector store, and runtime configuration to produce a ready-to-use context object for any LLM adapter.

Domain: RAG logic

Key Responsibilities:
- Initializes the minimum set of components required to test the `ContextBuilder`.
- Demonstrates loading a demo chat session using the existing `SessionStore`.
- Works with an existing attachment by making sure its chunks are stored in the vector store with proper metadata.
- Initializes the `ContextBuilder` instance using runtime configuration and shared `IntergraxVectorstoreManager`.
- Builds context for a single user question, obtaining reduced chat history, retrieved document chunks (RAG context), system prompt, and RAG debug information.
- Inspects the result of context building without making an LLM call.

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

Here's the documentation:

**Description:** This Jupyter notebook demonstrates using **DropInKnowledgeRuntime** with session-based chat, optional RAG (attachments ingested into a vector store), and live web search via **WebSearchExecutor**, achieving a "ChatGPT-like" experience with browsing.

**Domain:** Web Search Integration

**Key Responsibilities:**

- Initializes the core configuration for Drop-in Knowledge Runtime:
  - LLM adapter (Ollama backend via LangChain)
  - Embeddings + vector store
  - Web search integration
- Demonstrates how to use **DropInKnowledgeRuntime** with web search enabled:
  - Creates a fresh chat session for web search demo
  - Allows interactive testing with `ask(question: str)` helper

This notebook appears to be a demonstration or example code, likely part of the Intergrax framework's documentation.

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

**Description:** This notebook demonstrates the usage of the Drop-In Knowledge Runtime with a tools orchestration layer on top of conversational memory, optional RAG, and live web search context.

**Domain:** LLM adapters & Tools orchestration

**Key Responsibilities:**

* Configure Python path to import the `intergrax` package
* Load environment variables (API keys, etc.)
* Import core building blocks used by the Drop-In Knowledge Runtime
* Initialize non-tool configuration (LLM, embeddings, vector store, web search) in a single compact setup cell
* Define tools using the Intergrax tools framework:
	+ Implement demo tools (`WeatherTool`, `CalcTool`)
	+ Register them in a `ToolRegistry`
	+ Create an `IntergraxToolsAgent` instance that uses an Ollama-based LLM
	+ Attach this agent to `RuntimeConfig.tools_agent` for orchestration by the Drop-In Knowledge Runtime

**Note:** This file appears to be a demonstration notebook, and its code is likely not intended for production use.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

Description: This notebook provides a comprehensive example of building and using a hybrid multi-source RAG (Relevance Aware Generator) pipeline that combines local files, web search results, and LangGraph to answer user questions.

Domain: Hybrid Multi-Source RAG with Intergrax + LangGraph

Key Responsibilities:
- Combines multiple knowledge sources into a single in-memory vector index.
- Utilizes Intergrax components (loaders, splitter, embeddings, vectorstore) in conjunction with LangGraph to create an ephemeral knowledge graph for one-off research tasks.
- Demonstrates how to plug Intergrax components into a LangGraph `StateGraph` for a practical end-to-end RAG workflow.
- Provides environment and configuration setup for the hybrid multi-source RAG agent.

Note: This file appears to be well-documented, production-oriented code that showcases a clean pattern for combining multiple knowledge sources in a single RAG pipeline. It does not exhibit any characteristics of being experimental, auxiliary, legacy, or incomplete.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

**Description:** This notebook provides a minimal integration example between the Intergrax framework and LangGraph, demonstrating how to use an Intergrax LLM adapter as a node inside a LangGraph graph.

**Domain:** LLM adapters (LangGraph integration)

**Key Responsibilities:**

*   Initialize an OpenAI LLM adapter using the `OpenAIChatResponsesAdapter` class.
*   Define a simple state for the LLM QA example, which includes messages and an answer.
*   Implement a node that calls the Intergrax LLM adapter to generate answers.
*   Build a graph with a single node (the LLM answer node) and execute it using LangGraph.

**Notes:** This notebook is focused on demonstrating the integration between Intergrax and LangGraph, rather than providing a production-ready solution. The example code may need to be adapted for specific use cases and requirements.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

Description: This notebook demonstrates the construction of a practical web research agent using Intergrax and LangGraph components. It orchestrates a multi-step graph for user question normalization, web search, context building, and final answer generation.

Domain: Web Research Agent (Intergrax + LangGraph)

Key Responsibilities:
- Normalizes the user question.
- Executes web search using Intergrax WebSearchExecutor.
- Builds a compact textual context and citations from search results.
- Generates a final answer with sources using an LLM adapter.
- Demonstrates a realistic example of "no-hallucination" web-based Q&A inside a graph-based agent.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This notebook provides a demonstration of using the Integrax framework to interact with the OpenAI API, specifically for retrieving answers to questions and filling a VectorStore.

Domain: RAG (Reactive Accelerated Graph) logic, utilizing OpenAI adapters.

Key Responsibilities:
- Importing necessary libraries and setting up environment variables.
- Creating an instance of IntergraxRagOpenAI, which interacts with the OpenAI API.
- Ensuring the existence of a VectorStore, clearing it if necessary, and uploading local data to it.
- Testing query functionality by running questions through the RAG model.

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This Jupyter Notebook outlines the configuration and setup of a high-level hybrid chat agent utilizing the Integrax framework, which integrates various components for conversational AI.

**Domain:** RAG (Retrieval-Augmented Generation) logic, agents

**Key Responsibilities:**

* Initializes LLM adapters and conversational memory
* Registers tools and their corresponding arguments
* Configures vector store and RAG settings
* Sets up a RagAnswerer with retriever, llm, reranker, and config
* Defines a RagComponent for routing questions to the RAG answerer
* Creates a high-level hybrid chat agent combining LLM chat, tools, and RAG components

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter Notebook file defines various components of the Integrax framework, including a tool for retrieving weather information and a structured output schema for a RAG (Reactive Attention Guided) response. The code demonstrates how to use these components together to generate a human-readable answer and a parsed, validated Pydantic object.

**Domain:** RAG logic

**Key Responsibilities:**

* Define a `WeatherTool` class that accepts a city name via validated arguments and returns a dictionary compatible with the `WeatherAnswer` schema.
* Define the `WeatherArgs` and `WeatherAnswer` schemas using Pydantic models for structured output.
* Set up a conversational memory, tool registry, LLM adapter, and tools agent orchestrating the interaction between the LLM, tools, and conversational memory.
* Demonstrate how to use the `ToolsAgent` to run a natural-language question and request a structured result using the `WeatherAnswer` model as output_model.

Note: This file appears to be a working example rather than experimental or auxiliary code.

### `notebooks\rag\rag_custom_presentation.ipynb`

Description: This Jupyter notebook demonstrates loading and processing documents for use in a Retrieval Augmented Generation (RAG) pipeline, including splitting, embedding, and storing in a vector database.

Domain: RAG Logic

Key Responsibilities:
- Loading raw documents from directories
- Splitting large documents into smaller chunks
- Creating embeddings for document chunks using a specified model
- Initializing and interacting with the vector store database

### `notebooks\rag\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter Notebook script demonstrates the functionality of the Integrax framework, specifically focusing on multimodal retrieval and ingestion using Rag-based retrievers.

**Domain:** RAG logic

**Key Responsibilities:**

*   Loads documents from various sources (video, audio, images) using `DocumentsLoader` and `LangChainOllamaAdapter`.
*   Splits and embeds the loaded documents using `DocumentsSplitter` and `EmbeddingManager`.
*   Checks if the corpus is present in the VectorStore using `corpus_present` function.
*   If the corpus is not present, it ingests the embedded documents into the VectorStore using `add_documents` method.
*   Demonstrates retriever functionality using `RagRetriever`, retrieving top-k hits for a given question.

**Note:** The code appears to be a working example of the Integrax framework's multimodal retrieval capabilities.

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter Notebook script provides a demonstration of multimedia processing capabilities within the Integrax framework, including video and audio download from YouTube, transcription to VTT format, frame extraction and metadata retrieval from videos, and translation of audio content.

**Domain:** RAG logic (multimedia processing)

**Key Responsibilities:**

* Downloads video and audio files from YouTube using `yt_download_video` and `yt_download_audio` functions.
* Transcribes video to VTT format using `transcribe_to_vtt` function.
* Extracts frames and metadata from videos using `extract_frames_and_metadata` function.
* Translates audio content using `translate_audio` function.
* Uses Ollama model to describe images extracted from videos using `transcribe_image` function.

**Note:** The script appears to be a demonstration or example code, rather than a production-ready module. It is not clear if this code is part of the main Integrax framework or an auxiliary tool.

### `notebooks\rag\tool_agent_presentation.ipynb`

**Description:** This Jupyter notebook serves as a demonstration and setup for the Tools Agent in the Integrax framework. It defines two simple tools (WeatherTool and CalcTool) and demonstrates their usage within an integrated agent architecture.

**Domain:** RAG logic

**Key Responsibilities:**

* Defines a WeatherTool to simulate fetching current weather information
	+ Validates user-provided city name
	+ Returns mock current weather data for the specified city
* Defines a CalcTool to evaluate basic arithmetic expressions
	+ Validates and parses user-provided expression
	+ Safely evaluates the expression in a restricted environment
	+ Returns both the original expression and the calculated result
* Sets up an agent (ToolsAgent) that integrates with LLM adapters, tool registry, conversational memory, and other components to orchestrate task resolution

**Note:** This appears to be a demonstration or example code. While it does define functional tools and an agent setup, its primary purpose is likely educational or illustrative rather than production-ready.

### `notebooks\supervisor\supervisor_test.ipynb`

Description: This Jupyter notebook contains four example components for the Integrax framework, each implementing a different functionality: compliance checker, cost estimator, final summary generator, and financial audit.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Compliance Checker: verifies whether proposed changes comply with privacy policies and terms of service
  - uses mock decision (80% chance of success)
  - returns findings on compliance, policy violations, and notes for non-compliance
- Cost Estimator: estimates the cost of changes based on UX audit report
  - uses mock formula for calculation
  - returns estimated cost, currency, and method used
- Final Summary Report Generator: generates final consolidated summary using all collected artifacts
  - includes status pipeline, terminated by, terminate reason, PM decision, notes, UX report, financial report, and citations
- Financial Audit Agent: generates a mock financial report and VAT calculation (test data)
  - returns gross value, VAT rate, VAT amount, currency, and last quarter budget report

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This notebook is used to execute web searches using the Google Custom Search (CSE) API and Bing search, displaying search results in a formatted manner.

**Domain:** WebSearch

**Key Responsibilities:**

* Executing web searches using the Google CSE API and Bing search
* Displaying search results with relevant metadata (title, URL, snippet, etc.)
* Handling environmental variables for API keys and other configuration settings
* Importing necessary modules and loading environment variables from a `.env` file
