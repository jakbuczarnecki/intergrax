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
- `bundle_intergrax_engine.py`
- `generate_project_overview.py`
- `intergrax\__init__.py`
- `intergrax\chains\__init__.py`
- `intergrax\chains\langchain_qa_chain.py`
- `intergrax\chat_agent.py`
- `intergrax\globals\__init__.py`
- `intergrax\globals\settings.py`
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
- `intergrax\memory\stores\__init__.py`
- `intergrax\memory\stores\in_memory_conversational_store.py`
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
- `intergrax\runtime\drop_in_knowledge_mode\config.py`
- `intergrax\runtime\drop_in_knowledge_mode\context\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\context\context_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\context\engine_history_layer.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine\runtime.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine\runtime_state.py`
- `intergrax\runtime\drop_in_knowledge_mode\ingestion\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\ingestion\attachments.py`
- `intergrax\runtime\drop_in_knowledge_mode\ingestion\ingestion_service.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\reasoning\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\reasoning\reasoning_layer.py`
- `intergrax\runtime\drop_in_knowledge_mode\responses\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\responses\response_schema.py`
- `intergrax\runtime\drop_in_knowledge_mode\session\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\session\chat_session.py`
- `intergrax\runtime\drop_in_knowledge_mode\session\in_memory_session_storage.py`
- `intergrax\runtime\drop_in_knowledge_mode\session\session_manager.py`
- `intergrax\runtime\drop_in_knowledge_mode\session\session_storage.py`
- `intergrax\runtime\organization\__init__.py`
- `intergrax\runtime\organization\organization_profile.py`
- `intergrax\runtime\organization\organization_profile_instructions_service.py`
- `intergrax\runtime\organization\organization_profile_manager.py`
- `intergrax\runtime\organization\organization_profile_store.py`
- `intergrax\runtime\organization\stores\__init__.py`
- `intergrax\runtime\organization\stores\in_memory_organization_profile_store.py`
- `intergrax\runtime\user_profile\__init__.py`
- `intergrax\runtime\user_profile\session_memory_consolidation_service.py`
- `intergrax\runtime\user_profile\user_profile_debug_service.py`
- `intergrax\runtime\user_profile\user_profile_debug_snapshot.py`
- `intergrax\runtime\user_profile\user_profile_instructions_service.py`
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
- `INTERGRAX_ENGINE_BUNDLE.py`
- `main.py`
- `mcp\__init__.py`
- `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\06_session_memory_roundtrip_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\07_user_profile_instructions_baseline.ipynb`
- `notebooks\drop_in_knowledge_mode\08_user_profile_instructions_generation.ipynb`
- `notebooks\drop_in_knowledge_mode\09_long_term_memory_consolidation.ipynb`
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

Here is the documentation for the provided file:

**Description:** This module initializes and sets up the API package, providing a foundation for building and importing other modules.

**Domain:** API setup

**Key Responsibilities:**
* Initializes the API package
* Sets up import paths and dependencies
* Defines API namespace and entry points

### `api\chat\__init__.py`

Description: This module is the entry point for the chat API, responsible for initializing and configuring various components.

Domain: Chat API Initialization

Key Responsibilities:
• Initializes the chat API application
• Configures default settings and dependencies
• Exposes the main chat application instance

### `api\chat\main.py`

**Description:** This module provides the core functionality for the Integrax chat API, including handling user queries, uploading and indexing documents, and managing document storage.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Handles incoming user queries through the `/chat` endpoint
* Retrieves and sets up the RAG model based on user input
* Stores application logs and chat history for each session
* Uploads and indexes documents using Chroma, with support for multiple file formats
* Provides endpoints for listing all indexed documents and deleting specific documents

### `api\chat\tools\__init__.py`

DESCRIPTION: This package initializes the chat API tools, providing a foundation for integrating various functionality.

DOMAIN: Chat API Tools

KEY RESPONSIBILITIES:
• Initializes and configures the chat API toolset.
• Defines interfaces for integration with other components.

### `api\chat\tools\chroma_utils.py`

Description: This module provides utility functions for interacting with Chroma, a vector database used in the RAG pipeline of the Integrax framework.

Domain: RAG logic

Key Responsibilities:
- Loads and splits documents from a given file path.
- Indexes a document to Chroma using its file path and ID.
- Deletes a document from Chroma by its file ID.
Note: The function `load_and_split_documents` appears to be a duplicate definition, possibly an error or a placeholder for further implementation.

### `api\chat\tools\db_utils.py`

**Description:** This module provides database utilities for the Integrax framework, including schema creation and migration, as well as public API endpoints for interacting with the database.

**Domain:** Database Utilities

**Key Responsibilities:**

*   Provides low-level helpers for connecting to and managing a SQLite database
*   Offers functions for creating and migrating the database schema
*   Exposes public API endpoints for inserting, retrieving, and deleting messages and document records
*   Supports backward-compatibility with older application logs through entry points like `create_application_logs` and `insert_application_logs`
*   Includes utility functions for ensuring sessions and inserting documents into the database

Note: The code appears to be well-structured, and there are no clear indications of experimental or auxiliary functionality. However, some parts (like the backward-compatibility section) may be considered legacy code as they seem to have been superseded by newer interfaces.

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines Pydantic models for API requests and responses, specifically tailored for the Integrax framework's chat functionality.

**Domain:** LLM adapters

**Key Responsibilities:**
- Defines enumerations for model names (e.g., LLAMA_3_1, GPT_OSS_20B)
- Establishes data structures for query inputs (question, session_id, model) and responses (answer, session_id, model)
- Specifies the structure of document information (id, filename, upload_timestamp)
- Defines a request model for deleting files by ID

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides tools and utilities for building the RAG (Retriever-Augmented Generator) pipeline within the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides singletons for vector store, embedder, retriever, reranker, and answerers
* Implements lazy loading for these components to avoid redundant initialization
* Offers utility functions for retrieving and instantiating various components (e.g., `_get_vectorstore()`, `_build_llm_adapter(model_name)`)
* Defines default prompts for users (`_default_user_prompt()` and `_default_system_prompt()`)
* Exposes the `get_answerer()` function to retrieve a RagAnswerer instance, either by name or using the default model

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides a set of API utilities for interacting with the Integrax framework's backend services, including sending requests to chat and document management endpoints.

**Domain:** Chat API Utilities

**Key Responsibilities:**

* Sending POST requests to the chat endpoint to retrieve responses from the model
* Uploading files to the backend using HTTP POST requests
* Listing documents stored in the backend
* Deleting documents by their IDs

Note: This module appears to be a part of the Integrax framework's core functionality, and its code is well-structured and documented.

### `applications\chat_streamlit\chat_interface.py`

**Description:** This module provides a Streamlit-based interface for displaying chat interactions between the user and an AI assistant.
**Domain:** Chat Interface
**Key Responsibilities:**
- Displays chat messages sent by both the user and the AI assistant
- Handles user input via the chat interface and sends it to the API for processing
- Displays API responses received from the `get_api_response` function, including the generated answer and relevant metadata

### `applications\chat_streamlit\sidebar.py`

**Description:** This module is responsible for rendering the sidebar interface in the Chat Streamlit application, providing model selection, file upload/download management, and document listing capabilities.

**Domain:** RAG logic (Data Ingestion)

**Key Responsibilities:**
* Render a model selection component using Streamlit's `selectbox` widget
* Handle file uploads, including validation and uploading to a remote API endpoint
* List and display uploaded documents with corresponding metadata
* Provide delete functionality for selected documents

### `applications\chat_streamlit\streamlit_app.py`

Here is the documentation of the file:

**Description:** This module provides the main application logic for the Integrax RAG Chatbot, utilizing Streamlit for user interface management.

**Domain:** Application/Chat Interface

**Key Responsibilities:**
- Initializes Streamlit session state for messages and session ID
- Displays chat interface and sidebar using separate modules
- Sets up basic chatbot layout with title

### `applications\company_profile\__init__.py`

Description: The `__init__.py` file initializes the company profile application, setting up necessary configurations and imports.

Domain: Application configuration

Key Responsibilities:
• Defines the entry point for the company profile application
• Imports required modules and subpackages
• Sets up default configuration for the application
• Possibly performs other initialization tasks specific to this application

### `applications\figma_integration\__init__.py`

DESCRIPTION: This module serves as the entry point for Figma integration, handling application initialization and setup.

DOMAIN: Integration Adapters

KEY RESPONSIBILITIES:
* Initializes Figma integration components
* Sets up necessary configuration for API interactions
* Exposes APIs for custom integration implementation 
(Note: The content is very brief; additional information might be required to fully understand the module's responsibilities)

### `applications\ux_audit_agent\__init__.py`

Description: The UX Audit Agent is the entry point for the UX Audit functionality, responsible for initializing and configuring the audit process.

Domain: Agents

Key Responsibilities:
* Initializes the UX Audit Agent
* Configures audit settings and parameters
* Performs necessary setup and initialization for the audit process
* Provides a public API for triggering audits

### `applications\ux_audit_agent\components\__init__.py`

Description: This module serves as the entry point for the UX Audit Agent's components, enabling its setup and configuration.

Domain: Agents

Key Responsibilities:
- Initializes and configures the UX Audit Agent's components.
- Provides a mechanism for setting up and managing agent-specific settings.

### `applications\ux_audit_agent\components\compliance_checker.py`

Description: This module provides a compliance checking component to evaluate proposed changes against privacy policies and regulations, simulating a validation process.

Domain: RAG Logic (Regulatory, Accuracy, Governance)

Key Responsibilities:
- Evaluates proposed changes for compliance with privacy policies and regulatory rules.
- Returns simulated compliance results (80% chance of being compliant).
- Provides findings, including policy violations, required DPO review, and notes on the compliance status.
- Halts execution if non-compliant to ensure corrections are made before proceeding.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a cost estimation agent for UX-related changes based on an audit report, utilizing a mock pricing model.

**Domain:** RAG logic

**Key Responsibilities:**
- Estimates the cost of UX updates derived from an audit.
- Utilizes a mock pricing model to calculate costs.
- Returns a cost estimate, currency, and method used in calculation as part of the ComponentResult.

### `applications\ux_audit_agent\components\final_summary.py`

Description: This module provides a component for generating a final summary of the execution pipeline using all collected artifacts.

Domain: RAG logic

Key Responsibilities:
- Generates a complete summary of the entire execution pipeline.
- Utilizes all collected artifacts to provide detailed information about pipeline status and termination reasons.
- Returns a ComponentResult with an "ok" flag set to True, along with the generated final report.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module provides a financial audit component for generating mock financial reports and VAT calculations, intended for testing purposes.

**Domain:** Financial Agents

**Key Responsibilities:**
- Generates test data for financial computations
- Provides mock financial reports with net value, VAT rate, VAT amount, gross value, currency, and previous quarter budget
- Returns a `ComponentResult` object containing the generated report and related logs

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component for the Intergrax system, answering questions about its structure, features, and configuration.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Provides a general knowledge component that answers user queries
* Utilizes the `@component` decorator to register the function with the Intergrax framework
* Returns a structured response with answer, citations, and logs
* Includes mock data for demonstration purposes

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a PM (Project Manager) component for the UX audit pipeline, simulating a mock decision process based on random acceptance or rejection with comments.

**Domain:** RAG logic (Risk Assessment and Governance)

**Key Responsibilities:**
- Simulates a Project Manager's decision to accept or reject UX proposals
- Provides a 70% chance of approval and 30% chance of rejection
- Generates notes for the decision, which are conditional on whether the proposal is accepted or rejected
- Optionally stops pipeline execution if the proposal is rejected

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module defines a UX audit component that performs an analysis of UI/UX based on Figma mockups and returns a sample report with recommendations.

Domain: UX Audit Agent

Key Responsibilities:
- Performs a UX audit on Figma mockups
- Returns a sample report with recommendations
- Uses the ComponentContext and PipelineState to gather information
- Produces a UX report as output

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This Jupyter notebook contains a set of procedural tasks for conducting UX audits, policy compliance verification, and final report preparation.

Domain: UX Audit Agent / RAG Logic

Key Responsibilities:
- Perform UX audit on FIGMA mockups using the UX Auditor component.
- Verify changes comply with company policy using the Policy & Privacy Compliance Checker component.
- Prepare summary report for Project Manager using the Final Report component.
- Evaluate financial impact of changes using the Cost Estimation Agent component.
- Project Manager decision and project continuation.
- Final report preparation and synthesis.

Note: The notebook appears to be a comprehensive, well-structured workflow for conducting UX audits and policy compliance verification. However, some tasks seem to have missing inputs or outputs, which might require further investigation. Overall, the notebook is not marked as experimental, auxiliary, legacy, or incomplete.

### `bundle_intergrax_engine.py`

**Description:** This module provides utility functions for building a complete source code bundle of the Intergrax framework, including file metadata extraction, LLM instruction header generation, and bundling functionality.

**Domain:** Bundle Generation/LLM Instructions

**Key Responsibilities:**

* Extracts file metadata (rel path, module name, module group, sha256, lines, chars) from Python files
* Generates LLM instruction headers for the bundle
* Bundles source code into a single file with automatic ordering and dynamic module grouping
* Provides a module map and index for navigation within the bundle

### `generate_project_overview.py`

Description: This module is responsible for generating a Markdown file that provides an overview of the Intergrax project structure.

Domain: Utility modules

Key Responsibilities:
- Recursively scan the project directory.
- Collect all relevant source files (Python, Jupyter Notebooks, configurable).
- For each file, generate a structured summary using an LLM adapter.
- Build and write the Markdown file to disk.
- Optionally, use the generated document as context for LLM agents or other purposes.

Note: This module appears to be a core part of the Intergrax framework's utility modules.

### `intergrax\__init__.py`

DESCRIPTION: This is the entry point of the Intergrax framework, responsible for initializing and bootstrapping the application.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes Intergrax modules and dependencies
• Sets up application configuration and logging
• Defines main execution flow for the framework

### `intergrax\chains\__init__.py`

Description: This module initializes and configures the Intergrax chains, providing a foundation for building complex models.

Domain: Chain Initialization and Configuration

Key Responsibilities:
• Initializes chain components
• Configures chain settings and parameters
• Establishes connections between chain elements
• Sets up default behavior for chain operations

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module provides a flexible QA chain implementation using the LangChain framework, allowing for customization of retrieval, re-ranking, and prompt generation.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Builds a QA chain with hooks modifying data at stages:
	+ `on_before_build_prompt(payload): Dict -> Dict`
	+ `on_after_build_prompt(prompt_text, payload): str`
	+ `on_after_llm(answer_text, payload): str`
* Supports customization of retrieval, re-ranking, and prompt generation
* Includes support for citing sources in the output
* Provides a default prompt builder that can be overridden or extended
* Allows for optional use of a reranker and configuration of various parameters (e.g., top_k, min_score)

### `intergrax\chat_agent.py`

**Description:** This file implements the core logic of a chat agent, which provides a unified interface to interact with large language models and other tools. The chat agent makes routing decisions based on the input question and uses various components such as RAG (Retrieval-Augmented Generator) and tools agents to generate responses.

**Domain:** LLM Adapters / Agents

**Key Responsibilities:**

* Routing decisions via LLM
* Handling RAG and tool-based responses
* Managing memory and streaming functionality
* Generating structured output
* Providing a stable result with various metrics and statistics

### `intergrax\globals\__init__.py`

DESCRIPTION:
This module serves as the entry point for Intergrax's global settings and initialization.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
• Initializes global configuration settings.
• Imports other global modules.

### `intergrax\globals\settings.py`

Description: This module defines global configuration settings for the Intergrax framework, including language, locale, timezone defaults, model identifiers, and session memory/consolidation intervals.

Domain: Configuration

Key Responsibilities:
- Defines global default values for language, locale, and timezone.
- Provides default model identifiers for LLM models (Ollama and OpenAI).
- Sets default memory/consolidation intervals for user turns and cooldown seconds.
- Exposes these settings as a singleton instance (`GLOBAL_SETTINGS`) accessible throughout the framework.

### `intergrax\llm\__init__.py`

DESCRIPTION: 
This is the entry point for Intergrax's LLM adapters, responsible for initializing and configuring the available language models.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
• Initializes LLM adapter instances
• Defines interface for loading and registering custom adapters
• Configures default model settings

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This module provides a set of adapters for interacting with various large language models (LLMs) and associated logic.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines a protocol (`LLMAdapter`) for compatible adapters to implement.
* Provides an `OpenAIChatCompletionsAdapter` class that interacts with OpenAI's Chat Completions API.
* Includes various utility functions for message processing, tool support, and structured output validation.

Note: This module appears to be a collection of reusable components for integrating different LLMs into the Intergrax framework. It does not exhibit any obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm\messages.py`

**Description:** This module defines dataclasses for representing chat messages and their attachments within the Integrax framework.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Defines `AttachmentRef` dataclass for referencing attachments associated with a message or session
* Introduces `ChatMessage` dataclass, extending the OpenAI Responses API-compatible message format to support tool calls and attachments
* Provides `to_dict()` method for converting `ChatMessage` objects to dictionaries compatible with the OpenAI Responses API
* Implements custom reducer function `append_chat_messages()` for merging chat messages in LangGraph state updates

**Notes:** The module appears to be a key component of the Integrax framework's LLM adapters/RAG logic, providing essential data structures and utilities for representing and processing chat messages. No signs of experimental, auxiliary, legacy, or incomplete code are observed.

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point for Intergrax's LLM adapters, providing a centralized registry and import mechanism for various language model integrations.

Domain: LLM adapters

Key Responsibilities:
- Registers default adapter instances with the `LLMAdapterRegistry` for easy access
- Exports core classes and functions for working with LLM adapters, including base models and utility functions for data extraction and validation

### `intergrax\llm_adapters\base.py`

**Description:** 
This module provides a set of utilities and base classes for working with Large Language Models (LLMs) in the Intergrax framework, including token estimation and structured output generation.

**Domain:** LLM adapters

**Key Responsibilities:**

*   Provides a base class (`BaseLLMAdapter`) that includes shared utilities like token counting.
*   Offers a universal interface (`LLMAdapter`) for generating messages from given inputs.
*   Includes utility functions for estimating tokens, extracting JSON objects, and validating model output with Pydantic (v2/v1).
*   Supports structured output generation using specific models.

**Note:** This module appears to be well-structured, comprehensive, and production-ready. It does not show signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a minimal implementation of the Gemini chat adapter for the Intergrax framework, supporting basic chat usage without integrating additional tools.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an estimation of the context window size based on the model name
* Offers a `generate_messages` method to generate responses from the Gemini model
* Includes a `stream_messages` method for streaming responses
* Supports basic chat usage, but does not integrate tools or provide structured output

Note that this adapter is designed as a minimal implementation and lacks integration of additional tools.

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:**
This module provides a LangChain Ollama adapter, which enables the Intergrax framework to interact with Ollama language models through LangChain's ChatModel interface.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an adapter for Ollama models used via LangChain's ChatModel interface
* Estimates context window tokens based on model name (fallback path)
* Supports generating messages and streaming responses from the Ollama model
* Does not natively support tool-calling; instead, uses a planner-style pattern
* Offers structured output via prompt + validation functionality

Note: This file appears to be part of the Intergrax framework's standard implementation and is likely fully functional.

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an OpenAI LLM adapter for the Intergrax framework, utilizing the Responses API to generate chat responses.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a class `OpenAIChatResponsesAdapter` that extends `BaseLLMAdapter`
* Implements methods for generating single-shot and streaming completions using the OpenAI Responses API
* Supports tools integration and returns tool calls in a format compatible with Chat Completions
* Estimates context window tokens for various OpenAI models
* Caches maximum context window tokens for configured models

**Note:** This file appears to be part of the main Intergrax framework codebase, and is not marked as experimental or auxiliary.

### `intergrax\logging.py`

Description: This module configures and sets up the logging system for the Integrax framework.

Domain: Logging configuration

Key Responsibilities:
- Configures the logging level to display INFO messages and above.
- Sets a custom log format with timestamp, log level, and message.
- Forces override of any previous logging configurations.

### `intergrax\memory\__init__.py`

DESCRIPTION: This module initializes and sets up the memory management system within Intergrax, enabling storage and retrieval of data across various components.

DOMAIN: Memory Management

KEY RESPONSIBILITIES:
• Initializes memory storage mechanisms for efficient data caching.
• Defines interfaces for storing and retrieving data from memory. 
• Sets up default configuration for memory-related parameters.

### `intergrax\memory\conversational_memory.py`

**Description:** This module provides a universal in-memory conversation history component for the Integrax framework.

**Domain:** Conversational Memory Management

**Key Responsibilities:**

* Keep messages in RAM
* Provide API to add, extend, read and clear messages
* Optionally enforce max_messages limit
* Prepare messages for different model backends (get_for_model)
* No persistence capabilities (no files, no SQLite, no external storage)

Note: This class does not seem to be experimental or incomplete. It appears to be a core component of the Integrax framework.

### `intergrax\memory\conversational_store.py`

**Description:** This module provides an abstract interface and implementations for persistent storage of conversational memory, ensuring deterministic persistence, idempotent write operations, and safe interaction in async environments.

**Domain:** Conversational Memory Storage

**Key Responsibilities:**

* Abstracts the conversation history storage interface
* Ensures deterministic persistence and idempotent write operations
* Provides methods to load, save, append messages, and delete conversational memory sessions
* Supports swapping persistence backends without changing runtime logic

### `intergrax\memory\stores\__init__.py`

DESCRIPTION: This module initializes the memory store, providing a centralized location for data storage and retrieval within the Intergrax framework.

DOMAIN: Memory Store Management

KEY RESPONSIBILITIES:
• Initializes the memory store instance
• Defines interface for storing and retrieving data
• Sets up store configurations and logging mechanisms

### `intergrax\memory\stores\in_memory_conversational_store.py`

**Description:** This module provides an in-memory conversational store that enables local development, prototyping, and unit testing without requiring persistent storage.

**Domain:** Memory Stores - Conversational Store

**Key Responsibilities:**

* Provides an in-memory implementation of the `ConversationalMemoryStore` interface
* Supports loading and saving conversation histories using defensive copying
* Allows appending messages to existing conversations and persisting the updated state
* Enables deleting sessions with no-error semantics
* Offers optional diagnostic functionality for listing active session IDs

### `intergrax\memory\stores\in_memory_user_profile_store.py`

**Description:** This module implements an in-memory user profile store for the Intergrax framework, providing a simple and lightweight solution for storing and retrieving user profiles during unit testing, local development, or experimental purposes.

**Domain:** RAG logic (memory stores)

**Key Responsibilities:**
- Provides an in-memory implementation of `UserProfileStore` for unit tests, local development, or experiments.
- Stores user profiles in memory as dictionaries with user IDs as keys.
- Supports retrieving existing or default profiles, saving profiles to memory, and deleting stored profiles.
- Offers a debugging helper method to list stored user IDs.

### `intergrax\memory\user_profile_manager.py`

**Description:** The `user_profile_manager` module serves as a high-level facade for working with user profiles in the Intergrax framework. It provides convenient methods to manage user profiles, including loading and creating profiles, persisting profile changes, managing long-term user memory entries, and updating system-level instructions derived from the profile.

**Domain:** Memory management

**Key Responsibilities:**

* Provide methods to load or create a `UserProfile` for a given user ID
* Persist profile changes using the underlying `UserProfileStore`
* Manage long-term user memory entries (add, update, remove)
* Update system-level instructions derived from the profile
* Hide direct interaction with the underlying `UserProfileStore`

Note: The file appears to be complete and production-ready.

### `intergrax\memory\user_profile_memory.py`

**Description:** This module defines domain models for user and organization profiles, including memory entries, identities, preferences, and system instructions. These models are designed to be independent from storage or engine logic.

**Domain:** Memory Management

**Key Responsibilities:**

* Defines `MemoryKind` and `MemoryImportance` enums for categorizing memory entries
* Introduces `UserProfileMemoryEntry` dataclass for storing long-term user facts, insights, or notes
* Defines `UserIdentity` dataclass for representing high-level user descriptions
* Defines `UserPreferences` dataclass for storing stable user preferences influencing runtime behavior and LLM instructions
* Introduces `UserProfile` dataclass as a canonical aggregate of identity, preferences, system instructions, and memory entries

This module appears to be part of the main Intergrax framework, documenting core domain models related to user profile management.

### `intergrax\memory\user_profile_store.py`

**Description:** This module provides a protocol for persisting and retrieving user profiles in the Intergrax framework, decoupling backend storage from business logic.

**Domain:** User Profile Management

**Key Responsibilities:**
- Providing a storage interface for user profiles
- Loading and saving `UserProfile` aggregates
- Hiding backend-specific concerns for profile storage
- Implementing sane defaults for new users
- Ensuring backward compatibility for existing implementations

### `intergrax\multimedia\__init__.py`

Here is the documentation:

Description: Initializes and configures the multimedia module, handling imports and setup for other modules.

Domain: Multimedia Module Initialization

Key Responsibilities:
• Initializes the multimedia module with default settings
• Configures imports for other modules within the multimedia domain
• Establishes dependencies between multimedia-related components

### `intergrax\multimedia\audio_loader.py`

**Description:** The audio_loader.py module provides utilities for downloading and processing audio files from YouTube URLs, including the ability to extract audio and translate speech-to-text.

**Domain:** Multimedia

**Key Responsibilities:**

* Downloading audio files from YouTube URLs using yt_dlp
* Extracting audio from videos using FFmpeg
* Transcribing and translating speech-to-text using Whisper model
* Handling file output paths and creation for downloaded audio files

### `intergrax\multimedia\images_loader.py`

**intergrax\multimedia\images_loader.py**

Description: This module provides functionality to load and interact with multimedia content, specifically images, through the Integrax framework.

Domain: Multimedia Processing

Key Responsibilities:
- Loads images from file paths
- Utilizes ollama library for text-image interaction
- Defines a `transcribe_image` function to generate text descriptions based on images using specified models

Note: The presence of proprietary and confidential statements suggests that this module is part of the Integrax framework, which may limit its public availability or usage.

### `intergrax\multimedia\ipynb_display.py`

**Description:** 
This module provides utilities for displaying multimedia content (audio, images, and videos) within Jupyter notebooks.

**Domain:** Multimedia

**Key Responsibilities:**

* `display_audio_at_data`: Plays an audio file at a specified position.
* `_is_image_ext`: Checks if a path corresponds to an image file extension.
* `display_image`: Displays an image file or attempts to display it as an HTML image if the file exists but cannot be displayed by IPython.display.
* `_serve_path`: Serves a file from a designated directory, creating a copy with a unique filename if necessary.
* `display_video_jump`: Plays a video at a specified position, allowing for optional poster frame, autoplay, and playback rate control.

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides video loading, downloading from YouTube, video transcription, and frame extraction functionality.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloads videos from YouTube using the yt_dlp library
* Transcribes audio in video files using the Whisper model and WebVTT library
* Extracts frames from video files using OpenCV
* Saves extracted frames to a specified directory
* Saves metadata about extracted frames, including transcript text and timestamps
* Provides an optional limit on the number of frames to extract

### `intergrax\openai\__init__.py`

Description: The `__init__.py` file serves as the entry point for the OpenAI adapter within the Intergrax framework, facilitating its integration with other components.

Domain: LLM adapters

Key Responsibilities:
- Defines the API surface for the OpenAI adapter.
- Initializes and configures the adapter for use in the Intergrax framework.

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: 
This is the entry point for the RAG logic in the Intergrax framework, importing and initializing other components.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Imports necessary modules for RAG functionality
- Initializes and sets up RAG components and interfaces

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides a Rag (Retrieval-Augmented Generation) interface for OpenAI, enabling the creation and management of vector stores, file uploads, and retrieval of documents. It adheres to strict guidelines for knowledge retrieval and answer generation.

**Domain:** RAG logic / LLM adapters

**Key Responsibilities:**

* Create an instance of `IntergraxRagOpenAI` with an OpenAI client and a vector store ID
* Generate a prompt for the Rag interface, adhering to strict rules for knowledge retrieval and answer generation
* Ensure the existence of a vector store by its ID
* Clear all files loaded into the vector store
* Upload folders to the vector store, with optional file pattern filtering

### `intergrax\rag\__init__.py`

DESCRIPTION: The `__init__.py` file is responsible for initializing and setting up the RAG (Recurrent Attention Generator) logic within the Intergrax framework.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes the RAG module
- Sets up import paths and dependencies
- Exposes a public interface for interacting with the RAG module 

Note: This is a standard `__init__.py` file, indicating that the RAG logic is set up correctly within the Intergrax framework.

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust document loader with metadata injection and safety guards for various file types, including text files (.txt, .md), Office documents (.docx, .htm, .html), PDFs, Excel sheets, and images.

**Domain:** Data Ingestion/Document Loading

**Key Responsibilities:**

* Loads documents from various formats (text, office docs, pdf, excel, images) with customizable metadata extraction
* Supports OCR (Optical Character Recognition) for image and PDF files
* Provides extensible architecture via adapters and loaders for new file types
* Offers safety features like file size limits, max files per directory, and configurable logging
* Integrates with other Intergrax components, such as LLM adapters and multimedia loaders

### `intergrax\rag\documents_splitter.py`

**Description:** This module provides high-quality text splitting for RAG pipelines, generating stable chunk IDs and rich metadata.

**Domain:** RAG Logic

**Key Responsibilities:**

- Splits documents into semantically meaningful chunks
- Generates stable chunk IDs using anchors such as paragraph indices or page numbers
- Merges small tails and applies maximum chunk count per document
- Optionally calls custom metadata functions for each chunk

### `intergrax\rag\dual_index_builder.py`

**Description:** This module, `dual_index_builder.py`, is responsible for building two vector indexes: a primary index (`CHUNKS`) and an auxiliary index (`TOC`). It processes input documents, splits them into chunks, and embeds each chunk/document into the respective index.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

*   Builds two vector indexes: `CHUNKS` and `TOC`
*   Processes input documents, splitting them into chunks
*   Embeds each chunk/document into its respective index (`CHUNKS` or `TOC`)
*   Inserts embedded documents into the vector stores (`vs_chunks` and `vs_toc`)
*   Supports optional filtering of documents based on user-provided predicates

This module appears to be a crucial part of the Integrax framework, handling the construction of two key indexes for efficient retrieval and generation tasks.

### `intergrax\rag\dual_retriever.py`

**Description:** This module implements a Dual Retriever, which is a retrieval algorithm that first queries the Table of Contents (TOC) and then fetches local chunks from the same section/source.

**Domain:** RAG logic

**Key Responsibilities:**

*   Initializes the Dual Retriever with VectorstoreManager instances for TOC and CHUNKS
*   Provides methods to query both TOC and CHUNKS using embedding similarity
*   Implements expansion of context via TOC and search locally by parent_id
*   Merges, dedupes, and sorts hits based on similarity scores

### `intergrax\rag\embedding_manager.py`

**Description:** This module, `embedding_manager.py`, is a unified embedding manager that supports HuggingFace (SentenceTransformer), Ollama, or OpenAI embeddings. It provides features like provider switching, reasonable defaults for model names, batch/single text embedding with optional L2 normalization, and cosine similarity utilities.

**Domain:** RAG logic

**Key Responsibilities:**

* Unified embedding management for multiple providers (HuggingFace, Ollama, OpenAI)
* Provider switching with default models
* Embedding texts (batch or single) with optional L2 normalization
* Embedding for LangChain Documents (returns numpy array + aligned documents)
* Cosine similarity utilities and top-K retrieval
* Robust logging and shape validation
* Light retry mechanism for transient errors

Note: This file appears to be well-maintained, actively used, and is part of the main Intergrax framework.

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides a RAG (Retrieval-Augmented Generation) answerer that utilizes the Integrax framework to generate answers based on user queries. It incorporates retrieval, ranking, and generation capabilities.

**Domain:** RAG logic

**Key Responsibilities:**

- Retrieves relevant context fragments based on user queries
- Optionally re-ranks retrieved hits using a reranker model
- Builds context text from retrieved hits
- Generates citations for used hits
- Constructs system and user messages to be sent to the LLM (Large Language Model)
- Calls the LLM to generate answers, which can be in streaming or non-streaming modes
- Optionally generates structured output (e.g., output_model) based on the LLMS' capabilities

### `intergrax\rag\rag_retriever.py`

**Description:** This module implements a scalable, provider-agnostic RAG retriever for the Intergrax framework. It provides utilities for normalizing filters, embedding queries, and scoring similarities.

**Domain:** RAG logic

**Key Responsibilities:**

* Normalizes `where` filters for Chroma
* Normalizes query vector shape (1D/2D → [[D]])
* Unified similarity scoring:
	+ Chroma → converts distance to similarity = 1 - distance
	+ Others → uses raw similarity as returned
* Deduplication by ID + per-parent result limiting (diversification)
* Optional MMR diversification when embeddings are returned
* Batch retrieval for multiple queries
* Optional reranker hook (e.g., cross-encoder, re-ranking model)

**Note:** The file appears to be a core component of the Intergrax framework's RAG logic, and its functionality is well-documented. There is no indication that it is experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\re_ranker.py`

**Description:** This module implements a Re-Ranker class for ranking candidates based on cosine similarity to the query. It uses an embedding manager to embed texts in batches and performs L2-normalization. The Re-Ranker class supports score fusion with original retriever scores.

**Domain:** RAG (Relevant Answer Generation) logic

**Key Responsibilities:**

* Embeds texts in batches using an embedding manager
* Computes cosine similarities between query and candidate embeddings
* Provides a ReRanker class for ranking candidates based on cosine similarity
* Supports optional score fusion with original retriever scores
* Preserves the schema of input hits and adds 'rerank_score' and 'fusion_score' fields

Note: The file appears to be well-structured, implemented, and production-ready. There is no indication that it is experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module provides a unified interface for interacting with various vector stores, including ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**

* Initializes the target store and creates the collection/index (if needed)
* Upserts documents + embeddings with batching
* Queries top-K by cosine/dot/euclidean similarity
* Counts vectors
* Deletes by ids

Note that this module appears to be a part of the Intergrax framework, which is proprietary and confidential.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module provides a WindowedAnswerer class that builds upon the base Answerer to generate answers by processing retrieved candidates in windows. It includes functionality for memory-aware message construction and partial answer synthesis.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

*   Initializes the WindowedAnswerer with an Answerer, retriever, and optional verbosity
*   Builds context for each window using retrieved candidates
*   Constructs messages with memory-awareness, if available
*   Synthesizes final answer from partial answers generated for each window
*   Deduplicates sources collected during window processing
*   Stores the final answer and optional summary in memory, if configured

### `intergrax\runtime\__init__.py`

Description: The __init__.py file serves as the entry point for the Intergrax runtime, responsible for initializing and configuring the framework.

Domain: Framework initialization/entry point

Key Responsibilities:
• Initializes the Intergrax runtime environment
• Imports necessary modules and components
• Sets up global configuration and logging
• Exposes runtime-specific APIs and interfaces

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: The drop-in knowledge mode module is responsible for facilitating the integration of external knowledge into the Intergrax runtime without modifying its core functionality.

Domain: RAG logic

Key Responsibilities:
- Initializes and configures the knowledge graph for drop-in knowledge
- Integrates external knowledge sources into the Intergrax runtime
- Enables seamless incorporation of new knowledge without affecting existing functionality

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This file (`config.py`) contains configuration settings for the Intergrax framework's Drop-In Knowledge Runtime. It defines various parameters for interacting with tools, reasoning modes, and metadata.

**Domain:** LLM adapters and RAG logic

**Key Responsibilities:**

* Defines global configuration object `RuntimeConfig` for the Drop-In Knowledge Runtime
* Specifies core model and RAG backends (LLM adapter, embedding manager, vectorstore manager)
* Enables features like Retrieval-Augmented Generation (RAG), real-time web search, user profile memory
* Supports multi-tenancy with tenant ID and workspace ID
* Configures reasoning modes (direct, chain-of-thought internal) and allows for capturing reasoning metadata
* Provides metadata for app-specific instrumentation or tags

### `intergrax\runtime\drop_in_knowledge_mode\context\__init__.py`

Description: This module is responsible for initializing and managing the context in drop-in knowledge mode.

Domain: RAG logic

Key Responsibilities:
• Initializes the context for drop-in knowledge mode
• Sets up necessary components for RAG (Reactive Attention-based Generator) logic to function correctly
• Possibly loads or creates required data structures for the knowledge retrieval process

### `intergrax\runtime\drop_in_knowledge_mode\context\context_builder.py`

**Description:** This module provides context building functionality for Drop-In Knowledge Mode, responsible for deciding whether to use RAG and retrieving relevant document chunks from the vector store.

**Domain:** RAG logic

**Key Responsibilities:**

* Decide whether to use RAG for a given request
* Retrieve relevant document chunks from the vector store using session/user/tenant/workspace metadata
* Provide:
	+ A RAG-specific system prompt
	+ A list of retrieved chunks
	+ Debug metadata for observability

Note: This module is designed to be used in conjunction with other components, such as SessionStore and the runtime engine, to build conversation history and generate prompts.

### `intergrax\runtime\drop_in_knowledge_mode\context\engine_history_layer.py`

**Description:** This module encapsulates the logic for loading and preprocessing conversation history, applying token-based truncation, and updating RuntimeState with base history and debug information.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Loading raw conversation history from SessionStore
* Computing token usage for the raw history, if possible
* Applying per-request history compression strategies based on token budgets
* Updating RuntimeState with base history and debug information
* Handling degenerate cases where token budgets are extremely small or misconfigured.

### `intergrax\runtime\drop_in_knowledge_mode\engine\__init__.py`

**Description:** The `__init__.py` file initializes and sets up the drop-in knowledge mode engine, which enables external knowledge graphs to be integrated into the Intergrax system.

**Domain:** RAG logic

**Key Responsibilities:**
- Initializes the drop-in knowledge mode engine
- Sets up necessary components for integrating external knowledge graphs

### `intergrax\runtime\drop_in_knowledge_mode\engine\runtime.py`

**Description:** 
This module provides the core runtime engine for Intergrax's Drop-In Knowledge Mode, which enables a single entrypoint for various environments (FastAPI, Streamlit, CLI tools, etc.) to interact with the LLM.

**Domain:** RAG logic and conversational runtime engine

**Key Responsibilities:**

* Loads or creates chat sessions
* Appends user messages to the session
* Builds an LLM-ready context by combining system prompts, chat history, retrieved chunks from documents (RAG), web search context, and tools results
* Calls the main LLM adapter with the enriched context to produce a final answer
* Appends assistant messages to the session and returns a RuntimeAnswer object

The code appears to be well-structured, complete, and production-ready. There are no indications of experimental or auxiliary code.

### `intergrax\runtime\drop_in_knowledge_mode\engine\runtime_state.py`

Description: This module defines the RuntimeState class, which represents a mutable state object passed through the runtime pipeline of the Intergrax framework.

Domain: Drop-in Knowledge Mode Runtime State Management

Key Responsibilities:
- Aggregates request and session metadata.
- Stores ingestion results.
- Manages conversation history and model-ready messages.
- Tracks usage flags for RAG, websearch, tools, memory subsystems.
- Provides tools traces and agent answer.
- Supports full debug_trace for observability & diagnostics.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\__init__.py`

DESCRIPTION: This module initializes and configures the knowledge ingestion process for drop-in mode, responsible for importing and processing external knowledge into the Intergrax framework.

DOMAIN: Knowledge Ingestion

KEY RESPONSIBILITIES:
- Initializes knowledge ingestion configuration
- Imports required libraries and dependencies
- Sets up data processing pipelines for external knowledge integration

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\attachments.py`

**Description:** This module provides utilities for resolving attachments in the context of Intergrax's Drop-In Knowledge Mode, decoupling attachment storage from consumption by RAG components.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Defines an `AttachmentResolver` protocol to abstract attachment resolution
* Provides a minimal implementation (`FileSystemAttachmentResolver`) for local filesystem-based URIs
* Offers methods to resolve attachments into local file paths, raising exceptions for unsupported URI schemes or non-existent attachments

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\ingestion_service.py`

**Description:** This module implements an attachment ingestion pipeline for the Drop-In Knowledge Mode in the Intergrax framework, responsible for loading and processing attachments from sessions.

**Domain:** Data Ingestion (RAG-related)

**Key Responsibilities:**

* Resolves AttachmentRef objects into filesystem Paths using AttachmentResolver
* Loads documents using IntergraxDocumentsLoader
* Splits documents into chunks using IntergraxDocumentsSplitter
* Embeds chunks via IntergraxEmbeddingManager
* Stores vectors via IntergraxVectorstoreManager
* Returns a structured IngestionResult per attachment

### `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`

Description: This module contains initialization and setup for drop-in knowledge mode prompts within the Intergrax runtime.

Domain: RAG logic

Key Responsibilities:
* Initializes prompt-related configurations
* Sets up necessary dependencies for knowledge transfer
* Establishes framework connections for drop-in functionality

### `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`

Description: This module provides utilities for building history-related prompts in the Drop-In Knowledge Mode of the Intergrax framework.

Domain: LLM adapters

Key Responsibilities:
- Provides a `HistorySummaryPromptBuilder` strategy interface for customizing history summarization prompts.
- Includes a default implementation (`DefaultHistorySummaryPromptBuilder`) that generates a generic system prompt for summarizing older conversation turns.
- Allows for future extension and customization of the prompt based on request, strategy, and message inputs.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`

**Description:** This module provides a prompt builder for the Retrieval-Augmented Generator (RAG) logic, enabling the creation of contextual prompts for the model.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a strategy interface (`RagPromptBuilder`) for building the RAG-related part of the prompt
* Offers a default implementation (`DefaultRagPromptBuilder`) that injects retrieved chunks into system-level context messages
* Allows custom implementations to be provided and passed to `DropInKnowledgeRuntime` to control the exact system prompt text and chunk formatting

### `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`

**Description:** This module provides a mechanism for building web search-related prompts in the Intergrax framework, enabling users to interact with external knowledge sources.

**Domain:** LLM adapters

**Key Responsibilities:**
- Provides a `WebSearchPromptBuilder` protocol for custom implementation of web search prompt construction
- Offers a default `DefaultWebSearchPromptBuilder` class that summarizes web documents and constructs system-level messages
- Includes a `WebSearchPromptBundle` dataclass to hold the constructed prompt elements (context messages and debug information)
- Allows users to control aspects such as document summarization, result injection, and system message wording through custom implementation or configuration

Note: The provided file appears to be a well-documented and functional part of the Intergrax framework.

### `intergrax\runtime\drop_in_knowledge_mode\reasoning\__init__.py`

DESCRIPTION:
This module initializes and sets up the reasoning components for drop-in knowledge mode in the Intergrax framework.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes the reasoning engine
- Sets up the knowledge graph data structures
- Configures the reasoning parameters

### `intergrax\runtime\drop_in_knowledge_mode\reasoning\reasoning_layer.py`

**Description:** This module provides the reasoning layer for the Drop-In Knowledge Runtime, responsible for augmenting system instructions based on configured reasoning modes.

**Domain:** RAG logic

**Key Responsibilities:**

* Augment system instructions based on ReasoningMode.
* Never store or expose raw chain-of-thought content.
* Write lightweight observability info into RuntimeState.debug_trace.
* Apply configured reasoning mode to base system instructions.
* Update RuntimeState.reasoning_mode and debug_trace["reasoning"].
* Handle unknown ReasoningModes as DIRECT.

Note: This file appears to be a standard implementation, with no signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\drop_in_knowledge_mode\responses\__init__.py`

Description: The __init__.py file in the drop_in_knowledge_mode responses directory initializes and configures response handling for the Intergrax framework's knowledge dropping feature.

Domain: RAG logic

Key Responsibilities:
- Initializes response generation modules for drop-in knowledge mode
- Configures response handling pipelines
- Defines entry points for knowledge dropping functionality

### `intergrax\runtime\drop_in_knowledge_mode\responses\response_schema.py`

**Description:** This module defines data models for the Drop-In Knowledge Mode runtime in Intergrax framework. It provides structures for request and response data between applications (e.g., FastAPI, Streamlit) and the runtime.

**Domain:** RAG logic

**Key Responsibilities:**

* Defines high-level contract between applications and the DropInKnowledgeRuntime
* Exposes citations, routing information, tool calls, and basic statistics
* Provides dataclasses for request (`RuntimeRequest`) and response (`RuntimeAnswer`)
* Includes history compression strategies for conversation history
* Offers a simple strategy for compressing the conversation history before sending it to the LLM

### `intergrax\runtime\drop_in_knowledge_mode\session\__init__.py`

Description: This module initializes and manages session-specific knowledge storage in drop-in knowledge mode for the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
- Initializes the session's knowledge store
- Sets up data structures for storing and retrieving contextual information
- Establishes connections to relevant knowledge graphs or databases (assuming a graph database is used, if not then it will be different)

### `intergrax\runtime\drop_in_knowledge_mode\session\chat_session.py`

**Description:** This module contains the domain model and utility functions for managing chat sessions within the Intergrax framework.

**Domain:** Session Management

**Key Responsibilities:**

* Represents a single chat session with its state (open/closed, attachments, etc.)
* Provides utility methods for updating session timestamps, incrementing user turns, and marking sessions as closed
* Defines domain-level enums for session status and close reasons
* Isolated from I/O operations; persistence handled by manager/storage components

### `intergrax\runtime\drop_in_knowledge_mode\session\in_memory_session_storage.py`

**Description:** 
This module implements an in-memory session storage for handling conversational sessions within the Intergrax framework. It provides basic CRUD operations for managing chat sessions, as well as maintaining conversation histories.

**Domain:** Conversational Session Management

**Key Responsibilities:**

* Store and manage chat sessions in an in-process dictionary
* Maintain per-session conversation history using ConversationalMemory
* Apply a simple FIFO trimming policy via ConversationalMemory's max_messages setting
* Provide basic CRUD operations for managing chat sessions, including creation, retrieval, update, and deletion
* Support appending messages to conversation histories and retrieving ordered conversation histories
* Offer safety fallbacks for missing conversational memories or session metadata

### `intergrax\runtime\drop_in_knowledge_mode\session\session_manager.py`

**Description:** This module manages chat sessions, orchestrating their lifecycle and providing a stable API for the runtime engine (DropInKnowledgeRuntime). It integrates with user/organization profile managers to expose prompt-ready system instructions per session.

**Domain:** Session Management

**Key Responsibilities:**

* Orchestrate session lifecycle on top of a SessionStorage backend
* Provide a stable API for the runtime engine (DropInKnowledgeRuntime)
* Integrate with user/organization profile managers to expose prompt-ready system instructions per session
* Optionally trigger long-term user memory consolidation for a session
* Manage session metadata, including creation, saving, and closing sessions

### `intergrax\runtime\drop_in_knowledge_mode\session\session_storage.py`

**Description:** This module defines a low-level storage interface for chat sessions and conversation histories, providing basic CRUD operations.

**Domain:** Session Storage/Management

**Key Responsibilities:**

* Persist and load ChatSession objects
* Persist and load conversation history (ChatMessage sequences) for a given session
* Provide methods to create, read, update, and delete sessions and their metadata
* Support listing recent sessions for a user
* Append messages to the conversation history of a session
* Retrieve the ordered conversation history for a given session

### `intergrax\runtime\organization\__init__.py`

Description: This module serves as the entry point for the organization-related functionality within the Intergrax runtime.

Domain: Organization Management

Key Responsibilities:
* Initializes and configures organization-specific settings and services.
* Provides an interface for registering and managing organizations in the system.

### `intergrax\runtime\organization\organization_profile.py`

**Description:** This module provides classes and utilities for managing organization profiles within the Integrax framework.

**Domain:** Organization management

**Key Responsibilities:**

* Define a stable identification data class (`OrganizationIdentity`) for organizations
* Define an organization-level preferences class (`OrganizationPreferences`) influencing runtime behavior
* Implement an `OrganizationProfile` class as a single source of truth for an organization's long-term profile, mirroring the `UserProfile` structure
* Provide legacy fields for backwards compatibility and potential migration (e.g., summary_instructions, domain_summary)

### `intergrax\runtime\organization\organization_profile_instructions_service.py`

**Description:** This module provides a service to generate and update organization-level system instructions using an LLM (Large Language Model) adapter, based on an OrganizationProfile.

**Domain:** RAG logic

**Key Responsibilities:**

* Load an OrganizationProfile via the OrganizationProfileManager.
* Build an LLM prompt from the profile's identity, preferences, domain/knowledge summaries, and memory entries.
* Call the LLMAdapter to generate compact instructions.
* Persist the result via the OrganizationProfileManager.update_system_instructions() method.

Note: The file appears to be a crucial part of the Intergrax framework, responsible for generating organization-level system instructions using an LLM adapter.

### `intergrax\runtime\organization\organization_profile_manager.py`

**Description:** This module, `OrganizationProfileManager`, serves as a high-level facade for working with organization profiles in the Intergrax framework. It provides convenient methods to load, save, and manage organization profiles, while hiding direct interaction with the underlying storage.

**Domain:** Organization management

**Key Responsibilities:**

* Load an organization profile by ID
* Save changes to an organization profile
* Resolve system instructions for an organization (deterministic logic)
* Update system instructions for an organization
* Delete an organization's stored profile data

### `intergrax\runtime\organization\organization_profile_store.py`

**Description:** The `organization_profile_store` module defines a protocol for persistent storage of organization profiles in the Integrax framework. It abstracts away backend-specific concerns, providing a standardized interface for loading, saving, and deleting profiles.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Providing a protocol for persistent storage of organization profiles
* Loading and saving `OrganizationProfile` aggregates
* Returning initialized profiles with default values if no data exists yet
* Overwriting previously stored profiles when saving new ones
* Deleting profile data for a given organization ID
* Tolerating unknown organization IDs without error

### `intergrax\runtime\organization\stores\__init__.py`

Description: The __init__.py file serves as the entry point for the stores module, responsible for organizing and exporting store-related functionality within the Intergrax runtime.

Domain: Data Ingestion/Stores Management

Key Responsibilities:
- Initializes the stores module
- Exports store-related functions and classes
- Sets up necessary configurations and dependencies

### `intergrax\runtime\organization\stores\in_memory_organization_profile_store.py`

**Description:** This module provides an in-memory implementation of the OrganizationProfileStore interface, suitable for unit tests, local development, and experiments.

**Domain:** RAG logic / Organization management

**Key Responsibilities:**

* Provides a store for organization profiles in memory
* Supports retrieving existing or default profiles by organization ID
* Allows persisting or updating profiles in memory
* Enables deleting stored profiles by organization ID

### `intergrax\runtime\user_profile\__init__.py`

Description: This module initializes the User Profile component, providing essential functionality for managing user-related data within the Intergrax framework.

Domain: Configuration

Key Responsibilities:
- Initializes the User Profile component
- Sets up necessary configuration and dependencies
- Exposes user profile management interface

### `intergrax\runtime\user_profile\session_memory_consolidation_service.py`

**Description:** This module provides a service responsible for converting a single chat session history into structured long-term memory entries for the user profile.

**Domain:** RAG logic

**Key Responsibilities:**

* Takes the session conversation (ChatMessage sequence) and extracts:
	+ USER_FACT items (stable facts / goals)
	+ PREFERENCE items (communication / workflow preferences)
	+ optional SESSION_SUMMARY item (short global summary)
* Maps the extracted data into UserProfileMemoryEntry objects
* Persists them through UserProfileManager (add_memory_entry)
* Optionally refreshes user-level system instructions as part of the same pipeline

**Note:** This file appears to be a key component of the Intergrax framework, and its functionality is crucial for storing long-term memory entries for users.

### `intergrax\runtime\user_profile\user_profile_debug_service.py`

**Description:** This module provides a read-only service responsible for building UserProfileDebugSnapshot objects, which aggregate data from the user profile and recent chat sessions.

**Domain:** User Profile Debugging Service

**Key Responsibilities:**

* Retrieves UserProfile objects from the UserProfileManager
* Aggregates data from the UserProfile and SessionManager
* Builds UserProfileDebugSnapshot objects with relevant debug information
* Provides methods for exposing a "debug user profile" API endpoint, feeding an admin/developer UI panel, or ad-hoc diagnostics during development.

### `intergrax\runtime\user_profile\user_profile_debug_snapshot.py`

Description: This module provides debug views and utility functions for user profiles, including memory entry statistics and recent sessions.

Domain: User Profile Debugging

Key Responsibilities:
- Provides a `SessionDebugView` to display a single chat session in a lightweight format.
- Offers a `MemoryEntryDebugView` to expose relevant fields of a `UserProfileMemoryEntry`.
- Creates an immutable `UserProfileDebugSnapshot` with user profile state for debugging and observability, including recent memory entries and sessions.
- Implements utility functions like `build_memory_kind_counters` and `from_domain_session/from_memory_entry` to aid in constructing the snapshot.

### `intergrax\runtime\user_profile\user_profile_instructions_service.py`

**Description:** 
This module provides a high-level service for generating and updating user-level system instructions using an LLM adapter, based on the user's profile.

**Domain:** LLM adapters

**Key Responsibilities:**

- Load UserProfile via UserProfileManager.
- Build an LLM prompt using identity, preferences, and memory entries.
- Call LLMAdapter.generate_messages() to obtain a compact, stable system prompt.
- Persist the result via UserProfileManager.update_system_instructions().
- Handle user ID, force regeneration, and configuration options.

### `intergrax\supervisor\__init__.py`

DESCRIPTION: 
This is the entry point for the Intergrax supervisor, responsible for managing and orchestrating various tasks within the framework.

DOMAIN: Supervisor Logic

KEY RESPONSIBILITIES:
• Initializes the supervisor module
• Defines the entry points for other components to interact with the supervisor
• Possibly contains configuration or setup code for the supervisor

### `intergrax\supervisor\supervisor.py`

**Description:** This is the core module of the Intergrax framework's supervisor, responsible for planning and executing tasks based on user queries. It integrates with large language models (LLMs) to generate plans and assign components to execute them.

**Domain:** Supervision and task execution

**Key Responsibilities:**

*   Planning: generating a plan to achieve a goal based on user input
*   LLM integration: utilizing LLMs to decompose tasks, assign components, and retrieve results
*   Component management: registering and managing available components for task execution
*   Context creation: building execution contexts for component-based task execution
*   Plan analysis: analyzing generated plans to extract relevant information (e.g., source, analysis)
*   Heuristic planning: enabling heuristic fallback planning in case of LLM failure or insufficient data

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides the core components and functionality for managing pipeline steps within the Integrax framework.

**Domain:** Supervisor Components

**Key Responsibilities:**

* Defines dataclasses for `ComponentResult` and `ComponentContext`
* Provides a `Component` class representing a step in the pipeline, with attributes for name, description, usage conditions, and implementation function
* Implements the `run` method to execute a component, handling availability checks and exceptions
* Offers a decorator (`component`) for registering components with metadata

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides universal prompts and default template structures for the intergrax Supervisor, defining the planning process and data format requirements.

**Domain:** RAG logic / Planning / Unified Supervisor Prompts

**Key Responsibilities:**

- Define the planning process and its key principles (decomposition-first, correctness, observability).
- Specify output expectations from each step of the plan (e.g., outputs allowed by chosen components).
- Outline validation checks for a valid plan (correct assignment of methods and components, correct data flow, etc.).
- Provide default prompt templates for the intergrax Supervisor (plan system and user templates).

**Note:** The code appears to be well-documented, complete, and production-ready.

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module is responsible for transforming a plan into a runnable LangGraph pipeline. It provides utilities to create nodes from plan steps and build the graph topology.

**Domain:** Supervisor components, Plan execution, Graph building

**Key Responsibilities:**

*   Provides utility functions to ensure state defaults and append debug logs
*   Resolves inputs for plan steps based on artifacts and global state
*   Persists outputs from node executions into artifacts
*   Creates readable node names using slugification and node name composition
*   Constructs LangGraph nodes by executing plan steps with optional component execution
*   Builds the graph topology through stable topological ordering of plan steps

This module appears to be a critical part of the Integrax framework, responsible for transforming plans into executable graphs. The code is well-structured, and the use of type hints and docstrings makes it easy to understand the functionality. Overall, this module seems complete and not experimental or legacy.

### `intergrax\system_prompts.py`

Description: This module defines a specific RAG (Responsibly Assessed Guesswork) system instruction for the Integrax framework.

Domain: RAG logic

Key Responsibilities:
- Defines the rules and procedures for a strict RAG approach in the context of the Integrax framework.
- Specifies the guidelines for searching, verifying, and responding to user queries based on provided documents.
- Outlines the formatting requirements for responses, including citing sources and avoiding speculation.
- Emphasizes the importance of precision, accuracy, and transparency in response generation.

### `intergrax\tools\__init__.py`

DESCRIPTION: The `__init__.py` file is the entry point for the Intergrax tools package, responsible for initializing and organizing tool-related functionality.

DOMAIN: Tools initialization

KEY RESPONSIBILITIES:
- Initializes the tool suite
- Sets up package structure and imports
- Provides a central hub for tool-related operations

### `intergrax\tools\tools_agent.py`

**Description:** The `tools_agent.py` file provides a class-based interface for orchestrating tool usage within the Intergrax framework. It enables interaction with Large Language Models (LLMs) and utilizes a registry of available tools.

**Domain:** LLM adapters, Tools orchestration

**Key Responsibilities:**

*   **ToolsAgent Class:**
    *   Initializes an instance of `ToolsAgent` with an LLM adapter, tool registry, memory, configuration, and verbosity settings.
    *   Defines methods for:
        *   Pruning messages to conform to OpenAI's tool usage requirements
        *   Building output structures based on tool traces or extracted JSON from text
        *   Running the tools orchestration entrypoint with input data, context, stream mode, tool choice, and output model options
*   **ToolsAgentConfig Class:**
    *   Defines configuration settings for temperature, maximum answer tokens, maximum tool iterations, system instructions, and system context template.
*   **Helper Functions:**
    *   `_maybe_import_pydantic_base`: Tries to import Pydantic base class for v2/v1 compatibility.
    *   `_instantiate_output_model`: Creates an instance of the output model based on payload data, handling JSON strings, dictionaries, and Pydantic models.
    *   `_extract_json_from_text`: Tolerantly extracts the first JSON object from a given text.

### `intergrax\tools\tools_base.py`

**Description:** This module provides base classes and utilities for building and managing tools within the Integrax framework.

**Domain:** Tools management

**Key Responsibilities:**

* Defines a `ToolBase` class as a base for all tool implementations, providing common attributes and methods
* Offers a `ToolRegistry` class to store and manage registered tools, including exporting them in a format compatible with the OpenAI Responses API
* Includes utility functions like `_limit_tool_output` to safely truncate long tool outputs
* Enables strict validation of tool parameters using Pydantic (optional)

### `intergrax\websearch\__init__.py`

DESCRIPTION: This package initializes the web search functionality for the Intergrax framework, providing an entry point and importing necessary modules.

DOMAIN: Web Search API

KEY RESPONSIBILITIES:
* Initializes the web search application
* Imports relevant modules and configurations
* Sets up dependencies for web search functionality

### `intergrax\websearch\cache\__init__.py`

**Description:** This module implements a simple in-memory query cache for web search results with optional time-to-live (TTL) and maximum size.

**Domain:** Web Search Cache

**Key Responsibilities:**
- Provides an immutable `QueryCacheKey` class to describe unique web search configurations.
- Defines a `QueryCacheEntry` dataclass to store cached web documents and metadata.
- Implements the `InMemoryQueryCache` class, which stores query results in memory with optional TTL and maximum size.
  - Allows getting cached results for a given query key using `get()`.
  - Enables setting new cache entries using `set()`.

### `intergrax\websearch\context\__init__.py`

DESCRIPTION: This module initializes the web search context, handling setup and registration of relevant components.

DOMAIN: Web Search Context Initialization

KEY RESPONSIBILITIES:
• Registers necessary components for web search functionality.
• Sets up the initial context for subsequent operations.
• (Note: File appears to be a basic initialization module; more detailed responsibilities may exist elsewhere in the framework.)

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides a WebSearchContextBuilder class to build LLM-ready textual context and chat messages from web search results.

**Domain:** RAG logic

**Key Responsibilities:**

* Builds textual context strings from WebDocument objects or serialized dicts (as returned by WebSearchExecutor with serialize=True)
* Supports building context for top-N documents
* Includes options to include snippet, URL, and source labels in the context header
* Provides methods to build chat messages (system + user) for any chat-style LLM
* Enforces "sources-only" mode by using strict system prompts that require users to base their answers only on information contained in the 'Web sources' section.

### `intergrax\websearch\fetcher\__init__.py`

**Description:** This is the initialization module for the web search fetcher component, responsible for initializing and configuring the fetcher instance.

**Domain:** Web Search Fetching

**Key Responsibilities:**
* Initializes the fetcher instance
* Configures fetcher settings (e.g., URL, query parameters)
* Sets up logger for fetcher-related logs

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module provides lightweight and advanced HTML extraction utilities for web search functionalities.

**Domain:** Web Search Extractor Utilities

**Key Responsibilities:**

* Performs basic HTML metadata extraction on a PageContent instance:
	+ Extracts title.
	+ Extracts meta description.
	+ Extracts <html lang> attribute.
	+ Extracts Open Graph meta tags (og:*).
	+ Produces a plain-text version of the page.
* Performs advanced readability-based extraction on a PageContent instance:
	+ Removes obvious boilerplate elements (scripts, styles, iFrames, navigation).
	+ Prefers trafilatura (when available) to extract primary readable content.
	+ Fallbacks to BeautifulSoup plain-text extraction if trafilatura fails.
	+ Normalizes whitespace and reduces noise.

**Note:** The file appears to be a part of the Intergrax framework's web search functionality, providing essential utilities for metadata and text extraction.

### `intergrax\websearch\fetcher\http_fetcher.py`

**intergrax\websearch\fetcher\http_fetcher.py**

Description: This module provides an asynchronous HTTP fetcher for web pages, encapsulating the Intergrax framework's default headers and handling basic transport-level concerns.

Domain: Web search/HTTP fetching

Key Responsibilities:
- Perform HTTP GET requests with sane defaults (e.g., timeout, user agent)
- Capture URL, status code, raw HTML, and body size from the response
- Return a `PageContent` instance on success or None on failure

### `intergrax\websearch\integration\__init__.py`

DESCRIPTION: The __init__.py file serves as the entry point for web search integration, importing and initializing necessary components.

DOMAIN: Integration

KEY RESPONSIBILITIES:
• Initializes modules required for web search functionality
• Registers adapters for various data sources
• Sets up configuration for search query processing
• Provides a basic interface for integrating with other Intergrax components

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a LangGraph-compatible web search node wrapper that delegates search operations to an externally configured or internally created `WebSearchExecutor` instance.

**Domain:** LLM adapters / Web Search Integration

**Key Responsibilities:**

* Encapsulates configuration of `WebSearchExecutor`
* Provides sync and async node methods (`run` and `run_async`) operating on `WebSearchState`
* Delegates search operations to the `WebSearchExecutor` instance
* Offers a functional, synchronous wrapper (`websearch_node`) for simple integrations
* Offers a functional, async wrapper (`websearch_node_async`) suitable for LangGraph graphs and async environments

### `intergrax\websearch\pipeline\__init__.py`

DESCRIPTION: This module initializes and configures the web search pipeline.

DOMAIN: Pipeline initialization/configuration

KEY RESPONSIBILITIES:
• Initializes the web search pipeline with default settings.
• Loads pipeline configurations from external sources (e.g., files, databases).
• Performs sanity checks on configuration data before proceeding.

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module implements a search pipeline for web content using multiple providers. It orchestrates the search process, fetching, extraction, deduplication, and quality scoring of web documents.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Orchestrates multi-provider web search, fetching, and extraction
* Performs simple deduplication via text-based dedupe key
* Minimal and testable design with no direct LLM coupling
* Async fetching with rate limiting (TokenBucket)
* Sorts results by quality score and source rank

Note: The code appears to be well-structured and complete. There are no indications of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\websearch\providers\__init__.py`

**Description:** This module serves as the entry point for web search providers in the Intergrax framework, enabling external integration of various search engines and services.

**Domain:** LLM adapters

**Key Responsibilities:**
- Registers available web search provider modules.
- Exposes a standard interface for interacting with these providers.

### `intergrax\websearch\providers\base.py`

**Description:** This module defines a base interface for web search providers in the Integrax framework, providing a standardized way to execute searches and negotiate capabilities.

**Domain:** Web Search Adapters

**Key Responsibilities:**

* Provides an abstract base class `WebSearchProvider` with a stable interface for web search providers
* Accepts a provider-agnostic `QuerySpec` object as input for search requests
* Requires implementations to return a ranked list of `SearchHit` items, with optional support for language and freshness features
* Exposes capabilities for feature negotiation through the `capabilities()` method
* Allows for optional resource cleanup through the `close()` method

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search (v7) provider for the Intergrax framework, allowing users to perform web search queries using Bing's REST API.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Establishes a connection to Bing's Web Search API v7
* Supports language and region filtering
* Enables freshness filtering (Day, Week, Month)
* Provides safeSearch filtering (Off, Moderate, Strict)
* Handles API key management and authentication
* Returns search results in the format of `SearchHit` objects

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module provides a web search provider for Google Custom Search Engine (CSE), allowing the Intergrax framework to query and retrieve search results from Google's REST API.

**Domain:** WebSearch Providers

**Key Responsibilities:**

* Initializes connection to Google CSE using provided API key and engine ID
* Builds request parameters for search queries, including language filtering and UI language settings
* Sends requests to Google CSE API and retrieves search results in JSON format
* Parses and converts search results into `SearchHit` objects for processing by the framework
* Supports query capabilities, such as language and freshness filters
* Closes the session when no longer needed

**Status:** Not experimental, auxiliary, legacy, or incomplete.

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a Google Places provider for the Intergrax framework, allowing for text search and details retrieval from Google's Places API.

**Domain:** Web Search Providers

**Key Responsibilities:**

*   Provides a `GooglePlacesProvider` class that extends `WebSearchProvider`
*   Supports text search by arbitrary query (name + city, category, etc.)
*   Returns core business data, including name, address, location, rating, user ratings total, types, website, international phone number, and opening hours
*   Allows for fetching details of a single place ID with additional information
*   Maps results to `SearchHit` objects
*   Handles API key management and caching

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module implements a Reddit search provider for the Intergrax framework, utilizing the official OAuth2 API. It provides capabilities for searching posts and fetching top-level comments.

**Domain:** WebSearch Providers

**Key Responsibilities:**

* Authenticate using application-only OAuth2 (client_credentials) with environment variables REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT
* Perform search queries on the Reddit API with support for filtering by freshness and language (though language filter is currently ignored)
* Fetch top-level comments for each post
* Map search results to SearchHit objects
* Implement capabilities such as max page size and supports_freshness

### `intergrax\websearch\schemas\__init__.py`

Description: This is the entry point for the web search schema definitions, responsible for importing and registering all available schemas.

Domain: Schema Management

Key Responsibilities:
* Imports and registers web search schema definitions
* Provides a centralized interface for accessing registered schemas

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass `PageContent` to represent the fetched content of a web page, including its HTML and derived metadata.

**Domain:** Web search

**Key Responsibilities:**

* Represents the fetched and optionally extracted content of a web page.
* Encapsulates raw HTML and derived metadata for post-processing stages (extraction, readability, deduplication).
* Provides methods to filter out failed or empty fetches (`has_content`), generate a truncated text snippet for logging and debugging (`short_summary`), and estimate the approximate size of the content in kilobytes (`content_length_kb`).

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines the QuerySpec data class, which represents a canonical search query specification used by all web search providers. It encapsulates various parameters that can be used to customize search queries.

**Domain:** Web Search Providers / API Integration

**Key Responsibilities:**
- Defines a frozen data class `QuerySpec` with attributes for specifying search queries.
- Provides methods for normalizing the query string (`normalized_query`) and capping the number of top results (`capped_top_k`).

### `intergrax\websearch\schemas\search_hit.py`

Description: This module defines a data class for representing search result metadata, encapsulating provider-agnostic information for individual hits.

Domain: Websearch schemas

Key Responsibilities:
- Provides a `SearchHit` data class with various attributes (e.g., provider, query, rank, title, URL) and methods.
- Offers validation in the `__post_init__` method to ensure minimal correctness of the hit's properties.
- Includes utility functions like `domain()` for extracting domain information from the URL and `to_minimal_dict()` for creating a simplified representation of the hit.

### `intergrax\websearch\schemas\web_document.py`

**FILE PATH:** intergrax\websearch\schemas\web_document.py

**Description:** This module defines a unified data structure for representing and processing web documents, integrating provider metadata with extracted content and analysis results.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Represents a fetched and processed web document as a `WebDocument` data class.
* Integrates source search hit metadata with extracted page content and analysis results (deduplication key, quality score, source rank).
* Provides methods for document validation (`is_valid()`), combined textual content generation (`merged_text()`, with optional maximum length control), and summary line creation (`summary_line()`).

### `intergrax\websearch\service\__init__.py`

Description: This module initializes the web search service for Intergrax, setting up essential components and configurations.

Domain: Web Search Service

Key Responsibilities:
• Initializes service settings and configurations
• Sets up necessary dependencies and imports
• Registers web search endpoints and APIs
• Optionally loads additional configurations or modules

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level interface for web search and answering, integrating with Language Model (LLM) adapters to generate responses.

**Domain:** Web Search

**Key Responsibilities:**
- Runs web search via `WebSearchExecutor`
- Builds LLM-ready context/messages from web documents
- Calls any LLMAdapter to generate a final answer
- Provides synchronous and asynchronous interfaces for convenience

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module provides a high-level web search executor that constructs and executes queries, handling configuration and caching for efficient result retrieval.

**Domain:** Web Search Executor

**Key Responsibilities:**

* Construct QuerySpec from raw query and configuration
* Execute SearchAndReadPipeline with chosen providers
* Convert WebDocument objects into LLM-friendly dicts
* Handle configuration and caching for efficient result retrieval
* Support asynchronous execution of web search pipeline
* Provide optional query cache for serialized results

### `intergrax\websearch\utils\__init__.py`

Description: This utility module initializes and configures web search functionality within the Intergrax framework.
Domain: Web Search Utilities

Key Responsibilities:
- Initializes web search utilities for the Intergrax framework
- Configures necessary parameters and settings for web search functionality
- Provides a centralized entry point for web search-related operations

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides utilities for deduplication in the web search pipeline, specifically for detecting near-identical documents. It includes functions for normalizing text and generating stable SHA-256 based deduplication keys.

**Domain:** Web Search Utilities

**Key Responsibilities:**
* Normalizes input text for deduplication
* Generates a stable SHA-256 based deduplication key from normalized text

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a simple asyncio-compatible token bucket rate limiter, allowing for concurrent access and limiting the rate of requests.

**Domain:** RAG logic

**Key Responsibilities:**

*   Provides a TokenBucket class that implements a token bucket rate limiter.
*   Allows for specifying the average refill rate (tokens per second) and capacity.
*   Supports acquiring tokens in a non-blocking manner, with an optional wait if not enough tokens are available.
*   Designed to be used across concurrent coroutines within a single process.

### `INTERGRAX_ENGINE_BUNDLE.py`

**Description:** This is an auto-generated, complete source code bundle of the Intergrax framework containing all Python modules from the package directory.

**Domain:** LLM instructions / Framework bundle

**Key Responsibilities:**

* Contains all Python modules from the package directory
* Provides a single source of truth for the Intergrax framework
* Includes module groups such as chains, globals, llm, memory, multimedia, openai, rag, runtime, supervisor, tools, and websearch
* Offers navigation through the bundle using the MODULE MAP and INDEX
* Specifies rules for proposing changes to ensure preservation of existing architecture, naming, and conventions

### `main.py`

Here's the documentation for the `main.py` file:

Description: This module serves as the entry point for the Integrax framework, responsible for executing the core functionality of the application.

Domain: Framework Entrypoint

Key Responsibilities:
- Initializes and runs the primary function of the Integrax framework.
- Demonstrates basic functionality and logging.

### `mcp\__init__.py`

Description: This is the entry point for the Intergrax framework, responsible for initializing and configuring its core components.

Domain: Configuration

Key Responsibilities:
• Initializes the framework's internal state
• Configures key components and dependencies
• Defines the overall structure of the framework

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

Description: This notebook serves as a basic sanity-check for the Drop-In Knowledge Mode runtime within the Intergrax framework. It instantiates and tests various components, including session management, LLM adapters, embedding managers, and vector store management.

Domain: Drop-In Knowledge Mode

Key Responsibilities:
- Instantiate an in-memory session store using SessionManager.
- Initialize a LangChain Ollama adapter (LLM) with a placeholder for actual integration.
- Create a RuntimeConfig instance with placeholders for various components.
- Instantiate a DropInKnowledgeRuntime instance with the created config and session manager.
- Test basic functionality by creating a new session, appending user and assistant messages, building conversation history, and returning a RuntimeAnswer object.

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:** This Jupyter notebook demonstrates the Intergrax framework's Drop-In Knowledge Mode runtime, showcasing its functionality in handling sessions and attachments ingestion. The runtime is configured with Ollama as the LLM adapter, Chroma vector store manager, and various other components.

**Domain:** RAG logic / Vector Store Ingestion

**Key Responsibilities:**

- Demonstrates the Intergrax Drop-In Knowledge Mode runtime's functionality.
- Handles sessions and attachments ingestion using `AttachmentIngestionService`.
- Utilizes Ollama as the LLM adapter and Chroma vector store manager.
- Includes setup for in-memory session storage, embedding manager using Ollama, and vector store configuration.

This appears to be a standard Intergrax framework example showcasing its capabilities rather than an experimental or auxiliary component.

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of the `ContextBuilder` in Drop-In Knowledge Mode, showcasing how to wire it into the existing runtime using minimal components.

**Domain:** RAG (Relevance Aware Retrieval) logic

**Key Responsibilities:**

* Load a demo chat session and simulate conversation
* Work with an existing attachment, ingesting or retrieving it from the vector store
* Initialize the `ContextBuilder` instance using RuntimeConfig and IntergraxVectorstoreManager
* Build context for a single user question using reduced chat history, retrieved document chunks, system prompt, and RAG debug information
* Inspect the result of context building without LLM call

**Note:** This notebook appears to be a demonstration or tutorial code, focusing on showcasing the usage of `ContextBuilder` in Drop-In Knowledge Mode. It does not seem experimental, auxiliary, legacy, or incomplete.

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

Description: This notebook demonstrates how to use the DropInKnowledgeRuntime with session-based chat, optional RAG, live web search via WebSearchExecutor, and achieves a "ChatGPT-like" experience with browsing.

Domain: Chatbots / Conversational AI

Key Responsibilities:
- Initializes core runtime configuration (LLM + embeddings + vector store + web search)
- Demonstrates how to use DropInKnowledgeRuntime for chat engine with RAG and web search
- Sets up environment variables, including API keys for Google CSE and Bing Web Provider
- Configures session storage, LLM adapter, embedding manager, vector store configuration, and web search executor
- Initializes runtime and session manager instances

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

Description: This notebook demonstrates how to use the Drop-In Knowledge Runtime with a tools orchestration layer on top of conversational memory, optional RAG (attachments ingested into a vector store), and optional live web search context.

Domain: Tools Orchestration Layer

Key Responsibilities:
- Configure Python path for `intergrax` package import
- Load environment variables (API keys, etc.)
- Import core building blocks used by the Drop-In Knowledge Runtime
- Initialize runtime configuration (LLM, embeddings, vector store, web search) in a single compact setup cell
- Define tools using the Intergrax tools framework:
  - Implement demo tools (`WeatherTool`, `CalcTool`)
  - Register them in a `ToolRegistry`
  - Create an `IntergraxToolsAgent` instance that uses an Ollama-based LLM
  - Attach this agent to `RuntimeConfig.tools_agent` for orchestration in a ChatGPT-like flow

Note: This file appears to be a demo or example code, specifically designed to demonstrate the use of the Drop-In Knowledge Runtime with a tools orchestration layer.

### `notebooks\drop_in_knowledge_mode\06_session_memory_roundtrip_demo.ipynb`

**Description:** This Jupyter notebook serves as a test and demonstration environment for the Intergrax framework's Drop-In Knowledge Mode. It showcases the runtime's ability to create, reuse, persist, and load conversation history sessions.

**Domain:** LLM adapters, Runtime configuration, Session management, Vector store management, Debugging tools

**Key Responsibilities:**

*   Configures runtime settings, including language model adapters, embeddings, and vector stores.
*   Demonstrates session creation, reuse, persistence, and loading of conversation history.
*   Includes debugging tools for printing session history, runtime debug trace, and basic assertions.
*   Tests the framework's ability to handle a first request, including creating a new session, generating a valid answer, and storing debug information.

**Note:** This notebook appears to be a well-documented test and demonstration environment.

### `notebooks\drop_in_knowledge_mode\07_user_profile_instructions_baseline.ipynb`

**Description:** This notebook validates user-level system instructions injection from stored UserProfile into runtime prompt messages through Intergrax framework's DropInKnowledgeMode. It verifies instruction resolution, injection, and session history consistency.

**Domain:** LLM Adapters & Runtime Configuration

**Key Responsibilities:**

* Initialize Intergrax components (vectorstore manager, embedding manager, llm adapter)
* Set up runtime configuration with baseline settings
* Create a minimal user profile with system instructions to be injected
* Persist the user profile and retrieve it for injection into runtime prompt messages
* Run a baseline request through DropInKnowledgeRuntime and verify instruction injection
* Validate session history consistency

### `notebooks\drop_in_knowledge_mode\08_user_profile_instructions_generation.ipynb`

**Description:** This notebook demonstrates a production-safe, explicit mechanism to generate `UserProfile.system_instructions` using an LLM (Large Language Model), based on conversation history and existing user profile.

**Domain:** LLM adapters, User Profile Management

**Key Responsibilities:**

* Generate `UserProfile.system_instructions` using an LLM
* Persists generated instructions to the user profile
* Marks sessions as requiring refresh (`needs_user_instructions_refresh = True`)
* Demonstrates a separation of concerns between instruction generation and usage

### `notebooks\drop_in_knowledge_mode\09_long_term_memory_consolidation.ipynb`

**Description:** This notebook demonstrates the production-critical "long-term memory via consolidation" behavior in the Intergrax framework, testing the conversion of session history into UserProfile.memory_entries and system_instructions.

**Domain:** LLM adapters & Memory Management

**Key Responsibilities:**

* Tests the consolidation of session history into UserProfile.memory_entries
* Validates production-critical invariants for conversation history and profile state
* Demonstrates the use of Intergrax's LLM adapters and memory management components

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

**Description:** This notebook demonstrates the integration of Intergrax and LangGraph components for building a hybrid multi-source RAG (Regressive Attention Graph) agent, which combines local files and web search results in a single pipeline.

**Domain:** Hybrid Multi-Source RAG with Intergrax + LangGraph

**Key Responsibilities:**

* Ingest content from multiple sources:
	+ Local PDF files
	+ Local DOCX/Word files
	+ Live web results using the Intergrax `WebSearchExecutor`
* Build a unified RAG corpus by normalizing documents into a common internal format, attaching metadata (optionally), and splitting documents into chunks suitable for embedding.
* Create an in-memory vector index using an Intergrax embedding manager (e.g., OpenAI or Ollama-based) and store embeddings in an in-memory Chroma collection via the Intergrax vectorstore manager.
* Answer user questions with a RAG agent:
	+ Load → merge → index → retrieve → answer
	+ Generate a single, structured report: summary of relevant information, key insights, conclusions, and optionally recommendations or action items.

Note that this notebook appears to be a working example, showcasing the integration of Intergrax and LangGraph components for building a hybrid multi-source RAG agent.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This notebook demonstrates a basic integration between the Intergrax framework and LangGraph, showcasing how to use an Intergrax LLM adapter as a node within a LangGraph graph.

Domain: LLM adapters + LangGraph integration

Key Responsibilities:
- Initializes an OpenAI Chat Responses Adapter instance using the provided client and model.
- Defines a simple state for the LLM QA example, including messages and answer fields.
- Implements a LangGraph node (`llm_answer_node`) that calls the Intergrax LLM adapter to generate responses.
- Builds a StateGraph with a single node `llm_answer_node` and runs it on a sample input.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

**Description:**
This is a Python Jupyter Notebook demonstrating the Intergrax framework's capabilities in building a practical web research agent. It showcases how to integrate various components, such as web search providers, LLM adapters, and graph-based state management.

**Domain:** RAG logic (Reinforcement Learning Agents)

**Key Responsibilities:**

* Initializes OpenAI key and Google CSE API keys from environment variables
* Creates an OpenAIChatResponsesAdapter for the LLM adapter
* Initializes a WebSearchExecutor with GoogleCSEProvider and other default settings
* Defines a ContextBuilder for building a compact textual context + citations
* Defines an Answerer that uses the LLM adapter to answer questions based on web search results
* Demonstrates how to define a graph state (WebResearchState) and nodes for normalizing user questions, running web search, and generating answers

Note: The code in this notebook appears to be a demonstration of the Intergrax framework's capabilities rather than experimental or auxiliary. However, it may contain some placeholder or example code that is not production-ready.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This notebook provides a Jupyter presentation demonstrating the use of the Intergrax framework with OpenAI's RAG (Retrieval-Augmented Generation) model for text generation and vector store management.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Loads necessary dependencies, including OpenAI and dotenv libraries.
- Initializes the Vector Store using the provided ID.
- Clears the existing vector store and storage.
- Uploads a specified local folder to the vector store.
- Tests queries with sample questions and prints responses from the model.

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This notebook serves as a comprehensive example of how to integrate various components of the Intergrax framework, including LLM chat agents, RAG logic, vector stores, and tool registries. It demonstrates the creation of a hybrid chat agent that leverages both RAG-based document retrieval and tool-based response generation.

**Domain:** RAG (ReAdGeR) logic, Hybrid Agents

**Key Responsibilities:**
- Initializes an Ollama LLM model for chat interactions
- Configures a Conversational Memory instance
- Demonstrates the use of tools, such as the Weather Tool
- Sets up a Vector Store with Chroma and defines its configuration
- Creates an Embedding Manager and ReRanker instances
- Configures an Answerer component based on RAG settings
- Defines a RagComponent for handling document-based queries
- Deploys a high-level Hybrid Chat Agent combining LLM, tools, and RAG components

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates a simple RAG (Retrieval-Augmented Generation) setup for output structure presentation. It showcases how to create a structured response using the Integrax framework.

**Domain:** LLM adapters, RAG logic, and output structure presentation.

**Key Responsibilities:**

* Import necessary modules from the Integrax framework.
* Define Pydantic models (`WeatherAnswer` and `ExecSummary`) for structured output.
* Create a simple weather tool (`WeatherTool`) that returns demo data in a compatible format.
* Set up a tools agent with an Ollama-backed LLM, conversational memory, and tool registry.
* Demonstrate usage of the RAG setup by asking a natural-language question and requesting a structured response.

**Notes:** This notebook appears to be a working example and not experimental or auxiliary. However, it is worth noting that some parts (e.g., the `WeatherTool`) are stub implementations intended for testing purposes only.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter Notebook is a presentation of custom RAG (Reformer Augmented Generation) functionality within the Intergrax framework. It demonstrates loading documents, splitting them into chunks, and generating embeddings for each document.

**Domain:** RAG logic

**Key Responsibilities:**

* Load raw documents from a directory using DocumentsLoader
* Split loaded documents into smaller chunks using DocumentsSplitter
* Generate vector embeddings for each document chunk using EmbeddingManager
* Interact with VectorstoreManager to manage the underlying database and perform lightweight presence checks before ingestion.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

Description: This notebook demonstrates the loading, splitting, and embedding of multimodal documents using various Intergrax modules. It also tests a retriever on a sample query.

Domain: RAG (Retriever-Augmented Generator) logic

Key Responsibilities:
- Loads multimedia documents from files using DocumentsLoader
- Splits documents into chunks using DocumentsSplitter
- Embeds documents using EmbeddingManager
- Tests if corpus is present in Vectorstore, and ingests new data if necessary
- Demonstrates retriever functionality with a sample query

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This notebook provides an example usage of the Intergrax framework for multimedia processing, including downloading videos and audio from YouTube, transcribing video to VTT format, extracting frames and metadata from videos, translating audio, and describing images using a model.

**Domain:** Multimedia Processing

**Key Responsibilities:**
- Downloading videos and audio from YouTube
- Transcribing video to VTT format
- Extracting frames and metadata from videos
- Translating audio
- Describing images using a model

### `notebooks\rag\tool_agent_presentation.ipynb`

Description: This notebook showcases a tools agent that integrates with language models to perform tasks using external tools. It demonstrates the creation of a tools agent, registration of tools (weather and calculator), and example usage with input queries.

Domain: RAG logic

Key Responsibilities:
- Defines two custom tools: WeatherTool for retrieving mock weather data and CalcTool for basic arithmetic calculations.
- Sets up conversational memory to track dialogue context.
- Registers available tools in a ToolRegistry instance.
- Creates an LLM adapter using Ollama, which serves as the planner/controller for tool invocations.
- Initializes a ToolsAgent that orchestrates LLM reasoning, tool selection, and conversational memory updates.
- Demonstrates two test cases:
  - Queries weather information for Warsaw.
  - Performs a basic arithmetic calculation (235*17).

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This Jupyter Notebook file, `supervisor_test.ipynb`, appears to be a testbed for the Integrax framework's supervisor components. It contains several code cells that define and test various functions for data processing and pipeline management.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

- Defines mock compliance checker component using the `@component` decorator
  - Verifies whether proposed changes comply with privacy policies and terms of service (mock)
- Defines cost estimator component using the `@component` decorator
  - Estimates the cost of changes based on the UX audit report (mock)
- Defines final summary component using the `@component` decorator
  - Generates the final consolidated summary using all collected artifacts
- Defines financial audit component using the `@component` decorator
  - Generates a mock financial report and VAT calculation (test data)

**Notes:** This file appears to be part of the Integrax framework's testing infrastructure, focusing on supervisor components that handle compliance checking, cost estimation, final summarization, and financial audits. The content suggests that these functions are currently in development or experimental stages, given their mock nature.

### `notebooks\websearch\websearch_presentation.ipynb`

Description: This notebook demonstrates the usage of the Intergrax framework's web search capabilities, specifically with Google Custom Search and Bing. It showcases how to perform a query using the QuerySpec schema and retrieve search hits from multiple providers.

Domain: WebSearch

Key Responsibilities:
- Loads environment variables for Google Custom Search API key and custom search engine ID
- Initializes a GoogleCSEProvider instance
- Defines a QuerySpec object with specified query, top-k, locale, region, language, and safe search settings
- Executes the query using the provider's search method and prints the results
- Demonstrates usage of WebSearchExecutor with multiple search providers (Google Custom Search)
