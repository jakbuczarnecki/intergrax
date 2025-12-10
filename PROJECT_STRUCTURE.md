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
- `intergrax\runtime\drop_in_knowledge_mode\attachments.py`
- `intergrax\runtime\drop_in_knowledge_mode\config.py`
- `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine.py`
- `intergrax\runtime\drop_in_knowledge_mode\engine_history_layer.py`
- `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`
- `intergrax\runtime\drop_in_knowledge_mode\runtime_state.py`
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

DESCRIPTION: This file serves as the entry point for the Intergrax API, responsible for importing and initializing various components.

DOMAIN: API Initialization Module

KEY RESPONSIBILITIES:
* Initializes the API environment.
* Imports necessary modules and sub-packages.

### `api\chat\__init__.py`

Description: This is the entry point for the chat API, responsible for initializing and setting up the necessary components.

Domain: Chat API

Key Responsibilities:
* Initializes the chat API module
* Sets up default configurations and bindings
* Defines the main entry points for the chat API

### `api\chat\main.py`

**Description:** This is a main entry point for the chat API in the Integrax framework, handling user queries and document management.

**Domain:** LLM adapters/Chat API

**Key Responsibilities:**

* Handles user queries through POST requests to "/chat" endpoint
* Retrieves and sets up answerer with relevant model name and session history
* Logs application events and user queries for auditing purposes
* Manages document uploads, indexing, and deletion through corresponding endpoints ("upload-doc", "list-docs", "delete-doc")
* Provides basic authentication through use of UUID-based session IDs

### `api\chat\tools\__init__.py`

**api/chat/tools/__init__.py**

Description: Initializes the chat API tools package and sets up the necessary imports for downstream modules.

Domain: LLM adapters

Key Responsibilities:
- Registers LLM adapter classes
- Provides a central entry point for tool-specific initialization
- Imports and exposes utility functions for the chat API

### `api\chat\tools\chroma_utils.py`

Here's the documentation for the provided file:

**Description:** This module contains utility functions for interacting with Chroma, a vector store used in the Integrax framework for document indexing and retrieval.

**Domain:** Vector Store Utilities

**Key Responsibilities:**
- Load and split documents from a given file path
- Index documents to Chroma using a file path and ID
- Delete documents from Chroma by their ID

Note: The `load_and_split_documents` function has a recursive call, but it appears to be intended as a placeholder or a stub that needs implementation. Therefore, I've documented it as is, noting its potential for further development.

Also worth mentioning: This module seems to rely on functions imported from the `rag_pipeline` module, indicating that it's part of a larger pipeline or workflow in the Integrax framework.

### `api\chat\tools\db_utils.py`

**Description:** 
This module provides database utilities for managing a chat application's data. It includes functions for creating and migrating the schema, inserting and retrieving messages and documents.

**Domain:** Data Ingestion/Storage

**Key Responsibilities:**

* Creating and migrating database schema for sessions, messages, and documents
* Inserting and retrieving messages (user, assistant, system, tool roles) with optional model information
* Managing document store with insert, delete, and retrieval functions
* Backward-compatibility entry points for legacy application logs (currently only creating a new schema)
* Helper functions for getting database connections and ensuring session existence

### `api\chat\tools\pydantic_models.py`

**Description:** This module provides data models using Pydantic for serializing and deserializing API requests and responses in the chat tools section of the Integrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines an enumeration `ModelName` for specifying supported language models
* Provides a `QueryInput` model for representing incoming query inputs, including question text, session ID, and selected model
* Offers a `QueryResponse` model for encapsulating response data, such as answer text, session ID, and used model
* Includes a `DocumentInfo` model for managing uploaded documents with metadata like file ID, filename, and upload timestamp
* Specifies a `DeleteFileRequest` model for handling file deletion requests with a required file ID

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides utilities and singletons for the RAG (Reactor with Attention Guide) pipeline, including vector store management, embedding, retrieval, re-ranking, and LLM adapters.

**Domain:** RAG Logic

**Key Responsibilities:**

* Vectorstore management:
	+ Retrieves vector store instance
	+ Creates vector store instance if not already created
* Embedding management:
	+ Retrieves embedder instance
	+ Creates embedder instance if not already created
* Retriever management:
	+ Retrieves retriever instance
	+ Creates retriever instance if not already created
* Reranker management:
	+ Retrieves reranker instance
	+ Creates reranker instance if not already created
* LLM adapter creation:
	+ Builds LLM adapter for a given model name
* Singleton management: 
	+ Manages singleton instances for vector store, embedder, retriever, and reranker

Note that this module appears to be part of the Intergrax framework's RAG logic, which is a proprietary and confidential system.

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides Streamlit API utilities for interacting with the Integrax framework's backend API, including functions for making HTTP requests and handling responses.

**Domain:** RAG logic

**Key Responsibilities:**

- Makes POST requests to the "chat" endpoint with question data.
- Handles API response from "chat" endpoint (parses JSON).
- Makes POST request to upload a file to the "upload-doc" endpoint.
- Lists documents from the backend by making a GET request to the "list-docs" endpoint.
- Deletes a document from the backend by making a POST request to the "delete-doc" endpoint with the file ID.

Note: This file appears to be part of the main application code and not experimental or auxiliary.

### `applications\chat_streamlit\chat_interface.py`

Description: This module defines the user interface for interacting with the chat streamlit application, including displaying messages and generating responses through an API call.

Domain: LLM Adapters

Key Responsibilities:
- Displays chat interface using Streamlit
- Retrieves messages from session state
- Handles user input through a chat input field
- Makes API calls to retrieve response based on user prompt
- Updates session state with response data

### `applications\chat_streamlit\sidebar.py`

**Description:** This module implements a sidebar for the chat application, providing features such as model selection, document upload, listing uploaded documents, and deleting selected documents.

**Domain:** Chat Application UI

**Key Responsibilities:**

* Displays a model selection component in the sidebar
* Allows users to upload documents using a file uploader
* Lists uploaded documents with options to delete them
* Refreshes the list of uploaded documents on button click

### `applications\chat_streamlit\streamlit_app.py`

Here's the documentation for the provided file:

Description: This module serves as the entry point for the Integrax RAG Chatbot application, utilizing Streamlit to create a user interface with sidebar and chat functionality.

Domain: Chat Application Interface

Key Responsibilities:
- Initializes Streamlit session state for storing messages and session IDs
- Displays the chatbot's title
- Sets up event listeners for initial message storage and session ID tracking
- Calls functions to display sidebar and chat interface

### `applications\company_profile\__init__.py`

**File:** `applications/company_profile/__init__.py`

**Description:** This module initializes the company profile application, setting up necessary components and configurations.

**Domain:** Application Initialization/Configuration

**Key Responsibilities:**
- Imports and sets up dependencies for the company profile application
- Configures application-specific settings and constants
- Initializes any necessary application-level services or modules

### `applications\figma_integration\__init__.py`

Description: Initializes the Figma integration module and sets up necessary dependencies.
Domain: Integrations

Key Responsibilities:
• Sets up the Figma API client instance
• Defines the base URL for Figma API requests
• Exposes functions for connecting to Figma design files
• Configures any additional settings or parameters required for Figma integration

### `applications\ux_audit_agent\__init__.py`

Description: The `__init__.py` file is the entry point for the UX Audit Agent application, responsible for initializing and configuring the agent.

Domain: Application Initialization/Configuration

Key Responsibilities:
• Initializes the UX Audit Agent application
• Configures the agent with default settings and dependencies
• Sets up logging and other essential components

### `applications\ux_audit_agent\components\__init__.py`

DESCRIPTION: This module serves as the entry point and registry for UX audit agent components.

DOMAIN: Agents

KEY RESPONSIBILITIES:
* Registers all sub-components of the UX audit agent.
* Provides a centralized access point for these components.

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module defines a component for policy and privacy compliance checking, which evaluates proposed changes against the organization's privacy policy and regulations.

**Domain:** Compliance Checker

**Key Responsibilities:**

* Evaluates proposed UX changes against the organization's privacy policy and regulations.
* Returns findings on compliance, including any policy violations or required actions.
* Optionally stops execution if non-compliant to require corrections or DPO review.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a cost estimation component for UX-related changes, utilizing the audit report and implementing a mock pricing model.

**Domain:** RAG logic

**Key Responsibilities:**
- Estimates the cost of UX updates based on the audit report
- Utilizes a mock pricing model for calculation
- Returns a cost estimate with relevant metadata (currency and method)
- Integrates with the Intergrax supervisor components for pipeline execution

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module defines a component for generating a final report of the execution pipeline, using collected artifacts.

**Domain:** UX Audit Agent

**Key Responsibilities:**

* Generates a complete summary of the entire execution pipeline
* Collects and aggregates artifacts from previous stages
* Returns a final report as a dictionary with key details about pipeline status, termination reason, project manager decision, notes, and reports.

### `applications\ux_audit_agent\components\financial_audit.py`

Description: This module defines a component for generating mock financial reports and VAT calculations as part of the Integrax framework's audit agent.

Domain: RAG logic (Regulatory Audit Generator)

Key Responsibilities:
- Provides a Financial Agent component for testing financial computations and budget constraints.
- Generates mock financial reports with test data, including net value, VAT rate, VAT amount, gross value, currency, and previous quarter budget.

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component for the Intergrax system, allowing users to ask questions about its structure, features, and configuration.

**Domain:** RAG logic (Reactive Action Generation)

**Key Responsibilities:**

* Provides a general knowledge component for answering user queries
* Returns mock data as citations for demonstration purposes
* Handles user input and generates relevant responses based on the provided examples

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a component for project management that reviews UX reports and makes mock decisions based on random outcomes.

**Domain:** RAG (Risk, Action, Review) logic

**Key Responsibilities:**

* Reviews the UX report based on pipeline state and context
* Makes a mock decision to approve or reject the proposal with comments
* Produces outputs including decision, notes, and logs
* Optionally stops pipeline execution if rejected by project manager

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module defines a UX audit component that analyzes Figma mockups and generates a sample report with recommendations.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Performs a UX audit on Figma mockups
- Returns a sample report with recommendations
- Integrates with the Intergrax supervisor framework

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This is an interactive Jupyter notebook that outlines the process for conducting UX audits on FIGMA mockups, verifying compliance with company policy, and preparing summary reports. It provides step-by-step instructions, goals, inputs, outputs, and success criteria for each task.

Domain: RAG logic

Key Responsibilities:
- Perform UX audit on FIGMA mockups
  • Inputs: ['FIGMA mockups']
  • Outputs: ['ux_audit_report', 'findings']
- Verify changes comply with company policy
  • Inputs: ['ux_audit_report', 'findings']
  • Outputs: ['financial_report', 'last_quarter_budget']
- Prepare summary report for Project Manager
  • Inputs: ['financial_report', 'last_quarter_budget']
  • Outputs: ['final_report']
- Evaluate financial impact of changes
  • Inputs: ['final_report']
  • Outputs: ['cost_estimate', 'budget_notes']
- Project Manager decision and project continuation
  • Inputs: ['final_report', 'cost_estimate', 'budget_notes']
  • Outputs: []
- Final report preparation and synthesis
  • Inputs: ['final_report', 'cost_estimate', 'budget_notes']
  • Outputs: ['final_decision', 'rationale']

Note: The content appears to be a step-by-step guide for a UX audit process, utilizing various components and tools. It is not explicitly labeled as experimental or auxiliary, but rather seems to be a documented workflow for a specific project within the Integrax framework.

### `generate_project_overview.py`

**Description:** 
This module generates an automatic project structure overview documentation for the Intergrax framework.

**Domain:** Data Ingestion and Documentation Generation

**Key Responsibilities:**

- Recursively scans the project directory to collect all relevant source files.
- Reads each file's content and metadata, then sends it to a Language Model Adapter (LLM adapter) for summarization.
- Generates a structured summary for each file, including its purpose, domain, and key responsibilities.
- Creates a human-readable Markdown document that outlines the entire project structure, including file indices and detailed documentation for each file.

### `intergrax\__init__.py`

**intergrax/__init__.py**

Description: The main entry point for the Intergrax framework, responsible for initializing and configuring various components.

Domain: Framework initialization

Key Responsibilities:
* Initializes core modules and services
* Configures framework settings and environment variables
* Sets up logging and other essential infrastructure

### `intergrax\chains\__init__.py`

Description: The `__init__.py` file serves as the entry point for the Intergrax chains module, initializing and setting up its internal components.

Domain: Chains configuration

Key Responsibilities:
* Initializes chain-specific settings and constants
* Sets up required dependencies and imports
* Defines the module's API for external usage

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module provides a flexible QA chain implementation using LangChain, allowing for customization of the retrieval, re-ranking, and prompting stages with hooks.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Builds a QA chain consisting of:
	+ Retrieval stage (RagRetriever)
	+ Optional re-rank stage (ReRanker)
	+ Prompt building stage
	+ LLM execution stage
* Provides hooks for modifying data at each stage:
	+ `on_before_build_prompt`: Modifies payload before prompt building
	+ `on_after_build_prompt`: Modifies prompt text after building
	+ `on_after_llm`: Modifies answer text after LLM execution
* Supports customization of chain configuration, including retrieval and re-ranking settings

**Note:** This file appears to be a core part of the Intergrax framework, implementing a crucial component for QA functionality.

### `intergrax\chat_agent.py`

**Description:** This module defines the core functionality of the chat agent, responsible for routing and executing tasks based on user input.

**Domain:** Chat Agents

**Key Responsibilities:**

* Routing decisions via LLM (Language Model) with descriptions and tools enabled flag
* Execution of tasks in three modes: RAG (Retro-Agumented Generation), TOOLS, and GENERAL
* Handling memory, streaming, structured output, and stats for each task
* Providing a stable result format for each task execution

Note: The file appears to be well-structured and documented, but some parts of the code may be experimental or legacy based on the usage of deprecated libraries (e.g., `Literal` and `Union`).

### `intergrax\globals\__init__.py`

Description: This is the entry point for Intergrax's global module, handling initialization and setup.

Domain: Globals/Setup

Key Responsibilities:
* Initializes global settings and configurations.
* Sets up core modules and dependencies.
* Defines entry points for Intergrax's framework.

### `intergrax\globals\settings.py`

**Description:** This module provides a centralized mechanism for managing default settings and configuration across the Intergrax framework, including language, locale, timezone, LLM models, and session memory/consolidation intervals.

**Domain:** Configuration

**Key Responsibilities:**

* Provides a single source of truth for framework-wide default settings
* Allows local modules to import and use these defaults instead of hardcoding values
* Exposes environment variable-based fallbacks for user-specific overrides
* Defines default LLM models, including Ollama, OpenAI, and HuggingFace embeddings
* Configures session memory/consolidation intervals with optional cooldown seconds

### `intergrax\llm\__init__.py`

FILE PATH:
intergrax\llm\__init__.py

Description: The LLM adapter initialization module, responsible for setting up and configuring the Loebner Master (LLM) adapters within the Intergrax framework.

Domain: LLM Adapters

Key Responsibilities:
- Initializes and configures available LLM adapters
- Provides entry point for loading and using LLM adapters
- Sets default adapter configurations

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This module provides a set of utilities and adapters for interacting with Large Language Models (LLMs), specifically OpenAI's Chat Completions API.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides an interface to interact with various LLMs, including OpenAI's Chat Completions API
* Offers tools for handling structured output from LLMs, including validation and creation of model instances
* Defines a universal interface (`LLMAdapter`) that must be implemented by concrete adapters
* Includes helper functions for mapping `ChatMessage` objects to the OpenAI schema
* Implements an adapter for interacting with OpenAI's Chat Completions API

### `intergrax\llm\messages.py`

**Description:** This module provides utility classes for representing chat messages in Integrax framework, including dataclasses for ChatMessage and AttachmentRef. It also includes helper functions for managing chat message lists.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines the `ChatMessage` class with attributes for role, content, and additional metadata.
* Provides a `to_dict()` method to convert `ChatMessage` objects to dictionaries compatible with OpenAI Responses API / ChatCompletions.
* Defines an `AttachmentRef` dataclass for lightweight references to attachments associated with messages or sessions.
* Includes a function `append_chat_messages()` that appends new chat messages to existing ones in LangGraph state.

Note: The code appears well-structured and complete, no signs of experimental, auxiliary, legacy, or incomplete features.

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point for Intergrax's LLM adapters, providing a registry and default adapter registrations.
Domain: LLM Adapters
Key Responsibilities:
- Registers available LLM adapters with their corresponding factories.
- Exposes all adapter classes and utility functions to be imported and used elsewhere in the project.
- Sets up default adapter registrations for "openai", "gemini", and "ollama" using lambda functions.

### `intergrax\llm_adapters\base.py`

**Description:** This module defines utilities and interfaces for interacting with Large Language Models (LLMs) in the Integrax framework. It includes functions for token estimation, model validation, and structured output generation.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Token estimation:
	+ `estimate_tokens_for_messages`: estimates token count for a list of messages using a heuristic based on available encoding
* Model validation:
	+ `_validate_with_model`: validates and creates a model instance from a JSON string
* Structured output generation:
	+ `_model_json_schema`: returns the JSON schema for a given model class
	+ `generate_structured`: generates structured output for a list of messages using a specified output model
* LLM adapter interface:
	+ `LLMAdapter`: defines a protocol for LLM adapters, including methods for generating and streaming messages
* Utility functions:
	+ `_strip_code_fences`: removes Markdown code fences from a string
	+ `_extract_json_object`: extracts the first JSON object from a string
* Base class for LLM adapters:
	+ `BaseLLMAdapter`: provides shared utilities such as token counting

**Status:** This module appears to be production-ready, with well-documented functions and interfaces.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a basic implementation of the Gemini chat adapter, which allows integration with the Gemini large language model.

**Domain:** LLM adapters

**Key Responsibilities:**
* Estimates context window sizes for different Gemini models
* Splits system messages from conversational content
* Generates and streams chat responses using the Gemini model
* Supports structured output (not implemented in this version)
* Provides tools interface (not implemented in this version)

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides a LangChain Ollama adapter, enabling integration with the Ollama large language model via the LangChain framework. The adapter handles context window estimation and supports both single-shot and streaming generation.

**Domain:** LLM adapters

**Key Responsibilities:**
- Estimates context windows for supported Ollama models
- Supports single-shot and streaming text generation
- Adapts LangChain's ChatModel interface for Ollama integration
- Provides structured output via JSON schema validation (via prompt + post-hoc validation)
- Handles tool-calling in a planner-style pattern (no native tools support)

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an OpenAI adapter based on the Responses API, offering functionality similar to the previous Chat Completions adapter.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Estimates context window tokens for OpenAI models
* Converts chat completion messages to Responses API input items
* Extracts assistant output text from Responses API results
* Generates single-shot and streaming completions using Responses API
* Supports tools with potential function/tool calls in responses

### `intergrax\logging.py`

**Description:** This module sets up the global logging configuration for the Integrax framework, defining the log level and format.

**Domain:** Logging Configuration

**Key Responsibilities:**
- Sets the global logging level to INFO and above.
- Defines a custom log format with timestamp, log level, and message.
- Forces the logging configuration to override any previous settings.

### `intergrax\memory\__init__.py`

Description: The __init__.py file is the entry point for the memory domain, responsible for initializing and setting up the underlying memory management infrastructure.

Domain: Memory Management

Key Responsibilities:
• Initializes memory management components
• Sets up memory-related configurations
• Provides interfaces for accessing and managing in-memory data structures

### `intergrax\memory\conversational_memory.py`

**Description:** 
This module provides a universal in-memory conversation history component for storing and managing chat messages.

**Domain:** Conversational Memory

**Key Responsibilities:**

* Keep messages in RAM
* Provide a simple API to add, extend, read, and clear messages
* Optionally enforce a max_messages limit
* Prepare messages for different model backends (get_for_model)
* Delegate persistence to dedicated memory store providers

### `intergrax\memory\conversational_store.py`

**Description:** This module defines an abstract interface for persistent conversational memory storage, enabling various backends to be swapped without modifying runtime logic.

**Domain:** Conversational Memory Storage

**Key Responsibilities:**

* Provides a protocol for implementing persistent storage of conversational history
* Ensures deterministic persistence and idempotent write operations
* Enables loading, saving, appending messages, and deleting sessions from conversational memory
* Abstracts away business logic, trimming policies, and model-format conversions

### `intergrax\memory\stores\__init__.py`

Description: This module initializes and manages stores for the Intergrax memory.
Domain: Memory Management

Key Responsibilities:
* Initializes store instances
* Exposes store interfaces to users
* Sets up store configurations as needed

### `intergrax\memory\stores\in_memory_conversational_store.py`

**Description:** This module provides an in-memory conversational memory store implementation, suitable for local development, prototyping, or unit/integration testing.

**Domain:** Conversational Memory Store

**Key Responsibilities:**
- Stores conversation history in memory, isolated per Python interpreter.
- Provides methods to load and save conversational memory instances.
- Offers append and delete operations for messages and sessions.
- Includes an optional diagnostic method for listing active persisted session IDs.

### `intergrax\memory\stores\in_memory_user_profile_store.py`

**Description:** This module provides an in-memory implementation of the UserProfileStore, storing user profiles and their preferences within the application's process memory.

**Domain:** Memory Stores (In-Memory User Profile Store)

**Key Responsibilities:**
* Provides a simple in-memory storage for user profiles
* Allows retrieval, creation, and deletion of user profiles using user IDs
* Supports asynchronous operations for profile management
* Offers optional helper method for listing stored user IDs

Note: This module appears to be part of the mainline implementation, used primarily for unit testing, local development, or experimental purposes. It does not provide durability or cross-process sharing capabilities.

### `intergrax\memory\user_profile_manager.py`

**Description:** This module, `user_profile_manager.py`, serves as a high-level facade for working with user profiles in the Intergrax framework. It provides convenient methods for loading and creating user profiles, persisting profile changes, managing long-term user memory entries, and system-level instructions derived from the profile.

**Domain:** User Profile Management

**Key Responsibilities:**

* Provide convenient methods to:
	+ Load or create a UserProfile for a given user_id
	+ Persist profile changes
	+ Manage long-term user memory entries
	+ Manage system-level instructions derived from the profile
* Hide direct interaction with the underlying UserProfileStore

Note: The file appears to be well-structured and complete, with no obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\memory\user_profile_memory.py`

**Description:** This module provides core domain models for user and organization profiles, including identity, preferences, memory entries, and system instructions.

**Domain:** User Profile Management

**Key Responsibilities:**

* Defines the `UserProfileMemoryEntry` dataclass to represent long-term memory entries about a user.
* Provides the `UserIdentity` dataclass to describe who the user is at a high level.
* Offers the `UserPreferences` dataclass to store stable user preferences influencing system behavior.
* Combines these components into the `UserProfile` dataclass, aggregating identity, preferences, and long-term memory entries.
* Includes metadata hooks for versioning and other purposes.

Note: This file appears to be a core part of the Integrax framework's user profile management functionality.

### `intergrax\memory\user_profile_store.py`

Description: This module provides a protocol for persistent storage of user profiles, abstracting away backend-specific concerns and defining interface methods for loading, saving, and deleting profiles.

Domain: Memory Management

Key Responsibilities:
- Providing a standard interface for storing and retrieving user profiles
- Loading user profiles with default values if no data exists yet
- Saving user profile aggregates persistently
- Deleting stored profile data for associated user IDs

### `intergrax\multimedia\__init__.py`

Description: The __init__.py file is responsible for initializing the multimedia module and making it importable in other parts of the Intergrax project.

Domain: Multimedia

Key Responsibilities:
- Initializes the multimedia module.
- Defines imports or exports for other modules to interact with the multimedia functionality. 

Note: As this file is a standard Python package initializer, its main purpose is setup and organization, indicating it's not experimental, auxiliary, legacy, or incomplete at this point in the documentation.

### `intergrax\multimedia\audio_loader.py`

**intergrax\multimedia\audio_loader.py**

Description: This module provides utilities for loading and processing audio data from YouTube URLs.

Domain: Multimedia Processing

Key Responsibilities:
- Downloads audio files from specified YouTube URLs.
- Extracts audio from videos using `yt_dlp` and `ffmpeg`.
- Translates audio content into desired languages using the Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module is responsible for loading and processing images within the Integrax framework, enabling the integration of image-based data into AI models.

Domain: Multimedia Processing

Key Responsibilities:
- Loads images from file paths
- Integrates images with text prompts using ollama library
- Utilizes the llava-llama3 model for image-text generation

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, including audio, images, and videos, within IPython notebooks.

**Domain:** Multimedia Display Utilities

**Key Responsibilities:**

* `display_audio_at_data`: Displays an audio file at a specified position with optional autoplay and label.
* `_is_image_ext`: Checks if a given path has an image extension.
* `display_image`: Displays an image or attempts to display it as an HTML image tag if the original file is not found.
* `_serve_path`: Serves a local file by copying it to a temporary directory and returns its served URL.
* `display_video_jump`: Displays a video with a specified start position, poster frame, autoplay, muted state, label, maximum height, playback rate, and other options.

Note: This module appears to be part of the Intergrax framework's multimedia display utilities. The code is well-structured and follows good practices, suggesting it is not experimental or auxiliary.

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides utilities for loading, processing, and manipulating multimedia content, specifically videos. It includes functions for downloading YouTube videos, transcribing audio to text, extracting frames from videos, and saving metadata.

**Domain:** Multimedia Processing

**Key Responsibilities:**
- **yt_download_video**: Downloads a YouTube video using yt-dlp and returns the path to the downloaded file.
- **transcribe_to_vtt**: Transcribes an input media file (e.g., audio or video) to WebVTT format and saves it to a specified output file. It uses the Whisper model for transcription.
- **extract_frames_and_metadata**: Extracts frames from a video at specific time intervals, creates metadata for each frame, and saves the extracted frames and metadata to disk.
- **extract_frames_from_video**: Similar to extract_frames_and_metadata but extracts frames at regular time intervals (every `every_seconds`) instead of at mid-points of transcript segments.

Note: This module appears to be a collection of utility functions for multimedia processing tasks.

### `intergrax\openai\__init__.py`

Description: The `__init__.py` file serves as the entry point for the OpenAI adapters within the Intergrax framework.

Domain: LLM Adapters

Key Responsibilities:
* Imports and sets up necessary modules for OpenAI adapter functionality
* Provides a namespace for OpenAI-related classes and functions

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: This is the entry point for the RAG (Reformer-based Adapters for GPUs) logic within Intergrax, responsible for initializing and configuring RAG components.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes RAG components.
- Configures RAG settings and hyperparameters.
- Provides a consistent interface for working with RAG models.

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module, `rag_openai.py`, provides the RAG (Retrieval-Augmented Generation) logic for OpenAI integration in the Integrax framework. It enables interaction with the OpenAI API to perform tasks such as file search, vector store management, and uploading data.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Provides a prompt template for generating answers based on retrieved documents (via `rag_prompt` method)
* Ensures the existence of a vector store and retrieves it by its ID (via `ensure_vector_store_exists` method)
* Clears all files loaded into the vector store (via `clear_vector_store_and_storage` method)
* Uploads a folder's contents to the vector store, handling file preparation and upload status checks (via `upload_folder_to_vector_store` method)

**Note:** The code appears well-structured and functional. It's designed for specific use cases within the Integrax framework, so its purpose is clear and focused on RAG logic integration with OpenAI.

### `intergrax\rag\__init__.py`

DESCRIPTION: This module initializes and sets up the RAG logic, including importing necessary components and defining key interfaces.

DOMAIN: RAG Logic

KEY RESPONSIBILITIES:
- Initializes RAG components
- Defines core RAG interfaces
- Sets up RAG configuration and dependencies

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader with metadata injection and safety guards.

**Domain:** RAG logic / Document Loader

**Key Responsibilities:**

* Loads documents from various file formats (e.g., PDF, DOCX, XLSX) using different loading strategies
* Supports OCR (Optical Character Recognition) for PDFs and images
* Allows customization of loading settings (e.g., verbosity, file patterns, extensions map)
* Includes safety guards to prevent potential issues during document loading
* Provides metadata injection capabilities

**Note:** The code appears to be well-structured and comprehensive, with a focus on flexibility and customizability. There is no indication that the file is experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\documents_splitter.py`

**Description:** This module, `DocumentsSplitter`, provides high-quality text splitting capabilities for RAG pipelines. It aims to create stable chunk IDs with rich metadata while respecting the concept of semantic atoms.

**Domain:** RAG (Retrieve-Augment-Generate) logic

**Key Responsibilities:**

*   Provides a high-quality text splitter for RAG pipelines
*   Implements a 'semantic atom' policy for splitting documents
*   Generates stable, human-readable chunk IDs using available anchors
*   Adds metadata to chunks, including parent ID, source name, and page index if present
*   Optionally merges small tails of documents
*   Applies a maximum number of chunks per document (if specified)

### `intergrax\rag\dual_index_builder.py`

**Description:** This module is responsible for building two vector indexes: a primary index (CHUNKS) and an auxiliary index (TOC), using documents from the input list.

**Domain:** RAG Logic

**Key Responsibilities:**

* Builds two vector indexes:
	+ CHUNKS: all chunks/documents after splitting
	+ TOC: only DOCX headings within specified levels
* Embeds documents using an embedding manager and adds them to the respective index
* Handles batching and logging for performance and debugging purposes
* Optionally, filters documents based on a provided predicate function

**Note:** The file appears to be part of the Integrax framework's RAG logic module, which is used for building vector indexes. It seems well-maintained and functional, with clear documentation and handling of edge cases.

### `intergrax\rag\dual_retriever.py`

**Description:** This module provides a Dual Retriever class for retrieving documents from the Intergrax framework's vector store, utilizing both document metadata (TOC) and local chunks for enhanced context-based querying.

**Domain:** RAG logic

**Key Responsibilities:**

*   Initiates dual retriever operations using `VectorstoreManager` instances
*   Retrieves base hits from CHUNKS via query vectors
*   Expands context by TOC and searches locally based on parent IDs
*   Merges, deduplicates, and sorts retrieved documents by similarity scores
*   Trims results to a specified top-k value

### `intergrax\rag\embedding_manager.py`

**Description:** The Embedding Manager is a unified interface for loading and using various text embedding models, including Hugging Face (SentenceTransformer), Ollama, and OpenAI embeddings. It provides features such as provider switchability, reasonable defaults, batch/single text embedding with optional L2 normalization, and cosine similarity utilities.

**Domain:** RAG logic

**Key Responsibilities:**

* Loading and managing different text embedding models
* Providing a unified interface for accessing embeddings from various providers (Hugging Face, Ollama, OpenAI)
* Handling model loading errors and retrying attempts
* Offering batch/single text embedding functionality with optional L2 normalization
* Providing cosine similarity utilities and top-K retrieval functionality

### `intergrax\rag\rag_answerer.py`

**Description:** This module is responsible for answering questions based on the provided context, utilizing a combination of retrieval and ranking mechanisms with Large Language Models (LLMs). It allows for customization through configuration options.

**Domain:** RAG logic

**Key Responsibilities:**

* Retrieval: Identify relevant context fragments from the input question.
* Ranking (optional): Re-rank retrieved hits based on custom ranking criteria.
* Context building: Construct a cohesive context text using the ranked hits and limit it by character count.
* Citation generation: Create citations for used hits, including source and page information.
* Message building: Generate system and user messages for LLM processing, considering memory-aware or traditional message construction.
* LLM interaction: Pass constructed messages to an LLM adapter for generating answers.

### `intergrax\rag\rag_retriever.py`

**Description:** This module implements a scalable, provider-agnostic RAG retriever for the Integrax framework. It provides utilities for normalizing filters, converting similarity scores to unified formats, and performing batch retrieval.

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

**Note:** The file appears to be a main implementation module and is not experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\re_ranker.py`

**Description:** This module provides a ReRanker class for performing fast and scalable cosine re-ranking over candidate chunks, allowing for optional score fusion with the original retriever similarity.

**Domain:** RAG (Relevance-Aware Retrieval) logic

**Key Responsibilities:**

* Embeds texts in batches using an `intergraxEmbeddingManager`
* Performs lightweight in-memory caching of query embeddings
* Computes cosine similarities between the query and candidate chunks
* Optionally fuses scores with the original retriever similarity
* Preserves the schema of input hits, adding 'rerank_score', optional 'fusion_score', and 'rank_reranked' fields

**Note:** The code appears to be well-structured, readable, and follows good coding practices. There are no obvious issues or concerns that would suggest it is experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\vectorstore_manager.py`

**Description:** 
This module provides a unified vector store manager supporting ChromaDB, Qdrant, and Pinecone. It enables initialization of target stores, upsertion of documents with embeddings, querying by similarity, counting vectors, and deletion by IDs.

**Domain:** Vector Store Management

**Key Responsibilities:**

* Initialize and manage vector stores for ChromaDB, Qdrant, and Pinecone
* Upsert documents and their corresponding embeddings (with batching)
* Query top-K similar vectors using cosine, dot, or euclidean similarity
* Count the number of vectors in a collection
* Delete vectors by IDs

Note: This module appears to be production-ready as it has a comprehensive set of features and handles various edge cases. However, without further information on its usage and integration with other components of the Intergrax framework, it's difficult to determine its exact status.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module is responsible for implementing the Windowed Answerer layer on top of the base Answerer in the Integrax framework. It provides a mechanism to generate answers by processing context windows and synthesizing final responses.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Initializes the Windowed Answerer with an answerer, retriever, and optional verbose logging
* Processes context windows using the `_build_context_local` method to generate text excerpts
* Builds messages for each window using the `_build_messages_for_context` method to inject context-aware information
* Asks the LLM to generate partial answers for each window
* Synthesizes the final answer from partial responses using the `ask_windowed` method
* Optionally summarizes each window's partial response
* Deduplicates sources and appends the final answer (and optional summary) to the memory store if available

### `intergrax\runtime\__init__.py`

DESCRIPTION:
This module serves as the entry point for the Intergrax runtime, responsible for setting up and bootstrapping the framework.

DOMAIN: Runtime Initialization

KEY RESPONSIBILITIES:
* Initializes the Intergrax application context
* Defines the main entry points for the framework's execution

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: This module initializes the knowledge mode in Intergrax, a framework for integrating large language models with external data and agents.

Domain: Knowledge Mode Initialization

Key Responsibilities:
- Initializes the knowledge mode by loading relevant components.
- Sets up the necessary infrastructure for knowledge retrieval and processing.

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides utilities for resolving attachments in Drop-In Knowledge Mode, decoupling attachment storage and consumption.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Defines the `AttachmentResolver` protocol, which abstracts turning an `AttachmentRef` into a local `Path`.
* Provides the `FileSystemAttachmentResolver` implementation for handling local filesystem-based URIs.
* Enables decoupling of attachment storage and consumption.

Note that this module appears to be a designed component within the Intergrax framework, with clear responsibilities and interfaces.

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This module provides configuration options for the Drop-In Knowledge Runtime, defining how it interacts with tools, LLM adapters, RAG, and web search.

**Domain:** Configuration

**Key Responsibilities:**

- Defines the primary LLM adapter used for generation
- Configures RAG document indexing and retrieval
- Enables or disables features such as web search, long-term memory, and user profile memory
- Specifies the tools agent responsible for invoking tools and merging results into the final answer
- Determines the high-level policy defining whether tools may or must be used

This file appears to be a complete and stable part of the Intergrax framework.

### `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`

**Description:** This module is responsible for building the context in Drop-In Knowledge Mode, including deciding whether to use RAG (ReAdam Retrieval Augmentation) and retrieving relevant document chunks from the vector store.

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

**Notes:** The module is well-documented, and its responsibilities are clearly outlined. However, some parts of the code seem to be commented out or incomplete (e.g., `_should_use_rag` method). Additionally, the use of `async` methods suggests that this module is designed for asynchronous operation, but more context would be needed to confirm this assumption.

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

**Description:** This module defines the core runtime engine for Drop-In Knowledge Mode in the Intergrax framework.

**Domain:** LLM adapters, RAG logic, data ingestion, agents, configuration, utility modules

**Key Responsibilities:**

* Defines the `DropInKnowledgeRuntime` class
* Loads or creates chat sessions via `SessionManager`
* Appends user messages to the session
* Builds an LLM-ready context using various components (RAG, web search, tools)
* Calls the main LLM adapter to produce a final answer
* Returns a `RuntimeAnswer` object with the final answer text and metadata

This module appears to be fully functional and not experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\engine_history_layer.py`

**Description:** This module encapsulates the logic for managing conversation history, including loading raw history, computing token usage, applying history compression strategies, and updating the RuntimeState with preprocessed conversation history.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**
- Load raw conversation history from SessionStore
- Compute token usage for the raw history, if possible
- Apply token-based truncation according to the per-request history compression strategy
- Update RuntimeState with preprocessed conversation history and debug information
- Handle edge cases where token counting or budgeting is not possible

Note: The file appears to be well-maintained and functional, without any obvious issues.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

Description: This module provides a high-level ingestion service for attachments in the context of Drop-In Knowledge Mode, reusing existing Intergrax RAG building blocks to load, split, embed, and store documents.

Domain: Data Ingestion (RAG logic)

Key Responsibilities:
- Resolve AttachmentRef objects into filesystem Paths using AttachmentResolver.
- Load documents using IntergraxDocumentsLoader.load_document(...).
- Split them into chunks using IntergraxDocumentsSplitter.split_documents(...).
- Embed chunks (via IntergraxEmbeddingManager).
- Store vectors (via IntergraxVectorstoreManager).
- Return a structured IngestionResult per attachment.

Note: This service is intended to be called from orchestration layers when new attachments are added to a session.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`

DESCRIPTION:
This module initializes and manages prompts for drop-in knowledge mode in the Intergrax framework.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Initializes the prompt loader for drop-in knowledge mode.
- Defines a set of default prompts used by the framework.
- Exposes functionality to register custom prompts.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`

**Description:** This module provides a framework for building history-summary-related prompts in Drop-In Knowledge Mode, allowing customization of summarization strategies.

**Domain:** LLM adapters

**Key Responsibilities:**
- Provides an interface (`HistorySummaryPromptBuilder`) for customizing history summarization prompts
- Defines a default implementation (`DefaultHistorySummaryPromptBuilder`) that generates a static prompt for summarizing older conversation turns
- Allows future implementations to build on this structure and incorporate request, strategy, or message-specific logic

**Note:** The file appears to be part of the Intergrax framework's core functionality.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`

**Description:** This module defines a strategy for building prompts related to Retrieval-Augmented Generation (RAG) in the Intergrax framework, including generating system prompt text and injecting retrieved document context.

**Domain:** RAG logic

**Key Responsibilities:**
- Defines `RagPromptBundle` class to hold system prompt and context messages.
- Introduces `RagPromptBuilder` protocol for customizing RAG-related prompt building strategies.
- Provides default implementation (`DefaultRagPromptBuilder`) with responsibilities:
  - Using the system prompt from `BuiltContext` as-is.
  - Formatting retrieved chunks into a single additional system-level message.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`

Description: This module provides a utility for building web search prompts in the Drop-In Knowledge Mode of the Intergrax framework.

Domain: LLM adapters

Key Responsibilities:
- Provides a protocol for custom implementation of web search prompt builders.
- Defines a default prompt builder for web search results that takes a list of web documents and returns a system-level message with titles, URLs, and snippets.
- Offers basic debug info such as number of docs and top URLs.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

Description: This module defines data models for the Drop-In Knowledge Mode runtime in Intergrax, including requests and responses that represent high-level contracts between applications and the runtime.

Domain: Data Models / Runtime Interface

Key Responsibilities:
- Define dataclasses for request and response structures.
- Expose citations, routing information, tool calls, and basic statistics.
- Intentionally hide low-level implementation details.

### `intergrax\runtime\drop_in_knowledge_mode\runtime_state.py`

**Description:** The `RuntimeState` module is responsible for managing the state of the Intergrax framework during runtime, particularly in drop-in knowledge mode. It aggregates various metadata and results from different subsystems to provide a comprehensive view of the conversation context.

**Domain:** RAG logic

**Key Responsibilities:**

* Aggregates request and session metadata
* Stores ingestion results
* Manages conversation history and model-ready messages
* Tracks usage flags for RAG, websearch, tools, and memory subsystems
* Handles tools traces and agent answers
* Maintains a debug trace for observability and diagnostics

Note: The code appears to be production-ready, with clear documentation and type hints. No indications of experimental or auxiliary nature are present.

### `intergrax\runtime\drop_in_knowledge_mode\session\__init__.py`

Description: This module initializes sessions for drop-in knowledge mode, facilitating the integration of external knowledge sources within Intergrax.

Domain: RAG (Retrieval-Augmented Generation) logic

Key Responsibilities:
- Initializes session variables and settings
- Loads necessary dependencies for knowledge retrieval and injection
- Establishes connections to external knowledge databases or APIs

### `intergrax\runtime\drop_in_knowledge_mode\session\chat_session.py`

**Description:** 
This module defines the domain model for a chat session, encapsulating its metadata and state in the `ChatSession` dataclass. It provides methods for updating timestamps, marking sessions as closed, incrementing user turns, and accessing status-related properties.

**Domain:** Session management, Chat runtime

**Key Responsibilities:**
- Define the structure of a single chat session
- Provide status-related properties (open, closed)
- Methods to update timestamps and mark sessions as closed
- Increment user turns counter
- Domain helpers for updating session state in-memory

### `intergrax\runtime\drop_in_knowledge_mode\session\in_memory_session_storage.py`

**Description:** This module provides an in-memory implementation of SessionStorage for Intergrax's drop-in knowledge mode. It stores chat session metadata and conversation history within the process, using a simple FIFO trimming policy.

**Domain:** In-Memory Session Storage

**Key Responsibilities:**

* Stores ChatSession metadata in an in-process dictionary
* Maintains per-session conversation history using ConversationalMemory
* Applies a simple FIFO trimming policy via ConversationalMemory's max_messages setting
* Provides methods for creating, saving, and retrieving chat sessions
* Allows appending messages to the conversation history of a session
* Returns ordered conversation history for a given session ID

### `intergrax\runtime\drop_in_knowledge_mode\session\session_manager.py`

**Description:** This file implements the `SessionManager` class, responsible for managing chat sessions in the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Orchestrate session lifecycle on top of a SessionStorage backend
* Provide a stable API for the runtime engine (DropInKnowledgeRuntime)
* Integrate with user/organization profile managers to expose prompt-ready system instructions per session
* Optionally trigger long-term user memory consolidation for a session

### `intergrax\runtime\drop_in_knowledge_mode\session\session_storage.py`

**Description:** This module defines a low-level storage interface for chat sessions and their conversation history, providing methods for persisting and loading session metadata and conversation history.

**Domain:** Session Storage

**Key Responsibilities:**

* Persist and load ChatSession objects
* Persist and load conversation history (ChatMessage sequences) for a given session
* Provide low-level storage operations for session metadata CRUD (Create, Read, Update, Delete)
* Allow appending messages to the conversation history of a session
* Enable retrieving the ordered conversation history for a given session ID

### `intergrax\runtime\organization\__init__.py`

DESCRIPTION: This module serves as the initialization point for the organization package within Intergrax's runtime, responsible for setting up necessary components and modules.

DOMAIN: Organization Package Initialization

KEY RESPONSIBILITIES:
- Initializes the organization package.
- Sets up required sub-modules and utilities.
- Establishes relationships between organizational components.

### `intergrax\runtime\organization\organization_profile.py`

**Description:** This module defines data structures and classes for representing organization-level profiles in the Integrax framework. It includes stable identification data, preferences, system instructions, and long-term memory entries.

**Domain:** Organization management / configuration

**Key Responsibilities:**

* Provides `OrganizationIdentity` class to represent an organization's stable identification data.
* Offers `OrganizationPreferences` class to store organization-level settings that influence runtime behavior.
* Defines `OrganizationProfile` class as the single source of truth for an organization's long-term profile, containing identity, preferences, system instructions, and memory entries.

Note: This file appears to be part of the mainline codebase rather than experimental or auxiliary.

### `intergrax\runtime\organization\organization_profile_instructions_service.py`

**Description:** This module provides a service to generate and update organization-level system instructions for the Intergrax framework, utilizing an LLMAdapter based on the OrganizationProfile.

**Domain:** LLM adapters & Organization Profile Management

**Key Responsibilities:**

* Load OrganizationProfile via OrganizationProfileManager.
* Build an LLM prompt using identity, preferences, summaries, and memory entries.
* Call LLMAdapter.generate_messages() to obtain a compact, stable organization-level system prompt.
* Persist the result via OrganizationProfileManager.update_system_instructions().
* Handle regeneration of instructions based on configuration settings.

### `intergrax\runtime\organization\organization_profile_manager.py`

**Description:** This module provides a high-level facade for managing organization profiles in the Intergrax framework, hiding direct interaction with the underlying profile store.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Load an organization profile for a given `organization_id`
* Persist changes to the profile
* Resolve system instructions for the organization (deterministic logic)
* Update system instructions (policy concern for higher-level components)

Note: This file appears to be well-structured and production-ready, with clear responsibility separation and concise code.

### `intergrax\runtime\organization\organization_profile_store.py`

**Description:** This module defines a protocol for persistent storage of organization profiles in the Integrax framework.

**Domain:** Organization management

**Key Responsibilities:**
- Providing an interface for loading and saving organization profiles
- Hiding backend-specific concerns (e.g., JSON files, SQL DB)
- Ensuring implementations do not perform LLM prompt logic, RAG operations, or decide how the profile is injected into prompts
- Returning initialized profiles even if no data exists yet
- Allowing unknown organization IDs without error

### `intergrax\runtime\organization\stores\__init__.py`

DESCRIPTION: The __init__.py file initializes and sets up the stores module for the Intergrax runtime, providing a centralized entry point for interacting with data stores.

DOMAIN: Data Ingestion/Storage

KEY RESPONSIBILITIES:
- Initializes the stores module and its components.
- Provides an interface for registering and accessing various data store implementations.

### `intergrax\runtime\organization\stores\in_memory_organization_profile_store.py`

**Description:** This module provides an in-memory implementation of the OrganizationProfileStore, suitable for unit testing, local development, and experiments.

**Domain:** RAG logic (Relationship-Aggregation pattern)

**Key Responsibilities:**

* Provides an in-memory store for organization profiles
* Supports basic CRUD operations (get_profile, save_profile, delete_profile)
* Allows storing and retrieving profiles by organization ID
* Offers a default profile creation mechanism for new organizations

Note: This implementation is not intended for production use due to its lack of durability and cross-process sharing features.

### `intergrax\runtime\user_profile\__init__.py`

DESCRIPTION: The `__init__.py` file initializes the user profile module within the Intergrax framework, providing an entry point for importing and utilizing its functionality.

DOMAIN: User Profile

KEY RESPONSIBILITIES:
- Initializes the user profile module
- Defines the entry point for importing module functions and classes

### `intergrax\runtime\user_profile\session_memory_consolidation_service.py`

**Description:** This module provides a service for consolidating chat session history into long-term user profile memory entries and optionally refreshing system instructions.

**Domain:** LLM adapters, User Profile Management, Memory Consolidation

**Key Responsibilities:**

* Take the session conversation (ChatMessage sequence) as input
* Ask the LLM to extract USER_FACT items, PREFERENCE items, and optional SESSION_SUMMARY item
* Map the extracted data into UserProfileMemoryEntry objects
* Persist them through UserProfileManager (add_memory_entry)
* Optionally refresh user-level system instructions after storing new memory entries

### `intergrax\runtime\user_profile\user_profile_debug_service.py`

**Description:** This module provides a high-level service responsible for building debug snapshots of user profiles, including identity, preferences, and recent interactions.

**Domain:** User Profile Debugging

**Key Responsibilities:**

* Build UserProfileDebugSnapshot objects for a given user
* Aggregate data from UserProfileManager and SessionManager
* Expose a read-only API for debug purposes
* Provide detailed information about the user's profile, memory entries, and sessions
* Support ad-hoc diagnostics during development

### `intergrax\runtime\user_profile\user_profile_debug_snapshot.py`

**Description:** This module provides a lightweight, debug-friendly view of user profile state and recent sessions for observability purposes.

**Domain:** Runtime/Profile Management

**Key Responsibilities:**

* Provides `SessionDebugView` and `MemoryEntryDebugView` dataclasses to expose relevant fields for inspection and UI display.
* Offers `UserProfileDebugSnapshot` class to create an immutable snapshot of user profile state, including core identifiers, memory statistics, recent sessions, and a timestamp when the snapshot was generated.
* Includes helper methods (`build_memory_kind_counters`, `from_domain_session`, `from_memory_entry`) to facilitate the construction of debug views from domain objects.

### `intergrax\runtime\user_profile\user_profile_instructions_service.py`

**Description:** This module provides the UserProfileInstructionsService, responsible for generating and updating user-level system instructions using an LLMAdapter.

**Domain:** LLM adapters

**Key Responsibilities:**

* Load user profile via UserProfileManager.
* Build an LLM prompt from user identity, preferences, and memory entries.
* Call LLMAdapter to generate compact system instructions.
* Persist the result via UserProfileManager.
* Handle regeneration of instructions based on configuration settings.

### `intergrax\supervisor\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the supervisor component within Intergrax, responsible for managing and coordinating the overall system.

DOMAIN: Supervisor Management

KEY RESPONSIBILITIES:
• Initializes supervisor services
• Sets up event listeners and notification handlers
• Defines interface for interacting with the supervisor
• Provides configuration and logging functionality

### `intergrax\supervisor\supervisor.py`

**Description:** This is the main supervisor class in the Integrax framework, responsible for planning and executing tasks based on user queries. It uses large language models to generate plans and execute them by invoking specific components.

**Domain:** Supervisor/LLM adapters

**Key Responsibilities:**

* Planning tasks using two-stage approach (decomposition and per-step assignment)
* Using LLMs to decompose tasks into individual steps
* Assigning components to each step based on the plan
* Executing plans by invoking specific components
* Providing analysis of generated plans
* Handling user queries, meta-data, and configuration settings
* Fallback mechanisms for when the two-stage approach fails

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides the SupervisorComponents class, which defines components and their interfaces for integrating with the Integrax framework. It enables registering and running step implementations.

**Domain:** RAG (Retrieval-Augmented Generation) logic, Supervision layer

**Key Responsibilities:**
- Defines Component dataclass to represent individual steps.
- Provides ComponentContext dataclass for passing context to components.
- Offers the run method for executing registered component functions.
- Introduces syntactic sugar via a decorator for registering new components.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides a set of default prompt templates and planning guidelines for the Intergrax framework's Supervisor.

**Domain:** RAG logic / Planning

**Key Responsibilities:**

* Define universal prompts for Supervisor-Planner
* Provide default plan system and user template for planning
* Specify planning constraints (decomposition first, acyclic DAG, etc.)
* Define output format and schema for the generated plan
* Implement dataclass `SupervisorPromptPack` to store and manage prompt templates

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module provides utilities for constructing and executing the LangGraph pipeline within the Intergrax framework. It handles state management, node creation, graph building, and input/output resolution.

**Domain:** Supervision/Control Flow Management

**Key Responsibilities:**

* State schema and utility functions:
	+ Managing global state across the pipeline
	+ Resolving inputs for each plan step
	+ Persisting outputs from each plan step
* Node factory and creation:
	+ Generating unique node names based on plan steps
	+ Creating nodes as functions that execute a single plan step
* Graph building and ordering:
	+ Performing stable topological ordering of plan steps
	+ Constructing the LangGraph pipeline from a Plan object
* Utilities for graph construction:
	+ Slugifying text to create readable identifiers
	+ Making unique node names

**Notes:** The file appears well-structured, and its code is clean and concise. It seems like an essential module within the Intergrax framework, providing core functionality for LangGraph pipeline management.

### `intergrax\system_prompts.py`

Description: This module defines a system instruction for RAG (Reinforced Actor-Graph) logic within the Integrax framework.

Domain: LLM adapters/RAG logic

Key Responsibilities:
* Defines a strict RAG system instruction with guidelines for answering user questions based on document content.
* Specifies rules for citing sources, including formatting and referencing documents.
* Outlines procedures for verifying consistency, providing precise terminology, and avoiding speculation or referencing external knowledge.
* Lists dos and don'ts for responding to user queries, including using only available documents and acknowledging uncertainty when necessary.

Note: The file appears to be a well-documented and formalized set of guidelines rather than an executable code module.

### `intergrax\tools\__init__.py`

DESCRIPTION: This is the entry point for the tools module in Intergrax, responsible for package initialization.

DOMAIN: Tools Module Initialization

KEY RESPONSIBILITIES:
- Initializes package-level imports and configurations
- Registers available tools and utilities within the Intergrax framework
- Provides a standardized interface for tool access

### `intergrax\tools\tools_agent.py`

**Description:** This module defines a ToolsAgent class for orchestrating tool interactions in the Intergrax framework. It handles the integration of Large Language Models (LLMs) with external tools and provides a structured approach to tool invocation.

**Domain:** Tool Orchestration / LLM Integration

**Key Responsibilities:**

* Instantiates a ToolsAgent object, which manages the interaction between an LLM adapter, tool registry, and conversational memory
* Provides methods for running tool orchestration (e.g., `run`) that accept input data in various formats (string or list of ChatMessage)
* Offers tools-specific functionality, such as pruning messages for OpenAI compatibility and building output structures from tool traces
* Utilizes configuration settings (ToolsAgentConfig) to customize its behavior

### `intergrax\tools\tools_base.py`

**Description:** This module provides foundational tools and utilities for building and managing integrations within the Intergrax framework.

**Domain:** Configuration/Utility Modules

**Key Responsibilities:**

* Provides a base class `ToolBase` for defining and running integrations
	+ Allows for strict validation of input arguments using Pydantic (optional)
	+ Defines methods for getting parameters, running the tool, validating arguments, and converting to OpenAI-compatible schema
* Offers a registry `ToolRegistry` for storing and exporting tools in a format accepted by the OpenAI Responses API
	+ Enables registration, retrieval, and listing of tools
	+ Converts tool definitions to OpenAI-compatible JSON objects

Note: The file appears to be well-structured, complete, and production-ready.

### `intergrax\websearch\__init__.py`

DESCRIPTION: This module initializes the web search feature of Intergrax by importing necessary components and setting up the framework for external libraries.

DOMAIN: Web Search Setup

KEY RESPONSIBILITIES:
* Imports required modules and classes from other packages
* Initializes web search configuration and settings
* Sets up framework for integrating external web search libraries

### `intergrax\websearch\cache\__init__.py`

**Description:** This module provides an in-memory query cache with optional time-to-live (TTL) and maximum size for storing web search results.

**Domain:** Web Search Cache

**Key Responsibilities:**

*   Stores cached web search results in memory
*   Provides a simple way to cache and retrieve web documents
*   Allows setting TTL and maximum number of entries
*   Implements a basic eviction policy (LRU) for handling cache size limits
*   Uses immutable data classes for query keys and cache entries

Note: This module appears to be well-structured, complete, and production-ready.

### `intergrax\websearch\context\__init__.py`

Description: This module initializes the web search context, providing essential functionality for searching and retrieving data from the web.
Domain: Web Search Context Initialization

Key Responsibilities:
• Initializes the web search context with default settings
• Configures the search query parser and analyzer
• Sets up the caching mechanism for frequent queries

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides a utility class for building LLM-ready textual context and chat messages from web search results.

**Domain:** Web Search

**Key Responsibilities:**

* Builds a textual context string from WebDocument objects or serialized dicts produced by the WebSearchExecutor.
* Supports customization of context format, including inclusion of snippets, URLs, and character limits per document.
* Provides methods for building strict system prompts and user-facing prompts that wrap web sources, questions, and tasks.
* Generates chat messages in a typical pair format (system + user) suitable for chat-style LLMs.

Note: This file appears to be part of the Intergrax framework's web search functionality, and its code is well-structured and readable. There are no signs of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\websearch\fetcher\__init__.py`

Description: This module initializes the web search fetcher, responsible for retrieving relevant data from external sources.

Domain: Data Ingestion

Key Responsibilities:
• Initializes and configures the web search fetcher
• Defines interfaces for data retrieval and processing
• Sets up connections to external data sources (e.g., APIs)

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module contains the core functionality for HTML content extraction and processing within the Intergrax framework, including lightweight metadata extraction (`extract_basic`) and advanced readability-based extraction (`extract_advanced`).

**Domain:** Web Search / Content Extraction

**Key Responsibilities:**

- Perform lightweight HTML metadata extraction:
  - Extract title
  - Extract meta description
  - Extract HTML language attribute
  - Extract Open Graph meta tags (og:*)
  - Produce a plain-text version of the page
- Perform advanced readability-based extraction using trafilatura (if installed) or fallback to BeautifulSoup for non-content elements removal and text extraction
- Normalize whitespace and reduce noise in extracted text
- Attach metadata for debugging and analysis purposes, including usage of trafilatura and length comparisons before/after extraction

**Notes:** This module appears well-maintained and thoroughly documented. The code structure is logical, and the functions' responsibilities are clearly defined.

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module provides an asynchronous HTTP client for fetching web pages and encapsulating their content.

**Domain:** Web Search

**Key Responsibilities:**

* Performs asynchronous HTTP GET requests with customizable headers
* Captures final URL, status code, raw HTML, and body size of the response
* Returns a `PageContent` instance upon successful fetch or `None` on transport-level failure

### `intergrax\websearch\integration\__init__.py`

**Description:** This is the entry point for web search integration into the Intergrax framework, responsible for setting up and configuring the integration process.

**Domain:** Web Search Integration

**Key Responsibilities:**

* Initializes web search integration components
* Configures integration settings
* Defines interface for interacting with web search APIs

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module encapsulates a web search node for LangGraph-compatible applications. It provides a lightweight interface to delegate web search operations to the underlying WebSearchExecutor instance.

**Domain:** RAG logic / Data Ingestion

**Key Responsibilities:**

* Provides a LangGraph-compatible web search node wrapper (WebSearchNode)
	+ Encapsulates configuration of WebSearchExecutor
	+ Supports synchronous and asynchronous node methods
* Allows for customization of WebSearchExecutor through constructor parameters
* Offers a default, module-level node instance for convenience and backward compatibility
* Exposes functional, synchronous and async wrappers (websearch_node and websearch_node_async) around the default node instance

### `intergrax\websearch\pipeline\__init__.py`

Description: This module initializes the web search pipeline, handling setup and configuration for downstream processing.

Domain: Pipeline Initialization

Key Responsibilities:
- Initializes pipeline components
- Configures pipeline settings
- Establishes dependencies between modules

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module implements a pipeline for orchestrating web search, fetching, extraction, deduplication, and basic quality scoring.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Orchestrates multi-provider web search using the `search_all` method
* Fetches and extracts individual hits into WebDocument objects using the `_fetch_one` method
* Deduplicates documents based on a simple text-based dedupe key
* Sorts results by quality score (descending) and source rank (ascending)
* Provides asynchronous (`run`) and synchronous (`run_sync`) execution modes for pipeline operations

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: 
This module serves as the entry point for web search providers in Intergrax, handling instantiations and configurations.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
• Initializes available web search provider classes
• Configures web search providers
• Instantiates and returns configured web search providers

### `intergrax\websearch\providers\base.py`

**Description:** This module provides a base interface for web search providers to interact with the Integrax framework.

**Domain:** Web Search Providers

**Key Responsibilities:**

* Accepts a provider-agnostic QuerySpec
* Returns a ranked list of SearchHit items
* Exposes minimal capabilities for feature negotiation (language, freshness)
* Provides optional resource cleanup through the close method
* Implementations must honor top_k with provider-side caps and include 'provider' and 'query_issued' fields in hits

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search provider for the Intergrax framework, enabling search functionality through the Bing REST API.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

*   Initializes a Bing Web Search provider instance with optional API key, session, and timeout settings.
*   Provides capabilities of the provider, including language, freshness, and maximum page size support.
*   Builds headers for API requests with API key authentication.
*   Constructs parameters for API queries based on query specifications.
*   Parses search results from Bing's API responses and converts them into SearchHit objects.
*   Performs a search request to the Bing Web Search API with specified query and returns a list of SearchHit objects.
*   Closes the session after use.

**Status:** No indication of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module provides a Google Custom Search (CSE) provider for the Intergrax framework, enabling searching using the CSE REST API.

**Domain:** Websearch Providers

**Key Responsibilities:**

* Initializes a Google CSE provider with optional API key and CX search engine ID
* Supports language filtering via 'lr' parameter
* Supports UI language filtering via 'hl' parameter
* Ignores freshness parameter due to lack of native support in CSE
* Builds query parameters for the CSE REST API
* Performs a GET request to the CSE endpoint with built query parameters
* Parses JSON response from the CSE API and returns search hits
* Ensures stable 1-based rank ordering for returned search hits

### `intergrax\websearch\providers\google_places_provider.py`

Description: This module provides a Google Places API provider for the Intergrax web search framework, enabling text search and details retrieval for businesses.

Domain: Web Search Providers (RAG logic)

Key Responsibilities:
- Provides a Google Places API implementation for text search and business details
- Exposes environment variables and settings for Google Places API key and other parameters
- Supports query specification and parameter building for text search and details requests
- Fetches and maps place details from the Google Places API to SearchHit objects

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a full-featured Reddit search provider using the official OAuth2 API, enabling features like rich post metadata and optional top-level comment fetching.

**Domain:** WebSearch Providers

**Key Responsibilities:**

* Handles authentication with Reddit's OAuth2 client credentials
* Supports searching posts based on query specifications
* Maps query parameters to Reddit's API endpoints
* Fetches comments for a given post
* Returns SearchHit objects representing search results
* Implements capabilities like language and freshness filtering, as well as limiting the maximum page size

### `intergrax\websearch\schemas\__init__.py`

Description: This module initializes and exports the Web Search schemas for Intergrax.

Domain: Data Ingestion

Key Responsibilities:
- Initializes the web search schema module.
- Exports the necessary schemas for web search functionality.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass `PageContent` to represent the fetched and optionally extracted content of a web page, including metadata and derived information.

**Domain:** Web Search/Schemas

**Key Responsibilities:**

* Encapsulates raw HTML and derived metadata for post-processing stages
* Provides attributes for metadata such as title, description, language, Open Graph tags, schema.org data, and more
* Offers methods to check if the page contains content (`has_content`), generate a short summary of the text (`short_summary`), and calculate the approximate size of the content in kilobytes (`content_length_kb`)

### `intergrax\websearch\schemas\query_spec.py`

Description: This module defines a dataclass for canonical search query specifications used by web search providers, providing a structured way to describe user queries and their constraints.

Domain: Web Search Query Specification

Key Responsibilities:
- Defines the `QuerySpec` dataclass with attributes for query parameters.
- Provides methods for normalizing queries (`normalized_query`) and capping top results per provider (`capped_top_k`).

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a `SearchHit` dataclass to standardize and validate metadata for search results across different providers.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Provides a structured representation of a single search result entry
* Includes provider-agnostic fields (e.g., rank, title, URL) and optional provider-specific fields
* Enforces minimal safety checks on the `SearchHit` instance (e.g., valid rank, URL scheme, and netloc)
* Offers methods to extract domain information (`domain`) and a minimal, LLM-friendly representation of the hit (`to_minimal_dict`)

### `intergrax\websearch\schemas\web_document.py`

**Description:** This module defines a unified structure for representing fetched and processed web documents, connecting provider metadata with extracted content and analysis results.

**Domain:** Web Search

**Key Responsibilities:**
- Defines the `WebDocument` class as a dataclass, encapsulating search hit, page content, deduplication key, quality score, and source rank.
- Provides methods for document validation (`is_valid`), merged text generation (`merged_text`), and summary line creation (`summary_line`).

### `intergrax\websearch\service\__init__.py`

Description: This module serves as the entry point for the Intergrax web search service, responsible for initializing and setting up the necessary components.

Domain: Web Search Service

Key Responsibilities:
* Initializes the web search service
* Sets up necessary dependencies and configurations
* Exposes API endpoints for querying

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level helper class (`WebSearchAnswerer`) for conducting web searches, building context/messages from the search results, and generating answers using a language model adapter.

**Domain:** Web Search Service

**Key Responsibilities:**

* Runs web search via `WebSearchExecutor`
* Builds LLM-ready context/messages from web documents
* Calls any LLMAdapter to generate a final answer
* Provides synchronous and asynchronous API for non-async environments
* Handles system prompts and override configuration

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module provides a high-level, configurable web search executor that constructs QuerySpec from raw queries and configuration, executes the SearchAndReadPipeline with chosen providers, and converts WebDocument objects into LLM-friendly dictionaries.

**Domain:** Web search services integration

**Key Responsibilities:**
- Constructing QuerySpec from raw query and configuration
- Executing SearchAndReadPipeline with chosen providers
- Converting WebDocument objects into LLM-friendly dictionaries
- Managing query cache for serialized results

### `intergrax\websearch\utils\__init__.py`

Description: This module initializes and configures the web search utility package, providing a foundation for subsequent modules to build upon.

Domain: Web Search Utilities

Key Responsibilities:
- Initialize package-level constants and variables
- Import and configure dependent packages and modules
- Establish default settings and configurations for the web search utilities

### `intergrax\websearch\utils\dedupe.py`

**intergrax\websearch\utils\dedupe.py**

Description: This module provides utilities for deduplication in the web search pipeline, including text normalization and hash-based key generation.

Domain: Web Search Utilities

Key Responsibilities:
- Normalizes input text to facilitate deduplication
  * Treats None as empty string
  * Strips leading/trailing whitespace
  * Converts to lower case
  * Collapses internal whitespace sequences
- Generates a stable SHA-256 based deduplication key for the given text
  * Hex-encoded digest of the normalized text

Note: This module appears to be designed for specific use within the Intergrax web search pipeline and does not exhibit characteristics typically associated with experimental, auxiliary, or legacy code.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module implements a simple token bucket rate limiter, allowing for controlled and concurrent access to resources within the Intergrax framework.

**Domain:** Utility modules (specifically, rate limiting)

**Key Responsibilities:**

* Provides an asyncio-compatible implementation of the token bucket algorithm
* Allows for customizable rate limits and capacities
* Offers two main methods: `acquire` (blocking) and `try_acquire` (non-blocking)
* Designed to be used across concurrent coroutines within a single process

### `main.py`

Description: This module serves as the entry point for the Intergrax framework, responsible for executing the main program logic.
Domain: Framework Entry Point

Key Responsibilities:
* Defines the main function that runs when the script is executed directly
* Prints a welcome message indicating the execution of the Intergrax-ai application

### `mcp\__init__.py`

Here is the documentation for the given file:

**Description:** This module serves as the main entry point and package initializer for the MCP (Model Construction Platform) framework.

**Domain:** Configuration

**Key Responsibilities:**
* Initializes and sets up the MCP package structure
* Defines and registers key components, such as modules and services
* Provides a hook for custom initialization or setup
* Enables importing of other MCP packages and modules

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

Description: This notebook is a basic sanity-check implementation for the Drop-In Knowledge Mode runtime in the Intergrax framework, verifying its functionality and behavior.

Domain: RAG logic

Key Responsibilities:
- Verifies the creation or loading of a session
- Appends user and assistant messages to the conversation history
- Builds conversation history from SessionStore
- Returns a RuntimeAnswer object
- Keeps LLM integration as a placeholder for now

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

Description: This Jupyter notebook demonstrates the Drop-In Knowledge Mode runtime, specifically how it works with sessions and basic conversational memory, accepts attachments via `AttachmentRef`, and uses the `AttachmentIngestionService` to resolve attachment URIs to files, load and split documents with Intergrax RAG components, embed them, and store them in the vector store.

Domain: Drop-In Knowledge Mode Runtime

Key Responsibilities:
- Initializes the runtime using an in-memory session store, LLM adapter (Ollama + LangChain), embedding manager (Ollama embeddings), and vector store manager (Chroma collection).
- Demonstrates attachment ingestion by creating an `AttachmentRef` for a local project document.
- Verifies that attachments are correctly stored in the session, ingestion runs without errors, chunks are stored in the vector store, and ingestion details are visible in `debug_trace`.

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

Description: This Jupyter notebook demonstrates the use of the ContextBuilder component in Intergrax's Drop-In Knowledge Mode runtime, specifically for loading a demo chat session, working with an existing attachment, initializing the ContextBuilder, and building context for a single user question.

Domain: RAG logic

Key Responsibilities:
- Load a demo chat session using an existing SessionStore.
- Work with an existing attachment ingested from a previous notebook or prepare a new one and ingest it using the existing pipeline.
- Initialize the ContextBuilder instance using the shared IntergraxVectorstoreManager and RuntimeConfig.
- Build context for a single user question by creating a RuntimeRequest, calling ContextBuilder.build_context(session, request), and obtaining reduced chat history, retrieved document chunks, system prompt, and RAG debug information.
- Inspect the result of context building without performing an LLM call.

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

Here is the documentation for the provided file:

**Description:** This Jupyter Notebook demonstrates how to use the DropInKnowledgeRuntime with session-based chat, optional RAG, and live web search via WebSearchExecutor to achieve a "ChatGPT-like" experience with browsing.

**Domain:** **RAG logic** with **web search integration**

**Key Responsibilities:**
- Initializes core runtime configuration (LLM + embeddings + vector store + web search)
- Demonstrates session-based chat with optional RAG and live web search
- Uses LangChain Ollama adapter for LLM and Chroma vector store manager for vector store
- Configures web search executor to wrap Google CSE provider
- Creates a fresh chat session for the web search demo

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

Description: This notebook demonstrates the integration of tools with the Drop-In Knowledge Runtime, utilizing conversational memory and optional RAG and web search features.

Domain: Tools orchestration

Key Responsibilities:
- Configure Python path and load environment variables.
- Import core building blocks used by the Drop-In Knowledge Runtime.
- Initialize runtime configuration (LLM, embeddings, vector store, web search) in a single compact setup cell.
- Define demo tools using the Intergrax tools framework, including registration and agent attachment.
- Attach the IntergraxToolsAgent instance to RuntimeConfig.tools_agent for orchestration.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates an end-to-end hybrid multi-source RAG (Relevance Aware Generator) workflow using the Intergrax framework and LangGraph, combining multiple knowledge sources into a single in-memory vector index.

**Domain:** Hybrid Multi-Source RAG Logic

**Key Responsibilities:**

* Ingest content from multiple sources:
	+ Local PDF files
	+ Local DOCX/Word files
	+ Live web results using Intergrax `WebSearchExecutor`
* Build a unified RAG corpus:
	+ Normalize all documents into a common internal format
	+ Attach basic metadata about origin (pdf / docx / web)
	+ Split documents into chunks suitable for embedding
* Create an in-memory vector index:
	+ Use Intergrax embedding manager (e.g. OpenAI or Ollama-based)
	+ Store embeddings in an in-memory Chroma collection via Intergrax vectorstore manager
* Answer user questions with a RAG agent:
	+ The user provides a natural language question
	+ LangGraph orchestrates the flow: load → merge → index → retrieve → answer
	+ An Intergrax `RagAnswerer` generates a single, structured report

**Note:** This file appears to be a working notebook demonstrating the hybrid multi-source RAG logic using the Intergrax framework and LangGraph. It provides a practical example of combining multiple knowledge sources into a single in-memory vector index and answering user questions with a RAG agent.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This notebook demonstrates the integration between Intergrax and LangGraph by showcasing a simple LLM QA example.

Domain: Intergrax + LangGraph integration

Key Responsibilities:
- Integrates an Intergrax LLM adapter as a node inside a LangGraph graph.
- Defines a simple `State` that holds chat-style messages and the final answer returned by the node.
- Builds a `StateGraph` with a single node `llm_answer_node`.
- Runs the graph on a sample user question.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

Description: This notebook serves as a demonstration of the Intergrax framework's capabilities in building a practical web research agent. It integrates components such as WebSearchExecutor, WebSearchContextBuilder, and WebSearchAnswerer to provide a realistic example of "no-hallucination" web-based Q&A inside a graph-based agent.

Domain: LLM adapters, RAG logic

Key Responsibilities:
- Initializes the Intergrax LLM adapter (OpenAIChatResponsesAdapter) for answering strictly from the web context.
- Configures and initializes WebSearch components, including WebSearchExecutor, WebSearchContextBuilder, and WebSearchAnswerer.
- Demonstrates the flow of a multi-step graph within LangGraph, which orchestrates user question normalization, web search execution, context building, and final answer generation with sources.
- Defines the graph state (WebResearchState) for tracking relevant data throughout the process.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This Jupyter notebook provides an example use case for the Integrax framework, specifically demonstrating how to interact with OpenAI's Vector Store using the RAG (Retrieval-Augmented Generation) model.

Domain: RAG logic

Key Responsibilities:
- Initializes the OpenAI client and sets up environment variables.
- Creates an instance of IntergraxRagOpenAI and uses it to ensure the existence of a Vector Store, clear its contents, and upload a local folder to it.
- Tests queries using the RAG model by calling the run method with sample questions.

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This Jupyter notebook defines the configuration and setup for a hybrid chat agent combining RAG (Retrieval-Augmented Generator) with tools and LLM (Large Language Model) chat capabilities.

**Domain:** RAG logic, Agents

**Key Responsibilities:**

* Creates an instance of the `ChatAgent` class, integrating LLM and RAG components
* Sets up a vector store and RAG configuration for document-based queries
* Registers available tools, such as the `WeatherTool`, with the `ToolRegistry`
* Configures the RAG answerer with a retriever, reranker, and embedding manager
* Defines a high-level hybrid agent combining RAG, tools, and LLM chat functionality

**Note:** This file appears to be an implementation of the Intergrax framework's hybrid chat agent, showcasing its capabilities and configuration. The code demonstrates how to integrate various components, including RAG, tools, and LLM chat, to create a powerful conversational AI system.

### `notebooks\rag\output_structure_presentation.ipynb`

Description: This notebook provides an interactive demonstration of how to integrate a tools agent with structured output and RAG logic in the Intergrax framework.

Domain: Tools Agent Integration

Key Responsibilities:
- Define a custom tool (WeatherTool) that returns demo data.
- Use Pydantic models for structured output (e.g., WeatherAnswer, ExecSummary).
- Implement LLM adapters, vector store management, and conversational memory integration.
- Utilize the tools agent to orchestrate LLM reasoning, automatic tool selection, and output mapping.
- Demonstrate usage of the tools agent with a natural-language question.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter Notebook is an interactive presentation of custom RAG (Retrieval-Augmented Generation) components, showcasing loading and embedding documents in the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Load raw documents from a specified directory using the `DocumentsLoader` component
* Split loaded documents into smaller chunks using the `DocumentsSplitter` component
* Generate vector embeddings for each document chunk using the `EmbeddingManager` component
* Perform lightweight "probe" queries to check if the target corpus is already present in the vector store

Note: This notebook appears to be a demonstration or example code rather than an experimental, auxiliary, legacy, or incomplete implementation.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter Notebook, `rag_multimodal_presentation.ipynb`, contains a demonstration of the Intergrax framework's multimodal retrieval capabilities. It presents various examples of loading documents from different sources (video, audio, and images), splitting them into chunks, embedding these chunks, and storing them in a Vectorstore.

**Domain:** RAG logic

**Key Responsibilities:**

* Load documents from video, audio, and image files using `DocumentsLoader`
* Split documents into chunks using `DocumentsSplitter`
* Embed chunks using `EmbeddingManager` with Ollama model
* Store embedded documents in a Vectorstore using `VectorstoreManager`
* Perform retriever test using `RagRetriever`

**Note:** The file appears to be a demonstration or example code, as it contains print statements and does not follow a typical structure of a production-ready module. It may serve as a starting point for further development or experimentation.

### `notebooks\rag\rag_video_audio_presentation.ipynb`

Description: This notebook provides examples of multimedia processing tasks using the Intergrax framework, including video and audio download, transcription, frame extraction, and image translation.

Domain: Multimedia Processing (Video/Audio/Image)

Key Responsibilities:
- Download videos from YouTube
- Transcribe videos to VTT format
- Extract frames and metadata from videos
- Translate audio files
- Use ollama model to describe images
- Extract frames from videos and transcribe images

### `notebooks\rag\tool_agent_presentation.ipynb`

Description: This notebook demonstrates the creation and execution of a ToolsAgent, which integrates with an LLM (Large Language Model) to select and invoke tools for answering user questions. The agent is tested with two scenarios: getting the weather in Warsaw and performing a calculation.

Domain: RAG logic

Key Responsibilities:
- Creates a conversational memory shared across interactions
- Registers available tools for the agent, including a simple demo weather tool and an arithmetic calculator tool
- Configures the LLM (in this case, an Ollama-backed model) as the planner/controller for tools
- Sets up the ToolsAgent to orchestrate LLM reasoning, tool selection and invocation, and conversational memory updates
- Tests the agent with two scenarios:
  - Getting the weather in Warsaw should select the get_weather tool
  - Performing a calculation should select the calc_expression tool

Note: This file appears to be fully functional and not experimental, auxiliary, legacy, or incomplete.

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This Jupyter Notebook contains a collection of components, each representing a specific task or function within the Integrax framework. These components are designed to be executed in a pipeline and perform various tasks such as compliance checking, cost estimation, generating a final summary report, and financial audit.

**Domain:** RAG logic

**Key Responsibilities:**

- **Compliance Checker**: Verifies whether proposed changes comply with privacy policies and terms of service.
- **Cost Estimator**: Estimates the cost of changes based on the UX audit report (mock).
- **Final Summary Report Generator**: Generates the final consolidated summary using all collected artifacts.
- **Financial Audit Agent**: Generates a mock financial report and VAT calculation.

**Note:** The notebook appears to contain test or example code, as indicated by the "mock" and "test data" keywords.

### `notebooks\websearch\websearch_presentation.ipynb`

Description: This notebook demonstrates the usage of the WebSearchExecutor with Google and Bing Search, querying about LangChain and LangGraph.

Domain: LLM adapters

Key Responsibilities:
- Imports necessary modules and sets up environment variables.
- Defines a query specification using QuerySpec class from intergrax.websearch.schemas.query_spec module.
- Creates an instance of GoogleCSEProvider class from intergrax.websearch.providers.google_cse_provider module.
- Searches for results using the provider's search method and prints them out.

Note: This appears to be a demonstration or example code, likely part of the Integrax framework documentation.
