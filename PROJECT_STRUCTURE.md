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
- `intergrax\llm\messages.py`
- `intergrax\llm_adapters\__init__.py`
- `intergrax\llm_adapters\aws_bedrock_adapter.py`
- `intergrax\llm_adapters\azure_openai_adapter.py`
- `intergrax\llm_adapters\claude_adapter.py`
- `intergrax\llm_adapters\gemini_adapter.py`
- `intergrax\llm_adapters\llm_adapter.py`
- `intergrax\llm_adapters\llm_provider.py`
- `intergrax\llm_adapters\llm_provider_registry.py`
- `intergrax\llm_adapters\llm_usage_track.py`
- `intergrax\llm_adapters\mistral_adapter.py`
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
- `intergrax\runtime\drop_in_knowledge_mode\planning\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\planning\engine_plan_models.py`
- `intergrax\runtime\drop_in_knowledge_mode\planning\engine_planner.py`
- `intergrax\runtime\drop_in_knowledge_mode\planning\plan_builder_helper.py`
- `intergrax\runtime\drop_in_knowledge_mode\planning\step_planner.py`
- `intergrax\runtime\drop_in_knowledge_mode\planning\stepplan_models.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\user_longterm_memory_prompt_builder.py`
- `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`
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
- `intergrax\websearch\cache\query_cache.py`
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
- `intergrax\websearch\schemas\web_search_answer.py`
- `intergrax\websearch\schemas\web_search_result.py`
- `intergrax\websearch\service\__init__.py`
- `intergrax\websearch\service\websearch_answerer.py`
- `intergrax\websearch\service\websearch_config.py`
- `intergrax\websearch\service\websearch_context_generator.py`
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
- `notebooks\drop_in_knowledge_mode\06_session_memory_roundtrip_demo.ipynb`
- `notebooks\drop_in_knowledge_mode\07_user_profile_instructions_baseline.ipynb`
- `notebooks\drop_in_knowledge_mode\08_user_profile_instructions_generation.ipynb`
- `notebooks\drop_in_knowledge_mode\09_long_term_memory_consolidation.ipynb`
- `notebooks\drop_in_knowledge_mode\10_e2e_user_longterm_memory.ipynb`
- `notebooks\drop_in_knowledge_mode\11_chatgpt_like_e2e.ipynb`
- `notebooks\drop_in_knowledge_mode\12a_engine_planner.ipynb`
- `notebooks\drop_in_knowledge_mode\12b_engine_planner.ipynb`
- `notebooks\drop_in_knowledge_mode\13_engine_step_planner.ipynb`
- `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`
- `notebooks\langgraph\simple_llm_langgraph.ipynb`
- `notebooks\langgraph\simple_web_research_langgraph.ipynb`
- `notebooks\multimedia\rag_multimodal_presentation.ipynb`
- `notebooks\multimedia\rag_video_audio_presentation.ipynb`
- `notebooks\openai\rag_openai_presentation.ipynb`
- `notebooks\rag\chat_agent_presentation.ipynb`
- `notebooks\rag\output_structure_presentation.ipynb`
- `notebooks\rag\rag_custom_presentation.ipynb`
- `notebooks\rag\tool_agent_presentation.ipynb`
- `notebooks\supervisor\supervisor_test.ipynb`
- `notebooks\websearch\websearch_presentation.ipynb`

## Detailed File Documentation

### `api\__init__.py`

DESCRIPTION: The "__init__.py" file is the entry point for the API package, responsible for initializing and configuring the API framework.

DOMAIN: API Configuration

KEY RESPONSIBILITIES:
• Initializes the API package and sets up necessary imports.
• Configures default settings and logging mechanisms.
• Registers API endpoints and handlers.

### `api\chat\__init__.py`

Description: Initializes and configures the API for chat-related interactions.

Domain: API Configuration

Key Responsibilities:
- Imports necessary modules and components for chat functionality
- Sets up routes and handlers for chat API endpoints

### `api\chat\main.py`

**Description:** This is the main entry point for the API, handling chat interactions and document management within the Integrax framework.

**Domain:** Chat Service / Document Management

**Key Responsibilities:**

- Handles incoming chat requests with AI model answering capability
- Manages upload and indexing of documents to Chroma search engine
- Provides endpoints for listing and deleting stored documents
- Utilizes database and logging utilities for storage and debugging purposes

### `api\chat\tools\__init__.py`

Description: The __init__.py file initializes and sets up the chat tools module, making it available for import in other parts of the application.

Domain: API / Tools

Key Responsibilities:
* Registers all necessary modules and sub-modules within the chat tools package
* Sets up default configurations or parameters for the module
* Exposes the module's interface to be used by other components

### `api\chat\tools\chroma_utils.py`

**Description:** This module provides utility functions for managing documents within the Intergrax framework, specifically related to indexing and deletion.

**Domain:** RAG logic

**Key Responsibilities:**
- `load_and_split_documents`: Loads and splits documents from a given file path.
- `index_document_to_chroma`: Indexes a document in Chroma (vector store) with a provided file ID.
- `delete_doc_from_chroma`: Deletes a document from Chroma by its file ID.

Note: The functions seem to be functionally identical, with the exception of renaming some function names and variable types. This might indicate code duplication or the need for refactoring.

### `api\chat\tools\db_utils.py`

**Description:** This module provides utilities for interacting with a SQLite database, including schema creation and migration, as well as public API functions for managing sessions, messages, and documents.

**Domain:** Database Utilities

**Key Responsibilities:**

* Creating and migrating the database schema
* Managing sessions and ensuring they exist in the database
* Inserting, retrieving, and deleting messages (with optional model and timestamp information)
* Inserting, retrieving, and deleting document records
* Providing backward-compatibility entry points for legacy application logs

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines Pydantic models for API inputs and outputs related to chat functionality.

**Domain:** LLM adapters (Chat Tools)

**Key Responsibilities:**

* Defines enumerations for model names
* Models input data for queries, including question, session ID, and selected model
* Models output responses from queries, including answers, session IDs, and used models
* Models document information, including file IDs, filenames, and upload timestamps
* Defines a request model for deleting files by their IDs

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides various utility functions for the RAG pipeline, including singleton management and adapter creation.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**
- Singleton management for VectorstoreManager, EmbeddingManager, RagRetriever, ReRanker, and RagAnswerer
- Adapter creation for LLM (Large Language Model)
- Retrieval of relevant context using the vector store
- Validation of consistency across retrieved documents
- Response generation based on retrieved excerpts
- Citation rules for referencing source documents

### `applications\chat_streamlit\api_utils.py`

Description: This module provides a set of utility functions for interacting with the chat API and file management.

Domain: Chat Streamlit Integration

Key Responsibilities:

* Makes HTTP requests to the chat API endpoint for question-answering
* Uploads files to the server using HTTP POST requests
* Lists uploaded documents via HTTP GET request
* Deletes documents by their IDs through HTTP POST requests

### `applications\chat_streamlit\chat_interface.py`

Description: This module is responsible for creating and managing a chat interface within Streamlit applications, enabling users to interact with models through a graphical interface.

Domain: Chat Interface (LLM adapters)

Key Responsibilities:
- Initializes and displays a chat interface using Streamlit.
- Allows users to send queries through the interface.
- Sends user queries to an API endpoint for processing.
- Displays responses from the model in the chat interface, along with additional metadata (e.g., session ID).
- Handles errors during API calls.

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides the user interface components for a Streamlit application's sidebar, allowing users to interact with uploaded documents and models.

**Domain:** Configuration/Utility Modules

**Key Responsibilities:**

* Display model selection component
* Handle file uploads and updates
* List and display uploaded documents
* Enable deletion of selected documents

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module serves as the entry point for a Streamlit application, integrating chatbot functionality and a sidebar interface.
**Domain:** Chat Interface
**Key Responsibilities:**
* Initializes the Streamlit app with a title "intergrax RAG Chatbot"
* Sets up initial state variables in the session state: `messages` and `session_id`
* Calls functions to display the sidebar and chat interface

### `applications\company_profile\__init__.py`

Description: This module initializes and configures the company profile application.

Domain: Configuration

Key Responsibilities:
* Initializes the company profile application
* Sets up configuration for the application
* Provides access to company profile data 
* Registers application routes

### `applications\figma_integration\__init__.py`

Description: This module serves as the entry point for Figma integration, setting up necessary modules and configurations.

Domain: Integration Adapters

Key Responsibilities:
• Registers the Figma integration service
• Sets up configuration for the integration
• Imports and initializes other relevant components 
• Establishes connections to external services (experimental)

### `applications\ux_audit_agent\__init__.py`

Description: The __init__.py file is the entry point for the UX Audit Agent application, responsible for initializing and configuring the agent's functionality.

Domain: Agents

Key Responsibilities:
- Initializes the UX Audit Agent instance
- Configures logging settings
- Imports dependent modules and classes

### `applications\ux_audit_agent\components\__init__.py`

Description: This is the initialization module for the UX audit agent components, responsible for setting up and configuring the necessary components.

Domain: Components/Agent Setup

Key Responsibilities:
• Initializes all required components for the UX audit agent
• Sets up configuration and dependencies for the agent's functionality 
• Possibly imports other necessary modules or functions from parent directories

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module implements a compliance checker component for the Integrax framework, responsible for evaluating proposed changes against privacy policies and regulations.

**Domain:** RAG (Rules and Guidelines) logic

**Key Responsibilities:**

* Evaluates proposed changes for compliance with privacy policy and regulatory rules
* Returns a simulated compliance result (80% chance of being compliant)
* Generates findings, including policy violations, required DPO review, and notes
* Stops execution if non-compliant and corrections are needed

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module implements a component for estimating the cost of UX-related changes based on an audit report, using a mock pricing model.

**Domain:** UX Audit Agent Components

**Key Responsibilities:**
- Estimates the cost of UX updates derived from an audit.
- Uses a mock pricing model to calculate the estimated cost.
- Produces a result with the estimated cost and metadata.

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module defines a component for generating a complete final report of the task, using all collected artifacts. It is designed to be executed at the final stage of the pipeline.

**Domain:** RAG logic

**Key Responsibilities:**

* Generates a summary of the entire execution pipeline
* Collects and processes various artifacts (e.g., project manager decision, notes, UX report, financial report, citations)
* Returns a final report with a structured format
* Logs completion of the FinalSummary component

Note: This file appears to be part of the main framework functionality, rather than experimental or auxiliary.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module defines a component for generating mock financial reports and VAT calculations as part of the Intergrax framework's audit agent functionality.

**Domain:** Audit Agent Components

**Key Responsibilities:**
- Provides a Financial Agent component that generates mock financial data.
- Allows users to test financial computations, budget constraints, or cost reports.
- Offers example usage scenarios for generating specific types of financial reports.

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a basic knowledge component for the Intergrax system, answering general questions about its structure and configuration.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a basic knowledge component for the Intergrax system
* Answers general questions about the system's modules, architecture, and documentation
* Returns mock data as examples of potential responses

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a UX project management component that simulates PM decisions based on mock rules, providing functionality for approval or rejection with accompanying comments.

**Domain:** RAG logic (Risk, Action, Review)

**Key Responsibilities:**
- Simulates PM decision-making process
- Generates approval or rejection response with notes
- Optional stopping of pipeline execution upon rejection

### `applications\ux_audit_agent\components\ux_audit.py`

**Description:** This module provides a UX audit component for the Integrax framework, enabling analysis of UI/UX based on Figma mockups and generation of sample reports with recommendations.

**Domain:** LLM Adapters (specifically, UX Audit Logic)

**Key Responsibilities:**
- Performs UX audit on Figma mockups
- Returns sample report with recommendations
- Utilizes component decorator for registration in the supervisor

### `applications\ux_audit_agent\UXAuditTest.ipynb`

**Description:** This Jupyter notebook defines a workflow for performing a UX audit on FIGMA mockups, verifying compliance with company policy, preparing a summary report for the Project Manager, evaluating financial impact of changes, and making a final project decision. It appears to be an interactive guide for using the Integrax framework.

**Domain:** RAG (Reasoning And Goals) logic

**Key Responsibilities:**

* Perform UX audit on FIGMA mockups
	+ Assign score based on findings
	+ Generate UX report with recommendations
* Verify changes comply with company policy
	+ Use Policy & Privacy Compliance Checker tool
	+ Output financial report and last quarter budget
* Prepare summary report for Project Manager
	+ Use Final Report component
	+ Input financial report and cost estimate
* Evaluate financial impact of changes
	+ Use Cost Estimation Agent component
	+ Output cost estimate and budget notes
* Make final project decision based on previous steps

**Note:** The content appears to be a step-by-step guide for using the Integrax framework, rather than a simple code snippet or module. It includes interactive elements and diagnostics sections, suggesting it is an auxiliary file used in conjunction with other components of the framework.

### `bundle_intergrax_engine.py`

**Description:** This is the main entry point of the Intergrax framework's engine, responsible for generating a bundle file containing all Python modules from a specified package directory.

**Domain:** Engine / Bundling

**Key Responsibilities:**

*   Collects and processes Python files within a package directory
*   Computes metadata (e.g., SHA256 hashes, line counts) for each file
*   Builds an LLM instruction header to be included at the top of the bundle file
*   Assembles a module map and index for dynamic navigation within the bundle
*   Generates the final bundle file, which can be used by large language models (LLMs)
*   Supports filtering symbols from Python code for inclusion in the bundle

### `generate_project_overview.py`

Description: This module provides automatic project structure documentation generation for the Intergrax framework.

Domain: Documentation Generation

Key Responsibilities:
- Recursively scan the project directory.
- Collect all relevant source files (Python, Jupyter Notebooks, configurable).
- Generate a structured summary for each file via LLM adapter.
- Build and write the Markdown document to disk.
- Output the document with clear purpose, domain, and responsibilities.

### `intergrax\__init__.py`

DESCRIPTION:
This is the main entry point of the Intergrax framework, responsible for initializing and setting up various components.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes modules and sub-packages within the Intergrax package
• Sets up global configuration and logging facilities
• Performs any necessary imports or setup of framework dependencies

### `intergrax\chains\__init__.py`

Description: The __init__.py file serves as the entry point for the chains module, defining its interface and importing necessary components.

Domain: Chains/Graphs

Key Responsibilities:
- Defines the chains module's public API
- Imports and initializes core graph-related functionality

### `intergrax\chains\langchain_qa_chain.py`

Description: This module provides a flexible QA chain (RAG → [rerank] → prompt → LLM) using the LangChain framework, allowing for hooks to modify data at various stages.

Domain: RAG logic

Key Responsibilities:
- Builds a QA chain with reranking and hooks
- Provides default prompt builder and configuration options
- Allows for customization of the chain through hooks and configuration
- Supports asynchronous invocation
- Returns extended output including sources, prompt, and raw hits

### `intergrax\chat_agent.py`

**Description:** The chat_agent.py file implements a core component of the Intergrax framework, providing an interface for routing user queries to either RAG (Reasoning and Generation) components, tools, or general LLM (Large Language Model) operations.

**Domain:** Chat Agents / Routing Logic

**Key Responsibilities:**

* Provides a unified API for routing user queries
* Supports three main routes: RAG, tools, and general LLM operations
* Allows for manual override of the routing decision using specific flags
* Uses an LLM adapter to determine the route based on descriptions and tool availability
* Handles memory management and streaming capabilities for efficient processing
* Returns a stable result format containing answer, output structure, tool traces, sources, summary, messages, stats, route, and rag component information.

### `intergrax\globals\__init__.py`

DESCRIPTION: This module initializes the global state for the Intergrax framework, setting up essential configurations and dependencies.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
• Initializes global settings and constants
• Configures logging and error handling mechanisms
• Establishes connections to external services or databases (if applicable)
• Registers core modules and components with the framework

### `intergrax\globals\settings.py`

**Description:** This module provides a centralized storage for framework-wide configuration, including defaults for language, locale, timezone, LLM models, and session memory/consolidation intervals.

**Domain:** Configuration

**Key Responsibilities:**
* Stores global settings as a dataclass instance
* Exposes configurable fields with default values set via environment variables
* Provides fallbacks for runtime components in case of missing user/org-specific configurations
* Maintains singleton-style global settings instance used across the framework

### `intergrax\llm\__init__.py`

DESCRIPTION: The `__init__.py` file serves as the entry point for the LLM adapters in Intergrax, initializing and setting up various language model-related components.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Initializes the adapter framework
- Sets up default configurations and constants
- Imports and makes available various adapter classes

### `intergrax\llm\messages.py`

**Description:** This module provides classes and functions for working with chat messages in the Integrax framework, including utility classes for attachments and a custom reducer function for LangGraph state.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a `ChatMessage` class representing universal chat messages compatible with the OpenAI Responses API
* Defines an `AttachmentRef` class for lightweight references to message or session attachments
* Includes a custom reducer function `append_chat_messages` for merging new chat messages into existing state in LangGraph

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point and package initializer for LLM adapters in Intergrax.

Domain: LLM adapters

Key Responsibilities:
- Initializes the LLM adapter package.
- Defines exports and imports for related modules.

### `intergrax\llm_adapters\aws_bedrock_adapter.py`

**FILE PATH:** intergrax\llm_adapters\aws_bedrock_adapter.py

**Description:** This module provides codecs for interacting with Amazon Bedrock models, including Anthropic Claude Messages API, Meta Llama native completion format, and Mistral native format.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides codecs for different model families (Anthropic, Meta, Mistral) to build native requests and extract text from responses
* Supports streaming and non-streaming interactions with Bedrock models
* Includes specific implementations for Anthropic Claude Messages API, Meta Llama native completion format, and Mistral native format

**Note:** This file appears to be a part of the Intergrax framework's LLM adapters module, providing proprietary and confidential information. It is assumed that this code is experimental or auxiliary in nature, as there are no explicit indications otherwise. However, it seems to be well-documented with high-quality documentation for each codec implementation.

### `intergrax\llm_adapters\azure_openai_adapter.py`

**Description:** This file provides an implementation of the Azure OpenAI adapter for the Intergrax framework, enabling integration with the Azure OpenAI service.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides a wrapper around the official OpenAI Python SDK (AzureOpenAI) for interaction with the Azure OpenAI service
* Exposes methods for generating and streaming messages using the adapter
* Implements context window estimation and token counting for input and output
* Utilizes framework-wide settings for configuration
* Handles exceptions and error handling through usage tracking and logging

### `intergrax\llm_adapters\claude_adapter.py`

**Description:** This module provides a Claude (Anthropic) adapter for the Intergrax framework, enabling interaction with the Anthropic LLM model.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes the Claude adapter with optional client and model parameters
* Provides methods to generate messages (`generate_messages`) and stream messages (`stream_messages`)
* Estimates context window tokens based on the used model
* Maps chat messages to the format required by the Anthropic Messages API

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a Gemini chat adapter for the Intergrax framework, enabling interaction with the Gemini Large Language Model.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides an implementation of `LLMAdapter` for the Gemini model
* Supports generating messages using the `generate_messages` method
* Supports streaming responses from the model using the `stream_messages` method
* Estimates context window tokens based on the configured model
* Manages client and model configuration through `__init__`
* Offers internal methods for estimating tokens, splitting system text, building generation configs, and mapping history/contents

### `intergrax\llm_adapters\llm_adapter.py`

**Description:** This is the base interface class for LLM (Large Language Model) adapters in the Integrax framework. It defines a set of abstract methods and attributes that must be implemented by concrete adapter classes.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines a universal runtime interface for LLM adapters
* Provides shared base implementation for token counting
* Must be implemented by concrete adapter classes
* Abstract methods:
	+ `generate_messages(...)`
	+ `context_window_tokens`
* Optional methods (default to NotImplemented/False):
	+ `streaming`
	+ `tools`
	+ `structured output`

**Notes:** This class is a part of the Integrax framework, and its main purpose is to provide a common interface for various LLM adapters. The class is designed to be extensible and flexible, allowing different adapters to implement their own specific logic while maintaining a consistent interface.

### `intergrax\llm_adapters\llm_provider.py`

**Description:** This module defines a set of enumerations representing supported Large Language Model (LLM) providers for the Integrax framework.
**Domain:** LLM adapters
**Key Responsibilities:**
- Defines an enumeration class `LLMProvider` for various LLM service providers.
- Provides string-based representations for each provider type.

### `intergrax\llm_adapters\llm_provider_registry.py`

**Description:** This module provides a registry for LLM (Large Language Model) adapters, allowing for easy registration and creation of different adapter instances.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides a registry for LLM adapters
* Allows registering new adapters using the `register` method
* Enables creating adapter instances using the `create` method
* Normalizes provider names to ensure consistency across registrations
* Includes default registrations for various popular LLM providers

### `intergrax\llm_adapters\llm_usage_track.py`

**Description:** This module provides a usage tracker for LLM (Large Language Model) adapters, aggregating statistics across multiple adapters used during a single runtime run.

**Domain:** LLM adapters

**Key Responsibilities:**

* Aggregates usage statistics for multiple LLM adapters
* Provides a summary of total usage and per-label usage
* Allows registration and unregistration of adapters
* Builds a report with detailed statistics and aggregation by provider/model

### `intergrax\llm_adapters\mistral_adapter.py`

**Description:** This module provides a LLM (Large Language Model) adapter for the Mistral platform, enabling interaction with Mistral's API to generate and stream chat responses.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a MistralChatAdapter class that implements the LLMAdapter protocol
* Supports generating and streaming chat messages using the Mistral client
* Utilizes framework-wide defaults for configuration settings
* Estimates context window tokens for input and output text
* Merges system and conversation messages into a single payload for API requests

Note: This file appears to be part of a larger framework, with other modules and dependencies referenced throughout. The code is well-structured and follows standard practices, indicating that it is likely not experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides an adapter for Ollama models used via LangChain's ChatModel interface. It allows the agent to interact with various Ollama models and estimates context windows based on model names.

**Domain:** LLM adapters

**Key Responsibilities:**
- Estimates context windows based on Ollama model names
- Provides a unified interface for interacting with different Ollama models via LangChain's ChatModel interface
- Supports both single-shot and streaming interactions with the model
- Handles error cases and estimates input/output tokens accordingly

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module implements a chat adapter for OpenAI's Responses API, providing utilities to interact with their models using the Responses format.

**Domain:** LLM adapters

**Key Responsibilities:**

* Establishes an OpenAI client and model instance
* Estimates context window tokens for OpenAI models based on model name
* Converts Chat Completion style messages into Responses API input items
* Collects output text from Responses API results
* Provides streaming completion functionality using Responses API
* Implements single-shot completion (non-streaming) using Responses API

**Notes:** This file appears to be production-ready, with a comprehensive set of methods and utilities for interacting with OpenAI's Responses API. The code is well-structured and follows best practices.

### `intergrax\logging.py`

Here is the documentation for the `intergrax\logging.py` file:

**Description:** This module sets up a basic logging configuration for the Integrax framework.

**Domain:** Logging/Configuration

**Key Responsibilities:**
- Configures a global logger with basic settings (level, format).
- Enables logging of INFO and higher-level messages.
- Forces overwriting any previous logging configurations.

### `intergrax\memory\__init__.py`

DESCRIPTION: 
This module serves as the entry point for Intergrax's memory management functionality, handling initialization and setup.

DOMAIN: Memory Management

KEY RESPONSIBILITIES:
• Initializes and configures memory-related components
• Sets up necessary data structures for efficient memory usage
• Provides interface for external access to memory operations

### `intergrax\memory\conversational_memory.py`

**Description:** This module implements an in-memory conversation history component for storing and managing chat messages.

**Domain:** Conversational Memory

**Key Responsibilities:**

* Keep messages in RAM
* Provide a simple API to add, extend, read, and clear messages
* Optionally enforce a max_messages limit
* Prepare messages for different model backends (get_for_model)
* Delegated persistence is handled by dedicated memory store providers

### `intergrax\memory\conversational_store.py`

**Description:** This module defines the abstract interface for persistent storage of conversational memory, decoupling runtime logic from storage backends.
**Domain:** Conversational Memory Management
**Key Responsibilities:**

* Provide a protocol for persistent storage interfaces to interact with `ConversationalMemory` aggregates
* Ensure deterministic persistence, idempotent write operations, and no modification of semantic meaning
* Support async-safe interaction with conversational memory
* Define methods for loading, saving, appending messages, and deleting sessions

**Note:** This file appears to be a core part of the Intergrax framework's design, defining the contract for storage backends.

### `intergrax\memory\stores\__init__.py`

Description: This module initializes and configures the memory stores within Intergrax, providing a standardized interface for storing and retrieving data.

Domain: Memory management

Key Responsibilities:
* Initializes all memory store instances
* Exposes a unified API for interacting with memory stores
* Configures store-specific settings (e.g., capacity, persistence)
* Handles startup and shutdown of memory stores

### `intergrax\memory\stores\in_memory_conversational_store.py`

**Description:** This module implements an in-memory conversational store for the Intergrax framework, suitable for local development, prototyping, unit and integration testing, or environments where persistence is not required.

**Domain:** Memory/Storage (Conversational Store)

**Key Responsibilities:**
- Provides an in-memory implementation of `ConversationalMemoryStore`
- Allows loading and saving conversation history using defensive copying
- Enables appending messages to the conversational memory and persisting the updated state
- Supports deleting persistent session data without error handling
- Offers a diagnostic helper for listing active persisted sessions

### `intergrax\memory\stores\in_memory_user_profile_store.py`

Description: This module provides an in-memory user profile store implementation for the Integrax framework.

Domain: Memory Stores

Key Responsibilities:
- Provides an in-memory storage for user profiles.
- Supports retrieval, creation, update, and deletion of user profiles.
- Offers a basic data structure for storing and retrieving user information.

### `intergrax\memory\user_profile_manager.py`

**Description:** This module provides a high-level facade for working with user profiles, hiding direct interaction with the underlying UserProfileStore. It offers convenient methods to load or create a UserProfile, persist profile changes, manage long-term user memory entries, and manage system-level instructions derived from the profile.

**Domain:** Memory Management (User Profiles)

**Key Responsibilities:**

* Provide convenient methods for working with user profiles:
	+ Load or create a UserProfile
	+ Persist profile changes
	+ Manage long-term user memory entries
	+ Manage system-level instructions derived from the profile
* Hide direct interaction with the underlying UserProfileStore
* Optionally enable long-term RAG (Retrieval-Augmented Generation) dependencies for advanced vector-based retrieval over user's long-term memory entries

### `intergrax\memory\user_profile_memory.py`

Description: This module provides domain models for user profiles and prompt bundles, abstracting away storage or engine logic.

Domain: User Profile Management / LLM Prompt Generation

Key Responsibilities:
- Define core domain models for user/org profile and prompt bundles.
- Represent identities, preferences, and memory entries in a language-independent manner.
- Provide dataclasses for UserProfileMemoryEntry (long-term facts and notes), UserIdentity (high-level description of who the user is), UserPreferences (stable user preferences influencing runtime behavior), and UserProfile (canonical aggregate).
- Include metadata and unit-of-work flags to facilitate storage and retrieval.

### `intergrax\memory\user_profile_store.py`

**Description:** This module defines a protocol (interface) for persisting and retrieving user profiles in a scalable and abstract manner. The interface decouples storage concerns from application logic, making it reusable across different backend systems.

**Domain:** Memory Management

**Key Responsibilities:**
* Provide an interface for loading user profiles
* Implement sane defaults for new users
* Hide backend-specific storage concerns (JSON files, SQL DB, etc.)
* Allow for asynchronous profile retrieval and persistence operations

### `intergrax\multimedia\__init__.py`

DESCRIPTION:
This module serves as the entry point for multimedia-related functionality within the Intergrax framework.

DOMAIN: Multimedia Utilities

KEY RESPONSIBILITIES:
- Imports and sets up necessary dependencies for multimedia processing.
- Defines interface for integrating external multimedia libraries or services.

### `intergrax\multimedia\audio_loader.py`

**FILE PATH:** intergrax\multimedia\audio_loader.py

**Description:** This module provides functionality for downloading audio from YouTube URLs and translating the audio into a specific language using Whisper.

**Domain:** multimedia processing

**Key Responsibilities:**
- Downloads audio from specified YouTube URL with customizable output format.
- Translates downloaded audio into desired language using Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module is responsible for loading and processing multimedia content, specifically images, within the Intergrax framework.
Domain: Multimedia Processing

Key Responsibilities:
- Loads images from file paths
- Utilizes ollama library to process images with LLaMA-based models
- Transcribes image prompts into text through interaction with a large language model (LLM)

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, including audio, images, and videos with customizable playback options.

**Domain:** Multimedia Display

**Key Responsibilities:**

* Displays audio files with optional autoplay and label support
* Supports image display with error handling for invalid file paths
* Serves video files by copying them to a temporary directory and returning a served path
* Plays videos with jump-to-specific-timestamp functionality, including poster frame display and customizable playback settings

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides utilities for video processing within the Intergrax framework, including downloading videos from YouTube, transcribing audio to text, and extracting frames with associated metadata.

**Domain:** Multimedia

**Key Responsibilities:**
- Downloading videos from YouTube using yt_dlp library
- Transcribing video audio to text using Whisper model and webvtt library
- Extracting frames with associated metadata from a video file
- Resizing images to maintain aspect ratio

### `intergrax\openai\__init__.py`

DESCRIPTION: This module serves as the entry point for the Intergrax framework, importing and initializing other modules to enable functionality.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
* Initializes OpenAI adapter
* Imports necessary sub-modules for functionality

### `intergrax\openai\rag\__init__.py`

DESCRIPTION:
This module is the entry point for OpenAI-based Retrieval-Augmented Generation (RAG) logic within Intergrax.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes and sets up OpenAI API clients
- Configures RAG-specific settings and defaults
- Provides a public interface for interacting with the RAG module

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module implements a RagOpenAI class for integrating OpenAI's RAG (Retrieval-Augmented Generation) capabilities with the Intergrax framework. It provides functionality for vector store management, uploading files to the vector store, and generating prompts for the RAG model.

**Domain:** LLM adapters

**Key Responsibilities:**

*   Initialize the RagOpenAI class with an OpenAI client and a vector store ID.
*   Generate a prompt for the RAG model based on predefined rules.
*   Ensure that a specified vector store exists in the OpenAI system.
*   Clear all files loaded into a vector store.
*   Upload a folder containing files to a vector store using specific file patterns.

### `intergrax\rag\__init__.py`

Description: This is the entry point for the RAG (Relevant Attention Guided) logic in Intergrax.

Domain: RAG logic

Key Responsibilities:
• Initializes the RAG module and sets up necessary dependencies.
• Registers default RAG components.
• Exposes the main RAG interface for usage throughout the framework.

### `intergrax\rag\documents_loader.py`

**Description:** This module, `DocumentsLoader`, is responsible for robustly loading and processing various types of documents (text, images, videos) with metadata injection and safety guards. It provides extensibility through a default extensions map and the ability to customize loading behaviors.

**Domain:** Document loaders/RAG logic

**Key Responsibilities:**

- Load text files (.txt, .md, .markdown) using autodetected encoding
- Load DOCX files (.docx) into paragraphs or fulltext mode
- Load PDFs (.pdf) with optional OCR for extraction of text and layout data
- Load Excel spreadsheets (.xlsx, .xls) in various modes (rows, sheets, markdown)
- Load CSV files (.csv, .tsv) with customizable encoding and delimiter
- Load images (.jpg, .jpeg, .png, etc.) for text extraction via OCR or image captioning
- Load videos (.mp4, etc.) for transcription, metadata extraction, and frame processing

### `intergrax\rag\documents_splitter.py`

**Description:** This module provides a high-quality text splitter for RAG pipelines, implementing the 'semantic atom' policy to ensure stable chunk IDs and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**

* Splits documents into chunks based on semantic atoms (paragraphs, rows, pages, or images)
* Assigns stable, human-readable chunk IDs using available anchors
* Provides rich metadata for each chunk, including parent ID, source name, and source path
* Optionally merges tiny tails and applies a maximum chunk cap
* Supports custom metadata functions and merging of metadata from previous chunks

### `intergrax\rag\dual_index_builder.py`

**Description:** This module is responsible for building dual vector indexes (CHUNKS and TOC) using the documents provided. It utilizes the embedding manager to compute embeddings for each document and inserts them into the respective collections.

**Domain:** RAG logic / Indexing

**Key Responsibilities:**

* Builds two vector indexes: CHUNKS (all chunks/documents after splitting) and TOC (only DOCX headings within levels [toc_min_level, toc_max_level])
* Utilizes embedding manager to compute embeddings for each document
* Inserts documents into the respective collections (CHUNKS and TOC)
* Supports batch processing with adjustable batch size
* Allows for filtering of documents based on pre-defined criteria (prefilter function)
* Provides logging and verbosity options for debugging purposes

### `intergrax\rag\dual_retriever.py`

**Description:** The dual retriever module is responsible for retrieving relevant chunks and TOC (Table of Contents) sections from the Intergrax vector store. It first queries the TOC to identify relevant sections and then fetches local chunks from the same section/source.

**Domain:** RAG logic

**Key Responsibilities:**

*   Dual retrieval process:
    *   Query TOC for relevant sections
    *   Fetch local chunks from the same section/source
*   Normalization of hits from vector store query results
*   Merging user filters with parent ID constraint
*   Expansion of local chunks based on matched TOC sections

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for integrating different types of embeddings (HuggingFace, Ollama, OpenAI) into the Intergrax framework.

**Domain:** RAG logic / Embedding management

**Key Responsibilities:**

* Provides a unified interface for loading and managing different types of embeddings
* Supports HuggingFace (SentenceTransformer), Ollama, and OpenAI embeddings
* Offers features such as provider switch, reasonable defaults, batch/single text embedding, L2 normalization, cosine similarity utilities, and top-K retrieval
* Robust logging and shape validation
* Light retry for transient errors

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides a RagAnswerer class responsible for retrieving relevant context fragments, re-ranking them if necessary, and generating answers using a Large Language Model (LLM).

**Domain:** RAG (Relevant Answer Generation)

**Key Responsibilities:**

* Retrieval of relevant context fragments from the retriever
* Optional re-ranking of retrieved hits using a reranker
* Building of context text and citations
* Construction of system and user messages
* Generation of answers using an LLM
* Optionally, generating structured output in addition to text answer

### `intergrax\rag\rag_retriever.py`

**Description:** This module implements a scalable, provider-agnostic RAG (Relevance-Aware Generator) retriever for the Intergrax framework. It normalizes query vectors and filters, handles batch retrieval, and provides an optional reranker hook.

**Domain:** RAG logic

**Key Responsibilities:**
- Normalizes `where` filters for Chroma
- Normalizes query vector shape (1D/2D → [[D]])
- Unified similarity scoring
- Deduplication by ID + per-parent result limiting (diversification)
- Optional MMR diversification when embeddings are returned
- Batch retrieval for multiple queries
- Optional reranker hook

### `intergrax\rag\re_ranker.py`

**Description:** 
This module provides a re-ranking functionality for candidate chunks. It embeds texts in batches using an embedding manager and calculates cosine similarities to rank candidates.

**Domain:** RAG (Retriever-Augmented Generator) logic

**Key Responsibilities:**
- Embeds texts in batches using the `EmbeddingManager`
- Calculates cosine similarity between query and candidates
- Re-ranks candidates based on their similarity scores
- Supports optional score fusion with original retriever similarity
- Preserves schema of hits, adding 'rerank_score', 'fusion_score', and 'rank_reranked' fields

### `intergrax\rag\vectorstore_manager.py`

**Description:** The `vectorstore_manager.py` file provides a unified vector store manager supporting ChromaDB, Qdrant, and Pinecone. It enables initialization of target stores, upserting documents with embeddings, querying top-K similar vectors, counting vectors, and deleting by IDs.

**Domain:** Vector Stores / Embedding Management

**Key Responsibilities:**

* Initialize target vector store (ChromaDB, Qdrant, or Pinecone) based on configuration
* Upsert documents with embeddings (with batching)
* Query top-K most similar vectors using cosine, dot, or Euclidean similarity
* Count the number of vectors in the store
* Delete vectors by IDs

The file appears to be well-structured and follows best practices for documentation. The code is readable, and the use of type hints and docstrings makes it easy to understand the purpose and behavior of each function.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a windowed answerer on top of the base Answerer, allowing for incremental retrieval and synthesis of answers.

**Domain:** RAG Logic

**Key Responsibilities:**

*   Initializes the WindowedAnswerer with an underlying RagAnswerer, retriever, and optional verbose mode.
*   Builds context locally from hits in a window and sanitizes text data.
*   Constructs messages with memory-awareness, using either the standard or memory-optimized approach.
*   Asks for answers in a windowed fashion, performing broad retrieval, partial synthesis, and final reduction.
*   Collects sources and syntheses a final answer by combining partial results.

### `intergrax\runtime\__init__.py`

DESCRIPTION: The __init__.py file is the entry point for the Intergrax runtime, responsible for initializing and setting up the environment for other modules.

DOMAIN: Runtime Initialization

KEY RESPONSIBILITIES:
• Initializes the runtime environment.
• Imports and sets up necessary dependencies.
• Defines the application's entry points.

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: This module initializes the drop-in knowledge mode for the Intergrax runtime, enabling external knowledge to be easily integrated into the system.

Domain: RAG logic

Key Responsibilities:
- Initializes the drop-in knowledge mode
- Configures external knowledge integration
- Sets up necessary components for seamless knowledge incorporation

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This module defines the configuration for the Intergrax framework's Drop-In Knowledge Runtime, specifying settings for LLMs, RAG, web search, tools, and memory management.

**Domain:** Configuration / Utilities

**Key Responsibilities:**
- Defines configuration options for LLM adapters
- Specifies RAG (vectorstore-based retrieval) settings
- Enables or disables web search as an additional context source
- Controls the use of a tools agent for function/tool calling
- Configures memory management and validation features
- Provides an interface for arbitrary metadata instrumentation

### `intergrax\runtime\drop_in_knowledge_mode\context\__init__.py`

**intergrax/runtime/drop_in_knowledge_mode/context/__init__.py**

Description: This module initializes the context for drop-in knowledge mode, providing a centralized hub for accessing and managing contextual information.

Domain: RAG logic

Key Responsibilities:
- Initializes the Context object with default attributes
- Sets up data structures for storing and retrieving contextual data
- Defines interfaces for accessing and updating contextually relevant information

### `intergrax\runtime\drop_in_knowledge_mode\context\context_builder.py`

Description: This module provides context building functionality for Drop-In Knowledge Mode, responsible for deciding whether to use RAG and retrieving relevant document chunks from a vector store.

Domain: Context Builder

Key Responsibilities:
- Decide whether to use RAG for a given (session, request).
- Retrieve relevant document chunks from the vector store using session/user/tenant/workspace metadata.
- Provide a RAG-specific system prompt, list of retrieved chunks, and debug metadata for observability.

### `intergrax\runtime\drop_in_knowledge_mode\context\engine_history_layer.py`

**Description:** This module encapsulates the logic for loading, preprocessing, and compressing conversation history in Intergrax's drop-in knowledge mode.

**Domain:** RAG (Retrieve-Answer-Generate) logic

**Key Responsibilities:**

* Loading raw conversation history from SessionStore
* Computing token usage for the raw history
* Applying history compression strategies based on per-request settings
* Updating RuntimeState with base_history and debug information

### `intergrax\runtime\drop_in_knowledge_mode\engine\__init__.py`

Description: The __init__.py file initializes the drop-in knowledge mode engine, providing a central entry point for the module's functionality.

Domain: RAG logic

Key Responsibilities:
- Initializes the engine and sets up necessary components
- Defines and exports the engine's interface and dependencies 
- Possibly includes setup or configuration code for the knowledge engine

### `intergrax\runtime\drop_in_knowledge_mode\engine\runtime.py`

**Description:** This module defines the core runtime engine for Intergrax's Drop-In Knowledge Mode, responsible for handling chat sessions and generating responses using LLM adapters and other components.

**Domain:** RAG logic and runtime engine

**Key Responsibilities:**

* Defines the `DropInKnowledgeRuntime` class, which:
	+ Loads or creates chat sessions
	+ Appends user messages to the session
	+ Builds a conversation history for the LLM
	+ Augments context with RAG, web search, and tools
	+ Produces a `RuntimeAnswer` object as a high-level response
* Manages state through the `RuntimeState` class
* Utilizes various components, including LLM adapters, context builders, and ingestion services

**Notes:** This file appears to be part of Intergrax's core functionality and is not experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\engine\runtime_state.py`

**Description:** This module defines a mutable state object, `RuntimeState`, used to aggregate and pass various metadata, results, and flags throughout the runtime pipeline of Intergrax's drop-in knowledge mode.

**Domain:** Runtime State Management (drop-in knowledge mode)

**Key Responsibilities:**

- Aggregates request and session metadata.
- Collects ingestion results and conversation history.
- Tracks usage of RAG, websearch, tools, memory, and other subsystems.
- Stores tools traces and agent answers for observability and diagnostics.
- Maintains flags indicating the availability of certain capabilities (RAG, user LTM, attachments, websearch, tools).
- Provides a mutable state object to facilitate data exchange between runtime pipeline components.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\__init__.py`

Description: The __init__.py file initializes and configures the knowledge ingestion mechanism for drop-in mode in Intergrax's runtime.

Domain: Data Ingestion

Key Responsibilities:
- Initializes the knowledge ingestion process for drop-in mode
- Configures relevant settings and modules for data ingestion

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\attachments.py`

**Description:** This module provides utilities for resolving attachments in Drop-In Knowledge Mode, allowing the RAG pipeline to consume them from various storage systems.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Defines the `AttachmentResolver` protocol, which abstracts the process of turning an `AttachmentRef` into a local `Path`.
* Provides a minimal implementation (`FileSystemAttachmentResolver`) for handling local filesystem-based URIs.
* Decouples attachment storage from consumption in the RAG pipeline.
* Offers a way to download attachments from object storage (S3, GCS, etc.) or fetch them from a database and materialize as temporary files.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion\ingestion_service.py`

**Description:** This module provides a high-level service for ingesting attachments in Drop-In Knowledge Mode, reusing Intergrax RAG components.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Resolves AttachmentRef objects into filesystem Paths using AttachmentResolver
* Loads documents using IntergraxDocumentsLoader.load_document(...)
* Splits documents into chunks using IntergraxDocumentsSplitter.split_documents(...)
* Embeds chunks via IntergraxEmbeddingManager
* Stores vectors via IntergraxVectorstoreManager
* Returns a structured IngestionResult per attachment

Note: This module appears to be part of the Intergrax framework, specifically designed for Drop-In Knowledge Mode. It reuses existing RAG components and provides a clean, runtime-oriented API.

### `intergrax\runtime\drop_in_knowledge_mode\planning\__init__.py`

Description: This module initializes and configures the planning component for drop-in knowledge mode in Intergrax.

Domain: Planning

Key Responsibilities:
• Initializes planning component for drop-in knowledge mode
• Configures planning settings and parameters
• Sets up dependencies and interfaces for planning functionality

### `intergrax\runtime\drop_in_knowledge_mode\planning\engine_plan_models.py`

**Description:** This module defines data structures and functionality for planning in the Intergrax framework's drop-in knowledge mode. It provides a schema for engine plans, which outline the next steps to take based on user intent.

**Domain:** RAG logic (Retrieval-Augmented Generation)

**Key Responsibilities:**

* Defines typed plan schema with `EnginePlan` and `PlannerPromptConfig` dataclasses
* Specifies constraints and policies for planning, including:
	+ Intent definitions and field constraints
	+ User long-term memory policy
	+ Websearch vs Tools policy
	+ Clarify policy
* Provides methods for printing engine plans in a human-readable format

**Note:** This file appears to be part of the main Intergrax framework codebase, indicating it is likely stable and production-ready.

### `intergrax\runtime\drop_in_knowledge_mode\planning\engine_planner.py`

**Description:** This module provides a planner engine for Intergrax's drop-in knowledge mode, utilizing LLM-based planning to generate typed EnginePlans.

**Domain:** Planning (LLM-based)

**Key Responsibilities:**

*   Provides an EnginePlanner class that uses an LLM adapter to plan and output typed EnginePlans.
*   Validates the generated plans against capabilities to ensure they are feasible given the current runtime state.
*   Builds planner messages with minimal, low-variance input and strict JSON schema for output.
*   Utilizes capabilities as hard constraints (runtime will clamp again as a safety net).
*   Enforces explicit rules for next_step selection to reduce ambiguity.

**Note:** The code appears well-structured and comprehensive, with clear explanations of the planning process and constraint validation.

### `intergrax\runtime\drop_in_knowledge_mode\planning\plan_builder_helper.py`

**Description:** This module provides a helper function for building plans in drop-in knowledge mode, responsible for initializing the planning process with pre-configured settings.

**Domain:** Drop-in Knowledge Mode Planning Utilities

**Key Responsibilities:**

* Initializes the planning process by creating an `EnginePlanner` instance
* Creates a `RuntimeRequest` object from input parameters
* Sets up the `RuntimeState` object, including configuring LLM usage tracking and setting various capabilities based on configuration settings
* Calls the `plan` method of the `EnginePlanner` instance to generate a plan

### `intergrax\runtime\drop_in_knowledge_mode\planning\step_planner.py`

**Description:** This module provides a deterministic planner for building an ExecutionPlan from user input and engine hints, with the ability to handle different planning modes.

**Domain:** Planning (within Drop-in Knowledge Mode)

**Key Responsibilities:**

* Provides a `StepPlanner` class that takes in configuration and engine plan data
* Builds an ExecutionPlan based on user message, engine hints, and planning mode
* Handles different planning modes, including static and dynamic plans
* Ensures deterministic sequential execution for pre-steps
* Supports clarification with question functionality
* Integrates with other modules, such as web search and tools

**Notes:** This file appears to be a core component of the Intergrax framework, specifically designed for planning within Drop-in Knowledge Mode. The code is well-structured and follows best practices, indicating that it is not experimental or auxiliary. However, some parts of the code may require further investigation to understand their specific implementation details.

### `intergrax\runtime\drop_in_knowledge_mode\planning\stepplan_models.py`

**Description:** This module defines a data-driven approach for planning and executing steps within the Intergrax framework. It includes models for various step types, budgets, stop conditions, and validation.

**Domain:** Planning / Step Management

**Key Responsibilities:**

* Define enums for different step actions (e.g., ASK_CLARIFYING_QUESTION, USE_USER_LONGTERM_MEMORY_SEARCH)
* Create models for failure policies, budgets, stop conditions, and verification criteria
* Specify parameters for each step action using dedicated models (e.g., AskClarifyingParams, LtmSearchParams)
* Establish a mapping between step actions and their corresponding parameter models (_ACTION_TO_PARAMS_MODEL)
* Define the ExecutionStep model, which encapsulates the details of a single execution step

**Note:** This file appears to be a core component of the Intergrax framework's planning mechanism.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\__init__.py`

Description: This module is responsible for defining and initializing the prompts used in drop-in knowledge mode.

Domain: LLM adapters

Key Responsibilities:
* Defines and initializes prompt configurations for drop-in knowledge mode
* Provides a way to customize prompts based on user requirements
* Handles internal state management related to prompts

### `intergrax\runtime\drop_in_knowledge_mode\prompts\history_prompt_builder.py`

Description: This module provides a builder for constructing history-related prompts in Drop-In Knowledge Mode, allowing for customization of summarization strategies.

Domain: LLM adapters

Key Responsibilities:
- Provides a strategy interface (HistorySummaryPromptBuilder) for building the history-summary-related part of the prompt.
- Offers a default implementation (DefaultHistorySummaryPromptBuilder) that generates a generic system prompt for summarizing older conversation turns.
- Allows for customization through a custom implementation of the HistorySummaryPromptBuilder protocol.

### `intergrax\runtime\drop_in_knowledge_mode\prompts\rag_prompt_builder.py`

**Description:** This module provides a strategy for building the RAG-related part of the prompt in Drop-In Knowledge Mode, allowing for customization of prompt elements.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Provides a `RagPromptBuilder` protocol for custom implementation
* Offers a default `DefaultRagPromptBuilder` class that injects retrieved chunks into system-level context messages
* Enables global system instructions to be owned by the runtime
* Supports formatting of retrieved chunks into a compact, model-friendly text block

### `intergrax\runtime\drop_in_knowledge_mode\prompts\user_longterm_memory_prompt_builder.py`

**Description:** This module is responsible for building prompt messages that inject retrieved user long-term memory into the Large Language Model (LLM) context.

**Domain:** LLM adapters

**Key Responsibilities:**

* Building prompt messages to inject retrieved user long-term memory into the LLM context
* Constructing deterministic and compact prompts with traceable IDs
* Filtering out deleted entries and limiting the number of entries per prompt
* Handling character limits for prompt text
* Returning a `UserLongTermMemoryPromptBundle` object containing the built prompt message

### `intergrax\runtime\drop_in_knowledge_mode\prompts\websearch_prompt_builder.py`

**Description:** This module is responsible for building web search prompts within the Intergrax framework's Drop-In Knowledge Mode.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a `WebSearchPromptBundle` dataclass to containerize prompt elements related to web search.
* Defines a `WebSearchPromptBuilder` protocol for custom implementations of web search prompt building strategies.
* Includes a default implementation, `DefaultWebSearchPromptBuilder`, which delegates to the websearch module context generator and provides debug information.

The file appears to be part of the Intergrax framework's core functionality and is not marked as experimental or auxiliary.

### `intergrax\runtime\drop_in_knowledge_mode\responses\__init__.py`

Description: This module defines a drop-in knowledge mode for responses, allowing for easy integration of external knowledge sources into the Intergrax framework.

Domain: RAG (Reinforcement-aided generation) logic

Key Responsibilities:
* Defines the drop_in_knowledge_mode response type
* Enables the use of external knowledge sources in response generation
* Provides a flexible interface for integrating various knowledge sources

### `intergrax\runtime\drop_in_knowledge_mode\responses\response_schema.py`

**Description:** This module defines dataclasses and enums that represent the structure of request and response data exchanged between applications and the Drop-In Knowledge Mode runtime. It provides a high-level contract for applications to interact with the runtime, exposing citations, routing information, tool calls, and basic statistics.

**Domain:** RAG logic / Data exchange

**Key Responsibilities:**

* Define dataclasses for request (RuntimeRequest) and response (RuntimeAnswer) data structures
* Expose citation references (Citation), routing information (RouteInfo), tool call summaries (ToolCallInfo), and basic statistics (RuntimeStats)
* Provide an enum for history compression strategies (HistoryCompressionStrategy)

### `intergrax\runtime\drop_in_knowledge_mode\session\__init__.py`

Description: Initializes the knowledge session for drop-in knowledge mode, enabling external knowledge bases to be integrated seamlessly with the Intergrax runtime.

Domain: Knowledge Management

Key Responsibilities:
- Sets up the knowledge graph and database connections.
- Configures the knowledge retrieval mechanism.
- Establishes session-specific metadata.

### `intergrax\runtime\drop_in_knowledge_mode\session\chat_session.py`

**Description:** This file defines a `ChatSession` data class and related domain models for managing chat sessions in the Intergrax framework. The module provides utilities for session lifecycle management, persistence, and metadata handling.

**Domain:** LLM (Large Language Model) adapters / Session management

**Key Responsibilities:**

* Define the `ChatSession` data class with properties for session state, user, tenant, workspace, timestamps, attachments, and metadata.
* Provide domain-level status and close reason enumerations (`SessionStatus`, `SessionCloseReason`).
* Implement methods for updating session timestamps, marking sessions as closed, incrementing user turns, and accessing session metadata.
* Keep the session object I/O-free, relying on storage components or managers to persist changes.

### `intergrax\runtime\drop_in_knowledge_mode\session\in_memory_session_storage.py`

**Description:** This module implements an in-memory session storage for Intergrax's drop-in knowledge mode. It keeps chat session metadata and conversation history in an in-process dictionary, using a simple FIFO trimming policy.

**Domain:** Session Storage (in-memory)

**Key Responsibilities:**

* Store and retrieve chat sessions
* Maintain per-session conversation history using ConversationalMemory
* Apply a simple FIFO trimming policy via ConversationalMemory's max_messages setting
* Provide methods for appending messages to the conversation history and retrieving the ordered conversation history

Note that this implementation is not intended for production use in distributed or long-lived environments. It is suitable for development, testing, or single-process/single-node setups only.

### `intergrax\runtime\drop_in_knowledge_mode\session\session_manager.py`

**Description:** This module provides high-level management of chat sessions, including lifecycle operations, metadata retrieval, and integration with profile managers.

**Domain:** RAG logic (Relevance Aware Generation)

**Key Responsibilities:**

* Orchestrate session lifecycle on top of a SessionStorage backend
* Provide a stable API for the runtime engine (DropInKnowledgeRuntime)
* Integrate with user/organization profile managers to expose prompt-ready system instructions per session
* Optionally trigger long-term user memory consolidation for a session

The code appears to be well-structured and complete, without any obvious signs of being experimental, auxiliary, or legacy.

### `intergrax\runtime\drop_in_knowledge_mode\session\session_storage.py`

**Description:** This module provides a low-level storage interface for chat sessions and their conversation history, defining a protocol that can be implemented by various storage solutions.

**Domain:** Storage/Database

**Key Responsibilities:**

* Persist and load ChatSession objects
* Persist and load conversation history (ChatMessage sequences) for a given session
* Support CRUD operations on session metadata (create, get, update, delete)
* Provide methods to append messages to the conversation history and retrieve the entire history for a session

### `intergrax\runtime\organization\__init__.py`

Description: This module serves as the entry point and namespace for the Intergrax organization runtime, providing a standardized interface for accessing and configuring organizational components.

Domain: Organization Runtime

Key Responsibilities:
- Initializes and sets up the organization runtime environment
- Exposes configuration options and utility functions for organizational components
- Acts as a central hub for accessing various organizational modules and services

### `intergrax\runtime\organization\organization_profile.py`

**Description:** This module defines data classes representing organization profiles and their associated metadata within the Integrax framework. These profiles encapsulate stable identification, preferences, and system instructions that influence runtime behavior.

**Domain:** Organization Profiles/Configuration

**Key Responsibilities:**

* Define data class `OrganizationIdentity` for storing an organization's stable identification data.
* Introduce data class `OrganizationPreferences` to represent an organization-level set of preferences influencing runtime behavior.
* Implement data class `OrganizationProfile` as the single source of truth for an organization's long-term profile, incorporating identity, preferences, system instructions, and legacy summary fields.

**Notes:** The file appears well-structured, but there is mention of legacy fields (`summary_instructions`, `domain_summary`, etc.) which should be treated with caution or eventually deprecated.

### `intergrax\runtime\organization\organization_profile_instructions_service.py`

**Description:** This module provides a service for generating and updating organization-level system instructions using an LLMAdapter, based on OrganizationProfile.

**Domain:** RAG logic (Reasoning, Abstraction, Generation)

**Key Responsibilities:**

* Load OrganizationProfile via OrganizationProfileManager.
* Build an LLM prompt using identity, preferences, summaries, and memory entries.
* Call LLMAdapter.generate_messages() to obtain a compact, stable organization-level system prompt.
* Persist the result via OrganizationProfileManager.update_system_instructions().
* Generate and persist organization-level system instructions.
* Handle regeneration of instructions if present or forced.

### `intergrax\runtime\organization\organization_profile_manager.py`

Description: This module provides a high-level facade for working with organization profiles, hiding direct interaction with the underlying OrganizationProfileStore and providing convenient methods to load, persist, and manage system instructions.

Domain: Organization Profile Management

Key Responsibilities:
- Load an OrganizationProfile for a given organization_id
- Persist profile changes
- Resolve organization-level system instructions string for use in the runtime
- Hide direct interaction with the underlying OrganizationProfileStore
- Provide deterministic logic for building system instructions from organization profiles when explicit instructions are not set.

### `intergrax\runtime\organization\organization_profile_store.py`

**Description:** This module defines a protocol for persistent storage of organization profiles in the Integrax framework. It provides an interface for loading, saving, and deleting organization profiles while hiding backend-specific implementation details.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Providing a persistent storage interface for organization profiles
* Loading and saving `OrganizationProfile` aggregates
* Hiding backend-specific concerns (e.g., JSON files, SQL DB)
* Implementing sane defaults for new organizations
* Ensuring safe handling of unknown or non-existent profiles

### `intergrax\runtime\organization\stores\__init__.py`

Description: The __init__.py file serves as the entry point for the Intergrax organization stores, initializing and configuring store-related components.

Domain: Data Ingestion/Stores Management

Key Responsibilities:
- Initializes the store module
- Registers store-related components
- Configures data ingestion settings 
- Provides a single interface for accessing stores

### `intergrax\runtime\organization\stores\in_memory_organization_profile_store.py`

Description: This module provides an in-memory implementation of the OrganizationProfileStore interface.

Domain: Organization Profile Storage

Key Responsibilities:
- Provides a simple, volatile storage mechanism for organization profiles.
- Supports unit testing and local development use cases.
- Offers methods to retrieve, save, and delete organization profiles.
- Does NOT provide durability or cross-process sharing capabilities.

### `intergrax\runtime\user_profile\__init__.py`

Description: This module initializes the user profile functionality within the Intergrax runtime environment.
Domain: User Profile Management

Key Responsibilities:
* Initializes user profile settings and attributes
* Sets up default values for user preferences
* Establishes connections to relevant data storage systems for user data retrieval

### `intergrax\runtime\user_profile\session_memory_consolidation_service.py`

**Description:** This module is responsible for consolidating a single chat session into long-term user profile memory entries, optionally refreshing system instructions.

**Domain:** LLM adapters and user profile management

**Key Responsibilities:**

*   **Session Consolidation**: Converts a single chat session history into structured long-term memory entries for the user profile.
*   **LLM Interaction**: Asks the Language Model (LLM) to extract USER_FACT items, PREFERENCE items, and an optional SESSION_SUMMARY item from the conversation.
*   **Data Mapping**: Maps extracted data into UserProfileMemoryEntry objects.
*   **Persistence**: Persists memory entries through UserProfileManager (add_memory_entry).
*   **System Instructions Regeneration** (optional): Refreshes user-level system instructions after storing new memory entries.

Note: This file appears to be a part of the Intergrax framework and is not marked as experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\user_profile\user_profile_debug_service.py`

**Description:** This module provides a high-level service for building read-only, debug-oriented user profile snapshots. It aggregates data from the UserProfileManager and SessionManager to expose relevant information for administrative or developer use cases.

**Domain:** User Profile Debugging

**Key Responsibilities:**

* Build read-only user profile snapshots
* Aggregate data from UserProfileManager (identity, preferences, memory) and SessionManager (recent ChatSession metadata)
* Expose snapshot data through a public API method (`get_debug_snapshot`)
* Handle internal helpers for building identity and preference dictionaries, as well as recent memory entries and sessions

**Status:** This file appears to be a part of the main framework codebase, suggesting it is intended for production use.

### `intergrax\runtime\user_profile\user_profile_debug_snapshot.py`

**Description:** This module provides dataclasses and utility functions for creating an immutable snapshot of a user's profile state for debugging and observability.

**Domain:** User Profile Management

**Key Responsibilities:**

* Provide dataclasses for lightweight, debug-friendly views of ChatSession and UserProfileMemoryEntry.
* Offer static methods to build MemoryKind counters from memory entries and to create SessionDebugView and MemoryEntryDebugView instances from domain objects.
* Support constructing an immutable snapshot of user profile state, including identity, preferences, system instructions, memory statistics, recent memory entries, and recent sessions.

### `intergrax\runtime\user_profile\user_profile_instructions_service.py`

Description: This module provides high-level functionality to generate and update user-level system instructions for the Intergrax framework's conversational AI assistant.

Domain: LLM adapters, memory management, user profile instructions service

Key Responsibilities:
- Load UserProfile via UserProfileManager.
- Build an LLM prompt using identity, preferences, and memory entries.
- Call LLMAdapter.generate_messages() to obtain a compact, stable system prompt.
- Persist the result via UserProfileManager.update_system_instructions().

### `intergrax\supervisor\__init__.py`

DESCRIPTION: This module serves as the main entry point for the Intergrax supervisor, responsible for initializing and managing the overall workflow.

DOMAIN: Supervisor logic

KEY RESPONSIBILITIES:
• Initializes the supervisor component
• Sets up event listeners and workflows
• Provides a centralized interface for external interactions (e.g., command-line arguments)

### `intergrax\supervisor\supervisor.py`

**Description:** This module provides the core functionality of the Intergrax framework's supervisor component. It is responsible for planning and executing tasks based on user input, utilizing Large Language Models (LLMs) for decision-making.

**Domain:** Supervisor logic

**Key Responsibilities:**

*   Plan execution using LLMs
*   Decompose complex tasks into individual steps
*   Assign components to each step based on the plan
*   Execute tasks with assigned components and track progress
*   Provide analysis of executed plans and their outcomes

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides the core components and utilities for managing and executing tasks in the Integrax framework, including pipeline state management, component registration, and context passing.

**Domain:** Supervisor Components

**Key Responsibilities:**

* Defines a `PipelineState` dataclass for storing intermediate results and artifacts.
* Introduces the `ComponentResult` dataclass for representing individual task outputs.
* Establishes the `ComponentContext` dataclass for passing shared resources between tasks.
* Provides the `Component` class, which encapsulates a single task with metadata and an execution function (`fn`).
* Implements a decorator (`component`) for registering new components in a declarative manner.

Note: The code appears to be well-structured, documented, and production-ready.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module defines default prompt templates and rules for the Intergrax Supervisor, a hybrid RAG + Tools + General Reasoning engine. It provides guidance on task decomposition, component selection, and planning constraints.

**Domain:** LLM adapters / Unified Supervisor

**Key Responsibilities:**

* Defines default prompt templates for the Intergrax Supervisor (plan_system, plan_user_template)
* Provides rules for task decomposition and assignment
* Outlines planning constraints to ensure valid plans are generated
* Offers guidance on component selection based on step outputs and required inputs
* Specifies output formats and requirements for the Supervisor's response

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module is responsible for translating a supervisor plan into a LangGraph pipeline. It defines classes and functions to handle global state, node names, and graph construction.

**Domain:** Supervisor logic

**Key Responsibilities:**

*   Defines the `PipelineState` class as a typed dictionary to represent the global state of the LangGraph pipeline.
*   Implements utilities for ensuring state defaults, appending logs, and resolving inputs for plan steps.
*   Provides functions to slugify node names, create unique node names, and persist outputs.
*   Defines the `make_node_fn` function to generate a LangGraph node function that executes one plan step based on its method (GENERAL, TOOL, or RAG).
*   Implements topological ordering of plan steps for building a stable graph.
*   Exposes the `build_langgraph_from_plan` function to transform a Plan into a runnable LangGraph pipeline.

Note: The module appears to be well-structured and complete. There is no indication that it's experimental, auxiliary, legacy, or incomplete.

### `intergrax\system_prompts.py`

Description: This module defines a default RAG system instruction for strict RAGs in the Integrax framework.

Domain: RAG logic

Key Responsibilities:
- Defines a structured approach to answering user queries based on document search results.
- Outlines the steps and guidelines for providing accurate, precise, and detailed responses with proper citations.
- Specifies formatting requirements for answers, including summaries, detailed explanations, and references.

### `intergrax\tools\__init__.py`

**File:** intergrax\tools\__init__.py

**Description:** This module serves as the entry point for Intergrax's toolset, responsible for organizing and exposing various utility functions to other parts of the framework.

**Domain:** Tools/Utilities

**Key Responsibilities:**
- Provides an interface to initialize and configure tool dependencies
- Exposes utility functions and classes accessible throughout the framework

### `intergrax\tools\tools_agent.py`

**Description:** This module provides an interface for managing and executing tools within the Integrax framework. It enables users to leverage various tools, such as OpenAI's native tools or a JSON planner, to perform tasks.

**Domain:** LLM adapters & RAG logic

**Key Responsibilities:**

* Provides an interface for interacting with different language models (LLMs) and their tool sets.
* Offers support for native tools (e.g., OpenAI) and JSON planners (e.g., Ollama).
* Manages conversation flow, handling user inputs, system instructions, and context.
* Orchestrates the execution of tools, including error handling and output processing.
* Integrates with other modules, such as memory management and usage tracking.

Note: The file appears to be a primary implementation module for the ToolsAgent class.

### `intergrax\tools\tools_base.py`

**Description:** This module provides base classes and utilities for tool development within the Integrax framework. It includes functionality for defining tools, handling tool outputs, and registering tools for use with the OpenAI Responses API.

**Domain:** Tooling/Utilities

**Key Responsibilities:**

*   Provides a `ToolBase` class that serves as a foundation for developing custom tools.
*   Offers a `ToolRegistry` class for storing and exporting registered tools in a format compatible with the OpenAI Responses API.
*   Includes utilities for safely truncating tool outputs to prevent overflowing LLM context.
*   Supports Pydantic-based validation of tool arguments.

### `intergrax\websearch\__init__.py`

Here is the documentation for the provided file:

**Description:** The `__init__.py` file serves as the entry point for the web search module in Intergrax, responsible for setting up and configuring its internal components.

**Domain:** Web Search

**Key Responsibilities:**

* Initializes the web search module
* Sets up configuration and dependencies
* Exposes necessary functions and classes to other modules

### `intergrax\websearch\cache\__init__.py`

DESCRIPTION: This module initializes the web search cache module, providing a data storage system for cached results.

DOMAIN: Caching Module

KEY RESPONSIBILITIES:
• Initializes the cache storage mechanism.
• Defines interfaces and classes for caching and retrieval of search results.

### `intergrax\websearch\cache\query_cache.py`

Description: This module provides an in-memory query cache for web search results with optional time-to-live (TTL) and maximum size.

Domain: Web Search Cache

Key Responsibilities:
- Provides a simple, single-process in-memory query cache.
- Stores typed web search results as `QueryCacheEntry` objects.
- Supports TTL and max size configuration.
- Offers methods for getting and setting cached results.
- Includes basic eviction strategy (oldest entry removal) when max entries is reached.

### `intergrax\websearch\context\__init__.py`

Description: The context module initializes and manages the web search application context.
Domain: Web Search Infrastructure

Key Responsibilities:
• Initializes the application context
• Manages context dependencies
• Provides context services for web search functionality

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module is responsible for building LLM-ready textual context and chat messages from web search results. It creates system prompts enforcing strict rules, user-facing prompts wrapping web sources, questions, and concrete tasks, and builds a textual context string from typed WebSearchResult objects.

**Domain:** RAG (Relevance-Affinity-Generality) logic

**Key Responsibilities:**

* Builds system prompts enforcing "sources-only" mode
* Builds user-facing prompts for chat-style LLMs
* Constructs textual context strings from web search results
* Creates chat messages for system and user roles

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: This is the entry point for the web search fetcher module, responsible for initializing and configuring the web search functionality within Intergrax.

DOMAIN: Web Search Fetcher

KEY RESPONSIBILITIES:
* Initializes the web search fetcher module
* Configures the search engine API connections
* Defines the fetcher's data processing pipeline

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module contains functionality for extracting metadata and content from web pages, including title, description, language, Open Graph tags, and plain text. It includes two main functions: `extract_basic` for lightweight extraction and `extract_advanced` for more aggressive readability-based extraction.

**Domain:** Web Search Extraction

**Key Responsibilities:**

* Extracting metadata:
	+ Title
	+ Description (meta[name="description"] or meta[property="og:description"])
	+ Language (<html lang> attribute)
	+ Open Graph tags (meta[property^="og:"])
* Extracting plain text
* Handling trafilatura library usage for advanced readability extraction
* Fallback to manual HTML cleanup and extraction if trafilatura fails
* Normalizing whitespace and reducing noise in extracted text
* Attaching metadata to PageContent.extra for debugging and analysis

Note: This file appears to be a core component of the Intergrax framework, with well-structured code and clear documentation.

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module contains an asynchronous HTTP fetcher for web pages, providing a simple interface to retrieve HTML content from URLs.

**Domain:** Web Search Utility

**Key Responsibilities:**
- Perform HTTP GET requests with customizable headers and timeouts.
- Handle redirects and capture final URL information.
- Return a `PageContent` instance containing the fetched page's metadata.

### `intergrax\websearch\integration\__init__.py`

DESCRIPTION: 
This is the entry point for the web search integration module, responsible for setting up and configuring other components.

DOMAIN: Web Search Integration

KEY RESPONSIBILITIES:
- Imports necessary modules and sets up the integration framework.
- Registers available web search adapters with the main application.
- Provides a method to retrieve a registered adapter instance by name.

### `intergrax\websearch\integration\langgraph_nodes.py`

Description: This module provides a web search node implementation for the Intergrax framework, enabling integration with various search engines.

Domain: LLM adapters

Key Responsibilities:
- Encapsulates configuration and execution of web searches using WebSearchExecutor
- Implements LangGraph-compatible state management through WebSearchState class
- Provides synchronous and asynchronous node methods (run and run_async) for executing web searches
- Offers a default, lazily constructed WebSearchNode instance for convenience and backward compatibility

### `intergrax\websearch\pipeline\__init__.py`

DESCRIPTION: This module initializes the web search pipeline by setting up necessary components and configurations.

DOMAIN: Web Search Pipeline

KEY RESPONSIBILITIES:
* Initializes pipeline components
* Configures pipeline settings and parameters 
* Sets up data flow for web search functionality

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module implements a web search pipeline that can execute queries against multiple providers, fetch and extract relevant data, and deduplicate results. It is designed to be provider-agnostic and uses asynchronous fetching with rate limiting.

**Domain:** Web Search Pipeline

**Key Responsibilities:**
* Orchestrates multi-provider web search
* Fetches and extracts search hits into web documents
* Deduplicates results using a simple text-based key
* Provides a synchronous convenience wrapper for async execution
* Supports rate-limited HTTP fetching and provider-agnostic design

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: This module serves as the entry point for web search providers in the Intergrax framework, allowing for easy configuration and usage of various search engines.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
- Provides a standard interface for registering and using different web search providers.
- Allows for configuration and customization of search engine settings.

### `intergrax\websearch\providers\base.py`

**Description:** This module provides a base interface for web search providers in the Intergrax framework, defining a standard structure for querying and retrieving search results.

**Domain:** Web Search Providers

**Key Responsibilities:**
- Provides an abstract base class `WebSearchProvider` that defines a provider-agnostic interface.
- Exposes methods for searching (`search`) and obtaining capabilities (`capabilities`).
- Requires providers to implement the `search` method, which must return a ranked list of search hits.
- Includes optional resource cleanup functionality in the `close` method.

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module implements a Bing Web Search provider for the Intergrax framework.

**Domain:** RAG logic (Relevance Aware Retrieval)

**Key Responsibilities:**

* Provides Bing Web Search functionality using REST API
* Supports language, region, and freshness filtering
* Handles API key authentication and session management
* Retrieves search results and maps them to Intergrax's SearchHit schema
* Exposes methods for capabilities discovery, query execution, and session closure

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module is a custom search engine provider for the Intergrax framework, implementing the Google Custom Search (CSE) REST API.

**Domain:** Websearch Providers

**Key Responsibilities:**

* Provides configuration options for Google CSE via environment variables
* Supports language filtering and UI localization
* Ignores freshness parameter as it's not natively supported by the CSE API
* Validates URLs to ensure they're absolute
* Extracts publication date from page metadata (best effort)
* Infers source type from MIME type if provided
* Handles pagination with a cap of 10 results per request
* Provides search hits with relevant metadata and extra information

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module is a web search provider for the Intergrax framework, specifically designed to interface with Google Places API. It allows text search and details retrieval for businesses.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

- Provides a Google Places API interface
- Supports text search by arbitrary query
- Returns core business data (name, address, location, rating, user ratings, types, website, etc.)
- Allows fetching of additional details for each place
- Implements parameter building and mapping helpers for efficient querying

This file appears to be complete and production-ready.

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a Reddit search provider using the official OAuth2 API, allowing for full-featured search and post metadata retrieval.

**Domain:** Websearch providers

**Key Responsibilities:**

* Authenticates with Reddit's OAuth2 API using client credentials
* Handles search requests and returns results in the `SearchHit` format
* Fetches top-level comments for each post (optional)
* Maps user-specified freshness and language filters to Reddit API parameters
* Exposes capabilities (supports_language, supports_freshness)

### `intergrax\websearch\schemas\__init__.py`

Description: This file initializes the schema registration for web search functionality within Intergrax.

Domain: Web Search Schemas

Key Responsibilities:
• Registers schema definitions for web search components.
• Initializes and configures schema handlers.
• Provides entry point for web search schema management.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module provides a dataclass for representing the fetched and processed content of a web page, including metadata and derived information.

**Domain:** Web Search (Page Content)

**Key Responsibilities:**

* Encapsulates raw HTML and extracted metadata
* Provides access to derived information such as title, description, language, Open Graph tags, and schema.org data
* Offers methods for filtering out failed or empty fetches (`has_content`)
* Generates truncated text snippets for logging and debugging purposes (`short_summary`)
* Calculates the approximate size of the content in kilobytes (`content_length_kb`)

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines the `QuerySpec` dataclass, which represents a canonical search query specification used by web search providers.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Defines a minimal and provider-agnostic search query model
* Encapsulates query parameters such as user input, top results count, locale, region, language, freshness, site filter, and safe search
* Provides two utility methods:
	+ `normalized_query()`: Returns the query string with an applied site filter (if present)
	+ `capped_top_k(provider_cap)`: Returns a provider-safe 'top_k' that never exceeds the provider's cap and is >= 1

**Status:** Production-ready

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a `SearchHit` data class, which encapsulates metadata for a single search result entry, providing a standardized structure for provider-agnostic data.

**Domain:** Search Query Results Schema

**Key Responsibilities:**

* Provides a `SearchHit` data class with attributes for provider identifier, query string, rank, title, URL, and additional metadata.
* Includes validation checks in the `__post_init__` method to ensure the `rank` attribute is at least 1 and the `url` has a valid scheme and netloc.
* Offers utility methods:
	+ `domain()`: extracts the domain from the URL for grouping or scoring.
	+ `to_minimal_dict()`: returns a minimal representation of the hit, suitable for LLM prompts and logs.

### `intergrax\websearch\schemas\web_document.py`

**Description:** This module defines a unified data structure for representing processed web documents, including their source metadata and extracted content.

**Domain:** Web Search / Document Representation

**Key Responsibilities:**

* Represents a web document with its source search hit, extracted page content, and analysis results (deduplication key, quality score)
* Provides methods to validate the document's validity (textual content and URL presence), merge textual content for LLM or retrieval embedding, and generate a short summary line.

### `intergrax\websearch\schemas\web_search_answer.py`

**Description:** This module defines a dataclass for representing the result of a web search, including the final answer, underlying LLM messages, and associated web search results.

**Domain:** Web Search Schemas

**Key Responsibilities:**
- Represents typed result of WebSearchAnswerer
- Defines structure for storing final model answer, LLM-ready messages, and typed web search results

### `intergrax\websearch\schemas\web_search_result.py`

**Description:** This module defines a data class for representing web search results, encapsulating metadata and content from the provider.
 
**Domain:** Web Search Schemas
 
**Key Responsibilities:**
- Provides a structured representation of web search results.
- Includes essential metadata (provider, rank, quality score) and content attributes (title, URL, snippet).
- Includes optional fields for additional information (language, domain, publication details).

### `intergrax\websearch\service\__init__.py`

Description: This module initializes the web search service, responsible for providing API endpoints and managing search logic.
Domain: Web Search Service

Key Responsibilities:
* Initializes the web search service
* Defines API endpoint handlers
* Configures search logic and indexing mechanisms

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level helper class for generating answers to user queries by performing web searches, building context messages from search results, and utilizing Large Language Model (LLM) adapters.

**Domain:** Web Search

**Key Responsibilities:**
- Runs web searches using the `WebSearchExecutor` instance.
- Builds LLM-ready context/messages from web documents using the `WebSearchContextBuilder`.
- Calls an `LLMAdapter` to generate a final answer.
- Provides synchronous and asynchronous APIs for answering user queries.

### `intergrax\websearch\service\websearch_config.py`

Description: This module provides configuration and strategy definitions for the web search service, including LLM adapters and algorithmic parameters.

Domain: Web Search Configuration

Key Responsibilities:
- Defines the `WebSearchStrategyType` enum with four possible strategies.
- Provides the `WebSearchLLMConfig` dataclass to configure LLM adapters used in web search grounding steps.
- Offers the `WebSearchConfig` dataclass to set up algorithmic parameters and strategy for web search, including token budgets and chunking/reranking knobs.

### `intergrax\websearch\service\websearch_context_generator.py`

**Description:** This module provides a web search context generator, responsible for creating contextual information from search engine results. It supports multiple strategies (SERP_ONLY, URL_CONTEXT_TOPK, CHUNK_RERANK, and MAP_REDUCE) that can be configured through the WebSearchConfig.

**Domain:** RAG logic

**Key Responsibilities:**

*   Generate web search context based on the provided configuration
*   Support multiple context generation strategies (SERP_ONLY, URL_CONTEXT_TOPK, CHUNK_RERANK, MAP_REDUCE)
*   Provide debug information for each strategy
*   Handle token budgeting and truncation for excerpts in URL_CONTEXT_TOPK and MAP_REDUCE strategies
*   Raise errors if required configurations are missing for MAP_REDUCE strategy

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module implements a high-level, configurable web search executor that handles the construction of queries, execution of search pipelines, and conversion of web documents into LLM-friendly data structures.

**Domain:** Web Search Framework

**Key Responsibilities:**

* Constructs QuerySpec objects from raw queries and configuration
* Executes the SearchAndReadPipeline with chosen providers
* Converts WebDocument objects into LLM-friendly dicts (WebSearchResult)
* Provides methods for building query specifications and executing web searches asynchronously
* Supports caching of serialized results using an in-memory query cache

Note: The code appears to be complete and well-maintained, without any obvious signs of being experimental or auxiliary.

### `intergrax\websearch\utils\__init__.py`

**Description:** This utility module initializes and sets up the web search functionality within Intergrax.
**Domain:** Web Search Utilities
**Key Responsibilities:**
* Initializes web search components
* Sets up default search configurations
* Exposes initialization API for external usage

### `intergrax\websearch\utils\dedupe.py`

Description: This module provides functionality for deduplication of web search results, including normalization and hashing utilities.

Domain: Data Ingestion/Processing

Key Responsibilities:
* Normalizes input text to prepare it for deduplication
	+ Treats None as empty string
	+ Strips leading/trailing whitespace and converts to lower case
	+ Collapses internal whitespace sequences
* Generates a stable SHA-256 based deduplication key for the given text
	+ Hex-encoded digest of the normalized text

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a simple, asyncio-compatible token bucket rate limiter to enforce rate limits on concurrent requests.

**Domain:** Rate limiting utilities

**Key Responsibilities:**

* Initializes a token bucket with specified rate and capacity
* Waits until at least the required number of tokens are available before consuming them (acquire method)
* Non-blocking attempt to consume tokens, returns immediately if not enough tokens are available (try_acquire method)

### `main.py`

DESCRIPTION: This file serves as the entry point for the Integrax framework, responsible for executing the main program logic.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
* Initializes and runs the main program
* Prints a greeting message indicating successful execution of the Intergrax-ai framework.

### `mcp\__init__.py`

DESCRIPTION:
This is the top-level initialization file for the MCP package, responsible for setting up and configuring the framework's core components.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
- Initializes module dependencies and imports.
- Defines global constants and configuration settings.
- Sets up logging and error handling mechanisms.
- Establishes connections to other MCP modules.

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

Description: This notebook serves as a basic sanity-check for the Intergrax framework's Drop-In Knowledge Mode runtime. It tests the creation of a session, appending user and assistant messages, building conversation history, and returning a RuntimeAnswer object.

Domain: LLM (Large Language Model) adapters & Drop-In Knowledge Mode runtime

Key Responsibilities:
- Verify that the DropInKnowledgeRuntime can create or load a session.
- Test if the runtime can append user and assistant messages.
- Check if the conversation history is built from SessionStore.
- Ensure the RuntimeAnswer object is returned correctly.

Note: This notebook appears to be part of the Intergrax framework's testing infrastructure, specifically designed for Drop-In Knowledge Mode. It does not seem experimental, auxiliary, or legacy; however, it might require updates as the framework evolves.

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of Intergrax's Drop-In Knowledge Mode runtime for attachment ingestion. It showcases how to initialize the runtime, create an in-memory session store, and ingest attachments via a file system resolver.

**Domain:** LLM adapters & Runtime Configuration

**Key Responsibilities:**

* Initialize the Drop-In Knowledge Mode runtime with an in-memory session store
* Create an LLM adapter based on Ollama + LangChain
* Configure the runtime with settings for attachment ingestion
* Demonstrate ingestion of a local project document using an `AttachmentRef`
* Show how attachments are resolved and stored in the vector store

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of a ContextBuilder in the Drop-In Knowledge Mode runtime, providing a practical end-to-end demonstration of how to produce a ready-to-use context object for an LLM adapter.

**Domain:** RAG Logic / Context Builder

**Key Responsibilities:**

* Initializes minimum components required for testing ContextBuilder
	+ In-memory session store
	+ LLM adapter (Ollama-based)
	+ Embedding manager (same model as ingestion pipeline)
	+ Vector store manager (Chroma, same collection as before)
	+ Runtime config (RAG will be enabled in the next cell)
	+ Drop-In Knowledge runtime (used only as a dependency context)
* Initializes ContextBuilder instance using RuntimeConfig and shared IntergraxVectorstoreManager
* Demonstrates how to wire ContextBuilder into the Drop-In Knowledge Mode runtime using minimal components
* Provides end-to-end demonstration of producing a ready-to-use context object for an LLM adapter

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

Description: This notebook demonstrates the use of Intergrax framework's DropInKnowledgeRuntime with session-based chat, RAG (attachments ingested into a vector store), and live web search via WebSearchExecutor.

Domain: LLM adapters, Session management, Web search integration

Key Responsibilities:
- Initializes runtime configuration for drop-in knowledge mode
- Configures web search executor with Google CSE provider
- Sets up in-memory session storage and session manager
- Creates a DropInKnowledgeRuntime instance with configured settings
- Demonstrates usage of the runtime with interactive testing (ask(question: str) helper)
- Includes code to create a fresh chat session for web search demo

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

**Description:** This Jupyter notebook demonstrates how to use the Intergrax framework's Drop-In Knowledge Runtime with a tools orchestration layer, integrating conversational memory, RAG (attachments), and live web search context.

**Domain:** LLM adapters, RAG logic, data ingestion, agents, configuration, utility modules

**Key Responsibilities:**

* Configures Python path to import the `intergrax` package
* Loads environment variables (API keys, etc.)
* Imports core building blocks used by the Drop-In Knowledge Runtime
* Demonstrates how tools are integrated and used in a ChatGPT-like flow
* Initializes non-tool configuration (LLM, embeddings, vector store, web search) in a single compact setup cell
* Defines two demo tools (`WeatherTool`, `CalcTool`) using the Intergrax tools framework
* Registers them in a `ToolRegistry`
* Creates an `IntergraxToolsAgent` instance that uses an Ollama-based LLM
* Attaches this agent to `RuntimeConfig.tools_agent` so that the Drop-In Knowledge Runtime can orchestrate tools in a ChatGPT-like flow

**Note:** This notebook appears to be a demonstration or tutorial, rather than a production-ready implementation.

### `notebooks\drop_in_knowledge_mode\06_session_memory_roundtrip_demo.ipynb`

Description: This notebook demonstrates the Intergrax framework's capabilities for configurable, real adapters in a drop-in knowledge mode. It tests session creation, reuse, persistence, and loading of conversation history via SessionManager.

Domain: LLM Adapters, Drop-In Knowledge Mode

Key Responsibilities:
- Create a new session when session_id is None.
- Reuse an existing session when session_id is provided.
- Persist and load conversation history via SessionManager.get_history(...).
- Produce a consistent debug_trace["steps"].

Note: This file appears to be a part of the Intergrax framework's documentation and testing, specifically designed for drop-in knowledge mode with real adapters.

### `notebooks\drop_in_knowledge_mode\07_user_profile_instructions_baseline.ipynb`

**Description:** This notebook verifies the Intergrax framework's baseline user profile memory flow, which involves injecting system instructions from the stored UserProfile into the runtime prompt.

**Domain:** LLM adapters, Runtime configuration, User Profile Management

**Key Responsibilities:**

*   Persist a minimal user profile with system instructions
*   Verify that the final `system` instructions were built from the user profile
*   Validate that only user/assistant turns are stored in the persisted session history

### `notebooks\drop_in_knowledge_mode\08_user_profile_instructions_generation.ipynb`

**Description:** This notebook demonstrates a production-safe, explicit mechanism for generating `UserProfile.system_instructions` using an LLM based on conversation history, existing user profile, and optional session metadata.

**Domain:** LLM adapters (User Profile Instructions Generation)

**Key Responsibilities:**

* Generates `UserProfile.system_instructions` using an LLM
* Persists instructions to the user profile
* Marks sessions as requiring refresh (`needs_user_instructions_refresh = True`)
* Separates instruction generation from runtime usage
* Maintains data flow and invariants for explicit, auditable operation

Note: The file appears to be a notebook with clear documentation and separation of concerns. It provides a deterministic test case for the LLM-based generation service, which can be used to validate the instructions generation and persistence flow without relying on external APIs or model variability.

### `notebooks\drop_in_knowledge_mode\09_long_term_memory_consolidation.ipynb`

Description: This Jupyter notebook tests the "long-term memory via consolidation" behavior in the Intergrax framework, which involves consolidating session history into the user profile's long-term memory.

Domain: Long-Term Memory Consolidation

Key Responsibilities:
- Loads conversation history from the conversational store
- Triggers consolidation of session history into the user profile's long-term memory
- Validates production-critical invariants after consolidation

### `notebooks\drop_in_knowledge_mode\10_e2e_user_longterm_memory.ipynb`

Description: This notebook validates the engine-integrated user long-term memory path end-to-end by persisting profile information as JSON snapshots, modifying entries through edits and soft deletes, searching for entries via `SessionManager.search_user_longterm_memory`, injecting the engine via a SYSTEM message, and simulating multi-session interactions without refactors.

Domain: LLM adapters

Key Responsibilities:
- Validate user long-term memory path end-to-end
- Persist profile information as JSON snapshots
- Modify entries through edits and soft deletes
- Search for entries using `SessionManager.search_user_longterm_memory`
- Inject engine via SYSTEM message
- Simulate multi-session interactions

### `notebooks\drop_in_knowledge_mode\11_chatgpt_like_e2e.ipynb`

**Description:** This notebook is an integration/behavior test that exercises the Drop-In Knowledge Runtime end-to-end, in a way that resembles a real ChatGPT usage pattern.

**Domain:** RAG logic, LLM adapters, data ingestion, agents (Drop-In Knowledge Runtime)

**Key Responsibilities:**

* Tests multi-session behavior with isolated session history and shared user LTM
* Verifies user LTM persistence + recall across sessions
* Exercises session-level consolidation without cross-session leakage
* Tests RAG ingestion + Q&A over a document
* Demonstrates websearch as a context layer affecting the final answer
* Validates tools execution (tool + LLM) without breaking the user-last invariant
* Enables reasoning for observability, but does not persist into user-visible history

**Note:** This file appears to be an experimental/auxiliary notebook, used for testing and demonstration purposes.

### `notebooks\drop_in_knowledge_mode\12a_engine_planner.ipynb`

Description: This notebook is designed to test and improve the Intergrax Engine Planner in isolation from the engine execution pipeline. It allows for testing of plan generation and analysis without impacting the runtime.

Domain: RAG logic, LLM adapters, data ingestion, configuration, utility modules.

Key Responsibilities:
- Initialize an LLM adapter and RuntimeConfig
- Create a minimal RuntimeState for planner-only tests
- Run the EnginePlanner for different prompts
- Inspect and compare generated EnginePlan outputs
- Evaluate plan stability, correctness, and feasibility
- Refine the planner prompt, plan schema, and validation rules iteratively

### `notebooks\drop_in_knowledge_mode\12b_engine_planner.ipynb`

**Description:** This Jupyter notebook provides test utilities for the drop-in knowledge mode of the Intergrax framework. It includes functions to build a planner request, create a minimal RuntimeState instance, and run a single PlannerTestCase.

**Domain:** LLM adapters & RAG logic

**Key Responsibilities:**

* Building a planner request with user ID, session ID, message, instructions, and attachments
* Creating a minimal RuntimeState instance for planner-only tests
* Running a single PlannerTestCase and returning a structured result
* Extracting plan flags from an EnginePlan instance
* Defining test cases with question, expected intent, optional flags, and next step

### `notebooks\drop_in_knowledge_mode\13_engine_step_planner.ipynb`

Description: This notebook contains a series of tests for the Intergrax framework's knowledge mode, specifically for the StepPlanner component.

Domain: LLM adapters, RAG logic, data ingestion, agents, configuration, utility modules

Key Responsibilities:
* Test the integration between EnginePlanner and StepPlanner
* Validate the routing of plans based on user input and engine hints
* Demonstrate the use of different intent types (e.g. freshness, project architecture)
* Show how to build a plan using the `build_from_hints` method
* Test the generation of plans for various user queries

Note: The file appears to be well-maintained and not experimental, auxiliary, legacy, or incomplete.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

Description: This notebook demonstrates an end-to-end RAG workflow combining multiple knowledge sources into a single in-memory vector index and exposing it through a LangGraph-based agent.

Domain: Hybrid Multi-Source RAG with Intergrax + LangGraph

Key Responsibilities:
- Ingest content from multiple sources (local PDF files, local DOCX/Word files, live web results using the Intergrax `WebSearchExecutor`)
- Build a unified RAG corpus by normalizing documents into a common internal format, attaching basic metadata about origin (pdf / docx / web), and splitting documents into chunks suitable for embedding
- Create an in-memory vector index using an Intergrax embedding manager (e.g. OpenAI / Ollama) and storing embeddings in an **in-memory** Chroma collection via Intergrax vectorstore manager
- Answer user questions with a RAG agent by generating a single, structured report containing short summary of relevant information, key insights and conclusions, and optionally: recommendations/action items

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This Jupyter notebook demonstrates the integration of Intergrax and LangGraph for simple LLM QA.

Domain: Integrax-LangGraph adapters, RAG logic, data ingestion (LLM QA example)

Key Responsibilities:
- Import necessary libraries and modules from Intergrax and LangGraph.
- Define a simple State graph with a single node `llm_answer_node` that integrates an Intergrax LLM adapter as a LangGraph node.
- Run the graph on a sample user question, using the defined state and node implementation.
- This notebook serves as a starting point for more advanced examples, including web search, RAG, hybrid agents, etc.

Note: This appears to be a working example of integrating Intergrax with LangGraph, rather than experimental or auxiliary code.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

**Description:** This Jupyter Notebook file demonstrates a practical web research agent built from Intergrax components. It showcases how the framework can power "no-hallucination" web-based Q&A inside a graph-based agent.

**Domain:** RAG logic

**Key Responsibilities:**

* Initializes LLM adapter and WebSearch components
* Defines the graph state for the Web Research Agent
* Demonstrates nodes for:
	+ Normalizing user questions
	+ Running web search using Intergrax WebSearchExecutor
* Utilizes existing async API from WebSearchExecutor to execute web search

### `notebooks\multimedia\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter Notebook provides a demonstration of the Intergrax framework's capabilities for multimodal document processing and retrieval.

**Domain:** RAG (Retriever-Augmented Generator) logic, VectorStore management, Multimodal retrieval test

**Key Responsibilities:**

* Load and preprocess multimedia documents from various sources (video, audio, images)
* Split and embed documents using Intergrax's DocumentsSplitter and EmbeddingManager classes
* Manage VectorStore collection for multimodal document storage and querying
* Perform multimodal retrieval using the RagRetriever class

Note: This file appears to be a working example or a test script, rather than a core component of the Intergrax framework. It demonstrates how various components interact with each other but does not provide any groundbreaking features or innovations.

### `notebooks\multimedia\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the multimedia capabilities of the Intergrax framework by downloading, processing, and extracting video and audio content from YouTube.

**Domain:** Multimedia

**Key Responsibilities:**

* Downloads videos and audio from YouTube using `yt_download_video` and `yt_download_audio`
* Transcribes video to VTT format using `transcribe_to_vtt`
* Extracts frames and metadatas from video using `extract_frames_and_metadata`
* Translates audio using `translate_audio`
* Uses the ollama model to describe images using `transcribe_image`
* Extracts frames from video and transcribes images using `extract_frames_from_video`

### `notebooks\openai\rag_openai_presentation.ipynb`

**Description:** This Jupyter Notebook script demonstrates the usage of Intergrax framework for vector storage and querying, leveraging OpenAI's API.

**Domain:** RAG logic (Reinforcement Learning Augmented with Demonstrations)

**Key Responsibilities:**

*   Filling VectorStore using a local folder
*   Testing queries to retrieve answers from the vector store

Note that this script appears to be an example or demo code rather than a production-ready implementation.

### `notebooks\rag\chat_agent_presentation.ipynb`

Description: This notebook contains an example implementation of the Intergrax chat agent, which uses a combination of language models and tools to answer user queries. The code demonstrates how to create a chat agent that can handle various tasks, including answering questions about the weather, explaining concepts, and performing calculations.

Domain: Chat Agent

Key Responsibilities:
- Initializes the LLM adapter using the OLLAMA provider
- Creates an instance of the conversational memory
- Defines a demo weather tool that simulates a weather API response
- Registers available tools with the ToolRegistry
- Configures the vector store and RAG components
- Demonstrates how to use the chat agent to answer user queries.

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This notebook provides an example usage of Integrax's RAG (Reinforced Augmented Generation) components for answering complex questions with the help of a pre-trained LLM (Large Language Model).

**Domain:** RAG logic

**Key Responsibilities:**

* Demonstrates the interaction between Integrax's RAG and LLM adapters.
* Utilizes EmbeddingManager, RagRetriever, VectorstoreManager, ReRanker, and RagAnswerer components for generating structured answers.
* Presents a scenario where a question is asked to the agent using a natural language string.
* Displays the answer produced by the LLM as a human-readable text output.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the workflow of loading, splitting, and embedding documents within the Integrax framework. It showcases how to load documents from a directory, split them into chunks, and create vector embeddings for each chunk using Ollama.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Load documents from a specified directory
* Split loaded documents into smaller chunks
* Create vector embeddings for each document chunk using Ollama
* Store embedded vectors in a Chroma vector store
* Probe the vector store to check if the target corpus is already present

### `notebooks\rag\tool_agent_presentation.ipynb`

Description: This notebook provides a demo of the Integrax framework's ToolsAgent, showcasing its ability to select and invoke tools based on user input. It includes implementations for two simple tools: a weather tool and an arithmetic calculator.

Domain: RAG logic (Reasoning-Augmented Generation)

Key Responsibilities:
- Demonstrates how the ToolsAgent selects and invokes tools based on user input.
- Implements a WeatherTool that provides mock current weather information for a given city.
- Implements a CalcTool that safely evaluates basic arithmetic expressions using a restricted environment.
- Sets up the agent with a conversational memory, tool registry, LLM (LLMProvider.OLLAMA), and tools to orchestrate reasoning, tool selection, invocation, and memory updates.
- Tests the agent's capabilities by running two scenarios:
  - One that selects the get_weather tool based on user input about Warsaw's weather.
  - Another that selects the calc_expression tool for evaluating an arithmetic expression.

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This is a Jupyter Notebook file containing code for implementing components in the Integrax framework. It defines several functions that serve as agents in a pipeline, responsible for tasks such as compliance checking, cost estimation, generating a final summary report, and performing financial audits.

**Domain:** RAG logic (Reasoning Agents)

**Key Responsibilities:**
- Implementing compliance checker agent:
  - Verifies if proposed changes comply with privacy policies and terms of service.
- Cost estimator agent:
  - Estimates the cost of changes based on UX audit reports.
- Final summary report generator:
  - Creates a consolidated summary using all collected artifacts.
- Financial audit agent:
  - Generates mock financial reports and VAT calculations.

Note: The content appears to be fully functional, with detailed mock data used for testing purposes.

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This is an interactive Jupyter Notebook (.ipynb) file that serves as a presentation and demonstration of the WebSearchExecutor with Google and Bing Search capabilities in the Integrax framework.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Demonstrates how to use the WebSearchExecutor with Google Custom Search
* Provides an example query specification for searching with LangGraph and LangChain
* Shows how to handle search results, including provider, rank, title, URL, snippet, domain, and published date
* Includes a comparison of LangChain and LangGraph using a graph structure
