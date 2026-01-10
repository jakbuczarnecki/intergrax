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
- `intergrax\runtime\nexus\__init__.py`
- `intergrax\runtime\nexus\config.py`
- `intergrax\runtime\nexus\context\__init__.py`
- `intergrax\runtime\nexus\context\context_builder.py`
- `intergrax\runtime\nexus\context\engine_history_layer.py`
- `intergrax\runtime\nexus\engine\__init__.py`
- `intergrax\runtime\nexus\engine\runtime.py`
- `intergrax\runtime\nexus\engine\runtime_context.py`
- `intergrax\runtime\nexus\engine\runtime_state.py`
- `intergrax\runtime\nexus\ingestion\__init__.py`
- `intergrax\runtime\nexus\ingestion\attachments.py`
- `intergrax\runtime\nexus\ingestion\ingestion_service.py`
- `intergrax\runtime\nexus\pipelines\__init__.py`
- `intergrax\runtime\nexus\pipelines\contract.py`
- `intergrax\runtime\nexus\pipelines\no_planner_pipeline.py`
- `intergrax\runtime\nexus\pipelines\pipeline_factory.py`
- `intergrax\runtime\nexus\pipelines\planner_dynamic_pipeline.py`
- `intergrax\runtime\nexus\pipelines\planner_static_pipeline.py`
- `intergrax\runtime\nexus\planning\__init__.py`
- `intergrax\runtime\nexus\planning\engine_plan_models.py`
- `intergrax\runtime\nexus\planning\engine_planner.py`
- `intergrax\runtime\nexus\planning\plan_builder_helper.py`
- `intergrax\runtime\nexus\planning\plan_loop_controller.py`
- `intergrax\runtime\nexus\planning\runtime_step_handlers.py`
- `intergrax\runtime\nexus\planning\step_executor.py`
- `intergrax\runtime\nexus\planning\step_executor_models.py`
- `intergrax\runtime\nexus\planning\step_planner.py`
- `intergrax\runtime\nexus\planning\stepplan_models.py`
- `intergrax\runtime\nexus\prompts\__init__.py`
- `intergrax\runtime\nexus\prompts\history_prompt_builder.py`
- `intergrax\runtime\nexus\prompts\rag_prompt_builder.py`
- `intergrax\runtime\nexus\prompts\user_longterm_memory_prompt_builder.py`
- `intergrax\runtime\nexus\prompts\websearch_prompt_builder.py`
- `intergrax\runtime\nexus\responses\__init__.py`
- `intergrax\runtime\nexus\responses\response_schema.py`
- `intergrax\runtime\nexus\runtime_steps\__init__.py`
- `intergrax\runtime\nexus\runtime_steps\build_base_history_step.py`
- `intergrax\runtime\nexus\runtime_steps\contract.py`
- `intergrax\runtime\nexus\runtime_steps\core_llm_step.py`
- `intergrax\runtime\nexus\runtime_steps\ensure_current_user_message_step.py`
- `intergrax\runtime\nexus\runtime_steps\history_step.py`
- `intergrax\runtime\nexus\runtime_steps\instructions_step.py`
- `intergrax\runtime\nexus\runtime_steps\persist_and_build_answer_step.py`
- `intergrax\runtime\nexus\runtime_steps\profile_based_memory_step.py`
- `intergrax\runtime\nexus\runtime_steps\rag_step.py`
- `intergrax\runtime\nexus\runtime_steps\retrieve_attachments_step.py`
- `intergrax\runtime\nexus\runtime_steps\session_and_ingest_step.py`
- `intergrax\runtime\nexus\runtime_steps\setup_steps_tool.py`
- `intergrax\runtime\nexus\runtime_steps\tools.py`
- `intergrax\runtime\nexus\runtime_steps\tools_step.py`
- `intergrax\runtime\nexus\runtime_steps\user_longterm_memory_step.py`
- `intergrax\runtime\nexus\runtime_steps\websearch_step.py`
- `intergrax\runtime\nexus\session\__init__.py`
- `intergrax\runtime\nexus\session\chat_session.py`
- `intergrax\runtime\nexus\session\in_memory_session_storage.py`
- `intergrax\runtime\nexus\session\session_manager.py`
- `intergrax\runtime\nexus\session\session_storage.py`
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
- `notebooks\nexus\01_basic_memory_demo.ipynb`
- `notebooks\nexus\02_attachments_ingestion_demo.ipynb`
- `notebooks\nexus\03_rag_context_builder_demo.ipynb`
- `notebooks\nexus\04_websearch_context_demo.ipynb`
- `notebooks\nexus\05_tools_context_demo.ipynb`
- `notebooks\nexus\06_session_memory_roundtrip_demo.ipynb`
- `notebooks\nexus\07_user_profile_instructions_baseline.ipynb`
- `notebooks\nexus\08_user_profile_instructions_generation.ipynb`
- `notebooks\nexus\09_long_term_memory_consolidation.ipynb`
- `notebooks\nexus\10_e2e_user_longterm_memory.ipynb`
- `notebooks\nexus\11_chatgpt_like_e2e.ipynb`
- `notebooks\nexus\12a_engine_planner.ipynb`
- `notebooks\nexus\12b_engine_planner.ipynb`
- `notebooks\nexus\13a_engine_step_planner.ipynb`
- `notebooks\nexus\13b_engine_step_planner.ipynb`
- `notebooks\nexus\14_engine_planner_executor.ipynb`
- `notebooks\nexus\15_plan_loop_controller.ipynb`
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

DESCRIPTION: This is the top-level entry point for the Intergrax API, handling initialization and setup.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
• Initializes API configuration and settings.
• Sets up application-wide dependencies.
• Imports and registers core API modules.

### `api\chat\__init__.py`

**Description:** This module serves as the initialization point for chat-related functionality within the Intergrax framework, establishing a foundation for further development.

**Domain:** Chat API

**Key Responsibilities:**
* Initializes modules and imports necessary for chat functionality
* Sets up routing and other internal configuration

### `api\chat\main.py`

**Description:** This module implements the main logic for handling user queries and document interactions within the Integrax framework. It encompasses chat functionality, document uploading and indexing, as well as retrieving and deleting documents.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**
- Handles incoming chat requests from users, integrating with language models through adapters
- Provides endpoints for uploading and indexing documents
- Offers methods for listing all documents in the system and retrieving specific document information
- Allows for deletion of uploaded documents

### `api\chat\tools\__init__.py`

DESCRIPTION:
This is the initialization file for the chat API tools, responsible for setting up and importing necessary modules.

DOMAIN: Chat API tools

KEY RESPONSIBILITIES:
• Initializes tool imports
• Sets up module dependencies
• Provides entry points for tool usage

### `api\chat\tools\chroma_utils.py`

**Description:** The `chroma_utils.py` module provides utility functions for interacting with the Chroma vector store, including loading and splitting documents, indexing documents, and deleting documents.

**Domain:** RAG logic

**Key Responsibilities:**
- Load and split documents from a given file path
- Index documents to the Chroma vector store
- Delete documents by file ID

### `api\chat\tools\db_utils.py`

**Description:** This module provides database utilities for the Intergrax framework, including schema creation and migration, as well as public APIs for message and document storage.

**Domain:** Data Ingestion/Storage

**Key Responsibilities:**

* Provides low-level helpers for establishing a connection to the SQLite database
* Creates and migrates the database schema for messages and documents
* Offers public APIs for inserting, retrieving, and managing messages and documents
* Supports backward-compatibility with legacy application logs table

### `api\chat\tools\pydantic_models.py`

Description: This module provides data models for the chat API, utilizing Pydantic to define structured data classes.

Domain: LLM adapters

Key Responsibilities:
- Defines a ModelName enum for supported large language model variants.
- Specifies query input and response data structures (QueryInput and QueryResponse).
- Establishes a DocumentInfo class for file metadata representation.
- Creates a DeleteFileRequest class for removing files by ID.

### `api\chat\tools\rag_pipeline.py`

**Description:** This module is responsible for setting up and managing the components of a Relevance Aware Generator (RAG) pipeline, which is a key component in the Intergrax framework. The RAG pipeline enables the retrieval and ranking of relevant documents based on a user's query.

**Domain:** RAG logic

**Key Responsibilities:**

* Initializes VectorstoreManager and EmbeddingManager instances
* Provides methods for retrieving and re-ranking documents using Vectorstore and EmbeddingManager
* Builds LLM adapters using the Intergrax framework's registry
* Defines default prompts and settings for the answerer component

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides API utility functions for interacting with the Integrax framework's internal APIs, including chat and document management.

**Domain:** Data Ingestion / File Management

**Key Responsibilities:**
- Send requests to internal APIs for chat-related tasks (e.g., get API response)
- Upload files to the backend
- List uploaded documents
- Delete specific documents by ID

### `applications\chat_streamlit\chat_interface.py`

Description: This module provides a Streamlit interface for displaying and interacting with a chat application.

Domain: Chat Interface

Key Responsibilities:
- Displays a list of chat messages
- Handles user input via a text input field
- Makes an API request to generate a response to the user's query
- Updates the session state with the new session ID and generated answer
- Provides additional details about the model used and session ID

### `applications\chat_streamlit\sidebar.py`

Description: This module provides a user interface for the chat application's sidebar, including model selection and file upload functionality.

Domain: Chat Application Interface Components

Key Responsibilities:
- Displays model selection component
- Handles file uploads with API interactions (upload_document)
- Lists uploaded documents with refresh capability
- Enables document deletion

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module serves as the entry point for the Intergrax chatbot application, utilizing Streamlit to create a user interface with a sidebar and chat interface.

**Domain:** Chat UI

**Key Responsibilities:**

* Initializes the Streamlit application
* Sets up default session state variables (messages and session_id)
* Calls functions to display the sidebar and chat interface

### `applications\company_profile\__init__.py`

FILE PATH:
applications\company_profile\__init__.py

Description: The company profile initialization module sets up the foundation for accessing and managing company-related data within the Intergrax framework.

Domain: Data Access/Management

Key Responsibilities:
- Initializes and exposes company profile data access interfaces.
- Configures necessary dependencies for subsequent operations.

### `applications\figma_integration\__init__.py`

DESCRIPTION: 
This module initializes and configures the Figma integration for Intergrax.

DOMAIN: Figma Integration

KEY RESPONSIBILITIES:
- Initializes the Figma API client.
- Configures authentication with Figma credentials.
- Sets up event listeners for real-time updates from Figma.

### `applications\ux_audit_agent\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the UX Audit Agent application, responsible for initializing and configuring its functionality.

DOMAIN: Agents

KEY RESPONSIBILITIES:
* Initializes the audit agent
* Configures necessary dependencies and components

### `applications\ux_audit_agent\components\__init__.py`

DESCRIPTION: This module initializes and registers components for the UX audit agent, setting up its configuration and dependencies.

DOMAIN: UX Audit Agent Components

KEY RESPONSIBILITIES:
- Registers UX audit agent components
- Initializes component dependencies and configurations
- Sets up component registry for the UX audit agent

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module provides a compliance checker component for the Intergrax framework, which evaluates proposed changes against privacy policies and regulations.

**Domain:** Compliance & Validation

**Key Responsibilities:**

* Checks if proposed changes comply with privacy policy and regulatory rules
* Simulates a 80% chance of being compliant with random choices
* Returns findings on compliance status, policy violations, and required reviews or actions
* Stops pipeline execution if non-compliant

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a component for estimating the cost of UX-related changes based on an audit report, using a mock pricing model.

**Domain:** RAG logic

**Key Responsibilities:**
- Provides a Cost Estimation Agent component for calculating UX update costs.
- Utilizes a mock pricing model to estimate costs.
- Returns a cost estimate, currency, and method used in the calculation.
- Includes example usage scenarios.

### `applications\ux_audit_agent\components\final_summary.py`

**applications\ux_audit_agent\components\final_summary.py**

Description: This module defines a component that generates a complete summary of the entire execution pipeline using all collected artifacts.

Domain: Agent Components

Key Responsibilities:
- Generates final report based on collected artifacts
- Includes pipeline status, termination reason, and project manager decisions
- Produces a JSON object with the final report

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module provides a financial audit component for the Integrax framework, generating mock financial reports and VAT calculations for testing purposes.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**
- Defines a "Financial Agent" component using the `@component` decorator.
- Generates test data for financial computations, including net values, VAT rates, amounts, and gross values.
- Returns a mock financial report and VAT calculations as part of the component result.

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component that answers user questions about the Intergrax system, including its modules, architecture, and documentation.

**Domain:** LLM adapters

**Key Responsibilities:**

- Provides general knowledge responses to user queries about the Intergrax system.
- Utilizes the `@component` decorator to register the "General" component with specific metadata and usage guidelines.
- Returns a structured response containing the answer, citations (mock documents), and logs.

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a component for project management in the UX audit pipeline, which makes mock decisions on UX reports and can stop pipeline execution if necessary.

**Domain:** RAG logic

**Key Responsibilities:**
- Makes mock decisions on UX reports with 70% chance of acceptance
- Produces decision and notes as output
- Can stop pipeline execution if proposal is rejected
- Logs decision and reason for rejection (if applicable)

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module defines a component for performing UX audits on Figma mockups, generating sample reports with recommendations.

Domain: LLM adapters

Key Responsibilities:
- Defines the "UX Auditor" component using the `@component` decorator.
- Performs a UX audit based on Figma mockups and returns a sample report with recommendations.
- Returns a `ComponentResult` object containing the generated report and related metadata. 

Note: The provided content appears to be a well-structured, functional module, without any signs of being experimental, auxiliary, legacy, or incomplete.

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This Jupyter Notebook appears to be an interactive workflow that guides the UX audit process, integrating various components and tools to perform audits on FIGMA mockups and prepare final reports.

Domain: RAG logic (Reusable Activity Graph)

Key Responsibilities:
- Perform UX audit on FIGMA mockups using UX Auditor component
- Verify changes comply with company policy using Policy & Privacy Compliance Checker component
- Prepare summary report for Project Manager using Final Report component
- Evaluate financial impact of changes using Cost Estimation Agent component
- Project Manager decision and project continuation using Project Manager component
- Final report preparation and synthesis using Final Report component

Note: This file appears to be a working implementation, with no signs of being experimental, auxiliary, legacy, or incomplete.

### `bundle_intergrax_engine.py`

**Description:** This module provides the core functionality for bundling and managing Python files within the Intergrax framework. It handles tasks such as file collection, metadata extraction, and bundle generation.

**Domain:** File Management, Bundle Generation

**Key Responsibilities:**

* Collects Python files from a specified root directory while respecting excluded directories
* Extracts metadata (e.g., module name, group, SHA-256 hash) for each collected file
* Builds an LL&M header for the generated bundle, including instructions and metadata about the included modules
* Generates a bundle from a list of pre-selected .py file paths

Note: The code appears to be well-maintained and up-to-date, with proper documentation and adherence to coding standards. There is no indication that it's experimental, auxiliary, legacy, or incomplete.

### `generate_project_overview.py`

**Description:** This module provides a utility to automatically generate a Markdown file containing a project structure overview for the Intergrax framework.

**Domain:** Utility Modules

**Key Responsibilities:**

* Recursively scan the project directory
* Collect relevant source files (Python, Jupyter Notebooks, configurable)
* For each file:
	+ Read the source code
	+ Send content + metadata to an LLM adapter
	+ Generate a structured summary: purpose, domain, responsibilities
* Output: Creates `PROJECT_STRUCTURE.md` containing a clear, navigable, human-readable and LLM-friendly description of the entire project layout

**Note:** The module appears to be part of the Intergrax framework's official documentation generation pipeline.

### `intergrax\__init__.py`

**Intergrax Framework Initialization**

Description: The `__init__.py` file serves as the entry point for the Intergrax framework, responsible for initializing and configuring the overall architecture.

Domain: Core Framework Setup

Key Responsibilities:
- Initializes the framework modules
- Sets up the package structure
- Registers core components
- Defines the framework's entry points

### `intergrax\chains\__init__.py`

DESCRIPTION: This module serves as the entry point for Intergrax chain modules, providing a standardized interface and facilitating the loading of specific chain classes.

DOMAIN: Chain Management

KEY RESPONSIBILITIES:
• Defines the base class for chain modules.
• Loads available chain classes from submodules.
• Provides an entry point for registering new chain classes.

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module provides a flexible QA chain implementation using the LangChain framework, enabling users to build custom chains with hooks for modifying data at various stages.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Builds a QA chain with hooks for modifying data at stages:
	+ `on_before_build_prompt`: Modifies payload before building prompt
	+ `on_after_build_prompt`: Modifies prompt text after building
	+ `on_after_llm`: Post-processing of LLM answer text
* Creates a LangChain Runnable pipeline from the QA chain configuration
* Provides public API for invoking the chain and retrieving results

**Notes:** The module appears to be well-maintained, with clear documentation and structured code. However, some parts may require further investigation or clarification (e.g., specific implementation details in `_build_context` method).

### `intergrax\chat_agent.py`

**Description:** The `chat_agent` module is the main entry point for interacting with the Integrax framework, handling chat queries and routing them to various components such as RAG (Retrieval-Augmented Generation) endpoints or tools.

**Domain:** Chat Agent / LLM Adapters

**Key Responsibilities:**

* Routing chat queries through a combination of LLM adapters and RAG components
* Handling memory management and streaming for chat interactions
* Providing a stable result format for chat responses, including answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component
* Integrating with various components such as LLM adapters, tools, and RAG endpoints

**Notes:** The file appears to be a core part of the Integrax framework, handling the main logic for chat interactions. It provides a robust routing mechanism and integrates with various components to provide a comprehensive chat experience.

### `intergrax\globals\__init__.py`

Description: The `__init__.py` file serves as the entry point for the Intergrax framework, importing and initializing global modules.

Domain: Configuration

Key Responsibilities:
- Initializes global modules
- Sets up module imports for the rest of the framework

### `intergrax\globals\settings.py`

**Description:** This module provides a centralized configuration for the Intergrax framework, including default settings for language, locale, timezone, models, and session memory/consolidation intervals.

**Domain:** Configuration

**Key Responsibilities:**

* Expose global framework-wide configuration as a dataclass
* Define default values for language, locale, timezone, and models
* Allow customization of defaults through environment variables
* Provide singleton instance of GlobalSettings used across the framework

### `intergrax\llm\__init__.py`

FILE PATH:
intergrax\llm\__init__.py

DESCRIPTION: 
This module serves as the entry point for Intergrax's LLM (Large Language Model) adapters, providing a convenient interface to initialize and interact with various models.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
• Initializes and exports LLM adapter classes
• Defines default model configurations and settings
• Sets up model registration and loading mechanisms

### `intergrax\llm\messages.py`

**Description:** This module provides data classes and utilities for working with chat messages in the Integrax framework.

**Domain:** LLM adapters / Chat message models

**Key Responsibilities:**

* Defines a `MessageRole` enum to categorize messages by role (system, user, assistant, tool)
* Provides a `AttachmentRef` data class to store lightweight references to attachments associated with messages or sessions
* Introduces a `ChatMessage` data class that extends the universal chat message compatible with the OpenAI Responses API, supporting fields like tool calls and attachments
* Offers a `to_dict()` method for converting `ChatMessage` objects to dicts compatible with the OpenAI API
* Includes an `append_chat_messages()` function as a custom reducer for LangGraph state

### `intergrax\llm_adapters\__init__.py`

DESCRIPTION: This is the entry point for Intergrax's LLM (Large Language Model) adapters module, responsible for initializing and importing other related modules.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Initializes and imports other LLM adapters modules.
- Acts as a central hub for adapter-related functionality.
- Possibly contains configuration or setup code for the LLM adapters.

### `intergrax\llm_adapters\aws_bedrock_adapter.py`

**Description:** This module provides adapters for AWS Bedrock LLM models, handling model-specific codecs and data exchange between Integrax framework and the AWS Bedrock service.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Implementing model-family specific codecs (Anthropic Claude, Meta Llama, Mistral) for exchanging requests and responses with AWS Bedrock
* Building native request bodies for different model families based on system text, conversation history, temperature, and max tokens
* Extracting final text from invoke model response JSON
* Extracting streamed text chunks from invoke model with response stream event payload JSON

### `intergrax\llm_adapters\azure_openai_adapter.py`

**Description:** This module implements the Azure OpenAI adapter for the Intergrax framework, allowing it to interact with the Azure OpenAI service.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes an instance of the AzureOpenAIChatAdapter class, which provides a contract for interacting with the Azure OpenAI service.
* Defines methods for generating messages and streaming messages using the Azure OpenAI service.
* Estimates context window tokens based on deployment settings.
* Builds chat completion parameters for sending requests to the Azure OpenAI service.
* Splits system and conversation messages internally for processing.

The code appears to be well-structured and follows the standard Python style. However, it's worth noting that there are some potential issues with token estimation and context window management, which may require further investigation or optimization. Additionally, some methods have complex logic and may benefit from additional comments or documentation to improve readability.

### `intergrax\llm_adapters\claude_adapter.py`

**Description:** This module provides a Claude (Anthropic) adapter for the Intergrax framework, enabling integration with the Anthropic LLM.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes the Claude adapter using an optional Anthropic client and model ID
* Provides `generate_messages` and `stream_messages` methods to interact with the Claude API
* Estimates token usage for input messages and text output
* Implements internal helper functions for splitting system and conversation messages, mapping chat messages to Anthropic format, and handling errors.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This file defines a Gemini adapter for the Intergrax framework, utilizing the Google Gen AI SDK to interface with the Gemini model.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a GeminiChatAdapter class that inherits from LLMAdapter
* Supports generate_messages and stream_messages functions
* Utilizes genai.Client for interacting with the Gemini model
* Implements token estimation and context window management
* Handles error handling and logging through usage module

### `intergrax\llm_adapters\llm_adapter.py`

Description: This module provides a base interface and common functionality for integrating Large Language Models (LLMs) with the Intergrax framework.

Domain: LLM adapters

Key Responsibilities:

* Defines the universal runtime interface `LLMAdapter` for LLM adapters, providing strong runtime guarantees and shared base implementation.
* Implements token counting functionality through `count_messages_tokens` and `estimate_tokens_for_messages`.
* Supports optional features such as streaming, tools, and structured output via abstract methods that must be implemented by concrete adapter classes.

### `intergrax\llm_adapters\llm_provider.py`

Description: This module defines a set of enumerated values representing different LLM (Large Language Model) providers that can be used within the Integrax framework.
Domain: LLM adapters
Key Responsibilities:
- Defines an enumeration class `LLMProvider` for easy reference to supported LLM providers.
- Provides string-based representations for each provider, facilitating interoperability and consistency across the framework.

### `intergrax\llm_adapters\llm_provider_registry.py`

Description: This module provides a registry for various LLM (Large Language Model) adapters, allowing for easy registration and creation of instances based on provider specifications.

Domain: LLM Adapters

Key Responsibilities:
- Provides a central registry for LLM adapter factories
- Allows registering new providers with corresponding factory functions
- Facilitates instance creation using the registered providers
- Includes default registrations for major LLM providers

### `intergrax\llm_adapters\llm_usage_track.py`

**Description:** This module provides a usage tracking system for LLM (Large Language Model) adapters, aggregating statistics across multiple adapters used during a single runtime run.

**Domain:** LLM Adapters

**Key Responsibilities:**

*   Tracks adapter usage and statistics
*   Provides a report summarizing total usage and per-label statistics
*   Allows registering and unregistering adapters with the tracker
*   Supports aggregation by provider:model for deduplication of alias labels
*   Exports an immutable snapshot of usage stats for serialization and storage

### `intergrax\llm_adapters\mistral_adapter.py`

**Description:** This module provides a Mistral adapter for Intergrax's LLM adapters, allowing interaction with the Mistral model.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a `MistralChatAdapter` class that wraps the official Mistral Python SDK
* Supports two main methods:
	+ `generate_messages`: generates text responses from input messages using the Mistral model
	+ `stream_messages`: streams text responses from input messages using the Mistral model
* Estimates context window tokens for the configured model
* Splits system and conversation text from input messages
* Builds minimal, explicit Mistral Chat payloads
* Maps ChatMessage objects to Mistral chat completion message dicts

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides a LangChain Ollama adapter for interacting with Ollama models.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an adapter class (`LangChainOllamaAdapter`) that interacts with Ollama models via LangChain's `ChatModel` interface.
* Estimates context window tokens based on the model name or user-provided value.
* Converts internal `ChatMessage` list into LangChain message objects.
* Offers methods for generating and streaming messages using Ollama models.

**Classification:** This file appears to be part of the main codebase, well-documented, and actively used.

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** 
This module implements an adapter for OpenAI's Responses API, providing functionality to generate and stream chat responses.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an adapter class `OpenAIChatResponsesAdapter` that inherits from `LLMAdapter`
* Offers methods for single-shot completion (`generate_messages`) and streaming completion (`stream_messages`)
* Utilizes OpenAI's Responses API to generate and stream chat responses
* Supports context window estimation for various OpenAI models
* Implements input and output token handling for usage tracking
* Handles exceptions and error types during API calls

**Notes:** 
This file appears to be a production-ready module, with a clear structure and well-defined responsibilities.

### `intergrax\logging.py`

**Description:** This module configures and establishes the basic logging behavior for the Intergrax framework, setting a default log level and format.

**Domain:** Logging configuration

**Key Responsibilities:**
- Configures Python's built-in `logging` module with a default level of `INFO`
- Sets a custom log message format
- Forces the logging configuration (overrides any existing settings)

### `intergrax\memory\__init__.py`

FILE PATH:
intergrax\memory\__init__.py

DESCRIPTION: Initializes and sets up the memory management system within Intergrax.

DOMAIN: Memory Management

KEY RESPONSIBILITIES:
• Initializes memory-related classes and modules.
• Sets up default memory configurations.
• Defines memory-related constants and enumerations.

### `intergrax\memory\conversational_memory.py`

**Description:** This module provides a universal in-memory conversation history component for storing and managing chat messages.

**Domain:** Conversational Memory / Agents

**Key Responsibilities:**
- Store chat messages in RAM
- Provide an API to add, extend, read, and clear messages
- Optionally enforce a maximum number of messages limit
- Prepare messages for different model backends (e.g., OpenAI)
- Do not persist data (no files or external storage)

This module appears to be well-maintained and widely used within the Integrax framework.

### `intergrax\memory\conversational_store.py`

**Description:** This module defines the interface for persistent storage of conversational memory in the Intergrax framework, abstracting away underlying data stores and providing a unified way to interact with conversation histories.

**Domain:** Conversational Memory Storage

**Key Responsibilities:**
- Provide a protocol for abstracting storage backends (JSON, SQLite, Redis, etc.)
- Ensure deterministic persistence and idempotent write operations
- Load full conversational history or preload messages in chunks as needed
- Save entire state of conversational memory to persistent storage
- Append individual messages to persistent storage while updating the in-memory instance
- Permanently remove stored history for a given session

Note: The contents of this file suggest a well-designed and robust interface, but there is no indication that it's experimental, auxiliary, legacy, or incomplete.

### `intergrax\memory\stores\__init__.py`

DESCRIPTION: The `__init__.py` file initializes and defines the interface for memory stores in the Intergrax framework.

DOMAIN: Memory Management

KEY RESPONSIBILITIES:
• Initializes memory store instances.
• Defines the interface for interacting with memory stores.
• Possibly holds additional initialization or setup logic.

### `intergrax\memory\stores\in_memory_conversational_store.py`

**Description:** This module provides an in-memory conversational store implementation for Intergrax, suitable for local development, prototyping, unit testing, and environments where persistence is not required.

**Domain:** Conversational Memory Store

**Key Responsibilities:**

* Provides an in-memory storage for conversational data
* Supports loading and saving conversation history
* Allows appending messages to the conversation history
* Enables deleting sessions with no-error semantics
* Offers optional helper for listing active persisted session IDs

### `intergrax\memory\stores\in_memory_user_profile_store.py`

**Description:** This module implements an in-memory user profile store, providing a simple and lightweight solution for storing and retrieving user profiles within the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides an in-memory storage for user profiles
* Allows retrieval of existing profiles or default profiles if not present
* Enables persistence and update of profiles in memory
* Supports deletion of stored profiles by ID
* Offers optional debugging helper to list user IDs

### `intergrax\memory\user_profile_manager.py`

**Description:** This module provides a high-level facade for working with user profiles, hiding direct interaction with the underlying store and managing long-term user memory entries.

**Domain:** User Profile Management

**Key Responsibilities:**
- Load or create a UserProfile for a given user ID
- Persist profile changes
- Manage long-term user memory entries
- Manage system-level instructions derived from the profile
- Hide direct interaction with the underlying UserProfileStore
- Optionally enable long-term RAG (Recurrent Attention Graph) over user's long-term memory entries

### `intergrax\memory\user_profile_memory.py`

**Description:** This module provides core domain models for user/org profile and prompt bundles, serving as the foundation for storing and retrieving user-related information.

**Domain:** Memory Management

**Key Responsibilities:**

- Defines `MemoryKind` and `MemoryImportance` enums for categorizing memory entries.
- Introduces `UserProfileMemoryEntry`, a dataclass representing long-term memory entries for users, including stable facts, insights, or notes.
- Presents `UserIdentity`, a high-level description of who the user is, with attributes like display name, role, and domain expertise.
- Introduces `UserPreferences`, storing user preferences influencing how the runtime and LLM behave by default, such as answer language and style.
- Defines `UserProfile`, a canonical user profile aggregate separating identity, preferences, system instructions, and memory entries.

### `intergrax\memory\user_profile_store.py`

**intergrax\memory\user_profile_store.py**

Description: This module defines a protocol for persistent storage of user profiles, abstracting away backend-specific details and providing a standardized interface for loading, saving, and deleting user data.

Domain: Data Storage (RAG logic and LLM adapters are not part of this file's responsibility)

Key Responsibilities:
- Provides an asynchronous interface for loading, saving, and deleting user profiles
- Ensures implementations return initialized profiles with default settings when no data exists yet
- Allows for different backend storage mechanisms (JSON files, SQL DB, etc.) to be plugged in without changing the calling code

### `intergrax\multimedia\__init__.py`

DESCRIPTION: 
This is the entry point for the multimedia domain in the Intergrax framework, responsible for importing and exporting multimedia-related modules.

DOMAIN: Multimedia Domain

KEY RESPONSIBILITIES:
• Imports other multimedia-related modules and makes them available
• Exports multimedia-specific functionalities to be used elsewhere in the framework

### `intergrax\multimedia\audio_loader.py`

Description: This module provides utilities for loading and processing multimedia audio files within the Integrax framework.
Domain: Multimedia

Key Responsibilities:
- Downloads audio from a specified YouTube URL using yt_dlp library.
- Extracts audio from video file based on user-specified format (default: mp3).
- Translates audio content into a target language using Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module provides a utility for transcribing images using an LLM, specifically utilizing the Ollama model.

Domain: LLM adapters

Key Responsibilities:
- Transcribes image content into text based on user prompts and image paths
- Utilizes the Ollama model for processing

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, including audio, images, and videos, using IPython's display functionality.

**Domain:** Multimedia Display Utilities

**Key Responsibilities:**

* **Audio Display**: `display_audio_at_data` function displays an audio file with the ability to set playback position and autoplay.
* **Image Display**: `display_image` function displays an image file or URL, handling different image formats.
* **Video Display**: `display_video_jump` function displays a video file with the ability to set playback position, poster frame, and other options.

Note: The code appears to be well-maintained and production-ready. There are no signs of experimental or auxiliary functionality.

### `intergrax\multimedia\video_loader.py`

**Description:** 
This module provides functionality for handling multimedia content, specifically video files. It includes utilities for downloading videos from YouTube, transcribing audio to text (in VTT format), and extracting frames from videos along with associated metadata.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloading YouTube videos using `yt_dlp` library
* Transcribing audio in downloaded videos to text using Whisper model
* Extracting frames from videos, including resizing them for consistent output
* Saving extracted frames and their corresponding metadata (transcript text, timestamps) in a JSON file
* Option to limit the number of frames saved

### `intergrax\openai\__init__.py`

DESCRIPTION: 
This module serves as the entry point for importing other OpenAI-related modules within the Intergrax framework.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
• Initializes and exposes necessary functionality from the OpenAI library.
• Provides a standard interface for accessing various models and APIs.

### `intergrax\openai\rag\__init__.py`

FILE PATH:
intergrax\openai\rag\__init__.py

Description: Initializes the RAG (Retrieval-Augmented Generation) logic for OpenAI adapters.
Domain: LLM adapters

Key Responsibilities:
- Imports and sets up necessary modules for RAG functionality
- Defines entry points for RAG-based adapter initialization

### `intergrax\openai\rag\rag_openai.py`

Description: This module provides OpenAI adapter functionality for Integrax, enabling the use of OpenAI's vector store and file management capabilities within the framework.

Domain: LLM adapters

Key Responsibilities:
- Provides IntergraxRagOpenAI class to interact with OpenAI services
- Offers methods for retrieving vector stores, clearing vector stores and storage, and uploading folders to vector stores
- Defines a prompt template for strict RAG (Reinforced Active Retrieval) operations

### `intergrax\rag\__init__.py`

DESCRIPTION: This file defines the entry point for Intergrax's RAG logic, responsible for initializing and configuring the RAG components.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes RAG component instances
- Configures RAG components with necessary dependencies
- Sets up RAG-related constants and variables

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader with metadata injection and safety guards, supporting various file formats including PDF, Excel, CSV, DOCX, and images.

**Domain:** RAG logic (Documents Loader)

**Key Responsibilities:**

* Loads documents from various file formats
* Supports metadata injection and safety guards
* Provides options for handling specific file types (e.g., PDF OCR, Excel mode)
* Allows customization of loading behavior through various parameters (e.g., image captioning modes via framework adapter)

### `intergrax\rag\documents_splitter.py`

**Description:** This module provides a high-quality text splitter for RAG pipelines, implementing the 'semantic atom' policy to ensure stable chunk ids and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**

*   Split documents into chunks based on various separators (e.g., '\n\n', '\n', ' ', '')
*   Generate stable, human-readable chunk ids using available anchors (para_ix/row_ix/page_index) or falling back to index + content hash
*   Merge tiny tails and apply optional hard caps on the number of chunks per document
*   Add core source fields (parent_id, source_name, source_path) to each chunk's metadata
*   Infer page index from common loader keys when available

**Notes:** This file appears to be part of a larger framework for RAG pipelines, focusing on text splitting and chunk management. It demonstrates attention to detail in handling edge cases, such as merging tiny tails and applying optional caps. The code is well-structured, using clear variable names and following Python best practices.

### `intergrax\rag\dual_index_builder.py`

**Description:** This module is responsible for building and populating two vector indexes: a primary index (CHUNKS) and an auxiliary index (TOC). The primary index stores all chunks/documents after splitting, while the auxiliary index only stores DOCX headings within specific level ranges.

**Domain:** RAG logic

**Key Responsibilities:**
- Builds two vector indexes: primary (CHUNKS) and auxiliary (TOC)
- Prepares documents for indexing by filtering and processing text
- Computes embeddings for prepared documents using an EmbeddingManager instance
- Populates the primary index with chunked documents and their corresponding embeddings
- Optionally populates the auxiliary index with DOCX headings within specific level ranges
- Provides logging and status updates during the indexing process

### `intergrax\rag\dual_retriever.py`

**Description:** This module implements a dual retriever component that retrieves relevant documents using two strategies: section-based retrieval (TOC) and local chunk-based retrieval.

**Domain:** RAG logic

**Key Responsibilities:**

*   Dual retriever strategy implementation
*   Section-based retrieval (TOC)
*   Local chunk-based retrieval
*   Filtering and merging of user-provided filters with parent ID constraints
*   Normalization of query results
*   Embedding management for querying vector stores

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for various text embedding models, including Hugging Face's SentenceTransformer, Ollama, and OpenAI embeddings. It offers features such as provider switching, model loading, batch/single text embedding, and cosine similarity utilities.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Load text embedding models from various providers
* Provide a unified interface for embedding texts using different models
* Handle model loading failures with retry mechanisms
* Offer optional L2 normalization of embedded vectors
* Support cosine similarity calculations and top-K retrieval
* Log loading and execution information for debugging purposes

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides an answerer component for the Intergrax framework, integrating with Retrieval-Augmented Generation (RAG) and Large Language Models (LLM). It retrieves relevant context fragments, ranks them, builds messages, and generates answers using LLM.

**Domain:** RAG logic

**Key Responsibilities:**

* Retrieve relevant context fragments using a RagRetriever
* Rank retrieved hits using an optional reranker
* Build context from used hits
* Create system and user messages
* Generate answer using LLM, optionally with structured output
* Handle memory awareness and logging

### `intergrax\rag\rag_retriever.py`

**Description:** 
This module implements a RAG (Retrieval-Augmented Generation) retriever, providing utilities for scalable and provider-agnostic information retrieval.

**Domain:** RAG logic

**Key Responsibilities:**
- Normalizes `where` filters for Chroma
- Normalizes query vector shape to [[D]]
- Unified similarity scoring for various providers
- Deduplication by ID and per-parent result limiting (diversification)
- Optional MMR diversification when embeddings are returned
- Batch retrieval for multiple queries
- Optional reranker hook

### `intergrax\rag\re_ranker.py`

**Description:** This module provides a fast and scalable cosine re-ranker over candidate chunks, leveraging embedding management and optional score fusion with original retriever similarity.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**
- Embeds texts in batches using `EmbeddingManager`.
- Optional score fusion with original retriever similarity.
- Preserves schema of hits; adds 'rerank_score', 'fusion_score', and 'rank_reranked' fields.
- Re-ranks candidate lists by cosine similarity to the query.

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module provides a unified vector store manager for intergrax, supporting ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**
- Initializes target store (Chroma, Qdrant, or Pinecone) with provided configuration.
- Creates collection/index if needed (lazy for Qdrant/Pinecone).
- Upserts documents + embeddings with batching support.
- Queries top-K by cosine/dot/euclidean similarity.
- Counts vectors.
- Deletes by IDs.

The code appears to be well-maintained and production-ready, with clear documentation and structured organization. The VectorstoreManager class encapsulates the complex logic of interacting with different vector stores, providing a unified interface for users.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a windowed answerer on top of the base RagAnswerer, providing utilities for processing and synthesizing answers from retrieved candidates.

**Domain:** RAG Logic (Retrieval-Augmented Generation)

**Key Responsibilities:**

* Initializes the WindowedAnswerer with an Answerer and Retriever
* Builds context for each window using a local map-reduce approach
* Constructs messages for memory-awareness, including system prompts and contextual information
* Asks the LLM to generate partial answers for each window
* Synthesizes the final answer from the partials through a reduction step
* Collects sources and citations for each window
* Handles edge cases, such as no sufficiently relevant context found

### `intergrax\runtime\__init__.py`

Description: This is the main entry point for the Intergrax runtime, responsible for initializing and setting up the framework's core components.

Domain: Runtime Configuration

Key Responsibilities:
* Initializes the framework's core components
* Sets up the framework's configuration
* Defines the entry points for the application

### `intergrax\runtime\nexus\__init__.py`

**DESCRIPTION:** 
This module initializes and sets up the knowledge mode in Intergrax, allowing for drop-in integration of external knowledge.

**DOMAIN:** RAG logic (Reinforcement Learning with Augmented Data)

**Key Responsibilities:**
- Initializes knowledge mode configuration
- Sets up external knowledge integration
- Defines required dependencies and imports

### `intergrax\runtime\nexus\config.py`

**Description:** This file defines the configuration for the Intergrax framework's nexus Runtime, specifying how it interacts with various components such as LLM adapters, RAG (Retrieval-Augmented Generation) backend, web search, and tools agents.

**Domain:** Configuration & Settings

**Key Responsibilities:**

* Defines primary LLM adapter used for chat-style generation
* Configures RAG backend, including embedding manager and vectorstore manager
* Enables or disables Retrieval-Augmented Generation (RAG) and real-time web search features
* Specifies tenant ID and workspace ID for multi-tenancy support
* Configures maximum number of retrieved chunks per query and token budget reserved for RAG content
* Defines long-term memory retrieval configuration, including maximum number of entries and token budget
* Configures web search executor and pre-configured web search configuration
* Defines tools agent responsible for planning tool calls, invoking tools, and merging results into the final answer
* Specifies high-level policy defining whether tools may or must be used

### `intergrax\runtime\nexus\context\__init__.py`

Description: This file initializes the context for nexus mode in Intergrax's runtime.

Domain: RAG logic

Key Responsibilities:
• Initializes the context for nexus mode.
• Sets up necessary variables and data structures for knowledge retrieval.
• Provides a foundation for executing nexus queries.

### `intergrax\runtime\nexus\context\context_builder.py`

Description: This module provides a context builder for nexus Mode, responsible for deciding whether to use RAG and retrieving relevant document chunks from the vector store.

Domain: Context Builder/RAG Logic

Key Responsibilities:
- Decide whether to use RAG for a given request
- Retrieve relevant document chunks from the vector store using session/user/tenant/workspace metadata
- Provide a RAG-specific system prompt, retrieved chunks, and debug metadata for observability

### `intergrax\runtime\nexus\context\engine_history_layer.py`

Description: This module provides a history compression layer that preprocesses conversation history for the Intergrax framework.

Domain: nexus Mode

Key Responsibilities:

* Loading raw conversation history from SessionStore
* Computing token usage for the raw history, if possible
* Applying token-based truncation according to the per-request history compression strategy
* Resolving per-request settings and computing a token budget for history
* Updating RuntimeState with base_history and debug info

### `intergrax\runtime\nexus\engine\__init__.py`

Description: This is the entry point for the nexus mode engine, responsible for initializing and setting up the necessary components.

Domain: RAG (Reformulated Attention-based Generator) logic

Key Responsibilities:
- Initializes the engine's core components
- Sets up the knowledge retrieval mechanism
- Defines the interface for injecting custom knowledge sources

### `intergrax\runtime\nexus\engine\runtime.py`

**Description:** This module serves as the core runtime engine for nexus Mode, providing a high-level conversational interface using Intergrax components.

**Domain:** LLM (Large Language Model) adapters and RAG logic

**Key Responsibilities:**

* Defines the `RuntimeEngine` class for loading or creating chat sessions
* Appends user messages to the session and builds an LLM-ready context
* Calls the main LLM adapter with the enriched context to produce the final answer
* Returns a `RuntimeAnswer` object with the final answer text and metadata

This file appears to be a key component of the Intergrax framework, providing the core functionality for nexus Mode.

### `intergrax\runtime\nexus\engine\runtime_context.py`

**Description:** This module defines the RuntimeContext class, which encapsulates per-runtime context data and configuration for Intergrax.

**Domain:** Runtime Context Management

**Key Responsibilities:**

* Provides a centralized storage for runtime-specific configuration and dependencies.
* Manages LLTM usage records, including aggregation and printing of usage statistics.
* Offers methods to retrieve and clear LLTM usage runs.
* Allows for building a fully-resolved RuntimeContext instance using the same resolution rules as Runtime.__init__.

This file appears to be a crucial component of the Intergrax framework, providing essential functionality for runtime context management.

### `intergrax\runtime\nexus\engine\runtime_state.py`

**Description:** This module defines the `RuntimeState` class, which serves as a mutable state object passed through the runtime pipeline of Intergrax's nexus mode. It aggregates various metadata and results from different subsystems.

**Domain:** RAG logic and data ingestion (within the context of the nexus Mode)

**Key Responsibilities:**

* Aggregates request and session metadata, ingestion results, conversation history, model-ready messages, and flags indicating which subsystems were used
* Tracks usage and performance metrics for LLM adapters through the `LLMUsageTracker` class
* Provides methods for tracing events and setting debug sections/values
* Configures and finalizes the LLM tracker to collect usage data

**Notes:** The file appears to be a core component of the Intergrax framework, and its functionality is essential for managing runtime state and tracking LLM usage.

### `intergrax\runtime\nexus\ingestion\__init__.py`

Description: The `__init__.py` file is the entry point for the nexus mode ingestion module, responsible for initializing and configuring the ingestion process.

Domain: Data Ingestion

Key Responsibilities:
- Initializes the ingestion component
- Configures ingestion settings
- Sets up data processing pipelines

### `intergrax\runtime\nexus\ingestion\attachments.py`

**Description:** This module provides utility functions for resolving attachments in nexus Mode, decoupling attachment storage from the RAG pipeline's consumption.

**Domain:** Data Ingestion (Attachments)

**Key Responsibilities:**
* Defines the `AttachmentResolver` protocol for turning an `AttachmentRef` into a local file path.
* Implements the `FileSystemAttachmentResolver` class for resolving local filesystem-based URIs.
* Provides basic resolution functionality for local files and folders.

### `intergrax\runtime\nexus\ingestion\ingestion_service.py`

**Description:** This module provides a high-level service for ingesting attachments into the Intergrax framework's nexus Mode, allowing for efficient embedding and storage of documents.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Resolve AttachmentRef objects to loader-compatible paths using the AttachmentResolver component
* Load documents using IntergraxDocumentsLoader and split them into chunks with IntergraxDocumentsSplitter
* Embed chunks using IntergraxEmbeddingManager and store vectors in a vector database via IntergraxVectorstoreManager
* Return structured IngestionResult objects for each attachment
* Expose public API for ingesting attachments and searching session attachments

### `intergrax\runtime\nexus\pipelines\__init__.py`

Description: This is an initialization file for the nexus mode pipelines within the Intergrax runtime.

Domain: RAG logic

Key Responsibilities:
• Initializes pipeline components and configurations for nexus mode.
• Sets up necessary connections and dependencies between pipeline stages.

### `intergrax\runtime\nexus\pipelines\contract.py`

**Description:** This module defines a lightweight base class for pipeline runners in the Intergrax framework's nexus mode. It provides shared implementation and utilities for executing pipelines.

**Domain:** LLM adapters / Runtime Pipeline

**Key Responsibilities:**

* Provides a public entrypoint `run` method that executes a pipeline with shared validation and invariants.
* Defines an abstract method `_inner_run` that must be implemented by subclasses to perform pipeline-specific execution.
* Offers a utility function `build_default_planning_step_registry` to create a default registry for StepExecutor planning actions.
* Implements methods for validating the state and asserting the validity of the answer returned by the pipeline.

**Notes:** This file appears to be production-safe and well-maintained, with clear documentation and a structured implementation.

### `intergrax\runtime\nexus\pipelines\no_planner_pipeline.py`

Description: This module defines the NoPlannerPipeline class that serves as a nexus mode pipeline for Intergrax, responsible for executing a series of runtime steps to generate an answer.

Domain: RAG logic

Key Responsibilities:
- Defines the NoPlannerPipeline class as a subclass of RuntimePipeline
- Specifies a sequence of runtime steps ( SETUP_STEPS, EnsureCurrentUserMessageStep, RagStep, UserLongtermMemoryStep, RetrieveAttachmentsStep, WebsearchStep, ToolsStep, CoreLLMStep, PersistAndBuildAnswerStep ) to be executed in order
- Executes the pipeline using RuntimeStepRunner and returns the generated answer as a RuntimeAnswer object

### `intergrax\runtime\nexus\pipelines\pipeline_factory.py`

**Description:** This module, `pipeline_factory.py`, is responsible for creating pipelines for knowledge mode execution in the Intergrax framework. It utilizes a step planning strategy to determine which pipeline type to instantiate.

**Domain:** RAG logic

**Key Responsibilities:**

* Determines the appropriate pipeline type based on the step planning strategy
* Creates and returns the corresponding pipeline instance (NoPlannerPipeline, PlannerStaticPipeline, or PlannerDynamicPipeline)
* Builds and returns a default registry for StepExecutor planning actions in static/dynamic modes
* Includes explicit, production-safe bindings for various StepAction types to RuntimeStep instances

### `intergrax\runtime\nexus\pipelines\planner_dynamic_pipeline.py`

Description: This module implements a dynamic pipeline for the nexus mode in the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
- Defines a PlannerDynamicPipeline class that extends RuntimePipeline.
- Includes an inner_run method, which is currently unimplemented and raises a NotImplementedError.

### `intergrax\runtime\nexus\pipelines\planner_static_pipeline.py`

**Description:** This module defines a static plan pipeline for the Intergrax framework, which involves deterministic setup outside planner and minimal replanning loop.

**Domain:** RAG logic

**Key Responsibilities:**

* Validates configuration early to prevent planning/execution
* Performs deterministic setup using `RuntimeStepRunner` and `SETUP_STEPS`
* Builds components from config/state, including EnginePlanner, StepPlanner, and StepExecutor
* Plans and executes with minimal replanning loop (STATIC)
* Interprets stop_reason and returns runtime answer accordingly

### `intergrax\runtime\nexus\planning\__init__.py`

DESCRIPTION: This module is responsible for initializing and configuring the knowledge mode planning system in Intergrax.

DOMAIN: Knowledge Mode Planning

KEY RESPONSIBILITIES:
- Initializes the knowledge graph for planning.
- Configures planning parameters and settings.
- Sets up the planning engine for execution.

### `intergrax\runtime\nexus\planning\engine_plan_models.py`

Description: This module defines data structures and functionality for planning in the Intergrax framework, specifically for the nexus Runtime.

Domain: Engine Planning

Key Responsibilities:
- Defines `EnginePlan` data class to represent a plan with version, intent, and optional attributes for clarify questions, next steps, and soft preferences.
- Implements fingerprinting to generate a stable hash of the decision part of an `EnginePlan`.
- Provides methods for printing plans in a human-readable format using Pretty Print (pprint).
- Defines `PlannerPromptConfig` data class to configure planner prompts with version, system prompt, replan system prompt, next step rules prompt, and fallback clarify question.
- Includes default system prompts for the planner.

### `intergrax\runtime\nexus\planning\engine_planner.py`

**Description:** This module provides an LLM-based planner that outputs a typed `EnginePlan`. The planner takes in various inputs, including the LLM adapter, runtime state, and configuration, to generate a plan based on the model's output.

**Domain:** RAG logic

**Key Responsibilities:**

*   Provides an LLM-based planner for generating plans
*   Takes in various inputs, including the LLM adapter, runtime state, and configuration
*   Generates a plan based on the model's output, which is then validated against the capabilities of the current runtime
*   Allows for customization of prompts through `PlannerPromptConfig` objects

The file appears to be stable and production-ready.

### `intergrax\runtime\nexus\planning\plan_builder_helper.py`

**Description:** This module provides a helper function for building plans in nexus mode, utilizing the EnginePlanner to create an EnginePlan object.

**Domain:** RAG logic / Planning

**Key Responsibilities:**
- Initializes EnginePlanner with a specified LLM adapter
- Creates RuntimeRequest and RuntimeState objects for planning
- Configures and registers LLM usage tracker and other capabilities (RAG, user LTM, attachments, websearch, tools)
- Calls EnginePlanner.plan() to generate an EnginePlan object
- Returns the generated EnginePlan

### `intergrax\runtime\nexus\planning\plan_loop_controller.py`

**Description:** This module implements a production-grade plan execution controller for the Intergrax framework, handling bounded replanning with structured feedback and escalation strategy when stuck.

**Domain:** Planning

**Key Responsibilities:**
- Manages replanning loop limits
- Interprets stop reasons (HITL/REPLAN/FAILED/COMPLETED)
- Escalates to HITL when stuck
- Supports static mode execution with bounded replanning
- Integrates with EnginePlanner, StepPlanner, and StepExecutor for plan building and execution

**Note:** The file appears to be a key component of the Intergrax framework's planning module, implementing a critical functionality for production-grade performance.

### `intergrax\runtime\nexus\planning\runtime_step_handlers.py`

**Description:** This module provides utilities and adapters for integrating `RuntimeStep` instances with the step execution framework.

**Domain:** Planning/RAG logic

**Key Responsibilities:**

* Providing result helpers (`_ok_result` and `_failed_result`) to create deterministic StepExecutionResult objects.
* Defining a `RuntimeStepBinding` data class to represent an explicit binding between a StepAction and a factory that returns a RuntimeStep instance.
* Implementing the `make_runtime_step_handler` function, which adapts a RuntimeStep instance into a StepHandler by executing its `run` method with the provided state.
* Creating a `build_runtime_step_registry` function that takes a dictionary of bindings between StepActions and factories, and returns a StepHandlerRegistry object.

### `intergrax\runtime\nexus\planning\step_executor.py`

**Description:** This file defines a class `StepExecutor` responsible for executing an `ExecutionPlan` deterministically by using injected handlers and maintaining a store of step execution results.

**Domain:** RAG logic

**Key Responsibilities:**

* Initializes the executor with a handler registry and configuration (optional)
* Executes the plan's steps in order, handling dependencies and replanning as needed
* Maintains a store of step execution results for dependency resolution
* Stops plan execution if a clarifying question is requested or if a failure policy is triggered

### `intergrax\runtime\nexus\planning\step_executor_models.py`

**Description:** This module contains data structures and classes for planning in Intergrax, focusing on replanning, step execution, and error handling.

**Domain:** RAG logic

**Key Responsibilities:**

- Define enumerations for step status, replan codes, and plan stop reasons.
- Implement dataclasses for user input requests, failed steps, replan contexts, step errors, and plan execution reports.
- Establish protocols for step executor configurations and context execution.
- Introduce exceptions for step replan requests.

### `intergrax\runtime\nexus\planning\step_planner.py`

**Description:** 
The step planner module provides a deterministic planning mechanism to build an execution plan from user input and engine hints. It supports various modes, including static and dynamic planning.

**Domain:** RAG logic / Planning

**Key Responsibilities:**

*   Build an execution plan based on user message and engine hints
*   Support static and dynamic planning modes
*   Handle different step types (web search, tools, rag, synthesis, finalize)
*   Determine the intent of the plan (clarify, freshness, generic)
*   Ensure deterministic sequential execution for pre-steps
*   Generate a clarifying question if needed
*   Adapt engine hints from an engine plan to Intergrax format

**Note:** The module appears to be a key component of the Intergrax framework's planning mechanism and is likely intended for production use.

### `intergrax\runtime\nexus\planning\stepplan_models.py`

**Description:** This module defines a set of shared models and enums for planning steps in the Intergrax framework. These models represent parameters and budgets associated with various actions that can be taken during the planning process.

**Domain:** Planning

**Key Responsibilities:**

* Define enumerations for step actions, failure policies, web search strategies, output formats, plan modes, and other related concepts.
* Provide shared small models for:
	+ Failure policies
	+ Step budgets (top-k, max characters, etc.)
	+ Plan budgets (max total steps, tool calls, web queries, etc.)
	+ Stop conditions (max iterations, no progress, etc.)
	+ Verify criteria (id, description, severity)
* Define parameter models for specific actions:
	+ Ask Clarifying Question
	+ User Longterm Memory Search
	+ Attachments Retrieval
	+ RAG Retrieval
	+ Web Search
	+ Tools
	+ Synthesize Draft
	+ Verify Answer
	+ Finalize Answer
* Establish mappings between step actions and their corresponding parameter models.
* Define the ExecutionStep model, which represents a single planning step with associated properties (step ID, action, enabled status, dependencies, budgets, inputs, expected output type, rationale type, and on-failure policy).

### `intergrax\runtime\nexus\prompts\__init__.py`

Description: This file initializes the nexus mode prompts for Intergrax runtime.

Domain: LLM adapters

Key Responsibilities:
• Initializes prompt definitions for nexus mode
• Provides a standardized way to generate and manage knowledge-based prompts
• Integrates with other Intergrax components to enable seamless knowledge acquisition

### `intergrax\runtime\nexus\prompts\history_prompt_builder.py`

**Description:** This module provides functionality for building history-summary-related prompts in Intergrax's nexus Mode.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a strategy interface (`HistorySummaryPromptBuilder`) for building history-summary-related parts of the prompt.
* Offers a default implementation (`DefaultHistorySummaryPromptBuilder`) that generates a safe, generic system prompt for summarizing older conversation turns.
* Allows customization of the prompt through extension or modification of the `HistorySummaryPromptBuilder` protocol.

### `intergrax\runtime\nexus\prompts\rag_prompt_builder.py`

**Description:** This module provides a framework for building RAG-related prompts in the Intergrax nexus mode.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Provides a protocol (`RagPromptBuilder`) for building RAG-related prompt elements
* Offers a default implementation (`DefaultRagPromptBuilder`) that injects retrieved chunks into system-level context messages
* Allows customization of the prompt building process through the `RagPromptBuilder` interface
* Includes utility functions (e.g., `_format_rag_context`) for formatting retrieved chunk text into model-friendly format

Note: The file appears to be a self-contained module with a clear purpose and functionality, indicating it is not experimental or auxiliary.

### `intergrax\runtime\nexus\prompts\user_longterm_memory_prompt_builder.py`

**Description:** This module provides a prompt builder for injecting user long-term memory into large language models (LLMs) during knowledge retrieval in Intergrax.

**Domain:** LLM adapters

**Key Responsibilities:**
- Builds prompt messages to inject retrieved user long-term memory into the LLM context.
- Provides a default deterministic prompt builder with customizable output strategies.
- Handles filtering, limiting, and formatting of long-term memory entries for safe and compact injection into LLMs.

### `intergrax\runtime\nexus\prompts\websearch_prompt_builder.py`

**Description:** This module provides a prompt builder for web search results in nexus Mode.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Provides a strategy interface (`WebSearchPromptBuilder`) for building the web search part of the prompt
* Defines a default implementation (`DefaultWebSearchPromptBuilder`) that delegates to a websearch module context generator
* Builds a `WebSearchPromptBundle` containing system messages and debug information from web search results

### `intergrax\runtime\nexus\responses\__init__.py`

Description: The __init__.py file in the responses directory of the nexus mode is a package initializer for Python, responsible for setting up the necessary imports and configurations.

Domain: RAG logic

Key Responsibilities:
- Imports necessary components and modules for the response handling
- Defines the interface for generating responses in knowledge mode
- Sets up the configuration and environment for response generation

### `intergrax\runtime\nexus\responses\response_schema.py`

**Description:** This module defines data models for the nexus Mode runtime in Intergrax, which provides a high-level contract between applications and the RuntimeEngine. These dataclasses expose citations, routing information, tool calls, and basic statistics while hiding low-level implementation details.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Define data models for request and response structures:
	+ `RuntimeRequest`: High-level request structure containing user ID, session ID, message, attachments, metadata, instructions, history compression strategy, and maximum output tokens.
	+ `RuntimeAnswer`: High-level response structure containing final answer, citations, routing information, tool call summaries, basic statistics, and optional raw model output or intermediate artifacts.
* Define dataclasses for specific components:
	+ `Citation`: Represents a single citation/reference used in the final answer.
	+ `RouteInfo`: Describes how the runtime decided to answer the question.
	+ `ToolCallInfo`: Describes a single tool call executed during the runtime request.
	+ `RuntimeStats`: Basic statistics about a runtime call.
* Define an enumeration for history compression strategies:
	+ `HistoryCompressionStrategy`: Strategy for compressing conversation history before sending it to the LLM.

### `intergrax\runtime\nexus\runtime_steps\__init__.py`

Description: This module defines the initialization steps for Intergrax's knowledge graph when operating in drop-in mode.

Domain: nexus Mode

Key Responsibilities:
- Initializes the knowledge graph and related components
- Defines the setup process for drop-in operations
- Provides a hook for custom initialization routines

### `intergrax\runtime\nexus\runtime_steps\build_base_history_step.py`

Description: This module implements a runtime step responsible for building a base history in the Intergrax framework.

Domain: nexus Mode, Runtime Steps

Key Responsibilities:
- Builds base history using HistoryLayer
- Utilizes RuntimeState and HistoryLayer context to construct session (project/user/system seed) history

### `intergrax\runtime\nexus\runtime_steps\contract.py`

Description: This module defines the contract for runtime steps in Intergrax's nexus mode and provides a runner class to execute these steps.

Domain: Runtime Pipeline Execution

Key Responsibilities:
- Defines the `RuntimeStep` protocol with a `run` method.
- Implements `RuntimeStepRunner` to execute a list of steps asynchronously.
- Tracing events for each step execution.

### `intergrax\runtime\nexus\runtime_steps\core_llm_step.py`

**Description:** This module provides the CoreLLMStep class, responsible for interacting with the core LLM (Large Language Model) adapter and generating a final answer.

**Domain:** LLM adapters

**Key Responsibilities:**

* Calls the core LLM adapter to generate an answer
* Falls back to tools_agent_answer if available or fails to get a response from the LLM adapter
* Traces events and errors during the execution process

### `intergrax\runtime\nexus\runtime_steps\ensure_current_user_message_step.py`

**Description:** This module implements a runtime step for ensuring the current user's message is present as the last prompt message in the knowledge mode of the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Ensures the current user's message is appended to the list of messages for the LLM if it does not already exist or is different from the previous user's message
* If the request message is empty, does nothing
* If the list of messages for the LLM is empty, adds the current user's message
* Maintains user-last semantics by appending the current user's message to the end of the list of messages

### `intergrax\runtime\nexus\runtime_steps\history_step.py`

**Description:** This module, `HistoryStep`, is responsible for building conversation history for the Language Model (LLM) in Intergrax's nexus mode. It selects and shapes the conversational context from previous user and assistant turns.

**Domain:** RAG logic / Conversation History Management

**Key Responsibilities:**

* Build conversation history for the LLM
* Select and shape the conversational context from previous user and assistant turns
* Delegate history shaping to `ContextBuilder` if available
* Handle base history as-is in case of no additional history layer
* Trace history building step for debugging purposes

### `intergrax\runtime\nexus\runtime_steps\instructions_step.py`

**Description:** This module is responsible for injecting the final instructions into the LLM prompt as a system message, combining per-request instructions, user profile instructions, and organization profile instructions.

**Domain:** nexus Mode

**Key Responsibilities:**

* Combines per-request instructions, user profile instructions, and organization profile instructions
* Injects the combined instructions as the first system message in the LLM prompt
* Ensures that instructions are not persisted in SessionStore
* Prepends the system message to the messages for the LLM
* Trims/summarizes history before injection (must be called AFTER HistoryStep)

Note: This file appears to be a critical component of the nexus mode functionality, and its code is well-structured and documented. There are no indications that this file is experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\nexus\runtime_steps\persist_and_build_answer_step.py`

**Description:** This module implements a step in the Intergrax framework's nexus mode, responsible for persisting the assistant's message into the session and building a RuntimeAnswer object.

**Domain:** LLM adapters

**Key Responsibilities:**

* Persist the assistant's message into the session
* Build a RuntimeAnswer object including RouteInfo and RuntimeStats
* Determine the strategy label based on used components (RAG, websearch, tools)
* Handle tool calls for answer construction

### `intergrax\runtime\nexus\runtime_steps\profile_based_memory_step.py`

Description: This module defines a step in the Intergrax framework's knowledge mode, responsible for loading profile-based instruction fragments for a given request.

Domain: Knowledge Mode Planning

Key Responsibilities:
- Load user and organization profile memory if enabled in RuntimeConfig
- Extract prebuilt 'system_prompt' strings from profile bundles
- Store resulting fragments in RuntimeState for later merging into system messages
- Set debug information about the memory layer step
- Record a trace event to indicate that the memory layer has been processed

### `intergrax\runtime\nexus\runtime_steps\rag_step.py`

Description: This module defines a runtime step for building the Retrieval-Augmented Generator (RAG) layer in the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
- Ensure ContextBuilder result exists, or build it if missing
- Build RAG context messages from retrieved chunks using RagPromptBuilder
- Inject RAG context messages before the last user message
- Prepare compact RAG text for tools agent (state.tools_context_parts)
- Set debug fields and trace event

### `intergrax\runtime\nexus\runtime_steps\retrieve_attachments_step.py`

Description: This module provides a runtime step for retrieving relevant chunks from session-ingested attachments and injecting them into the LLM context.

Domain: RAG logic / nexus mode

Key Responsibilities:
- Retrieve relevant chunks from session-ingested attachments using the ingestion service.
- Inject the retrieved chunks into the LLM context as system messages.
- Provide a textual form of the injected context for tools agents.
- Record the execution of this step in the trace event log.

Note: This file appears to be part of an experimental or advanced feature (nexus mode) and is not marked as legacy or incomplete.

### `intergrax\runtime\nexus\runtime_steps\session_and_ingest_step.py`

**Description:** This module defines a runtime step in the Intergrax framework that loads or creates a session, ingests attachments (RAG), and appends user messages to the session history.

**Domain:** nexus Mode

**Key Responsibilities:**
* Load or create a session based on the provided session ID
* Ingest attachments into a vector store for the current session
* Append user messages to the session history
* Initialize debug trace with session and ingestion metadata
* Update the runtime state with ingested results and debug trace

### `intergrax\runtime\nexus\runtime_steps\setup_steps_tool.py`

Description: This module defines a collection of setup steps that can be executed in knowledge mode, encapsulating the necessary initialization and configuration for a successful run.

Domain: Knowledge Mode Setup

Key Responsibilities:
- Initializes key components such as session, ingest, memory, history, and instructions
- Ensures current user message is set up correctly
- Builds base history step to initialize relevant data structures

### `intergrax\runtime\nexus\runtime_steps\tools.py`

**Description:** This module provides tools for handling knowledge mode in the Intergrax framework, including inserting context messages and formatting RAG (Retriever-Generator) context.

**Domain:** Knowledge Mode Tools

**Key Responsibilities:**

* Inserting context messages before last user message
	+ `insert_context_before_last_user` function
* Formatting RAG context for display
	+ `format_rag_context` function
* Utility functions for extracting text and metadata from chunk objects
	+ `_chunk_text` and `_chunk_meta` functions

**Note:** The code appears to be a part of the main Intergrax framework, with no signs of being experimental, auxiliary, or legacy.

### `intergrax\runtime\nexus\runtime_steps\tools_step.py`

**Description:** This module provides a runtime step for the Intergrax framework, allowing tools to be executed and their results integrated into the conversation. The tools agent can be configured to run in different modes.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Runs the tools agent (planning + tool calls) if configured
* Stores output from tools execution in RuntimeState:
	+ `state.used_tools`
	+ `state.tool_traces`
	+ `state.tools_agent_answer`
* Decides whether to reuse tools agent answer or use it as system context for core LLM
* Injects executed tool calls as system context for core LLM when necessary

The file appears to be part of the main Intergrax framework implementation, and its functionality seems complete.

### `intergrax\runtime\nexus\runtime_steps\user_longterm_memory_step.py`

**Description:** This module implements a runtime step for the Intergrax framework, specifically handling user long-term memory retrieval and injection as context messages.

**Domain:** RAG logic

**Key Responsibilities:**

* Retrieves user long-term memory (LTM) using either cached results or session manager search
* Injects LTM as context messages into the conversation flow
* Manages debug tracing and logging for LTM-related events
* Supports configuration options for enabling/disabling LTM and customizing retrieval parameters

### `intergrax\runtime\nexus\runtime_steps\websearch_step.py`

**Description:** This module provides a runtime step for executing web search and injecting its context into the LLM prompt, as part of the Intergrax framework's nexus mode.

**Domain:** RAG logic (Runtime Agents and Generators)

**Key Responsibilities:**

* Execute websearch using state.messages_for_llm and tools context
* Build web context messages with websearch_prompt_builder
* Inject web context messages before the last user message
* Append compact web context text into state.tools_context_parts (for tools agent)
* Debug and tracing

### `intergrax\runtime\nexus\session\__init__.py`

Description: Initializes a knowledge session for Intergrax's drop-in mode, managing the flow of external knowledge into the system.

Domain: RAG logic

Key Responsibilities:
- Initializes the session state
- Sets up event listeners for knowledge injection and processing
- Establishes connections to relevant components (e.g., knowledge bases)

### `intergrax\runtime\nexus\session\chat_session.py`

**Description:** This module defines a domain model for chat sessions within the Intergrax framework, encapsulating session metadata and state.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Defines `SessionStatus` and `SessionCloseReason` enumerations to represent session lifecycle states
* Introduces the `ChatSession` data class, which serves as a domain model for chat sessions
	+ Stores session metadata, such as user, tenant, and workspace identifiers, timestamps, and attachments
	+ Maintains core domain state (session status and closed reason)
	+ Tracks user turns and consolidation-related state
* Provides methods to interact with the `ChatSession` object:
	+ `touch`: refreshes modification timestamp without persisting changes
	+ `is_closed`: checks whether the session is marked as closed at the domain level
	+ `mark_closed`: marks the session as closed, optionally specifying a reason
	+ `increment_user_turns`: increments and returns the per-session counter of user turns

This module appears to be part of the Intergrax framework's core logic, likely used in conjunction with other components for managing chat sessions.

### `intergrax\runtime\nexus\session\in_memory_session_storage.py`

**Description:** This module provides an in-memory implementation of SessionStorage for storing and retrieving chat session metadata and conversation history.

**Domain:** Runtime / Session Management

**Key Responsibilities:**

* Store and manage chat session metadata (e.g., user ID, tenant ID, workspace ID) in an in-process dictionary.
* Maintain per-session conversation history using ConversationalMemory, applying a simple FIFO trimming policy via max_messages setting.
* Provide methods for creating, saving, getting, listing, and appending messages to sessions.
* Support retrieval of conversation history for a given session.

**Note:** This implementation is suitable for development, testing, or single-process setups but not intended for production use in distributed environments.

### `intergrax\runtime\nexus\session\session_manager.py`

Description: This module implements the high-level management of chat sessions, including creation, persistence, retrieval, and closure. It provides a stable API for the runtime engine and integrates with profile managers to expose system instructions per session.

Domain: Session Management

Key Responsibilities:
- Orchestrate session lifecycle on top of a SessionStorage backend.
- Provide a stable API for the runtime engine (RuntimeEngine).
- Integrate with user/organization profile managers to expose prompt-ready system instructions per session.
- Optionally trigger long-term user memory consolidation for a session.
- Retrieve and store chat session metadata, including history, creation, and closure events.

### `intergrax\runtime\nexus\session\session_storage.py`

**Description:** This module defines a low-level storage interface for chat sessions and conversation histories, providing operations for persisting and loading session metadata and conversation data.

**Domain:** Session Storage

**Key Responsibilities:**

* Persist and load ChatSession objects
* Persist and load conversation history (ChatMessage sequences) for a given session
* Provide CRUD operations for session metadata
	+ Get session by ID
	+ Create new session with optional metadata
	+ Save changes to an existing session
	+ List recent sessions for a user
* Append single messages to the conversation history of a session
* Retrieve the ordered conversation history for a given session

**Note:** This file appears to be part of the Intergrax framework's core functionality, and its interface is intentionally minimal. The SessionManager layer built on top of this storage is responsible for higher-level domain logic such as profile instructions, memory synthesis, and counters.

### `intergrax\runtime\organization\__init__.py`

**Note:** The content provided is truncated. However, I can provide a possible interpretation based on the given file path and the context of Intergrax framework.

Description: This module serves as the entry point for the organization runtime in Intergrax, providing necessary initialization and setup for organizational components.

Domain: Organization Runtime

Key Responsibilities:
- Initializes organizational structures and dependencies.
- Sets up relevant runtime configurations.
- Provides access to organizational functionality.

### `intergrax\runtime\organization\organization_profile.py`

**Description:** This module defines data classes and structures for representing organization profiles in the Intergrax framework. It provides stable identification data, preferences, system instructions, and legacy summary fields.

**Domain:** Organization Profiles (RAG logic)

**Key Responsibilities:**

* Define OrganizationIdentity class with attributes for stable identification
* Define OrganizationPreferences class with attributes for output/communication preferences, runtime capabilities, sensitive topics, hard constraints, and soft guidelines
* Define OrganizationProfile class as the single source of truth for an organization's long-term profile, containing identity, preferences, system instructions, and legacy summary fields

### `intergrax\runtime\organization\organization_profile_instructions_service.py`

**Description:** This module provides a service for generating and updating organization-level system instructions using an LLM (Large Language Model) adapter.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Load organization profile via OrganizationProfileManager
* Build an LLM prompt from identity, preferences, domain/knowledge summaries, and memory entries
* Call LLMAdapter to generate compact system instructions
* Persist the result via OrganizationProfileManager update_system_instructions()
* Handle regeneration of instructions based on configuration and existing values

### `intergrax\runtime\organization\organization_profile_manager.py`

**Description:** This module provides a high-level facade for working with organization profiles, abstracting away direct interactions with the underlying store.

**Domain:** Organization Profile Management

**Key Responsibilities:**

* Load an OrganizationProfile for a given organization_id
* Persist profile changes
* Resolve organization-level system instructions string for use in the runtime
* Hide direct interaction with the underlying OrganizationProfileStore
* Provide convenient methods for managing profiles and system instructions

Note: The file appears to be part of a larger framework, and its main functionality is focused on providing a high-level interface for working with organization profiles. It does not perform LLM calls or RAG operations directly.

### `intergrax\runtime\organization\organization_profile_store.py`

**Description:** This module provides a protocol interface for persistent storage of organization profiles in the Integrax framework.

**Domain:** Organization Storage

**Key Responsibilities:**
- Load and save organization profiles as aggregates
- Provide default values for new organizations
- Hide backend-specific concerns (e.g., JSON files, SQL DB)
- Support getting, saving, and deleting profiles asynchronously

### `intergrax\runtime\organization\stores\__init__.py`

Description: This module initializes the organization stores, responsible for handling and persisting organizational data within the Intergrax framework.

Domain: Organization Stores

Key Responsibilities:
• Initializes the organization store instances
• Configures the storage mechanisms (e.g., databases, file systems) 
• Sets up the data persistence and retrieval logic

### `intergrax\runtime\organization\stores\in_memory_organization_profile_store.py`

Description: This module implements an in-memory store for organization profiles within the Integrax framework.

Domain: In-Memory Organization Store

Key Responsibilities:
- Provides an implementation of `OrganizationProfileStore` for unit testing and local development.
- Stores organization profiles in memory, with no durability or cross-process sharing capabilities.
- Supports basic CRUD operations (get, save, delete) on organization profiles.
- Includes a helper function to list stored organization IDs for debugging purposes.

### `intergrax\runtime\user_profile\__init__.py`

**intergrax/runtime/user_profile/__init__.py**

Description: Initializes and configures the user profile management system, providing a foundation for storing and retrieving user-specific data.

Domain: User Profile Management

Key Responsibilities:
- Imports necessary modules for user profile handling
- Defines the user profile class and its attributes
- Establishes connections to storage services (e.g., databases)

### `intergrax\runtime\user_profile\session_memory_consolidation_service.py`

**Description:** This module provides a service for consolidating chat session history into structured long-term memory entries for the user profile.

**Domain:** RAG logic

**Key Responsibilities:**

- Take a single chat session history and convert it into structured long-term memory entries
- Ask the LLM to extract USER_FACT, PREFERENCE, and optional SESSION_SUMMARY items from the conversation
- Map extracted data into UserProfileMemoryEntry objects
- Persist them through UserProfileManager
- Optionally refresh user-level system instructions as part of the same pipeline

**Note:** The file appears to be a complete implementation, with clear responsibilities and a well-defined interface.

### `intergrax\runtime\user_profile\user_profile_debug_service.py`

**Description:** This module provides a read-only service to build debug snapshots of user profiles, aggregating data from the UserProfileManager and SessionManager.

**Domain:** User Profile Management

**Key Responsibilities:**

* Building debug snapshots for given users
* Aggregating data from UserProfileManager (identity, preferences, memory, system instructions) and SessionManager (recent ChatSession metadata)
* Exposing a "debug user profile" API endpoint
* Feeding an admin / developer UI panel
* Ad-hoc diagnostics during development

### `intergrax\runtime\user_profile\user_profile_debug_snapshot.py`

**Description:** This module provides data classes and utility functions for creating debug snapshots of user profiles, including recent sessions and memory entries.

**Domain:** Debugging/Observability Tools

**Key Responsibilities:**

* Providing a lightweight view of a single ChatSession (`SessionDebugView`)
* Exposing relevant fields from `UserProfileMemoryEntry` for inspection (`MemoryEntryDebugView`)
* Creating an immutable snapshot of user profile state for debugging purposes (`UserProfileDebugSnapshot`)
* Calculating memory entry counters by kind (`build_memory_kind_counters` method)
* Converting domain objects to debug views (`from_domain_session` and `from_memory_entry` methods)

### `intergrax\runtime\user_profile\user_profile_instructions_service.py`

**Description:** This module provides a high-level service to generate and update user-level system instructions using an LLM adapter, based on a user's profile.

**Domain:** RAG (Reactive AI Gateway) logic

**Key Responsibilities:**

* Load UserProfile via UserProfileManager
* Build an LLM prompt using identity, preferences, and memory entries
* Call LLMAdapter to generate compact system instructions
* Persist the result via UserProfileManager.update_system_instructions()
* Handle regeneration of instructions based on configuration settings

### `intergrax\supervisor\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the supervisor component, responsible for managing and coordinating various Intergrax systems.

DOMAIN: Supervisor component

KEY RESPONSIBILITIES:
• Initializes and sets up the supervisor instance
• Defines interfaces and APIs for interacting with other components
• Manages the lifecycle of dependent services and systems

### `intergrax\supervisor\supervisor.py`

**Description:** This module provides a supervisor component for the Intergrax framework, responsible for planning and executing tasks using Large Language Models (LLMs). It enables two-stage planning: decomposing tasks into steps, assigning each step to an appropriate model or component, and re-executing if necessary.

**Domain:** Supervisor & RAG logic

**Key Responsibilities:**

* Planning tasks through LLMs
* Decomposing tasks into individual steps
* Assigning each step to a suitable model or component
* Two-stage planning: decompose-assign-recompute
* Executing plan using supervisor components and LLMs
* Handling fallback and semantic fallback scenarios
* Providing analysis of generated plans
* Managing configuration for the supervisor

**Note:** This file appears to be well-maintained, up-to-date, and production-ready. It is not marked as experimental, auxiliary, legacy, or incomplete.

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides a framework for defining and managing components in the Integrax pipeline, allowing for flexible and reusable step implementations.

**Domain:** Supervisor logic / Component management

**Key Responsibilities:**

- Defines a `Component` class representing individual steps with metadata and implementation functions
- Provides a `run` method to execute component functions with context and state
- Introduces `ComponentContext` and `PipelineState` dataclasses for encapsulating relevant information
- Offers a decorator (`component`) for convenient registration of new components

### `intergrax\supervisor\supervisor_prompts.py`

Description: This module provides default prompt templates and constraints for the Intergrax framework's unified Supervisor, ensuring a structured approach to generating plans.

Domain: RAG logic / Plan generation

Key Responsibilities:
- Defines default plan system and user templates as strings.
- Provides the `SupervisorPromptPack` class with dataclass attributes for plan system and user template, initialized with default values.
- Specifies constraints for plan generation, including decomposition-first mandate, component selection policy, output→method guards, component output contracts, strict rules, and validation checklist.

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module provides utilities and functions to transform a Plan into a runnable LangGraph pipeline. It handles state management, node creation, and graph building.

**Domain:** LangGraph pipeline construction

**Key Responsibilities:**

* State schema definition (PipelineState)
* Utility functions for state management (_ensure_state_defaults, _append_log, _resolve_inputs, _persist_outputs)
* Node factory function (make_node_fn) that executes a plan step
* Topological ordering of steps to build a stable graph (topo_order)
* Building the LangGraph pipeline from a Plan (build_langgraph_from_plan)

Note: The code appears to be well-structured and complete. There are no obvious signs of experimental, auxiliary, legacy, or incomplete status.

### `intergrax\system_prompts.py`

Description: This module defines a strict RAG system instruction for answering user questions based on document content.

Domain: LLM adapters/RAG logic

Key Responsibilities:
- Defines the RAG (Role and Accountability) system instruction for strict adherence to document-based knowledge.
- Specifies procedures for understanding, searching, verifying, and responding to user queries.
- Outlines guidelines for citing sources, handling ambiguity, and maintaining a neutral tone.
- Provides formatting recommendations for response structure and content.

### `intergrax\tools\__init__.py`

**intergrax\tools\__init__.py**

Description: The entry point for the Intergrax tools module, responsible for initializing and exposing utility functions to other parts of the framework.

Domain: Configuration / Utility Modules

Key Responsibilities:
- Initialize and register tool submodules
- Provide access to shared utility functions and constants

### `intergrax\tools\tools_agent.py`

**Description:** This module implements the ToolsAgent class, which is responsible for orchestrating tool execution and interacting with a Large Language Model (LLM) to provide answers. The agent supports both native tools (OpenAI) and a JSON planner (Ollama).

**Domain:** LLM adapters / Agents

**Key Responsibilities:**

* Initializes the ToolsAgent instance with an LLM adapter, tool registry, and optional memory and configuration
* Supports native tools (OpenAI) or a JSON planner (Ollama)
* Prunes messages for OpenAI to ensure tool calls appear only after the last assistant message with tool calls
* Builds output structure by preferring tools over extracted JSON from answer text
* Provides public API for running tools orchestration, including input data handling and context injection
* Tracks LLM usage through an optional tracker

Note: The code appears to be well-structured and commented. There is no indication of it being experimental, auxiliary, legacy, or incomplete.

### `intergrax\tools\tools_base.py`

**Description:** This module provides base classes and utilities for tool development within the Intergrax framework.

**Domain:** Utility modules

**Key Responsibilities:**

* Provides a `ToolBase` class serving as a foundation for all tools, with attributes like name, description, and schema model.
* Offers methods to derive JSON Schema for 'parameters' from Pydantic models.
* Enables tool registration through the `ToolRegistry` class.
* Allows for validation of arguments using Pydantic (if schema model is available).
* Exports tools to a format compatible with the OpenAI Responses API.

Note: The file appears to be a critical part of the Intergrax framework, providing essential functionality for tool development and management.

### `intergrax\websearch\__init__.py`

Description: Initializes the web search functionality within the Intergrax framework, setting up necessary components for interacting with external web services.

Domain: RAG logic

Key Responsibilities:
* Registers web search modules
* Initializes web search configuration
* Sets up data processing pipelines for web search results

### `intergrax\websearch\cache\__init__.py`

Description: This module initializes the cache for web search functionality within Intergrax.

Domain: Web Search Cache Initialization

Key Responsibilities:
• Initializes a cache system for storing and retrieving web search results
• Defines cache expiration policies and eviction strategies 
• Possibly integrates with external caching mechanisms (e.g., Redis)

### `intergrax\websearch\cache\query_cache.py`

**Description:** This module provides a simple in-memory query cache for web search results. It allows storing cached values for given query keys and has optional TTL (time-to-live) and max size configurations.

**Domain:** Web Search Cache Management

**Key Responsibilities:**
* Provides `QueryCacheKey` class to create canonical cache keys from web search configuration parameters.
* Offers `InMemoryQueryCache` class for simple in-memory caching with optional TTL and max size settings.
* Implements methods for getting, setting, and clearing cached values.

### `intergrax\websearch\context\__init__.py`

DESCRIPTION: This module defines the initial context for web search-related functionality within Intergrax.

DOMAIN: Web Search Context Setup

KEY RESPONSIBILITIES:
• Initializes the web search context instance
• Sets up default configuration for web search operations
• Registers required dependencies and services for web search functionality

### `intergrax\websearch\context\websearch_context_builder.py`

Description: This module builds LLM-ready textual context and chat messages from web search results, adhering to strict "sources-only" mode rules.

Domain: RAG logic

Key Responsibilities:
- Builds a textual context string from typed WebSearchResult objects.
- Constructs system prompts that enforce the use of only web sources, no hallucinations, single concise answers, and explicit handling of missing information.
- Builds user-facing prompts that wrap web sources, questions, and concrete tasks.
- Returns chat messages (system + user) for chat-style LLMs from typed web search results.

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: This module serves as the entry point for the web search fetcher component, responsible for initializing and configuring its functionality.

DOMAIN: Web Search Fetcher

KEY RESPONSIBILITIES:
- Initializes the fetcher's core components
- Sets up configuration settings for web search operations
- Defines interfaces for integration with other Intergrax components

### `intergrax\websearch\fetcher\extractor.py`

Description: This module provides utilities for web content extraction and analysis, including lightweight and advanced readability-based extraction.

Domain: Web Search Extraction Utilities

Key Responsibilities:
- Lightweight HTML extraction with basic metadata extraction.
- Advanced readability-based extraction using trafilatura (if installed) or BeautifulSoup fallback.
- Text normalization and formatting adjustments.
- Metadata attachment for debugging and observability purposes.

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module provides an asynchronous HTTP client for fetching web pages and extracting their content. It allows for customized headers, timeout, and redirection handling.

**Domain:** Web Search Fetcher (HTTP Client)

**Key Responsibilities:**
* Perform an asynchronous HTTP GET request with sane defaults
* Capture final URL, status code, raw HTML, and body size
* Keep higher-level concerns (robots, throttling, extraction) outside the client module

### `intergrax\websearch\integration\__init__.py`

DESCRIPTION: The __init__.py file in the websearch integration directory serves as the entry point for importing modules within this package, allowing users to easily integrate various search functionalities with Intergrax.

DOMAIN: Integration Module

KEY RESPONSIBILITIES:
- Exports classes and functions that facilitate interaction between the user interface and external search APIs.
- Serves as the primary import point for websearch integration-related modules.
- Enables modular design by providing a central hub for importing and managing related functionality.

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a web search node implementation for the Intergrax framework, enabling integration with external web search services. It encapsulates configuration and delegate search operations to a WebSearchExecutor instance.

**Domain:** RAG logic (Reinforcement Learning Augmented Graphs) / Web Search Integration

**Key Responsibilities:**

* Provides a LangGraph-compatible web search node wrapper (WebSearchNode class)
	+ Encapsulates configuration of WebSearchExecutor
	+ Implements synchronous and asynchronous node methods operating on WebSearchState
	+ Delegates search logic to the provided WebSearchExecutor instance
* Offers a default, module-level node instance for convenience and backward compatibility
* Provides functional wrappers around the default WebSearchNode instance (websearch_node and websearch_node_async functions)
	+ Suitable for simple integrations with custom configuration or async environments

**Status:** This file appears to be production-ready, with a clear and well-structured implementation. However, it might benefit from additional documentation on usage examples and potential edge cases.

### `intergrax\websearch\pipeline\__init__.py`

Description: Initializes the web search pipeline, providing a standardized interface for executing search queries and processing results.

Domain: Web Search Pipeline

Key Responsibilities:
• Initializes pipeline components
• Defines search query execution flow
• Sets up result processing and output mechanisms
• Ensures compatibility with various data sources and formats

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module implements a web search pipeline that orchestrates multi-provider searching, fetching, extraction, deduplication, and basic quality scoring. It aims to be provider-agnostic, async-friendly with rate limiting, and minimally coupled to the underlying LLM/RAG layers.

**Domain:** Web Search Pipeline (RAG/LLM integration)

**Key Responsibilities:**

* Orchestrates multi-provider web search
* Fetches and extracts search hits into WebDocument objects
* Performs deduplication based on simple dedupe key computation
* Applies basic quality scoring to WebDocuments
* Supports synchronous and asynchronous execution modes

The code appears to be well-structured, comprehensive, and production-ready. There is no indication that it's experimental, auxiliary, legacy, or incomplete.

### `intergrax\websearch\providers\__init__.py`

Description: This is the entry point for web search providers within the Intergrax framework, responsible for initializing and configuring available provider classes.

Domain: Web Search Providers

Key Responsibilities:
* Initializes available web search provider classes
* Configures and registers providers for use in the application
* Provides a standardized interface for interacting with various web search services

### `intergrax\websearch\providers\base.py`

**Description:** This module defines the base interface for web search providers in the Integrax framework, encapsulating provider-agnostic query handling and result ranking.

**Domain:** Web Search Providers

**Key Responsibilities:**
* Accept a provider-agnostic QuerySpec
* Return a ranked list of SearchHit items
* Expose minimal capabilities for feature negotiation (language, freshness)
* Execute a single search request with query validation and sanitization
* Provide capability map for feature negotiation
* Optional resource cleanup (HTTP sessions, clients, caches)

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search provider for the Intergrax framework, enabling search functionality with features like language and region filtering.

**Domain:** LLM adapters

**Key Responsibilities:**
- Provides implementation of the Bing Web Search API (v7) as a REST client.
- Supports query filtering by language and region.
- Offers freshness filtering with options "Day", "Week", or "Month".
- Includes safe search functionality with strict mode enabled by default.
- Utilizes environment variables for API key management.
- Exposes methods for querying, result pagination, and provider metadata.

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module implements a Google Custom Search (CSE) provider for the Intergrax framework, enabling search operations via REST API.

**Domain:** WebSearch Providers

**Key Responsibilities:**

* Initializes a Google CSE provider with optional parameters (API key, CX, session, timeout)
* Builds query parameters for CSE API requests
* Performs search queries to the CSE API and returns results as SearchHit objects
* Extracts metadata from search results, including title, URL, snippet, published time, and source type
* Handles errors and edge cases during search operations

The GoogleCSEProvider class is a concrete implementation of WebSearchProvider, inheriting common methods for building query parameters, searching, and closing the session. It utilizes environment variables for API key and CX values, adhering to Intergrax's configuration approach.

This module appears to be production-ready, with proper error handling and usage of type hints.

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a Google Places API provider for the Intergrax web search framework, allowing text search and details retrieval from Google's business database.

**Domain:** Web Search Providers

**Key Responsibilities:**

* Provides an interface to the Google Places API for text search and details retrieval
* Handles environment variables for API keys and other configuration options
* Exposes methods for building search queries and mapping results to Intergrax's SearchHit format
* Supports language and region filtering, as well as fetching additional place details

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module implements a Reddit API search provider for the Intergrax framework, enabling full-featured searches with rich metadata and optional comment fetching.

**Domain:** LLM adapters / WebSearch providers

**Key Responsibilities:**

* Authenticates using client credentials (OAuth2) to obtain an access token
* Handles search queries, mapping query specifications to Reddit's API endpoints
* Fetches top-level comments for each post, if enabled
* Provides a `capabilities` method for describing the provider's features and limitations
* Offers methods for building and refreshing authentication tokens
* Maps comment data from Reddit's API response to a standardized format

### `intergrax\websearch\schemas\__init__.py`

Description: This file defines the schema for web search queries within the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
* Defines a schema for web search queries
* Exposes attributes and methods for query construction and validation
* Integrates with other modules for query execution and result processing

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass `PageContent` to represent the content of a web page, encapsulating both raw HTML and derived metadata for post-processing stages.

**Domain:** Web Search

**Key Responsibilities:**
- Represents web page content with relevant metadata
- Provides methods for filtering out empty fetches (`has_content`)
- Generates short summaries and approximates content size

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a dataclass for canonical search query specifications used by web search providers.

**Domain:** Query specification schema

**Key Responsibilities:**

* Defines the `QuerySpec` dataclass with attributes for query filtering and metadata
* Provides methods for normalizing queries and capping top results per provider

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a dataclass `SearchHit` to represent metadata for a single search result entry, providing a standardized structure for searching and processing results across various providers.

**Domain:** Search/Query

**Key Responsibilities:**

* Defines the `SearchHit` dataclass with attributes for provider identifier, query string, rank, title, URL, snippet, displayed link, publication datetime, source type, and extra fields.
* Implements minimal safety checks in the `__post_init__` method to enforce valid rank values and ensure URLs have a scheme and netloc.
* Provides methods to extract the domain from the URL (`domain`) and return a minimal, LLM-friendly representation of the hit as a dictionary (`to_minimal_dict`).

### `intergrax\websearch\schemas\web_document.py`

**FILE PATH:** intergrax\websearch\schemas\web_document.py

**Description:** This module defines a unified structure for representing processed web documents, combining provider metadata with extracted content and analysis results.

**Domain:** Web Search

**Key Responsibilities:**

* Provides a dataclass `WebDocument` to store processed web document information
	+ Combines original search hit (provider metadata) with extracted content and analysis results
	+ Includes fields for deduplication, quality scores, and source rank adjustments
* Offers methods to validate documents (`is_valid`) and generate merged text for LLM or retrieval embedding (`merged_text`)
* Allows generating a summary line used in logs or console outputs (`summary_line`)

**Note:** This file appears well-maintained and not experimental or auxiliary.

### `intergrax\websearch\schemas\web_search_answer.py`

**Description:** This module defines a dataclass for storing the result of a Web Search, including the final answer and related context.

**Domain:** LLM adapters/Web Search

**Key Responsibilities:**

* Defines a `WebSearchAnswer` dataclass to represent search results
* Includes fields for the final model answer, LLM-ready messages, and typed web search results
* Leverages type hinting and dataclasses for structured data representation

### `intergrax\websearch\schemas\web_search_result.py`

Description: This module defines a data class for representing search results from the web, encapsulating relevant metadata and content.

Domain: Web Search Schemas

Key Responsibilities:
- Defines the `WebSearchResult` data class with frozen attributes.
- Includes fields for provider, rank, quality score, title, URL, snippet, description, language, domain, publication date, fetch date, and associated document.

### `intergrax\websearch\service\__init__.py`

DESCRIPTION: This module serves as the entry point for the web search service, responsible for initializing and setting up the necessary dependencies.

DOMAIN: Web Search Service

KEY RESPONSIBILITIES:
• Initializes the web search service
• Sets up dependencies for the service
• Provides an interface for other modules to interact with the web search functionality

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module implements the WebSearchAnswerer class, a high-level helper that integrates web search results with Large Language Model (LLM) capabilities to generate answers.

**Domain:** LLM adapters

**Key Responsibilities:**
- Runs web searches via WebSearchExecutor
- Builds LLM-ready context/messages from web documents using WebSearchContextBuilder
- Calls any LLMAdapter to generate a final answer
- Provides both asynchronous and synchronous APIs for answer generation

### `intergrax\websearch\service\websearch_config.py`

**Description:** This module defines configuration classes for the web search service, including settings for search strategies, token budgets, and LLM adapters.

**Domain:** RAG logic

**Key Responsibilities:**
- Defines an enum for Web Search Strategy types
- Provides dataclasses for configuring web search:
  - `WebSearchLLMConfig`: specifies LLM adapters for map, reduce, and rerank steps
  - `WebSearchConfig`: configures overall strategy, token budgets, chunking settings, and run ID

**Note:** This file appears to be a well-documented, stable part of the Intergrax framework.

### `intergrax\websearch\service\websearch_context_generator.py`

**Description:** This module is responsible for generating context text based on web search results, utilizing different strategies such as SERP only, URL context top-k, chunk re-rank, and MAP reduce.

**Domain:** Web Search Context Generation

**Key Responsibilities:**

*   Provides a protocol for generating web search context
*   Implements different web search strategies (SERP only, URL context top-k, chunk re-rank, MAP reduce)
*   Utilizes data classes to encapsulate web search results and context generation metadata
*   Uses adapters for LLM map and reduce operations in the MAP reduce strategy

**Note:** The ChunkRerankContextGenerator class is currently not implemented.

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module provides a high-level, configurable web search executor that constructs query specifications and executes the search pipeline with chosen providers.

**Domain:** Web Search API

**Key Responsibilities:**

* Construct QuerySpec from raw queries and configuration
* Execute SearchAndReadPipeline with chosen providers
* Convert WebDocument objects into LLM-friendly dicts
* Manage provider signatures for cache invalidation
* Cache serialized web search results when a query cache is configured

### `intergrax\websearch\utils\__init__.py`

Description: This utility module initializes and exports various components for web search functionality within the Intergrax framework.

Domain: Web Search Utilities

Key Responsibilities:
• Initializes necessary components for web search operations
• Exports these components as part of the intergrax.websearch.utils namespace
• Provides a foundation for building upon in other parts of the web search module

### `intergrax\websearch\utils\dedupe.py`

**intergrax\websearch\utils\dedupe.py**

Description: This module provides functions for deduplicating web search results by normalizing and hashing text.

Domain: Web Search Utilities

Key Responsibilities:
* Normalizes text to prepare it for deduplication
	+ Treats `None` as an empty string
	+ Strips leading/trailing whitespace
	+ Converts to lower case
	+ Collapses internal whitespace sequences to a single space
* Generates stable SHA-256 based deduplication keys for the given text

Note: This file appears to be part of the main Intergrax framework codebase, and its purpose is clear.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a token bucket rate limiter, which allows for controlled and concurrent access to resources.
**Domain:** LLM adapters
**Key Responsibilities:**
* Implements a simple asyncio-compatible token bucket rate limiter
* Provides methods for acquiring and trying to acquire tokens within the allowed capacity and rate limits
* Designed for use across concurrent coroutines in a single process

Note: The code appears to be well-documented, complete, and production-ready.

### `main.py`

FILE PATH:
main.py

Description: The entry point for the Integrax framework, responsible for bootstrapping and initializing the application.
Domain: Application Entry Point

Key Responsibilities:
* Initializes the application
* Prints a message indicating that the Intergrax-ai is running

### `mcp\__init__.py`

DESCRIPTION:
This module serves as the entry point for the Intergrax framework, containing essential initialization code.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes the core components of the framework.
• Sets up the application configuration and dependencies.
• Defines the main entry points for the framework.

### `notebooks\nexus\01_basic_memory_demo.ipynb`

Description: This Jupyter notebook provides a basic sanity-check demo for the nexus Mode runtime in the Intergrax framework.

Domain: LLM adapters and Runtime Engine logic

Key Responsibilities:
- Creates or loads a session
- Appends user and assistant messages to the session
- Builds conversation history from SessionStore
- Returns a RuntimeAnswer object

### `notebooks\nexus\02_attachments_ingestion_demo.ipynb`

**Description:** This notebook demonstrates the Intergrax framework's nexus Mode runtime, specifically how it works with sessions and basic conversational memory, accepts attachments via `AttachmentRef`, and ingests them using the `AttachmentIngestionService`.

**Domain:** LLM (Language Model) adapters, RAG (Retrieval-Augmented Generation) logic, data ingestion.

**Key Responsibilities:**

* Initializes an in-memory session store for notebook experimentation
* Configures Intergrax Embedding Manager (Ollama embeddings)
* Configures Intergrax Vectorstore Manager (Chroma as a vector store)
* Creates a `RuntimeConfig` and `RuntimeEngine` instance with RAG turned off for ingestion purposes only
* Demonstrates attachment ingestion using the `AttachmentIngestionService`
* Prepares an `AttachmentRef` for a local project document and simulates its ingestion

### `notebooks\nexus\03_rag_context_builder_demo.ipynb`

**Description:** This Jupyter notebook demonstrates the usage of the `ContextBuilder` in nexus Mode, showcasing how to build context for a single user question using RAG retrieval from the vector store. The notebook also initializes various components required for testing the `ContextBuilder`.

**Domain:** RAG logic

**Key Responsibilities:**

* Initialize an in-memory session store and LLM adapter (Ollama-based)
* Create an embedding manager and vector store connection
* Configure runtime settings, including enabling RAG
* Demonstrate building context for a single user question using the `ContextBuilder`
* Show how to inspect the result of context building without making an LLM call

**Note:** This notebook appears to be a demo or tutorial file, focusing on showcasing specific functionality rather than serving as a production-ready code snippet.

### `notebooks\nexus\04_websearch_context_demo.ipynb`

Description: This notebook demonstrates how to use the Intergrax framework's RuntimeEngine with session-based chat, optional RAG (attachments ingested into a vector store), and live web search via WebSearchExecutor to achieve a "ChatGPT-like" experience with browsing.

Domain: nexus mode

Key Responsibilities:
- Initializes the core configuration for web search.
- Configures the LLM adapter, embeddings, vector store, runtime config, session manager, and web search executor.
- Demonstrates how to create a fresh chat session and interactively test the web search functionality using the RuntimeEngine's ask method.

### `notebooks\nexus\05_tools_context_demo.ipynb`

Description: This notebook demonstrates how to use the nexus Runtime with a tools orchestration layer, on top of conversational memory, RAG (attachments ingested into a vector store), and live web search context.

Domain: LLM adapters & Tools integration

Key Responsibilities:
- Initialize the Python path for importing the `intergrax` package.
- Load environment variables (API keys, etc.).
- Import core building blocks used by the nexus Runtime.
- Define tools using the Intergrax tools framework (ToolBase & ToolRegistry, IntergraxToolsAgent, ToolsAgentConfig, IntergraxConversationalMemory).
- Register demo tools (WeatherTool, CalcTool) and create an IntergraxToolsAgent instance.
- Attach this agent to RuntimeConfig.tools_agent for orchestration in a ChatGPT-like flow.

### `notebooks\nexus\06_session_memory_roundtrip_demo.ipynb`

**Description:** This Jupyter notebook demonstrates the functionality of Intergrax's nexus mode, specifically focusing on session and memory roundtrip capabilities. It creates a new session, reuses an existing one, persists and loads conversation history, and produces a consistent debug trace.

**Domain:** LLM adapters, RAG logic, data ingestion, agents

**Key Responsibilities:**

* Creates and reuses sessions using `SessionManager`
* Persists and loads conversation history via `SessionManager.get_history(...)`
* Produces a consistent debug trace
* Demonstrates the use of real adapters (no fakes or fallbacks)
* Configures runtime settings for LLM, embeddings, and vector store implementations

Note: This notebook appears to be a demonstration or test case rather than an experimental or auxiliary file. The code is well-structured and follows established patterns within the Intergrax framework.

### `notebooks\nexus\07_user_profile_instructions_baseline.ipynb`

**Description:** This Jupyter notebook demonstrates the Intergrax framework's user profile instructions baseline functionality. It introduces a minimal, production-ready memory flow for injecting user-level system instructions into runtime prompts.

**Domain:** LLM adapters / nexus Mode / Runtime Engine

**Key Responsibilities:**

* Validates that user-level system instructions can be resolved from stored `UserProfile` and injected as the first `system` message in the runtime prompt.
* Verifies that a persisted `UserProfile` is used to build the final `system` instructions.
* Tests that only user/assistant turns are stored in the session history, with no saved system instructions.

This notebook appears to be production-ready and part of a larger framework for managing user profiles and injecting instructions into runtime prompts.

### `notebooks\nexus\08_user_profile_instructions_generation.ipynb`

**Description:** This notebook demonstrates a production-safe, explicit mechanism for generating user profile instructions using an LLM (Large Language Model), based on conversation history and existing user profile. The goal is to separate instruction generation from usage.

**Domain:** LLM adapters

**Key Responsibilities:**

* Define the minimal LLM contract used by the generation service
* Implement a deterministic LLM stub for testing the generation flow
* Demonstrate the notebook-local generation service, which takes history and existing instructions as input and generates new instructions using the LLM
* Persist the generated instructions to the user profile and mark the session as requiring refresh

### `notebooks\nexus\09_long_term_memory_consolidation.ipynb`

Description: This Jupyter notebook tests the production-critical "long-term memory via consolidation" behavior in the Intergrax framework. It validates the correct execution of this mechanism by simulating a conversation, running consolidation, and checking invariants.

Domain: Long-term Memory Consolidation (RAG logic)

Key Responsibilities:
- Sets up a test environment with user profile and session identifiers
- Seeds baseline conversation history into the conversational memory store
- Runs consolidation service using the simulated history to populate user profile's memory entries
- Validates production-critical invariants after consolidation (e.g., number of messages, memory entry counts)

### `notebooks\nexus\10_e2e_user_longterm_memory.ipynb`

Description: This notebook validates the engine-integrated user long-term memory path end-to-end, covering persistence, modifications, semantic retrieval, and engine injection.

Domain: LLM adapters

Key Responsibilities:
- Initialize various managers (LLM adapter, EmbeddingManager, VectorstoreManager) for user profile management.
- Configure RuntimeConfig with specific settings for LTM enabled, RAG disabled, and tools mode set to "off".
- Create RuntimeEngine instance with the configured context and session manager.
- Perform STEP 1: Seed LTM entries + search using UserProfileManager.

### `notebooks\nexus\11_chatgpt_like_e2e.ipynb`

Description: This notebook serves as an integration/behavior test for the nexus Runtime, mimicking a real ChatGPT usage pattern. It exercises various aspects of the runtime's functionality.

Domain: LLM adapters, RAG logic, data ingestion, agents, configuration, utility modules

Key Responsibilities:
- Exercises multi-session behavior with isolated session history and shared user LTM.
- Tests user LTM persistence and recall across sessions.
- Verifies session-level consolidation without cross-session leakage.
- Ingests RAG content and performs Q&A over a document.
- Utilizes websearch as a context layer affecting the final answer.
- Executes tools (tool + LLM) without breaking the "user-last invariant."
- Enables reasoning for observability but does not persist it into user-visible history.

Note: This notebook appears to be a test suite for the Intergrax framework, specifically designed to exercise various aspects of its functionality. It is not experimental, auxiliary, or legacy code, and appears to be in a stable state.

### `notebooks\nexus\12a_engine_planner.ipynb`

Description: This notebook is a test environment for the Intergrax Engine Planner, allowing isolated testing and iterative improvement of the planner without impacting the runtime execution pipeline.

Domain: LLM (Large Language Model) driven planning

Key Responsibilities:
- Run `EnginePlanner.plan(...)` for various user prompts
- Inspect and compare generated `EnginePlan` outputs
- Evaluate plan stability, correctness, and feasibility
- Refine the planner prompt, plan schema, and validation rules iteratively
- Improve the planner safely without impacting runtime behavior

### `notebooks\nexus\12b_engine_planner.ipynb`

**Description:** This notebook provides a set of utilities for testing the Intergrax framework's nexus mode planner, including test case definitions and helper functions for building planner requests and states.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Define test cases for the nexus mode planner
* Provide helper functions for building planner requests and states
* Utilize Intergrax framework components such as `LLMAdapterRegistry`, `RuntimeConfig`, and `EnginePlanner`
* Include experimental code to simulate memory-layer instruction fragments (currently set to None)
* Define utility functions for extracting plan flags from EnginePlan objects

**Note:** The file appears to be a collection of test utilities and does not seem to be experimental, auxiliary, legacy, or incomplete. However, the simulation of memory-layer instruction fragments is currently disabled.

### `notebooks\nexus\13a_engine_step_planner.ipynb`

Description: This notebook provides an example of using the Intergrax framework to plan and execute tasks, specifically demonstrating the integration between the LLM-based EnginePlanner and the deterministic StepPlanner.

Domain: LLM adapters, RAG logic, Agents, Configuration, Utility modules, Planning

Key Responsibilities:
- Demonstrates the usage of LLM adapters (e.g., OLLAMA) with the Intergrax framework.
- Shows how to configure and use the StepPlanner to plan tasks based on user input.
- Validates the integration point between EnginePlanner and StepPlanner through a test case.
- Utilizes various Intergrax modules, including LLm adapters, planning step models, and runtime configuration.
- Includes example usage of RuntimeConfig and LLMAdapterRegistry.
 
Note: The notebook appears to be a comprehensive example rather than experimental or auxiliary code. It is designed to demonstrate the capabilities and integration of various Intergrax components.

### `notebooks\nexus\13b_engine_step_planner.ipynb`

**Description:**
This Jupyter notebook is a testbed for the StepPlanner module in Intergrax, which plans and executes tasks using LLM-driven planning. It validates the behavior of StepPlanner without executing any tools.

**Domain:** RAG logic, LLM adapters, data ingestion, configuration

**Key Responsibilities:**

* Validates StepPlanner behavior without executing any tools
* Tests LLM-driven planning for both STATIC and DYNAMIC planning modes
* Verifies correctness of built plans (order, depends_on, mode)
* Ensures consistency between STATIC and DYNAMIC planning for the same step types
* Sets up a planner context with explicit configuration and typed inputs
* Utilizes LLM adapters to interact with LLMs

Note: This file appears to be a test notebook for the StepPlanner module, and is not experimental or auxiliary.

### `notebooks\nexus\14_engine_planner_executor.ipynb`

**Description:** This is a Jupyter notebook that validates the StepExecutor's behavior in isolation, testing its sequential execution, dependencies, retry policies, controlled replan, handler contract enforcement, and final output selection.

**Domain:** LLM (Large Language Model) adapters / RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Validates StepExecutor behavior through integration tests
* Tests sequential execution, dependencies, retry policies, and controlled replan
* Verifies handler contract enforcement and final output selection
* Provides a happy path test for DRAFT → VERIFY → FINAL step execution

Note: This file appears to be an implementation of unit tests for the StepExecutor component within the Intergrax framework. The code is written in Python and uses Jupyter Notebook cells to define and execute test cases.

### `notebooks\nexus\15_plan_loop_controller.ipynb`

Description: This Jupyter notebook defines a test suite for the PlanLoopController, a key component of the Intergrax framework that orchestrates planning and execution loops in nexus Runtime mode.

Domain: LLM adapters, RAG logic, data ingestion, agents, configuration, utility modules (notebook testing)

Key Responsibilities:
- Validates PlanLoopController behavior end-to-end in STATIC planning mode
- Tests REPLAN_REQUIRED loop iteration, replan_ctx injection into EnginePlanner prompt, bounded replanning, and HITL escalation when limits are exceeded
- Exercises the Controller's ability to generate a plan, execute steps until an event requiring replanning or reaching max replans/repeats occurs
- Demonstrates Human-in-the-Loop (HITL) intervention if stuck in planning/execution loop.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

Description: This notebook demonstrates an end-to-end RAG workflow using Intergrax and LangGraph components, combining multiple knowledge sources into a single in-memory vector index.

Domain: Hybrid Multi-Source RAG with Intergrax + LangGraph

Key Responsibilities:
- Ingest content from multiple sources (local PDF files, local DOCX/Word files, live web results)
- Build a unified RAG corpus by normalizing documents and attaching metadata
- Create an in-memory vector index using Intergrax embedding manager and vectorstore manager
- Answer user questions with a RAG agent through LangGraph orchestrating the flow

Note: This notebook appears to be a practical demonstration of combining multiple knowledge sources into a single in-memory vector index, rather than a production-ready codebase.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates a minimal integration between the Intergrax and LangGraph frameworks for simple LLM QA.

**Domain:** RAG logic (LLM adapters, graph building)

**Key Responsibilities:**

- Integrate an Intergrax LLM adapter as a node inside a LangGraph graph.
- Define a `State` that holds chat-style messages and a final answer returned by the node.
- Build a `StateGraph` with a single node `llm_answer_node`.
- Run the graph on a sample user question.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates the integration of Intergrax components with LangGraph to create a practical web research agent. It showcases how to build a multi-step graph for web-based Q&A using an LLM adapter and web search providers.

**Domain:** Web Research Agent, LLM Adapters, RAG Logic

**Key Responsibilities:**

* Importing necessary libraries and setting up environment variables
* Initializing LLM adapter and WebSearch components (WebSearchExecutor, WebSearchContextBuilder, WebSearchAnswerer)
* Defining the graph state (WebResearchState) with fields for user question, normalized question, search results, context text, citations, and answer
* Implementing nodes for normalizing user questions and running web searches using Intergrax WebSearchExecutor
* Demonstrating the workflow of the web research agent through a multi-step graph

### `notebooks\multimedia\rag_multimodal_presentation.ipynb`

**Description:**
This notebook provides a demonstration of using the Integrax framework for multimodal document retrieval and indexing.

**Domain:** RAG (Relational Augmented Graph) logic, multimodal processing

**Key Responsibilities:**

* Loading documents from various sources (video, audio, image)
* Splitting and embedding documents
* Indexing documents in Vectorstore
* Retrieving relevant documents using a retriever
* Displaying results

### `notebooks\multimedia\rag_video_audio_presentation.ipynb`

**Description:** This notebook provides a set of multimedia processing tasks, including video and audio download, transcription, frame extraction, and translation.

**Domain:** Multimedia Processing

**Key Responsibilities:**
- Download videos and audios from YouTube using `yt_download_video` and `yt_download_audio`.
- Transcribe video to VTT format using `transcribe_to_vtt`.
- Extract frames and metadatas from a video using `extract_frames_and_metadata`.
- Translate audio using the `translate_audio` function.
- Use the ollama model to describe an image using `transcribe_image`.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This Jupyter Notebook demonstrates the usage of Intergrax's RAG (Retrieval-Augmented Generation) functionality with OpenAI, showcasing how to create a VectorStore, upload documents, and run queries.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Creates a VectorStore using IntergraxRagOpenAI
- Uploads a local folder of documents to the VectorStore
- Runs queries on the uploaded documents to retrieve answers from OpenAI

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This notebook contains the implementation of a chat agent using the Integrax framework, which integrates with various tools and models to provide conversational responses.

**Domain:** Chat Agent Presentation

**Key Responsibilities:**

* Importing necessary libraries and modules from the Integrax framework
* Defining classes for LLM adapters, Rag retriever, and answerer
* Creating instances of LLM providers, embedding managers, vector store managers, and re-ranking algorithms
* Registering available tools with the tool registry
* Implementing example tools (e.g., weather) to demonstrate how the chat agent can interact with external APIs or services
* Using conversational memory to track user interactions and provide context for future conversations

Note: This file appears to be a working implementation of a chat agent, rather than an experimental or auxiliary module.

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the usage of the Intergrax framework by showcasing a simple weather tool and its integration with LLM adapters and conversational memory.

**Domain:** RAG logic (Relevant Answer Generation)

**Key Responsibilities:**

* Defines a WeatherAnswer Pydantic model for structured output.
* Implements a WeatherTool class that accepts city names as input and returns demo weather data in the WeatherAnswer format.
* Demonstrates how to use the ToolsAgent to orchestrate LLM reasoning, tool selection, invocation, and response generation with structured output using the WeatherAnswer model.
* Integrates with conversational memory for multi-turn interactions.
* Uses an Ollama-backed LLM adapter via LangChain's ChatOllama.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter notebook appears to be a Rag Custom Presentation script that provides utilities for loading, splitting, and embedding documents. It contains code snippets for document loading, chunking, and vector store management using various components from the Integrax framework.

**Domain:** RAG (Relevance Aware Retrieval) logic

**Key Responsibilities:**
* Loading documents from a directory with metadata support
* Splitting loaded documents into smaller chunks
* Generating embeddings for document sets using Ollama as the provider
* Managing vector stores and performing lightweight "probe" queries

### `notebooks\rag\tool_agent_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the use of the Integrax framework to create a tools agent that can perform various tasks, including fetching weather information and performing basic arithmetic calculations. The agent uses a planner/controller (LLM) to decide which tool to invoke based on user input.

**Domain:** RAG logic (Reasoning-Augmentation-Generation)

**Key Responsibilities:**
* Define tools for specific tasks (e.g., weather, calculator)
* Create a registry of available tools
* Set up an LLM-based planner/controller to select tools and integrate outputs
* Demonstrate tool selection and invocation with example user inputs

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This Jupyter Notebook file contains a collection of Python functions, each implementing a component for the Intergrax framework. These components are designed to perform specific tasks such as compliance checking, cost estimation, generating final summaries, and conducting financial audits.

**Domain:** RAG (Reasoning And Generation) logic

**Key Responsibilities:**

* Compliance Checker:
	+ Verifies whether proposed changes comply with privacy policies and terms of service.
	+ Returns findings, policy violations, and a decision on compliance.
* Cost Estimator:
	+ Estimates the cost of changes based on UX audit reports.
	+ Uses a mock formula to calculate costs.
* Final Summary:
	+ Generates a final consolidated summary using all collected artifacts.
	+ Includes status pipeline, terminated by, terminate reason, PM decision, and other relevant information.
* Financial Audit:
	+ Generates a mock financial report and VAT calculation.
	+ Returns a report with net value, VAT rate, VAT amount, gross value, currency, and budget for the last quarter.

**Status:** This file appears to be production-ready code with proper documentation and structure.

### `notebooks\websearch\websearch_presentation.ipynb`

Description: This Jupyter Notebook module demonstrates the usage of Intergrax's web search functionality, specifically utilizing Google Custom Search and Bing Search providers.

Domain: Web Search Executor

Key Responsibilities:
- Loads environment variables from .env file for API keys.
- Initializes a QuerySpec object with predefined parameters.
- Creates an instance of the GoogleCSEProvider class to perform searches.
- Executes the search query and prints results, including provider, rank, title, URL, snippet, domain, and published date.
- Demonstrates web search capabilities using Intergrax's framework.
