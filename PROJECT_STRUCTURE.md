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
- `intergrax\llm\conversational_memory.py`
- `intergrax\llm\llm_adapters_legacy.py`
- `intergrax\llm_adapters\__init__.py`
- `intergrax\llm_adapters\base.py`
- `intergrax\llm_adapters\gemini_adapter.py`
- `intergrax\llm_adapters\ollama_adapter.py`
- `intergrax\llm_adapters\openai_responses_adapter.py`
- `intergrax\logging.py`
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

FILE PATH:
api\__init__.py

DESCRIPTION: This file serves as the entry point for the API module, responsible for initializing and configuring API-related functionality.

DOMAIN: API Initialization

KEY RESPONSIBILITIES:
• Initializes API settings and configurations
• Sets up API routing and endpoints
• Imports necessary modules and packages for API operations

### `api\chat\__init__.py`

Description: This module initializes the chat API and sets up dependencies for interactions with the chat system.

Domain: LLM adapters

Key Responsibilities:
• Initializes the chat API and its dependencies
• Sets up configuration for chat interactions
• Establishes connections with other necessary systems or modules
 

Note: Based on the provided content, this file appears to be a standard module used in the development of the Intergrax framework, as it initializes an essential part of the chat functionality.

### `api\chat\main.py`

**Description:** This module implements the core logic for the Integrax chat API, handling user queries and interactions with models, as well as document management and indexing.

**Domain:** Chat API

**Key Responsibilities:**

* Handles incoming user queries through the `/chat` endpoint
	+ Retrieves relevant history pairs from the database
	+ Passes query to an answerer (model) for processing
	+ Returns response to the user
* Manages document uploads and indexing
	+ Validates file extensions
	+ Uploads documents to a temporary location
	+ Indexes documents in Chroma
* Provides endpoints for listing and deleting documents
	+ Retrieves list of all documents from the database
	+ Deletes documents from both Chroma and the database

### `api\chat\tools\__init__.py`

DESCRIPTION: This module serves as the entry point for tools and utilities related to chat APIs.

DOMAIN: API Tools

KEY RESPONSIBILITIES:
- Provides a centralized initialization point for chat-related tooling.
- Imports and sets up various utility modules within the chat tools scope.

### `api\chat\tools\chroma_utils.py`

Here is the documentation for the provided file:

Description: This module provides utilities for interacting with the Chroma search engine within the Intergrax framework.

Domain: RAG (Retrieval-Augmented Generation) logic

Key Responsibilities:
- Load and split documents from a given file path.
- Index documents to the Chroma vector store.
- Delete documents by their ID in the Chroma vector store.

### `api\chat\tools\db_utils.py`

**Description:** This module provides a database utility for managing chat sessions and documents within the Integrax framework. It includes functionality for schema creation, migration, message insertion, retrieval, and document management.

**Domain:** Database Utilities

**Key Responsibilities:**

* **Database Connection Management**
	+ Establishing connections to the `rag_app.db` SQLite database
	+ Ensuring session existence and creation if necessary
* **Schema Creation and Migration**
	+ Creating schema for sessions, messages, and documents tables
	+ Migrating from legacy application logs table to the new message structure
* **Message Management**
	+ Inserting user and assistant messages into the `messages` table
	+ Retrieving messages by session ID with optional limit filtering
	+ Returning pairs of user and assistant text for chat history
* **Document Management**
	+ Inserting document records into the `document_store` table
	+ Deleting document records by file ID
	+ Retrieving all documents in descending upload timestamp order

Note: The code appears to be well-structured, with clear functions and docstrings. However, some functionality seems optional or legacy (e.g., `create_application_logs`, `insert_application_logs`, `get_chat_history`), suggesting that it may be incomplete or for backward compatibility purposes only.

### `api\chat\tools\pydantic_models.py`

Description: This module defines data models for API interactions, utilizing Pydantic to provide a structured and type-safe interface.
Domain: LLM adapters & RAG logic

Key Responsibilities:
- Defines enum for model names (e.g., LLAMA_3_1, GPT_OSS_20B)
- Provides Pydantic models for query inputs (question, session ID, model), query responses (answer, session ID, model), document information (ID, filename, upload timestamp), and delete file requests (file ID)

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides utilities for building and managing components of the RAG (Retrieval-Augmented Generation) pipeline, including vector store management, embedding managers, retrievers, rerankers, and answerers. It also defines prompts and settings for interacting with users.

**Domain:** RAG logic

**Key Responsibilities:**
- Manages singletons for vector stores, embedders, retrievers, rerankers, and answerers.
- Provides utility functions for retrieving and configuring these components.
- Defines default user and system prompts for interactions.
- Enables building and managing LLM adapters.
- Offers settings and environment variables for persistence directories and model names.

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides a set of API utility functions for interacting with the Integrax framework, including chat functionality and document management.

**Domain:** Chat Streamlit API Utilities

**Key Responsibilities:**
- Makes POST requests to the "chat" endpoint to retrieve responses from models.
- Handles file uploads via the "upload-doc" endpoint.
- Lists available documents via the "list-docs" endpoint.
- Deletes documents by ID using the "delete-doc" endpoint.

Note: The code is stable and functional, with proper error handling and logging.

### `applications\chat_streamlit\chat_interface.py`

Description: This module provides a Streamlit-based chat interface for interacting with an external API.
Domain: LLM adapters/Integration with external APIs

Key Responsibilities:
- Displays user input and receives API responses in a chat format
- Utilizes `get_api_response` function to fetch answers from the API
- Handles session management by storing and updating session IDs and messages
- Provides expandable details section for generated answer, model used, and session ID

### `applications\chat_streamlit\sidebar.py`

**Description:** This module is responsible for implementing a sidebar in the Streamlit application, providing functionality for model selection, uploading documents, listing uploaded documents, and deleting selected documents.

**Domain:** Chat Application Utilities

**Key Responsibilities:**

* Provides a model selection component using `st.sidebar.selectbox`
* Allows users to upload documents using `st.sidebar.file_uploader` and `upload_document` API call
* Displays a list of uploaded documents with delete options
* Refreshes the document list upon button click
* Enables deletion of selected documents using `delete_document` API call

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module provides the core functionality for the Streamlit-based chat interface application.

**Domain:** Chat Interface/Streamlit App

**Key Responsibilities:**
* Initializes and configures the Streamlit app
* Displays the sidebar and chat interface components
* Manages session state (messages, session ID)

### `applications\company_profile\__init__.py`

DESCRIPTION: 
This module is the entry point for the company profile application, initializing and configuring its components.

DOMAIN: Application initialization & configuration

KEY RESPONSIBILITIES:
- Initializes the company profile application
- Configures the necessary modules and services

### `applications\figma_integration\__init__.py`

DESCRIPTION:
This module serves as the entry point for Figma integration, encapsulating the necessary functionality to establish a connection with Figma and facilitate data exchange.

DOMAIN: Figma Integration

KEY RESPONSIBILITIES:
- Initializes the Figma API client
- Defines the interface for interacting with Figma
- Provides methods for retrieving design files and nodes

### `applications\ux_audit_agent\__init__.py`

Description: The UX Audit Agent is a utility module responsible for bootstrapping the audit agent application, providing essential initialization and setup.

Domain: Agents

Key Responsibilities:
• Initializes the audit agent application
• Sets up necessary configurations and dependencies
• Enables subsequent import of other modules within the application

### `applications\ux_audit_agent\components\__init__.py`

DESCRIPTION: The __init__.py file is the entry point for the UX audit agent components, importing and initializing necessary modules for auditing user experience.

DOMAIN: Agents

KEY RESPONSIBILITIES:
- Initializes and imports relevant component modules.
- Sets up logging and configuration for the audit agent.

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module defines a component for evaluating compliance with privacy policies and regulations by simulating validation checks. It can be used in the UX audit pipeline to verify if proposed changes comply with existing rules.

**Domain:** Compliance Checker (RAG logic)

**Key Responsibilities:**
- Provides a component for policy and regulatory compliance checking
- Simulates compliance validation (80% chance of being compliant)
- Returns findings, including policy violations and recommendations for correction or DPO review
- Can stop pipeline execution if non-compliant changes are detected

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a cost estimation component for UX audit reports, implementing a mock pricing model to estimate costs based on the number of issues identified.

**Domain:** RAG logic

**Key Responsibilities:**
- Estimates the cost of UX-related changes using a mock pricing model
- Accepts pipeline state and context as inputs
- Returns cost estimates, currency, and method used in the calculation
- Logs the estimated cost for auditing purposes

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module generates a complete final report of the task execution pipeline using all collected artifacts.

**Domain:** UX Audit Agent Components

**Key Responsibilities:**

* Generates a final report at the last stage of the pipeline
* Collects and aggregates various artifacts, such as project manager decisions, notes, UX reports, financial reports, and citations
* Returns a `ComponentResult` object containing the generated summary

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module implements a financial audit agent component for the Integrax framework, generating mock financial reports and VAT calculations.

**Domain:** Agents

**Key Responsibilities:**
- Provides a "Financial Agent" component for testing financial computations.
- Generates mock financial reports with test data (e.g., net values, VAT rates, amounts).
- Supports use cases like testing budget constraints or cost reports.

### `applications\ux_audit_agent\components\general_knowledge.py`

Description: This module defines a component that provides general knowledge about the Intergrax system, including its architecture and features.

Domain: LLM adapters

Key Responsibilities:
- Provides a general knowledge component for answering user queries about Intergrax.
- Returns mock knowledge responses with fake documentation citations. 

Note: The file appears to be a part of the main framework and does not exhibit any characteristics of being experimental, auxiliary, legacy, or incomplete.

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module defines a component that simulates a project manager's decision-making process for UX reports, providing mock approvals or rejections with associated comments.

**Domain:** RAG logic (Risk, Action, Review)

**Key Responsibilities:**
- Simulates project manager's decision based on predefined weights
- Returns decision and corresponding notes to the pipeline execution
- Optionally stops pipeline execution if rejected

### `applications\ux_audit_agent\components\ux_audit.py`

**FILE PATH:** applications\ux_audit_agent\components\ux_audit.py

**Description:** This module provides a UX audit component that analyzes Figma mockups and returns a sample report with recommendations.

**Domain:** RAG logic (Recommendations, Analysis, Generation)

**Key Responsibilities:**
* Performs UX audit on Figma mockups
* Generates a sample report with recommendations
* Utilizes the Integrax framework's component registration system
* Returns a ComponentResult object containing the report and logs

Note: The file appears to be a well-structured and functional part of the Integrax framework, without any obvious signs of being experimental, auxiliary, or incomplete.

### `applications\ux_audit_agent\UXAuditTest.ipynb`

**Description:** This Jupyter Notebook script appears to be part of the Integrax framework, implementing an audit pipeline for user experience (UX) assessments. It outlines a series of steps and tasks aimed at ensuring compliance with company policies.

**Domain:** UX Audit Agent

**Key Responsibilities:**

* Perform UX audits on FIGMA mockups
* Verify changes comply with company policy using specialized tools
* Prepare summary reports for Project Managers
* Evaluate financial impact of changes
* Project Manager decision-making and project continuation
* Final report preparation and synthesis

### `generate_project_overview.py`

Description: This module is a utility for automatically generating documentation on the project structure and layout for the Intergrax framework.

Domain: Utilities/Project Structure Documentation

Key Responsibilities:
- Recursively scans the project directory.
- Collects relevant source files (Python, Jupyter Notebooks) based on configurable extensions and exclusions.
- Sends file content to an LLM adapter for summary generation.
- Creates a structured Markdown document (`PROJECT_STRUCTURE.md`) describing the project layout, including file summaries generated via LLM.

Note: This module appears to be part of the main framework codebase, not experimental or auxiliary.

### `intergrax\__init__.py`

DESCRIPTION: The main entry point of the Intergrax framework, responsible for initializing and setting up the necessary components.

DOMAIN: Framework initialization

KEY RESPONSIBILITIES:
- Initializes Intergrax core components
- Sets up framework configuration and dependencies

### `intergrax\chains\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the Intergrax chains module, defining its interface and functionality.

DOMAIN: Chains logic

KEY RESPONSIBILITIES:
• Initializes the chains module's components.
• Defines the chains module's API.

### `intergrax\chains\langchain_qa_chain.py`

Description: This module implements a QA chain using the LangChain library and the Integrax framework. It enables users to build a flexible QA pipeline with various hooks for modifying data at different stages.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Builds a QA chain with hooks for modifying data at different stages
- Supports various LangChain LLMS (e.g., ChatOllama, ChatOpenAI)
- Provides an interface for users to build and invoke the QA chain
- Includes default prompt builder and customizable configuration options

### `intergrax\chat_agent.py`

**Description:** The `intergrax.chat_agent.py` file implements a chat agent API that utilizes the LLM (Large Language Model) to route user queries to either RAG (Retrieval-Augmented Generation), tools, or general functionality.

**Domain:** LLM adapters and routing logic

**Key Responsibilities:**

*   Provides a chat agent API with LLM-based routing
*   Supports RAG, tools, and general functionality routing
*   Allows for manual override of routing decisions
*   Handles memory management and streaming
*   Returns structured output with answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component information

Note: The file appears to be well-structured, and its purpose is clearly defined. There are no indications of it being experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm\__init__.py`

FILE PATH:
intergrax\llm\__init__.py

Description: The LLM (Large Language Model) package initialization module, responsible for importing and setting up the necessary components.

Domain: LLM adapters

Key Responsibilities:
- Initializes the LLM adapter registry
- Imports all available LLM adapter modules
- Sets up the adapter factory for easy instantiation

### `intergrax\llm\conversational_memory.py`

**Description:** This module provides a universal conversation memory system that works independently of LLMs/adapters. It keeps chat messages in RAM and has separate save/load methods to files (JSON/NDJSON) and SQLite.

**Domain:** Conversational Memory, Chat State Management

**Key Responsibilities:**

*   Keeps a list of chat messages in RAM with support for filtering and limiting the number of stored messages
*   Provides methods for adding and extending message history
*   Offers save/load functionality to JSON and NDJSON files as well as SQLite
*   Supports preparation of messages for sending to models, including removal of older 'tool' messages if necessary

### `intergrax\llm\llm_adapters_legacy.py`

Description: This module provides adapters for interacting with LLM (Large Language Model) services, specifically OpenAI's Chat Completions API.

Domain: LLM Adapters

Key Responsibilities:
- Provides a universal interface `LLMAdapter` for LLM adapters to follow.
- Implements an `OpenAIChatCompletionsAdapter` class for interaction with the OpenAI Chat Completions API.
- Offers helper functions for message mapping and JSON schema extraction.

### `intergrax\llm_adapters\__init__.py`

**Description:** This module serves as the entry point for LLM adapters in the Intergrax framework, providing a centralized registry and import mechanism for various language model adapters.

**Domain:** LLM adapters

**Key Responsibilities:**

* Registers and imports available LLM adapters (e.g., OpenAI, Gemini, LangChain Ollama)
* Provides access to adapter classes through the `__all__` list
* Registers default adapters with the `LLMAdapterRegistry` for easy usage
* Enables lazy loading of adapters through lambda functions in the registry

### `intergrax\llm_adapters\base.py`

**Description:** This module provides the foundation for interacting with Large Language Models (LLMs) through a unified interface, including adapters and a registry. It also includes utility functions for structured output handling.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides a protocol (`LLMAdapter`) for adapters to implement
* Defines adapter classes (`BaseModel`, `LLMAdapterRegistry`)
* Includes utility functions for:
	+ Extracting JSON objects from text
	+ Validating and creating model instances from JSON strings
	+ Converting internal chat messages to OpenAI-compatible message dictionaries
	+ Handling structured output through Pydantic v2/v1 or plain dataclasses/classes

This file appears to be a core part of the Intergrax framework, providing the necessary building blocks for interacting with LLMs. It is not marked as experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a minimal adapter for the Gemini Large Language Model (LLM) to interact with the Intergrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes the Gemini chat model and stores it as an instance variable
* Splits system messages from conversation messages
* Generates responses to input messages using the Gemini model
* Streams generated responses through a simple fallback mechanism (single-shot only)
* Provides placeholder methods for tool support that are not implemented in this adapter

### `intergrax\llm_adapters\ollama_adapter.py`

Description: This module provides an adapter class for interacting with Ollama models via LangChain's ChatModel interface.

Domain: LLM adapters

Key Responsibilities:
- Converts internal ChatMessage list into LangChain message objects
- Maps generation parameters (temperature and max_tokens) to Ollama options dictionary
- Provides public API methods for generating and streaming messages, including support for structured output via prompt + validation
- Indicates that there is no native tool-calling support in this adapter and suggests using a planner-style pattern instead

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** 
This module provides adapters for interacting with the OpenAI chat interface using their new Responses API, offering functionality similar to the legacy Chat Completions adapter.

**Domain:** LLM Adapters

**Key Responsibilities:**
- Provides adapters for OpenAI's Responses API.
- Offers public interfaces for generating messages, streaming responses, and working with tools.
- Supports structured JSON output generation based on JSON schema validation.
- Integrates with client instances to create requests for the Responses API.

### `intergrax\logging.py`

**intergrax\logging.py**

Description: This module sets up global logging configurations for the Integrax framework, specifying log level and format.

Domain: Logging Configuration

Key Responsibilities:
- Configures basic logging with a specified level (INFO) and format.
- Forces new configuration to overwrite any previous ones.

### `intergrax\multimedia\__init__.py`

DESCRIPTION: The `__init__.py` file serves as the entry point for the Intergrax multimedia module, responsible for setting up and organizing related components.

DOMAIN: Multimedia Framework Utilities

KEY RESPONSIBILITIES:
- Initializes and configures the multimedia module.
- Defines interfaces for dependent modules to interact with multimedia functionality.
- Optionally includes other setup or initialization tasks specific to this module.

### `intergrax\multimedia\audio_loader.py`

Description: This module provides functionality for downloading audio from YouTube URLs and translating it using the Whisper model.

Domain: Audio Processing

Key Responsibilities:
- Downloads audio files from specified YouTube URLs in various formats (defaulting to mp3).
- Translates downloaded audio into text using the Whisper model, with options for selecting models and languages.

### `intergrax\multimedia\images_loader.py`

Description: This module provides functionality for loading and processing images within the Intergrax framework.

Domain: Multimedia

Key Responsibilities:
- Loads images from file path.
- Utilizes ollama library to perform image-based tasks (e.g., transcription).
- Integrates with specific models (e.g., "llava-llama3:latest") for processing.

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides functionality for displaying multimedia content, including audio, images, and videos, using IPython's display capabilities.

**Domain:** Multimedia Display

**Key Responsibilities:**

* Displaying audio files with customizable start time and autoplay options
* Displaying image files as IPython widgets
* Serving media files from a temporary directory to ensure they are not deleted during rendering
* Displaying video files with customizable playback start time, poster frames, and other controls

### `intergrax\multimedia\video_loader.py`

**Description:** 
The Intergrax framework provides a suite of multimedia processing tools, including video loading, transcription to VTT format, and frame extraction with metadata.

**Domain:** Multimedia Processing

**Key Responsibilities:**
- Loads videos from YouTube URLs using `yt_dlp`
- Transcribes input media into VTT format using Whisper model
- Extracts frames from videos at specified intervals or with a specified height, saving them as images in JPEG format
- Stores extracted frame metadata (timestamp, transcript, etc.) in JSON files

### `intergrax\openai\__init__.py`

DESCRIPTION: This is the entry point for the OpenAI integration within Intergrax, serving as a namespace and importer for related modules.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
• Provides a centralized import mechanism for other OpenAI-related components.
• Defines the root namespace for the OpenAI integration.
• Exposes functionality through __all__ attribute.

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: This file serves as the entry point for the RAG logic within Intergrax, providing an interface to initialize and interact with various components of the framework.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes the RAG component
- Defines interfaces for interacting with RAG modules
- Possibly handles setup or configuration for underlying RAG functionality

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides a Rag OpenAI class for integrating the openai library with Intergrax's RAG logic, enabling interaction with Vector Stores and file management. It offers methods for uploading folders to vector stores, clearing existing data, and ensuring the existence of a specified vector store.

**Domain:** LLM adapters

**Key Responsibilities:**

*   Provides a Rag OpenAI class for integrating openai library with Intergrax's RAG logic
*   Offers methods for interacting with Vector Stores:
    *   Ensures the existence of a specified vector store
    *   Uploads folders to vector stores, supporting various file patterns and formats
    *   Clears existing data from a vector store
*   Utilizes openai library for file management in vector stores

Note: The provided content appears to be well-structured, but there are some syntax errors in the code snippet. It is recommended to review and correct these issues to ensure proper functionality.

### `intergrax\rag\__init__.py`

**Description:** 
This module serves as the entry point for Intergrax's Retrieval-Augmented Generation (RAG) logic, providing a centralized initialization mechanism for RAG components.

**Domain:** RAG logic
**Key Responsibilities:**
• Initializes RAG modules and sets up their configurations
• Registers available RAG components with the framework

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader with metadata injection and safety guards. It supports various file formats, including text, images, audio, and videos.

**Domain:** Document loaders/RAG logic

**Key Responsibilities:**

* Load documents from various file formats (text, images, audio, and videos)
* Support OCR for images and PDFs
* Inject metadata into loaded documents
* Provide safety guards to handle exceptions and errors
* Allow customization of loading settings through a configuration object
* Supports extensibility via adapters for specific document types or providers

Note: This file appears to be well-maintained and actively used, with no obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\documents_splitter.py`

**Description:** This module provides high-quality text splitting functionality for RAG pipelines, generating stable chunk IDs and rich metadata.

**Domain:** RAG logic (Retroactively Applicable Generation)

**Key Responsibilities:**

*   High-quality text splitting based on a "semantic atom" policy
*   Generating stable chunk IDs using available anchors (para_ix/row_ix/page_index)
*   Rich metadata creation, including core source fields and indexing information
*   Optional customization of metadata through user-provided functions

This module appears to be production-ready.

### `intergrax\rag\dual_index_builder.py`

**Description:** 
This module is responsible for building two vector indexes: a primary index (CHUNKS) and an auxiliary index (TOC), used in the context of the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

- Builds two vector indexes: primary (CHUNKS) and auxiliary (TOC)
- CHUNKS index contains all documents after splitting
- TOC index contains only DOCX headings within specified levels
- Embeddings are computed for each document in the primary index using an embedding manager
- Documents are added to their respective vector stores in batches for performance reasons
- Optional: A prefilter can be applied to documents before processing

### `intergrax\rag\dual_retriever.py`

**Description:** The `intergrax\dual_retriever.py` file implements a dual retriever class for fetching relevant information from a vector store. It first queries the Table of Contents (TOC) to identify sections related to the user's query, and then expands this context by searching locally within these identified sections.

**Domain:** RAG logic

**Key Responsibilities:**

*   Initialize the dual retriever with a Vector Store Manager for chunks and optionally TOC
*   Implement helper functions for normalizing hits and querying the vector store
*   Expand context via TOC, search locally by parent ID, and merge results
*   Retrieve relevant information based on user queries

**Note:** This file appears to be part of an experimental or proprietary framework (Integrax) and may contain confidential information.

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for various models (HuggingFace, Ollama, OpenAI) and offers features such as provider switching, model loading, batch/single text embeddings, L2 normalization, cosine similarity utilities, and robust logging.

**Domain:** RAG logic (Reinforcement Learning with Augmented Data)

**Key Responsibilities:**

* Load and manage different embedding models (HuggingFace, Ollama, OpenAI)
* Provide a unified API for batching and single text embeddings
* Support L2 normalization of embeddings
* Offer cosine similarity utilities (e.g., top-K retrieval)
* Perform robust logging and error handling
* Handle model loading and dimension detection with retries

### `intergrax\rag\rag_answerer.py`

**Description:** This module implements an answerer for the Intergrax framework, leveraging Retrieval-Augmented Generation (RAG) to provide accurate and informative responses.

**Domain:** RAG Logic

**Key Responsibilities:**

- **Configuration**: Exposes a `IntergraxAnswererConfig` dataclass that defines various hyperparameters and settings for the answerer.
- **Data Models**: Defines several dataclasses, including `AnswerSource`, which represents the source of an answer with metadata like score and preview text.
- **Main Answerer**: The `IntergraxRagAnswerer` class is responsible for answering questions using RAG. It takes various components as inputs (retriever, LLM adapter, reranker) and returns a dictionary containing the answer and supporting information.
- **Message Processing**: Provides methods for building context text from retrieved hits, making citations, and generating system/user messages for the LLM.
- **LLM Integration**: Handles interactions with the LLM adapter to generate answers and support structures.

### `intergrax\rag\rag_retriever.py`

**Description:** This module provides a scalable, provider-agnostic RAG retriever for intergrax, enabling efficient and effective retrieval of relevant documents.

**Domain:** RAG (Retriever-Augmented Generator) logic

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

**Notes:** The file appears to be a primary module within the RAG logic domain and is not marked as experimental or auxiliary. However, its implementation and functionality suggest that it may require further optimization and testing to ensure its robustness in production environments.

### `intergrax\rag\re_ranker.py`

**Description:** The ReRanker module is responsible for re-ranking a list of candidate documents based on their similarity to a given query, using cosine similarity on L2-normalized vectors.

**Domain:** RAG logic

**Key Responsibilities:**

* Accepts hits from the retriever (dict) or raw LangChain Documents
* Embeds texts in batches using the `intergraxEmbeddingManager`
* Optional score fusion with original retriever similarity
* Re-ranks candidates based on cosine similarity to the query
* Preserves schema of hits; adds 'rerank_score', optional 'fusion_score', and 'rank_reranked' fields

Note: The code appears well-structured, but some sections could be refactored for better readability. Additionally, there are some magic numbers and hard-coded values (e.g., `256` in the LRU cache) that might benefit from being replaced with named constants or configurable parameters.

### `intergrax\rag\vectorstore_manager.py`

Here is the requested documentation:

**Description:** This module provides a unified vector store manager for Intergrax, supporting ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**

* Initialize target store (Chroma, Qdrant, or Pinecone) based on configuration
* Create collection/index (lazy initialization for Qdrant/Pinecone)
* Upsert documents + embeddings with batching
* Query top-K by cosine/dot/euclidean similarity
* Count vectors
* Delete by IDs

**Note:** This file appears to be a core component of the Intergrax framework, providing vector store management functionality. It is not marked as experimental or auxiliary, suggesting it is intended for production use.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module provides a windowed (map→reduce) layer on top of the base Answerer for retrieving and synthesizing answers from large datasets.

**Domain:** RAG logic

**Key Responsibilities:**

*   Retrieves candidates from the retriever
*   Builds windows of candidates
*   Synthesizes partial answers within each window using LLM
*   Combines partial answers to form a final answer
*   Deduplicates sources and returns final answer, sources, and statistics

### `intergrax\runtime\__init__.py`

DESCRIPTION: This is the top-level initialization module for the Intergrax runtime environment, responsible for bootstrapping and importing core components.

DOMAIN: Runtime Initialization

KEY RESPONSIBILITIES:
• Imports and initializes core modules and sub-packages
• Sets up global configurations and constants
• Defines entry points for the application

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: Initializes and sets up the drop-in knowledge mode functionality for the Intergrax runtime.
Domain: LLM adapters

Key Responsibilities:
• Registers the necessary components for drop-in knowledge mode
• Sets up the knowledge graph and related data structures
• Configures the interaction between the LLM adapter and the knowledge graph

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides utilities for resolving attachments in the Intergrax framework's Drop-In Knowledge Mode.

**Domain:** RAG (Retrieve, Annotate, Generate) logic & Utilities

**Key Responsibilities:**

* Defines the `AttachmentResolver` protocol to abstract attachment resolution.
* Implements `FileSystemAttachmentResolver` to handle local filesystem-based URIs.
* Provides utility functions for resolving attachments into local file paths.
* Decouples attachment storage from consumption by Intergrax RAG components.

Note: This module appears to be a standard part of the Intergrax framework, and its purpose is clear. There are no indications of it being experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This configuration file defines the settings and dependencies of the Drop-In Knowledge Mode runtime in Intergrax. It outlines the interaction with various components such as LLM adapters, RAG vectorstore managers, web search executors, and tools agents.

**Domain:** Runtime Configuration

**Key Responsibilities:**

- Defines the primary LLM adapter for generation.
- Specifies the embedding manager for RAG/document indexing and retrieval.
- Configures the vectorstore manager for semantic search over stored chunks.
- Enables/disables various features such as Retrieval-Augmented Generation, real-time web search, long-term memory, and short-term user profile memory.
- Manages multi-tenancy settings with tenant ID and workspace ID.
- Sets limits for chat history messages and tokens.
- Configures RAG and web search parameters including token budgets and score thresholds.
- Specifies the tools agent execution policy with optional invocation of tools.

### `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`

**Description:** This module provides a context builder for Drop-In Knowledge Mode in the Intergrax framework, responsible for preparing an LLM-ready context based on user requests and session metadata.

**Domain:** RAG (Regressive Attention-based Generator) logic, Context building

**Key Responsibilities:**

- Deciding whether to use RAG for a given request
- Retrieving relevant document chunks from the vector store for the current session
- Building the system prompt, reduced chat history, and list of retrieved chunks with debug information
- Returning a `BuiltContext` object that can be translated into OpenAI-style messages by the runtime engine

**Note:** This module appears to be intentionally minimal and intended as an extension point for RAG. It does not handle LLM adapter implementation or prompt serialization concerns, which are left to the runtime engine.

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

Description: This module implements the core runtime engine for Intergrax's Drop-In Knowledge Mode, which enables a high-level conversational interface with LLM adapters, RAG, web search, and tools.

Domain: Conversational Runtime Engine

Key Responsibilities:
- Loads or creates chat sessions.
- Appends user messages to the session.
- Builds a conversation history for the LLM.
- Integrate with LLM adapters (currently left as TODO).
- Ingest attachments and index them for RAG.
- Build rich context from memory, RAG, web search, and tools.
- Support agentic flows via a supervisor.
- Expose observability hooks and cost tracking.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

**Description:** 
This module provides an attachment ingestion pipeline for Drop-In Knowledge Mode, which resolves AttachmentRef objects to loader-compatible paths, loads and splits documents using Intergrax RAG components, embeds them, and stores vectors in a vector database.

**Domain:** Data Ingestion

**Key Responsibilities:**

- Resolve AttachmentRef objects into filesystem Paths via AttachmentResolver
- Load documents using IntergraxDocumentsLoader.load_document(...)
- Split documents into chunks using IntergraxDocumentsSplitter.split_documents(...)
- Embed chunks (via IntergraxEmbeddingManager)
- Store vectors (via IntergraxVectorstoreManager)
- Return a structured IngestionResult per attachment

This module appears to be part of the main functionality and does not appear to be experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\drop_in_knowledge_mode\rag_prompt_builder.py`

Description: This module provides utilities for building the RAG-related part of a prompt in Drop-In Knowledge Mode.

Domain: RAG logic

Key Responsibilities:
- Provide a strategy interface (RagPromptBuilder) for custom implementation and control over the prompt.
- Offer a default implementation (DefaultRagPromptBuilder) that constructs the system prompt using BuiltContext and formats retrieved chunks into a single additional message.
- Format the context text in a model-friendly way, avoiding internal markers and focusing on semantic context.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

**Description:** This module defines dataclasses for request and response structures used in the Drop-In Knowledge Mode runtime.

**Domain:** RAG logic

**Key Responsibilities:**

* Define high-level contracts between applications (FastAPI, Streamlit, CLI, MCP, etc.) and the DropInKnowledgeRuntime
* Expose citations, routing information, tool calls, and basic statistics
* Provide dataclasses for request (`RuntimeRequest`) and response (`RuntimeAnswer`) structures
* Hide low-level implementation details while maintaining enough structure for exposing relevant metadata

Note: This module appears to be well-structured and complete. No indications of being experimental, auxiliary, legacy, or incomplete are found.

### `intergrax\runtime\drop_in_knowledge_mode\session_store.py`

**Description:** This module provides session storage abstractions and utilities for the Intergrax framework's Drop-In Knowledge Mode runtime.

**Domain:** Session Management / Data Storage

**Key Responsibilities:**
* Define data classes for chat sessions (`ChatSession`) and messages (`SessionMessage`)
* Establish a protocol for session persistence (`SessionStore`)
* Implement an in-memory `InMemorySessionStore` for quick experiments and notebooks
* Provide methods for loading, creating, saving, and appending to sessions
* Enable listing recent sessions for a given user

### `intergrax\runtime\drop_in_knowledge_mode\websearch_prompt_builder.py`

Description: This module provides utilities for building prompts related to web search results in the Drop-In Knowledge Mode of the Intergrax framework.

Domain: LLM adapters

Key Responsibilities:
- Provides a strategy interface (WebSearchPromptBuilder) for building web search prompts.
- Defines a default implementation (DefaultWebSearchPromptBuilder) that takes a list of web documents and builds a system-level message listing titles, URLs, and snippets.
- Offers basic debug information such as the number of documents and top URLs.

### `intergrax\supervisor\__init__.py`

**File Path:** intergrax\supervisor\__init__.py

**Description:** The supervisor module initializes and sets up the Intergrax framework's supervision logic, responsible for overseeing various components and their interactions.

**Domain:** Framework Core

**Key Responsibilities:**
- Initializes the supervisor component
- Sets up event listeners and handlers
- Defines supervisor-related configuration options
- Provides API for registering and unregistering supervisors

### `intergrax\supervisor\supervisor.py`

**Description:** This module implements the Intergrax Supervisor, responsible for planning and executing tasks based on user input. It provides utilities for interacting with Large Language Models (LLMs) and managing task components.

**Domain:** Agent/Supervisor

**Key Responsibilities:**

*   Planning tasks using two-stage decomposition:
    *   Stage 1: Decompose the query into a plan using LLMs
    *   Stage 2: Assign components to each step in the plan
*   Managing task components and their execution context
*   Interacting with LLMs for text extraction, JSON parsing, and score calculation
*   Analyzing plans and computing diagnostics (e.g., router scores)
*   Registering and listing available components
*   Setting up prompts and prompts packs for the supervisor

### `intergrax\supervisor\supervisor_components.py`

Description: This module defines the core components and utilities for the Integrax framework's supervisor, including component registration, context management, and execution.

Domain: Supervisor Components

Key Responsibilities:
- Defines data structures for PipelineState and ComponentResult.
- Establishes a Component class with attributes (name, description, etc.) and methods (run).
- Provides a decorator (component) for registering components.
- Offers utility classes (ComponentContext, ComponentResult) for component execution.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides default prompt templates for the Intergrax framework's Supervisor, including system and user-defined prompts. The Supervisor is responsible for generating a precise, auditable DAG (Directed Acyclic Graph) plan that an executor will run.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Defines default prompt templates for the Supervisor
* Includes explanations of universal en-prompt principles and validation rules
* Provides example prompts with placeholders for user inputs and component outputs
* Contains strict rules for resource flags, DAG structure, gates, synthesis steps, reliability, and decomposition coverage and integrity
* Lists validation checklist for self-checking before returning a plan
* Returns only one JSON object with a specific schema

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module provides utilities for transforming a plan into a runnable LangGraph pipeline. It handles state management, node creation, and graph construction.

**Domain:** Supervisor Logic

**Key Responsibilities:**

*   State Management:
    *   Ensures default state values are set (artifacts, step status, debug logs, scratchpad)
    *   Appends log messages to the debug logs
    *   Resolves inputs for plan steps based on artifacts and state
    *   Persists node results into artifacts
*   Node Factory:
    *   Creates a LangGraph node function that executes one plan step
    *   Handles different execution methods (GENERAL, TOOL, RAG)
    *   Calls component instance execute method when applicable
*   Graph Construction:
    *   Performs stable topological ordering of plan steps based on dependencies
    *   Builds the LangGraph pipeline from a plan and components

The module appears to be well-structured and provides a clear interface for transforming plans into executable pipelines. The code is clean, and the comments are concise and informative.

### `intergrax\system_prompts.py`

Here's the requested documentation for the file `intergrax\system_prompts.py`:

**Description:** This module defines a default RAG (Restricted Answer Generation) system instruction template for the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a structured response format for answering user questions based on available documents
* Includes instructions on how to verify consistency, cite sources, and handle uncertainty
* Defines guidelines for formatting responses, including using precise terminology and avoiding speculation
* Outlines the importance of transparency and providing clear references to supporting documentation

### `intergrax\tools\__init__.py`

DESCRIPTION: This module serves as the entry point for Intergrax's tools package, responsible for initializing and configuring other tool modules.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
• Initializes tool modules
• Configures inter-module dependencies 
• Sets up global tool context

### `intergrax\tools\tools_agent.py`

**Description:** This module provides a high-level interface for integrating various tools with a conversational AI system. It acts as an orchestrator, managing the interaction between these tools and the LLM.

**Domain:** Tools Integration / LLM Adapters

**Key Responsibilities:**

*   Manages tool invocation and output processing
*   Provides a JSON-based planner for tool orchestration
*   Supports native tools (e.g., OpenAI) and non-native tools (e.g., Ollama)
*   Offers a flexible way to handle input data, including both user inputs and pre-built conversation contexts
*   Enables the creation of an output structure based on various formats (Pydantic models or regular classes)

**Note:** This module appears to be well-structured and complete. It is not marked as experimental or auxiliary.

### `intergrax\tools\tools_base.py`

**intergrax\tools\tools_base.py**

Description: This module provides base classes and utilities for Intergrax tools, including tool registration and output handling.

Domain: Tools/Utilities

Key Responsibilities:
* Provides a `ToolBase` class with basic attributes (name, description, schema model) and methods (get parameters, run, validate args, to OpenAI schema).
* Implements a `ToolRegistry` class for storing and exporting tools in a format compatible with the OpenAI Responses API.
* Includes a `_limit_tool_output` function for safely truncating tool outputs.
* Offers optional Pydantic support using stubs if the library is not installed.

### `intergrax\websearch\__init__.py`

Description: This module initializes and configures the web search functionality within the Intergrax framework.

Domain: Web Search Integration

Key Responsibilities:
- Initializes the web search module
- Configures search parameters (e.g., API keys, indexing settings)
- Registers web search-related utility functions

### `intergrax\websearch\cache\__init__.py`

**Description:** 
This module provides a simple in-memory query cache for web search results with optional time-to-live (TTL) and maximum size.

**Domain:** Web Search Cache

**Key Responsibilities:**

* Stores serialized web documents using immutable QueryCacheKey instances
* Provides methods for getting cached results by key and setting new cache entries
* Supports TTL and max size configuration for cache eviction
* Utilizes a dictionary-based in-memory store, suitable for single-process applications (e.g., notebooks, local development)
* Aims to be replaced with a distributed backend (e.g., Redis) in the future

**Note:** This module is a part of the Intergrax framework and appears to be a core component of its web search functionality.

### `intergrax\websearch\context\__init__.py`

Description: This module initializes the web search context, providing essential components for web-related operations.

Domain: Web Search Context

Key Responsibilities:
- Initializes the web search context
- Sets up necessary dependencies and configurations
- Exposes contextual information for downstream components

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module, `websearch_context_builder.py`, is responsible for constructing a textual context and chat messages from web search results. It provides methods to build LLM-ready context strings and user-facing prompts.

**Domain:** RAG logic (Reinforcement Learning of Alternatives with Generation)

**Key Responsibilities:**

* Building a textual context string from WebDocument objects
* Building a textual context string from serialized web documents
* Constructing system and user prompts for chat-style LLMs
* Enforcing strict "sources-only" mode rules in system prompts

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: This module initializes the web search fetcher component of Intergrax, responsible for retrieving data from external sources.

DOMAIN: Web Search Fetcher

KEY RESPONSIBILITIES:
* Initializes the web search engine API connection
* Configures fetcher settings and parameters
* Defines exception handling for API errors

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module provides utilities for extracting metadata and content from web pages, including title, description, language, Open Graph tags, and plain text.

**Domain:** Web search, data ingestion

**Key Responsibilities:**

* Perform lightweight HTML extraction on a `PageContent` instance (extract_basic function)
	+ Extract <title>
	+ Extract meta description
	+ Extract <html lang> attribute
	+ Extract Open Graph meta tags (og:)
	+ Produce a plain-text version of the page
* Perform advanced readability-based extraction on a `PageContent` instance (extract_advanced function)
	+ Remove obvious boilerplate elements (scripts, styles, iFrames, navigation)
	+ Prefer trafilatura (when available) to extract primary readable content
	+ Fallback to BeautifulSoup plain-text extraction if trafilatura fails
	+ Normalize whitespace and reduce noise

**Note:** This file appears to be part of the Intergrax framework's web search module, and its purpose is to provide a robust way to extract metadata and content from web pages.

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module provides a high-level interface for fetching web pages over HTTP.

**Domain:** Web Search Fetcher

**Key Responsibilities:**

* Performs an asynchronous HTTP GET request to fetch a single URL
* Merges custom headers with default headers
* Captures final URL, status code, raw HTML, and body size
* Returns a PageContent instance on success or None on transport-level failure

### `intergrax\websearch\integration\__init__.py`

DESCRIPTION: This module serves as the entry point for web search integration within Intergrax, handling imports and initialization of external libraries.

DOMAIN: Integration

KEY RESPONSIBILITIES:
• Initializes and sets up external dependencies required for web search functionality
• Defines the import structure for integration with other modules 
• Possibly handles configuration or setup related to web search capabilities

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a LangGraph-compatible web search node implementation, encapsulating configuration and delegation to an external WebSearchExecutor instance. It supports both synchronous and asynchronous operation modes.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**
- Encapsulates configuration for the WebSearchExecutor
- Provides synchronous (`run`) and asynchronous (`run_async`) methods operating on `WebSearchState` instances
- Delegates search tasks to the external WebSearchExecutor instance
- Offers a functional, synchronous wrapper (`websearch_node`) and an async wrapper (`websearch_node_async`) around the default node instance

### `intergrax\websearch\pipeline\__init__.py`

DESCRIPTION: This is the entry point for the Intergrax web search pipeline, responsible for initializing and configuring the pipeline components.

DOMAIN: Web Search Pipeline

KEY RESPONSIBILITIES:
* Initializes the pipeline with default settings
* Configures the pipeline components based on environment variables
* Sets up logging and monitoring for the pipeline

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module defines the SearchAndReadPipeline class, which orchestrates multi-provider web search, fetching, extraction, deduplication, and quality scoring.

**Domain:** Websearch pipeline

**Key Responsibilities:**

* Orchestrates multi-provider web search
* Fetches and extracts documents from search hits
* Performs basic HTML extraction, quality scoring, and dedupe key computation
* Supports rate-limited HTTP GETs with asyncio
* Deduplicates documents based on dedupe_key
* Sorts results by quality_score (descending) then source_rank (ascending)

This file appears to be a core component of the websearch pipeline in the Intergrax framework. It provides a comprehensive set of functionalities for searching, fetching, and processing web documents. The code is well-structured, and the use of asyncio and rate limiting ensures efficient execution of tasks.

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: This module initializes the web search providers for Intergrax, allowing for multiple search engine integrations.
DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
- Initializes the web search provider instances
- Registers available search engines with their respective configurations
- Provides access to the registered providers for use in other parts of the framework

### `intergrax\websearch\providers\base.py`

Description: This module defines a base interface for web search providers in the Integrax framework.

Domain: Web Search Provider Interface

Key Responsibilities:
- Provides an abstract interface for various web search providers (Google, Bing, DuckDuckGo, etc.)
- Accepts a provider-agnostic QuerySpec
- Returns a ranked list of SearchHit items
- Exposes minimal capabilities for feature negotiation (language, freshness)
- Allows individual provider implementations to override specific methods (search, close)

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module implements a Bing Web Search provider for the Intergrax framework, allowing integration with the Bing v7 REST API.

**Domain:** Web search providers

**Key Responsibilities:**

* Initializes and authenticates with the Bing v7 API using an API key
* Supports language and region filtering through environment variables or query specifications
* Enables freshness filtering (Day, Week, Month) through query specifications
* Supports safe search filtering (Off, Moderate, Strict)
* Retrieves search results from the Bing API, parsing and transforming them into SearchHit objects

Note: The file appears to be part of a larger framework and seems complete, with no obvious signs of being experimental or legacy.

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** 
This module implements a custom web search provider for Intergrax, utilizing the Google Custom Search (CSE) REST API.

**Domain:**
Websearch providers

**Key Responsibilities:**

* Initializes the Google CSE provider with environment variables for API key and search engine ID
* Builds parameters for each query, including language filtering, content filtering, and UI language handling
* Performs a GET request to the Google CSE API using the built parameters
* Extracts relevant data from the API response and returns it as SearchHit objects
* Handles errors and exceptions during the search process

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a Google Places API provider for web search, allowing users to query business data and map locations.

**Domain:** Web Search Providers (RAG logic)

**Key Responsibilities:**

* Provides a Google Places API provider class (`GooglePlacesProvider`)
* Handles text search queries and returns core business data
* Supports fetching extended details for single places using the `fetch_details` parameter
* Maps search results to `SearchHit` objects
* Utilizes environment variables and optional parameters for configuration

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a Reddit search provider for the Intergrax framework, utilizing the official OAuth2 API.

**Domain:** Web search providers

**Key Responsibilities:**
- Establishes connection with Reddit's OAuth2 API
- Handles authentication and token management
- Supports searching through title and body content
- Returns rich post metadata (score, num_comments, upvote_ratio, nsfw)
- Fetches top-level comments for each post (optional)

### `intergrax\websearch\schemas\__init__.py`

**Description:** The `__init__.py` file in the web search schema package is responsible for initializing and exporting schema-related functionality.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Initializes schema package
* Exports schema-related functions and classes

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module provides a dataclass for representing the content of a web page, including its metadata and optional extracted information.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Encapsulates raw HTML and derived metadata of a web page
* Provides fields for various types of metadata (e.g., title, description, language)
* Offers methods for filtering out empty or failed fetches (`has_content`)
* Enables truncation of text snippets for logging and debugging purposes (`short_summary`)
* Computes the approximate size of content in kilobytes (`content_length_kb`)

### `intergrax\websearch\schemas\query_spec.py`

**intergrax\websearch\schemas\query_spec.py**

Description: This module defines a data model (QuerySpec) for standardizing search queries across various web providers.

Domain: Web Search Query Schema

Key Responsibilities:
- Defines a canonical query specification with minimal, provider-agnostic, and stable fields.
- Provides methods to normalize the query string with site filtering and cap the top results per provider.

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a dataclass `SearchHit` for representing provider-agnostic metadata of a single search result entry.

**Domain:** RAG logic (Richer Abstract Generation)

**Key Responsibilities:**

* Provides a structured representation of a search result entry
* Includes metadata such as provider ID, query string, rank, title, URL, snippet, and publication datetime
* Performs minimal safety checks on the `rank` and `url` fields
* Offers utility methods for extracting domain information from the URL and creating a minimal dictionary representation of the hit

### `intergrax\websearch\schemas\web_document.py`

**Description:** This module defines a unified structure for representing fetched and processed web documents, connecting search hit metadata with extracted content and analysis results.

**Domain:** Web Search Schema

**Key Responsibilities:**

* Provides a `WebDocument` dataclass to represent a unified web document structure
* Connects original search hit metadata with extracted content and analysis results
* Offers methods for:
	+ Validating the document's content and URL (`is_valid`)
	+ Merging textual content for LLM or retrieval embedding (`merged_text`)
	+ Generating a short summary line used in logs or console outputs (`summary_line`)

### `intergrax\websearch\service\__init__.py`

Description: This is the entry point for the web search service, responsible for initializing and configuring the underlying components.

Domain: Web Search Service

Key Responsibilities:
- Initializes the web search engine instance
- Sets up configuration for web search queries
- Registers necessary dependencies and services
- Provides an interface for external interaction with the web search functionality

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** 
This module provides a high-level helper class, `WebSearchAnswerer`, which runs web searches via an executor, builds LLM-ready context/messages from web documents, and calls any LLMAdapter to generate a final answer.

**Domain:** Web search service

**Key Responsibilities:**
- Runs web searches using a `WebSearchExecutor` instance
- Builds LLM-ready messages from web documents using a `WebSearchContextBuilder`
- Calls an `LLMAdapter` to generate a final answer
- Provides synchronous and asynchronous interfaces for answering questions

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module contains the core web search functionality for the Intergrax framework, enabling users to execute web searches and retrieve relevant results.

**Domain:** Web Search Service

**Key Responsibilities:**

* Construct a `QuerySpec` object from a raw query and configuration settings.
* Execute the `SearchAndReadPipeline` with chosen providers (e.g., Google CSE or Bing Web).
* Convert `WebDocument` objects into LLM-friendly dictionaries for easy processing.

The `WebSearchExecutor` class serves as the primary entry point for web search operations, handling various configuration options and provider settings. It also includes caching mechanisms to improve performance by storing serialized results in memory.

**Notes:** This file appears well-maintained and production-ready, with clear documentation and type hints throughout. The code structure is logical, and the use of static methods (e.g., `_build_default_providers`) helps encapsulate specific tasks.

### `intergrax\websearch\utils\__init__.py`

Description: This module initializes and exports utility functions for web search operations within the Intergrax framework.

Domain: Utility modules

Key Responsibilities:
- Initializes global configuration settings for web search utilities
- Exports a singleton instance of the WebSearchUtil class for use across the application
- Defines any necessary constants or variables for web search functionality

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides simple deduplication utilities for web search pipeline, including text normalization and SHA-256 based key generation.

**Domain:** RAG logic (deduplication)

**Key Responsibilities:**

* Normalizes input text for deduplication
	+ Treats None as empty string
	+ Strips leading/trailing whitespace
	+ Converts to lower case
	+ Collapses internal whitespace sequences
* Generates a stable SHA-256 based deduplication key from normalized text

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module provides a simple asyncio-compatible token bucket rate limiter for controlling the rate of operations.
**Domain:** RAG logic (Rate limiting and Access control)
**Key Responsibilities:**
- Provides a `TokenBucket` class for rate limiting
- Offers methods to acquire tokens (`acquire`) and try to acquire tokens without blocking (`try_acquire`)
- Ensures safe concurrent usage across coroutines in the same process

### `main.py`

Description: This module serves as the entry point for the Integrax framework, responsible for executing the primary program flow when run directly.

Domain: Configuration/Initialization

Key Responsibilities:
- Provides a main execution path for the Intergrax framework.
- Contains the initial setup and initialization logic.
- Serves as an entry-point for testing or direct execution of the framework.

### `mcp\__init__.py`

DESCRIPTION: This is the main entry point of the Intergrax framework, responsible for setting up and initializing its components.

DOMAIN: Framework Configuration

KEY RESPONSIBILITIES:
• Initializes core modules and services
• Registers available adapters and integrations
• Defines default configuration settings and paths
• Imports and sets up framework-wide utilities

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

Description: This notebook is a basic sanity-check for the Drop-In Knowledge Mode runtime in the Intergrax framework, verifying its ability to create or load a session, append user and assistant messages, build conversation history from SessionStore, and return a RuntimeAnswer object.

Domain: RAG logic / Agents

Key Responsibilities:
- Instantiate an InMemorySessionStore
- Instantiate a LangChainOllamaAdapter as LLM adapter
- Initialize the vector store at startup
- Create a minimal RuntimeConfig instance
- Instantiate a DropInKnowledgeRuntime instance with the created config and session store
- Perform a basic sanity-check by creating a new session and appending user messages

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the functionality of the Intergrax framework's Drop-In Knowledge Mode, specifically focusing on attachments ingestion and basic conversational memory handling.

**Domain:** LLM adapters, RAG logic, Data Ingestion, Agents (conversational memory management)

**Key Responsibilities:**

* Initializes an in-memory session store for testing purposes
* Sets up an LLM adapter using Ollama + LangChain
* Configures embedding and vector store managers
* Creates a RuntimeConfig instance to initialize the Drop-In Knowledge Mode runtime
* Demonstrates how to prepare an AttachmentRef for local file ingestion

**Notes:**

This notebook appears to be an experimental or demonstration code, focusing on specific components of the Intergrax framework rather than providing a comprehensive overview. It is likely meant to serve as a reference or starting point for further development and testing within the framework.

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

Description: This is a Jupyter Notebook that demonstrates the usage of the `ContextBuilder` in Intergrax's Drop-In Knowledge Mode runtime.

Domain: RAG logic, Data Ingestion, Runtime Configuration

Key Responsibilities:
- Initializes session store and LLM adapter.
- Sets up embedding manager and vector store connection.
- Configures runtime with enablement of RAG functionality.
- Demonstrates usage of `ContextBuilder` to build context for a single user question.

### `notebooks\drop_in_knowledge_mode\04_websearch_context_demo.ipynb`

**Description:** This Jupyter Notebook demonstrates the use of the Intergrax framework's Drop-In Knowledge Runtime with web search functionality, showcasing how to integrate live web search via `WebSearchExecutor` into a chat session.

**Domain:** RAG logic, websearch integration, LLM adapters

**Key Responsibilities:**

* Initializes core runtime configuration for Drop-In Knowledge Runtime
	+ Session store (in-memory storage for chat messages and metadata)
	+ LLM adapter (Ollama through LangChain adapter)
	+ Embedding manager (matching the model used during ingestion)
	+ Vector store configuration (same collection as in RAG demo)
* Initializes web search executor with Google CSE provider
* Configures runtime settings, including enabling RAG and web search features
* Demonstrates creating a fresh chat session for web search demo
* Provides helper function `ask(question: str)` for interactive testing

### `notebooks\drop_in_knowledge_mode\05_tools_context_demo.ipynb`

Description: This notebook demonstrates the integration of tools into the Drop-In Knowledge Runtime, showcasing how to use a tools orchestration layer on top of conversational memory, RAG (attachments ingested into a vector store), and live web search context.

Domain: RAG logic, data ingestion, agents

Key Responsibilities:
- Configures Python path for importing `intergrax` package
- Loads environment variables (API keys)
- Imports core building blocks used by the Drop-In Knowledge Runtime
- Initializes session store (in-memory storage for chat messages & metadata)
- Implements two demo tools (`WeatherTool`, `CalcTool`) using the Intergrax tools framework
- Registers tools in a `ToolRegistry`
- Attaches an `IntergraxToolsAgent` instance to `RuntimeConfig.tools_agent`

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates an end-to-end RAG workflow that combines multiple knowledge sources into a single in-memory vector index and exposes it through a LangGraph-based agent.

**Domain:** Hybrid Multi-Source RAG with Intergrax + LangGraph

**Key Responsibilities:**
* Ingest content from multiple sources (local PDF files, local DOCX/Word files, live web results using the Intergrax `WebSearchExecutor`)
* Build a unified RAG corpus by normalizing documents into a common internal format and attaching basic metadata
* Create an in-memory vector index using an Intergrax embedding manager and vectorstore manager
* Answer user questions with a RAG agent that generates a single, structured report

**Note:** The file is experimental/incomplete as it appears to be a demonstration of a complex workflow rather than a production-ready implementation.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This Jupyter Notebook demonstrates the integration between Intergrax and LangGraph by creating a simple LLM QA graph. It showcases how to use an Intergrax LLM adapter as a node inside a LangGraph graph.

Domain: LLM adapters, Graph composition, Integration examples

Key Responsibilities:
- Initializes an OpenAI Chat Responses Adapter instance.
- Defines a `SimpleLLMState` class to hold chat messages and the final answer produced by the LLM node.
- Creates a `llm_answer_node` function that calls the Intergrax LLM adapter to generate responses.
- Builds a simple StateGraph with a single node (`llm_answer_node`) and demonstrates its functionality.

Note: This code is part of a larger Jupyter Notebook, and some sections might be missing due to formatting constraints.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

**Description:** This is a Jupyter Notebook that demonstrates the implementation of a practical web research agent using the Intergrax framework. The notebook showcases how to build and execute a multi-step graph-based agent for "no-hallucination" web-based Q&A.

**Domain:** Web Research Agent (Intergrax WebSearch + LangGraph)

**Key Responsibilities:**

* Initializes LLM adapter and WebSearch components
* Defines the graph state for the Web Research Agent
* Implements nodes:
	+ Normalize user question
	+ Run web search using Intergrax WebSearchExecutor
* Demonstrates how to use Intergrax components:
	+ `WebSearchExecutor` – orchestrates web search providers (Google, Bing, Reddit, etc.)
	+ `WebSearchContextBuilder` – builds a compact textual context + citations
	+ `WebSearchAnswerer` – uses an LLM adapter to answer strictly from the web context

**Status:** This notebook appears to be a well-documented example of how to build and execute a Web Research Agent using Intergrax. It is likely intended for demonstration or educational purposes, rather than being a production-ready codebase.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This Jupyter Notebook demonstrates the use of the Intergrax framework with OpenAI's RAG (Reactive Agent) functionality to interact with a Vector Store.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Initializes an OpenAI client and loads environment variables.
- Creates an instance of `IntergraxRagOpenAI` to interact with the Vector Store.
- Ensures the existence of the Vector Store, clears it, and uploads a local folder to the store.
- Runs queries on the Vector Store using the RAG logic.

### `notebooks\rag\chat_agent_presentation.ipynb`

Description: This Jupyter notebook demonstrates the creation of a hybrid chat agent utilizing the Intergrax framework, integrating RAG logic, LLM adapters, and tool-based functionality.

Domain: RAG (Retriever-Answerer) logic and Chat Agent implementation

Key Responsibilities:
- Initializes an Ollama LLM adapter with a specific model.
- Creates an instance of the Conversational Memory component.
- Defines a demo Weather tool for simulating weather API responses.
- Configures Vector Store and RAG components using Chroma as the provider.
- Sets up an answerer configuration with top-k retrieval, minimum score filtering, and re-ranking parameters.
- Initializes a RagComponent instance for responding to document-based queries.
- Creates a high-level hybrid chat agent combining RAG, tools, and LLM chat functionality.

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the integration of various components within the Integrax framework, including LLM adapters, RAG logic, and tools agents. It showcases how these components can be combined to provide structured output for user queries.

**Domain:** RAG (Retriever-Augmented Generator) logic

**Key Responsibilities:**

* Demonstrates the usage of Intergrax's RAG components, such as embedding manager, retriever, re-ranker, and answerer.
* Shows how to integrate these components with an LLM adapter (in this case, Ollama) and a tools agent.
* Provides examples of using Pydantic for structured output, including defining schemas for the output model.
* Includes a stub implementation of a weather tool that returns demo data.

**Note:** The notebook appears to be well-documented and includes clear explanations of each component's purpose. However, it is marked as a "demo" or "stub" in some places, indicating that this might not be production-ready code.

### `notebooks\rag\rag_custom_presentation.ipynb`

Description: This notebook contains a collection of code cells that demonstrate how to load documents, split them into chunks, and generate vector embeddings for each chunk using the Intergrax framework. The code showcases the integration of various components such as document loaders, splitters, and embedding managers.

Domain: RAG (Retrieval-Augmented Generation) pipeline utilities

Key Responsibilities:
- Load documents from a specified directory in various formats.
- Split loaded documents into smaller chunks based on paragraphs or tokens.
- Generate vector embeddings for each document chunk using an embedding manager.
- Initialize a vector store at startup and perform lightweight presence checks before ingestion.
- Create an embedding manager for probe queries to check if the target corpus is already present in the vector store.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter notebook script demonstrates various components of the Intergrax framework, including document loading, splitting, embedding, and vector store management. It also includes a retriever test.

**Domain:** RAG (Reformer-based Architecture for Generative Models) logic

**Key Responsibilities:**

* Load documents from different sources (video, audio, images)
* Split and embed documents
* Manage vector store, including checking corpus presence and ingesting documents if necessary
* Deduplicate IDs after loading documents
* Add documents to the vector store with associated metadata

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter notebook script utilizes the Intergrax framework to perform multimedia processing tasks, including video and audio downloading from YouTube, transcription of videos into VTT format, extraction of frames and metadata from videos, translation of audio, and image description using a model.

**Domain:** Multimedia processing

**Key Responsibilities:**

* Downloading video and audio from YouTube
* Transcribing videos to VTT format
* Extracting frames and metadata from videos
* Translating audio
* Describing images using a model

### `notebooks\rag\tool_agent_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates a tools agent in the Integrax framework, showcasing its ability to reason and interact with various tools to answer user queries. It presents two test cases: one for retrieving weather information and another for performing arithmetic calculations.

**Domain:** RAG (Retrieval-Augmented Generation) logic / Tools Agent

**Key Responsibilities:**

* Demonstrates the integration of tools within the Integrax framework
* Defines and registers tools (e.g., `WeatherTool` and `CalcTool`) with their respective schemas and functionality
* Configures the LLM adapter using OpenLLama model for planning and control
* Orchestrates tool selection, invocation, and conversational memory updates through the IntergraxToolsAgent
* Presents two test cases to demonstrate the agent's ability to reason and interact with tools to answer user queries

**Note:** This file is part of an experimental setup, as indicated by the presence of alternative configurations (e.g., OpenAI) commented out for now.

### `notebooks\supervisor\supervisor_test.ipynb`

Description: This notebook contains several components that can be used in the Integrax framework to automate and streamline tasks, including compliance checking, cost estimation, generating a final summary report, and financial audits.

Domain: RAG logic ( Reasoning, Action, Goal)

Key Responsibilities:
- Provides a compliance checker component that verifies whether proposed changes comply with privacy policies and terms of service.
- Offers a cost estimator component that calculates the estimated cost of changes based on the UX audit report.
- Includes a final summary generator component that produces a consolidated summary using all collected artifacts.
- Implements a financial audit component that generates a mock financial report and VAT calculation.

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This Jupyter Notebook file demonstrates the use of the WebSearchExecutor with Google and Bing Search, showcasing how to search for web content using the Integrax framework.

**Domain:** RAG logic (Reactive Agents)

**Key Responsibilities:**

* Demonstrates searching for web content using Google Custom Search
* Utilizes the GoogleCSEProvider class to perform searches
* Executes a query against Google Custom Search with specific parameters
* Prints the results of the search, including provider, rank, title, URL, and snippet
* Includes an example of using Bing Search (although not fully implemented)
* Uses environment variables to set API keys for Google Custom Search
