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
- `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`
- `intergrax\runtime\drop_in_knowledge_mode\session_store.py`
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

DESCRIPTION: This is the entry point for the API, responsible for initializing and setting up the application's routes and services.

DOMAIN: API configuration

KEY RESPONSIBILITIES:
- Initializes the API application instance
- Sets up routing and URL handling mechanisms
- Configures service registration and injection

### `api\chat\__init__.py`

**api/chat/__init__.py**

Description: This module initializes and sets up the chat API, including importing necessary components and registering them with the framework.

Domain: Chat API Initialization

Key Responsibilities:
- Imports required modules and sub-packages for the chat API
- Registers the chat API components with the Intergrax framework

### `api\chat\main.py`

**Description:** This module is responsible for handling chat-related functionality within the Integrax framework. It defines a FastAPI application with endpoints for initiating conversations, uploading and indexing documents, listing existing documents, and deleting documents.

**Domain:** Chat API

**Key Responsibilities:**
- Provides a RESTful API for user-machine interactions through the `/chat` endpoint.
- Allows users to upload and index documents into Chroma using the `/upload-doc` endpoint.
- Offers functionality to list all uploaded documents via the `/list-docs` endpoint.
- Enables document deletion through the `/delete-doc` endpoint.

This module appears to be a core component of the Integrax framework, providing essential functionality for user interaction.

### `api\chat\tools\__init__.py`

Description: This module initializes and sets up the tools namespace within the chat API, providing a foundation for various utility functions to be imported and used.

Domain: LLM Adapters/Tools Utilities

Key Responsibilities:
• Initializes the tools namespace
• Sets up the tooling infrastructure for the chat API 
• Imports necessary utility modules

### `api\chat\tools\chroma_utils.py`

**Description:** This module provides tools for interacting with the Chroma database, allowing for document indexing, deletion, and management.

**Domain:** RAG (Recurrent Attention Graph) logic

**Key Responsibilities:**
- Load and split documents from a file path
- Index a document to the Chroma vector store
- Delete a document by its ID from the Chroma database

### `api\chat\tools\db_utils.py`

**Description:** This module provides a set of utilities for interacting with the Integrax framework's database, including schema creation and migration, message insertion and retrieval, and document management.

**Domain:** Database Utilities (RAG logic & Data Ingestion)

**Key Responsibilities:**

* Schema creation and migration using SQLite
* Message insertion and retrieval for user, assistant, system, and tool roles
* Document insertion and deletion with metadata storage
* Backward-compatibility entry points for application logs (now redundant)
* Public API for messages and documents

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines data models using Pydantic for serializing and deserializing API request and response objects in the chat application.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines enums for model names (e.g., LLAMA 3.1, GPT-OSS 20B)
* Creates Pydantic models for:
	+ Query input data (question, session ID, model selection)
	+ Query response data (answer, session ID, used model)
	+ Document metadata (ID, filename, upload timestamp)
	+ Delete file request data (file ID)

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides utilities and singletons for the RAG (Retriever-Augmented Generator) pipeline in the Integrax framework, including vector store management, embedding manager, retriever, reranker, and answerer instances.

**Domain:** RAG logic / Pipeline tools

**Key Responsibilities:**

* Provides singleton instances of vector store, embedder, retriever, reranker, and answerer
* Offers methods for getting these instances (_get_vectorstore, _get_embedder, _get_retriever, _get_reranker)
* Defines the default user prompt and system prompt templates for interacting with the RAG pipeline
* Allows building LLM adapters using the `_build_llm_adapter` function

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides utility functions for interacting with the API endpoints of the Integrax framework, including sending requests, uploading and managing documents.

**Domain:** Data Ingestion / File Management

**Key Responsibilities:**

* Provides a function to get an API endpoint URL based on a given name
* Functionality to send POST requests to the chat endpoint with JSON data
* Function to upload files using HTTP POST request
* Function to list uploaded documents by sending a GET request to the list-docs endpoint
* Function to delete documents by sending a POST request to the delete-doc endpoint

### `applications\chat_streamlit\chat_interface.py`

**Description:** This module provides a Streamlit interface for interacting with the chat functionality, allowing users to input queries and receive responses.

**Domain:** RAG logic (Chat Interface)

**Key Responsibilities:**
* Initializes and displays a chat interface using Streamlit
* Handles user input via `st.chat_input` and appends it to the message list
* Sends API requests for responses using `get_api_response`
* Updates the session state with response data, including session ID and generated answer
* Displays error messages when API requests fail

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides a Streamlit sidebar for interacting with the Intergrax framework's API, including model selection, document upload, listing, and deletion.

**Domain:** Chat Interface Utilities

**Key Responsibilities:**
- Provides a model selection dropdown in the Streamlit sidebar.
- Allows users to upload documents via the sidebar file uploader.
- Lists uploaded documents with their IDs and timestamps.
- Enables deleting selected documents from the list.

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module implements the user interface for a chatbot application using Streamlit, integrating a sidebar and chat interface.

**Domain:** RAG logic (chatbot)

**Key Responsibilities:**
* Initializes the Streamlit app with title "intergrax RAG Chatbot"
* Sets default state values for messages and session ID
* Displays the sidebar and chat interface

Note: The code appears to be part of a larger framework, possibly still under development or not fully documented.

### `applications\company_profile\__init__.py`

DESCRIPTION: This module initializes and configures the company profile application, setting up its components and dependencies.

DOMAIN: Application Initialization

KEY RESPONSIBILITIES:
* Initializes the CompanyProfile application instance
* Configures application-wide settings and constants
* Registers necessary modules and services for the application
* Sets up dependency injection for the application's components

### `applications\figma_integration\__init__.py`

**Description:** This module serves as the entry point and interface for the Figma integration with the Intergrax framework.

**Domain:** Integration Utilities

**Key Responsibilities:**
* Initializes the Figma integration
* Exposes methods for interacting with Figma APIs
* Sets up connections between Figma and other Intergrax components

### `applications\ux_audit_agent\__init__.py`

DESCRIPTION: The __init__.py file is a special Python file that serves as the entry point for the ux_audit_agent application.

DOMAIN: Agents

KEY RESPONSIBILITIES:
• Initializes the audit agent application.
• Defines the main entry points and interfaces for the application.
• Possibly contains metadata or configuration for the agent.

### `applications\ux_audit_agent\components\__init__.py`

DESCRIPTION: This module initializes the UX audit agent components, handling setup and configuration for various tasks.

DOMAIN: Agents

KEY RESPONSIBILITIES:
• Initializes component modules within the UX audit agent.
• Provides a central entry point for component registration.
• Handles configuration and setup for downstream tasks.

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module provides a policy and privacy compliance checker, which evaluates proposed changes against the privacy policy and regulations.

**Domain:** RAG logic (Regulatory Audit and Governance)

**Key Responsibilities:**
- Evaluates proposed UX changes for compliance with privacy policy and regulations.
- Simulates compliance result with an 80% chance of being compliant.
- Generates findings on policy violations, requires DPO review, and notes for non-compliant scenarios.
- Stops execution if non-compliant, requiring corrections or DPO review.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module defines a cost estimation agent that provides a mock calculation for the estimated cost of UX-related changes based on an audit report. The agent is registered as a component in the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**
- Provides a mock pricing model to estimate the cost of UX updates.
- Retrieves the audit report from the pipeline state and extracts issues.
- Calculates the estimated cost based on the number of issues found.
- Returns a ComponentResult with the estimated cost, currency, and method used.

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module defines a component for generating a final report of the execution pipeline, including collected artifacts.

**Domain:** RAG (Results, Analysis, and Governance) logic

**Key Responsibilities:**
- Generates a complete summary of the entire execution pipeline using all collected artifacts.
- Produces a "final_report" with relevant details such as pipeline status, terminated reason, project manager decision, and citations.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module provides a component for generating mock financial reports and VAT calculations, which can be used in testing environments. The component is designed to simulate realistic financial data and computations.

**Domain:** Financial Agent

**Key Responsibilities:**

* Generates mock financial reports with test data
* Calculates VAT amounts based on predefined rates
* Provides example use cases for testing financial computations and budget constraints
* Returns a ComponentResult object containing the generated report and logs

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component for the Intergrax system, answering questions about its modules, architecture, and documentation.
**Domain:** LLM adapters
**Key Responsibilities:**
- Provides answers to general questions about the Intergrax system's structure, features, or configuration.
- Uses predefined examples of such questions as input.
- Returns mock knowledge responses, including citations and logs.

Note: The file appears to be part of a larger, proprietary framework (Intergrax), with clear licensing restrictions.

### `applications\ux_audit_agent\components\project_manager.py`

**Description:** This module implements a project manager component for UX audit pipelines in the Integrax framework. It simulates a PM's decision based on random mock scenarios.

**Domain:** LLM adapters/RAG logic (UX Audit)

**Key Responsibilities:**

* Provides a Project Manager component to simulate approval or rejection of UX reports
* Generates mock decisions with accompanying notes and logs
* Allows for pipeline execution to be stopped if the project manager rejects the proposal

### `applications\ux_audit_agent\components\ux_audit.py`

**Description:** This module provides a UX audit component for the Integrax framework, which analyzes Figma mockups and generates sample reports with recommendations.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a UX audit component that can be used in pipeline analysis
* Performs UX audit based on Figma mockups
* Generates sample report with recommendations
* Returns a `ComponentResult` object with the generated report and other metadata

### `applications\ux_audit_agent\UXAuditTest.ipynb`

**Description:** This Jupyter Notebook file, `UXAuditTest.ipynb`, contains a step-by-step process for performing a UX audit on FIGMA mockups. The notebook outlines the tasks involved in conducting the audit, verifying changes comply with company policy, preparing a summary report for Project Manager, evaluating financial impact of changes, and finally, project continuation.

**Domain:** RAG (Risk, Assessment, Governance) logic

**Key Responsibilities:**
- Perform UX audit on FIGMA mockups
- Verify changes comply with company policy using Policy & Privacy Compliance Checker tool
- Prepare summary report for Project Manager using Final Report component
- Evaluate financial impact of changes using Cost Estimation Agent
- Project Manager decision and project continuation
- Final report preparation and synthesis

**Note:** This file appears to be part of the Integrax framework's RAG logic, providing a structured process for UX audit and related tasks.

### `generate_project_overview.py`

**Description:** This module generates a Markdown file containing an overview of the Intergrax project structure. It recursively scans the project directory, collects relevant files, and uses Large Language Models (LLM) to generate summaries for each file.

**Domain:** Project Structure Documentation Generator

**Key Responsibilities:**

* Recursively scan the project directory to collect source files
* Filter excluded directories and files based on configurable settings
* Use LLM adapters to generate summaries for each file
* Generate a Markdown file containing a structured overview of the project layout
* Provide a blueprint for architectural decision-making and context for LLM agents

### `intergrax\__init__.py`

DESCRIPTION:
This is the top-level initialization module for the Intergrax framework, responsible for setting up the overall structure and importing necessary components.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
• Initializes the Intergrax framework by defining its namespace and package structure.
• Imports other essential modules and sub-packages required for the framework's functionality.
• Sets up any initial configuration or logging mechanisms.

### `intergrax\chains\__init__.py`

Description: The `__init__.py` file serves as the entry point for the Intergrax chains module, defining its namespace and imports.

Domain: Chain management

Key Responsibilities:
- Defines the top-level interface for working with chains in Intergrax.
- Imports and exposes chain-related functionality to other parts of the framework.

### `intergrax\chains\langchain_qa_chain.py`

Description: This module provides a flexible QA chain builder for LangChain-based applications, enabling the integration of various components such as RAG retrievers, re-rankers, prompt builders, and LLM models.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Builds a QA chain with customizable components (RAG retriever, reranker, prompt builder, LLM model)
- Allows for hooks to modify data at various stages of the chain
- Supports input filtering using the "where" parameter
- Returns a LangChain Runnable object that can be used for invoking or asynchronous invocation
- Includes default prompt builder and configurable options for building context and citations

### `intergrax\chat_agent.py`

**Description:** This module defines a chat agent API that integrates multiple components to provide a unified interface for various tasks. It utilizes LLM adapters, RAG logic, and tool registry to perform routing decisions, execute RAG requests, and handle general inquiries.

**Domain:** Chat Agent / Routing

**Key Responsibilities:**

*   Provides a unified API for different chat-related tasks
*   Utilizes LLM adapters for routing decisions based on descriptions and tools enabled flag
*   Supports execution of RAG (ReAdamining Generator) requests with multiple options
*   Handles general inquiries using the base LLM without external tools or vector stores
*   Integrates tool registry to perform live, actionable operations
*   Offers configurable settings for chat agent behavior

### `intergrax\llm\__init__.py`

FILE PATH:
intergrax\llm\__init__.py

Description: The LLM adapters package initializer, responsible for loading and configuring language model adapters.
Domain: LLM Adapters

Key Responsibilities:
- Loads available LLM adapters from disk
- Configures adapter settings
- Exposes adapters to the main application

### `intergrax\llm\conversational_memory.py`

**Description:** This module provides utilities and classes for handling conversational memory in the Intergrax framework. It includes data structures for representing chat messages and a class for managing conversation history.

**Domain:** LLM adapters / Conversational Memory

**Key Responsibilities:**

*   Provides a `ChatMessage` dataclass to represent universal chat messages with various fields (e.g., role, content, created_at).
*   Offers an `IntergraxConversationalMemory` class to manage conversation history in memory and persist it to files or SQLite.
*   Implements methods for adding, extending, retrieving, and clearing conversation history.
*   Includes save/load functionality for conversation history from JSON/NDJSON files or SQLite database.
*   Provides filtering capabilities for OpenAI/Ollama responses and other native tools.

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This module defines a set of adapters and utilities for interacting with Large Language Models (LLMs), specifically OpenAI's Chat Completions API.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines the `LLMAdapter` protocol, which specifies the interface for all LLM adapters
* Provides the `OpenAIChatCompletionsAdapter` class, a concrete implementation of an LLM adapter for interacting with OpenAI's Chat Completions API
* Includes utilities for:
	+ Converting chat messages to the format required by the OpenAI API
	+ Generating and streaming responses from the LLM
	+ Handling tools (optional) for more complex interactions

Note: This file appears to be a part of the main codebase, with no clear indicators of being experimental or auxiliary.

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point for LLM adapters in the Intergrax framework, providing utilities and default registrations for various language model integrations.

Domain: LLM Adapters

Key Responsibilities:
- Exposes a registry of available LLM adapters
- Registers default adapter instances with the registry
- Imports and exports adapters for OpenAI, Gemini, and LangChain Ollama

### `intergrax\llm_adapters\base.py`

Description: This module provides a base implementation for LLM (Large Language Model) adapters, including utilities for structured output and an interface for adapters.

Domain: LLM Adapters

Key Responsibilities:
- Provides a universal interface (`LLMAdapter` protocol) for adapters to generate responses
- Offers tools for extracting JSON objects from text and mapping internal message objects to OpenAI-compatible message dictionaries
- Implements an adapter registry (`LLMAdapterRegistry`) for easy management of available adapters

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a minimal Gemini adapter for the Integrax framework, allowing for simple chat usage without wire tools.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes the Gemini chat adapter with a model and optional defaults
* Splits system messages from conversational messages
* Generates responses to input messages using the provided model
* Streams generated responses
* Intentionally does not implement tool functionality (e.g., wiring tools)

Note: This implementation appears to be a skeleton or a minimal version, as indicated by the absence of tool functionality and a note stating that it intentionally does not wire tools.

### `intergrax\llm_adapters\ollama_adapter.py`

Description: This module provides an Ollama model adapter using LangChain's ChatModel interface.

Domain: LLM adapters

Key Responsibilities:
- Adapts the Ollama model to work within the Integrax framework.
- Provides methods for generating and streaming messages from the model.
- Offers support for structured output generation with validation against a specified schema.
- Indicates that native tool-calling is not supported in this adapter, suggesting a planner-style pattern should be used instead.

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides OpenAI responses adapter for Integrax framework, enabling integration with OpenAI's Responses API. It supports both plain chat and tool-based interactions.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an interface to generate messages using OpenAI's Responses API
* Supports streaming completion using the Responses API
* Enables interaction with tools (functions or external services) through the adapter
* Generates structured JSON output based on a provided schema
* Validates the generated output against the specified model's JSON schema

### `intergrax\logging.py`

Here is the documentation for the `logging.py` file:

Description: This module provides configuration and setup for logging in the Integrax framework.

Domain: Logging Configuration

Key Responsibilities:
* Sets up basic configuration for the logger with a specified log level (INFO) and format.
* Configures the logger to display timestamps, log levels, and messages in a specific format.
* Forces the new configuration to override any previous ones.

### `intergrax\multimedia\__init__.py`

DESCRIPTION: This module serves as the entry point and namespace manager for multimedia-related functionality within the Intergrax framework.

DOMAIN: Multimedia Module

KEY RESPONSIBILITIES:
- Registers core multimedia components
- Defines interface for external multimedia modules to interact with the framework's infrastructure

### `intergrax\multimedia\audio_loader.py`

**intergrax\multimedia\audio_loader.py**

Description: This module provides functionality for downloading audio files from YouTube and translating them using the Whisper library.

Domain: Multimedia Processing

Key Responsibilities:
- Downloads audio files from YouTube using yt_dlp.
- Converts downloaded videos to audio-only files in specified format (default: mp3).
- Translates audio files using the Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module provides functionality for loading and processing images within the Intergrax framework, enabling integration with LLMs for tasks such as image-to-text transcription.

Domain: Image Processing/LLM Adapters

Key Responsibilities:
- Loads images from file paths
- Utilizes Ollama to send images along with text prompts for processing by LLM models (e.g., "llava-llama3")
- Returns transcribed text responses from the LLM model

### `intergrax\multimedia\ipynb_display.py`

Description: This module provides utilities for displaying multimedia content in Jupyter notebooks, including audio, images, and videos.

Domain: Multimedia

Key Responsibilities:
- Convert file paths to base64 encoded URLs for inline display
- Display audio files with optional autoplay and labeling
- Display image files
- Serve video files from a temporary directory and jump to a specific timestamp

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides a set of functions for loading, processing, and extracting multimedia content from video files.

**Domain:** Multimedia Processing

**Key Responsibilities:**
- Download videos from YouTube using the `yt_dl` library
- Extract video frames with metadata (time stamps) using OpenCV
- Transcribe video captions to WebVTT format using Whisper speech recognition model
- Save extracted frames and metadata as JSON file

### `intergrax\openai\__init__.py`

DESCRIPTION: This is the main entry point for the OpenAI adapter within the Intergrax framework.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
* Imports necessary components from other modules
* Exposes a public interface for interacting with the OpenAI API
* Possibly initializes or configures related dependencies or settings

### `intergrax\openai\rag\__init__.py`

Description: The `__init__.py` file serves as the entry point and initializer for the RAG (Retrieval-Augmented Generator) logic within the Intergrax framework, enabling the registration of RAG models with OpenAI adapters.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Registers RAG models with their corresponding OpenAI adapter classes.
- Initializes the RAG module and makes its components available for use.

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides a Rag OpenAI class, which enables the integration of the Intergrax framework with the OpenAI API for vector stores and file operations.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes an IntergraxRagOpenAI object with an OpenAI client and vector store ID
* Generates a RAG prompt based on the provided guidelines
* Ensures the existence of a specified vector store
* Clears all files loaded into the vector store
* Uploads folders to a vector store, following specific patterns

### `intergrax\rag\__init__.py`

Description: The __init__.py file is the entry point for the RAG (Retrieval-Augmented Generation) logic within the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
- Initializes the RAG module and its dependencies.
- Sets up default configuration and parameters for the RAG component.
- Provides a public API for interacting with the RAG logic.

### `intergrax\rag\documents_loader.py`

**Description:**
This module provides a robust and extensible document loader for the Intergrax framework, supporting various file formats and enabling metadata injection and safety guards.

**Domain:** RAG (Reinforced Adversarial Generation) logic / Document loading utilities

**Key Responsibilities:**

* Loading documents from various file formats (.txt, .md, .docx, .htm, .html, .pdf, .xlsx, .xls, .csv)
* Supporting metadata injection and safety guards
* Configurable settings for document loading (e.g., OCR options, image processing)
* Integration with Intergrax's multimedia loaders (video and audio) through adapter framework
* Experimental features include captioning images via LLM adapters

### `intergrax\rag\documents_splitter.py`

**Description:** 
This module provides a high-quality text splitter for RAG pipelines, ensuring stable chunk IDs and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**
- Provides a `IntergraxDocumentsSplitter` class that splits documents into chunks based on various criteria
- Implements the 'semantic atom' policy to avoid splitting small semantic units (paragraphs, rows, pages, images)
- Generates stable chunk IDs using available anchors or hash-based fallback
- Allows customization of metadata and chunking behavior through optional arguments and functions
- Merges tiny tail chunks and applies hard caps on maximum number of chunks per document

### `intergrax\rag\dual_index_builder.py`

**Description:** This module provides functionality for building dual indexes in the Intergrax framework. It enables the creation of primary and auxiliary vector stores for efficient document indexing.

**Domain:** RAG logic (Relationship-Aware Graphs)

**Key Responsibilities:**

* Building two vector indexes: primary (CHUNKS) and auxiliary (TOC)
* Splitting documents into chunks
* Creating auxiliary index from DOCX headings within specified levels
* Computing embeddings for documents in both indexes
* Inserting batches of embedded documents into the vector stores
* Handling logging and verbosity options

**Note:** This file appears to be a core component of the RAG logic, and its functionality is essential for indexing documents in the Intergrax framework.

### `intergrax\rag\dual_retriever.py`

**Description:** This module implements a dual retriever for Intergrax, which combines the results of two vector store queries to retrieve relevant information.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Dual retriever: first query TOC (sections), then fetch local chunks from the same section/source
* Uses `intergraxVectorstoreManager` to perform queries
* Embeds input text using an embedding manager and performs similarity searches
* Merges and deduplicates search results
* Returns a list of relevant hits with their metadata and similarity scores

**Note:** This file appears to be part of the Intergrax framework, which is proprietary and confidential. The code seems well-structured and functional, but its exact purpose and functionality require more context.

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for integrating various text embedding models, including HuggingFace (SentenceTransformer), Ollama, and OpenAI embeddings.

**Domain:** RAG logic / Embedding Management

**Key Responsibilities:**

* Unified provider switch between HuggingFace, Ollama, and OpenAI
* Model loading with reasonable defaults for model names
* Batch/single text embedding with optional L2 normalization
* Embedding support for LangChain Documents (returns np.ndarray + aligned docs)
* Cosine similarity utilities and top-K retrieval
* Robust logging, shape validation, and light retry for transient errors

**Note:** The module appears to be a robust and well-structured implementation of an embedding manager, with clear responsibilities and functionality. There is no indication that it is experimental, auxiliary, or incomplete.

### `intergrax\rag\rag_answerer.py`

**Description:** The IntergraxRagAnswerer module is responsible for retrieving and ranking relevant context fragments from a knowledge base, generating an answer using a language model, and producing a structured output.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Retrieve top-k relevant context fragments from the knowledge base
* Apply local similarity threshold filter on retrieved hits
* Re-rank retrieved hits using an optional re-ranker module
* Build context text by concatenating relevant fragments
* Generate citations for used hits
* Build system and user messages
* Send messages to language model for generating answer (text or structured output)
* Optionally, generate structured output in addition to the textual answer

### `intergrax\rag\rag_retriever.py`

**Description:** The Intergrax RAG retriever is a scalable, provider-agnostic module for retrieving relevant documents from a vector store. It normalizes filters, unifies similarity scoring, deduplicates results, and supports optional reranking and MMR diversification.

**Domain:** RAG (Relevance-Aware Generator) logic

**Key Responsibilities:**

* Normalizing `where` filters for Chroma
* Unifying similarity scoring across different providers
* Deduplication by ID and per-parent result limiting
* Optional reranking using cross-encoder or re-ranking models
* MMR diversification when embeddings are returned
* Batch retrieval for multiple queries

### `intergrax\rag\re_ranker.py`

**Description:** The ReRanker module is a fast and scalable cosine re-ranker that takes candidate chunks as input, embeds them using the EmbeddingManager, and returns a list of hit dictionaries with additional reranked scores.

**Domain:** RAG logic

**Key Responsibilities:**

* Embeds texts in batches using intergraxEmbeddingManager
* Calculates cosine similarities between query and documents
* Returns a list of hit dictionaries with additional reranked scores
* Supports input types: hits from the retriever (dict) or raw LangChain Documents
* Optional score fusion with original retriever similarity
* Lightweight in-memory cache for query embeddings

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module provides a unified vector store manager for Intergrax, supporting ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**

* Initialize target store and create collection/index (lazy initialization for Qdrant/Pinecone)
* Upsert documents + embeddings with batching
* Query top-K by cosine/dot/euclidean similarity
* Count vectors
* Delete by IDs

Note that this file appears to be a complete implementation, without any clear indicators of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module is a component of the Intergrax framework's Retrieval-Augmented Generation (RAG) logic. It implements a "windowed" answerer, which divides the retrieval process into smaller chunks or windows to improve performance and context-awareness.

**Domain:** RAG Logic

**Key Responsibilities:**

* Divides the retrieval process into smaller window-sized chunks
* Processes each window separately using the provided answerer and retriever components
* Builds a final answer by synthesizing partial answers from each window
* Deduplicates sources across windows
* Appends the final answer (and optional summary) to memory if available

### `intergrax\runtime\__init__.py`

DESCRIPTION:
This is the entry point for Intergrax's runtime module, responsible for initializing and configuring the framework.

DOMAIN: Runtime Initialization

KEY RESPONSIBILITIES:
* Initializes the Intergrax runtime environment
* Registers necessary components and services
* Sets up logging and other basic infrastructure

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

Description: This module initializes the drop-in knowledge mode for the Intergrax framework, enabling external knowledge sources to be integrated into the system.

Domain: RAG logic

Key Responsibilities:
- Initializes and configures the drop-in knowledge mode.
- Integrates external knowledge sources with the Intergrax framework.
- Sets up necessary data structures and hooks for seamless integration.

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides attachment resolution utilities for Intergrax's Drop-In Knowledge Mode, allowing the framework to decouple attachment storage and consumption.

**Domain:** Attachment Resolution (Drop-In Knowledge Mode)

**Key Responsibilities:**

* Defines the `AttachmentResolver` protocol, an abstraction that resolves an `AttachmentRef` into a local `Path`.
* Implements the `FileSystemAttachmentResolver`, a minimal resolver for local filesystem-based URIs.
* Provides utility functions to resolve attachments from various sources and return their corresponding file paths.

Note: This module appears to be stable and complete, providing essential functionality for attachment resolution in Intergrax's Drop-In Knowledge Mode.

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This module defines configuration objects for the Drop-In Knowledge Mode runtime in Intergrax, including settings for LLM adapters, embeddings, vector stores, and feature toggles.

**Domain:** Configuration

**Key Responsibilities:**

* Define main knobs for controlling the runtime behavior
* Integrate with existing Intergrax components (LLM adapters, embedding managers, vector store managers)
* Configure feature toggles (RAG, web search, tools, long-term memory)
* Set token budgets and limits for context construction
* Provide optional arbitrary metadata and app-specific configuration

**Status:** This file appears to be a critical part of the Intergrax framework, and its contents suggest that it is well-documented and widely used. There are no indications of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\runtime\drop_in_knowledge_mode\context_builder.py`

**Description:** This module is responsible for building the context for the Intergrax Drop-In Knowledge Mode, which includes deciding whether to use RAG, retrieving relevant document chunks from the vector store, reducing chat history, and composing a system prompt.

**Domain:** LLM adapters / Context Builder

**Key Responsibilities:**

* Decide whether to use RAG for a given (session, request)
* Retrieve relevant document chunks from the vector store using session/user/tenant metadata
* Reduce chat history to a manageable window
* Return a BuiltContext object that the engine can translate into OpenAI-style messages

This module appears to be a core part of the Intergrax framework's functionality and is not marked as experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

**Description:** This module implements the core runtime engine for Drop-In Knowledge Mode, providing a high-level conversational runtime for the Intergrax framework.

**Domain:** LLM adapters and conversational runtime

**Key Responsibilities:**

* Loads or creates chat sessions via `SessionStore`
* Appends user messages to the session
* Builds a conversation history for the LLM (limited by `RuntimeConfig`)
* Calls the configured LLM adapter
* Produces a `RuntimeAnswer` object with the final answer text and metadata

Note: The engine is currently a skeleton, missing RAG/web search/tools integration and other features.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

**Description:** This module provides a high-level service for ingesting attachments in the context of Drop-In Knowledge Mode, leveraging existing Intergrax RAG building blocks to resolve, load, split, embed, and store documents.

**Domain:** Data Ingestion

**Key Responsibilities:**

* Resolve AttachmentRef objects into filesystem Paths using an AttachmentResolver
* Load documents using IntergraxDocumentsLoader.load_document(...)
* Split documents into chunks using IntergraxDocumentsSplitter.split_documents(...)
* Embed chunks via IntergraxEmbeddingManager
* Store vectors via IntergraxVectorstoreManager
* Return a structured IngestionResult per attachment

Note: This file appears to be well-structured and production-ready, with clear documentation and separation of concerns.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

**Description:** This module defines data classes for request and response structures used in the Drop-In Knowledge Mode runtime of the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Define data classes for high-level contract between applications (FastAPI, Streamlit, CLI, MCP, etc.) and the DropInKnowledgeRuntime
* Expose citations, routing information, tool calls, and basic statistics in a structured way
* Intentionally hide low-level implementation details while keeping enough structure to support applications
* Provide data classes for request (`RuntimeRequest`) and response (`RuntimeAnswer`) structures
* Define data classes for specific components:
	+ `Citation`: represents a single citation/reference used in the final answer
	+ `RouteInfo`: describes how the runtime decided to answer the question
	+ `ToolCallInfo`: describes a single tool call executed during the runtime request
	+ `RuntimeStats`: basic statistics about a runtime call

### `intergrax\runtime\drop_in_knowledge_mode\session_store.py`

**Description:** This module provides a session storage abstraction for the Drop-In Knowledge Mode runtime, enabling persistence and retrieval of chat sessions across various backends.

**Domain:** Session Storage

**Key Responsibilities:**

* Defines data classes for `SessionMessage` and `ChatSession` to represent individual messages and top-level chat sessions
* Establishes the `SessionStore` protocol for session persistence, with methods for getting, creating, saving, appending, and listing sessions
* Includes an in-memory implementation (`InMemorySessionStore`) for testing and quick experiments

### `intergrax\supervisor\__init__.py`

DESCRIPTION: The __init__.py file serves as the entry point for the supervisor module in Intergrax, responsible for setting up and initializing its internal components.

DOMAIN: Supervisor Module

KEY RESPONSIBILITIES:
- Initializes internal components of the supervisor
- Defines exportable functions and classes
- Sets up necessary configurations

### `intergrax\supervisor\supervisor.py`

**Description:** The supervisor module is responsible for planning and executing tasks within the Intergrax framework. It coordinates with Language Model (LLM) adapters, component execution, and other system services to determine task plans and their execution.

**Domain:** Task Planning / Execution Management

**Key Responsibilities:**

*   Planning tasks using LLM-based decomposition and heuristic methods
*   Executing plan steps through component management and resource allocation
*   Managing LLM adapters for text processing and analysis
*   Registering and managing components for task execution
*   Analyzing plan results and providing diagnostic information
*   Supporting two-stage planning with decomposed step assignment

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module defines the core components and utility classes for building, managing, and executing pipeline steps in the Integrax framework.

**Domain:** Supervisor Components

**Key Responsibilities:**
- Defines `Component` class for representing individual steps with metadata and executable logic.
- Provides `ComponentContext` class for passing contextual information to components.
- Introduces `PipelineState` dataclass for encapsulating pipeline state, including query results, artifacts, and debug logs.
- Offers a decorator (`component`) for easy registration of new components with their respective functions.
- Supplies utility classes like `ComponentResult` for handling component output and diagnostic logs.

Note: This module appears to be a core part of the Integrax framework, providing essential functionality for building and executing pipeline steps.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module defines default prompt templates for the Intergrax Supervisor, used to guide planning and decomposition of tasks. The prompts are structured as JSON objects that contain information about task intent, required components, user inputs, and validation rules.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Define default prompt templates for the Intergrax Supervisor
* Provide a unified structure for planning and decomposition tasks
* Include domain-specific knowledge and constraints in the prompts
* Support validation checks to ensure plan correctness and integrity

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module is responsible for building and executing the LangGraph pipeline from a given plan. It handles state management, node creation, and graph construction.

**Domain:** RAG logic

**Key Responsibilities:**

* Manages global state traveling through the LangGraph pipeline using the `PipelineState` dataclass.
* Provides utilities for ensuring default state values, appending logs, resolving inputs, and persisting outputs.
* Creates readable node names using the `_slugify` function.
* Builds LangGraph nodes from plan steps using the `make_node_fn` function.
* Performs topological ordering of plan steps to construct a stable graph.
* Transforms a plan into a runnable LangGraph pipeline using the `build_langgraph_from_plan` function.

### `intergrax\system_prompts.py`

Description: This module defines a default RAG system instruction for strict RAG protocols in the Integrax framework.

Domain: LLM adapters/RAG logic

Key Responsibilities:
- Defines the structure and guidelines for responding to user queries based on documents stored in the vector store.
- Outlines the steps for verifying consistency, citing sources, and providing accurate and precise answers.
- Specifies the formatting and style requirements for responses, including citations and references.

### `intergrax\tools\__init__.py`

Description: This module initializes and imports the necessary tools within the Intergrax framework.

Domain: Utility modules

Key Responsibilities:
• Initializes the tooling environment for the framework.
• Imports and registers essential utility functions.

### `intergrax\tools\tools_agent.py`

**Description:** This module provides the IntergraxToolsAgent class, which acts as a bridge between LLM adapters and tools for handling conversations.

**Domain:** Conversational Agents with integrated tool-calling functionality

**Key Responsibilities:**

* Initializes an LLM adapter and a tool registry instance
* Manages the conversation flow based on user input and system instructions
* Supports native tool-calling (OpenAI) or JSON planner-based interaction (Ollama)
* Prunes messages to ensure OpenAI compatibility when necessary
* Builds output structures from tool traces, answer text, or extracted JSON
* Provides a `run` method for handling conversations with tools integration

### `intergrax\tools\tools_base.py`

**Description:** This module provides a set of base classes and utilities for defining and registering tools within the Integrax framework. It includes functionality for validating tool outputs, serializing tools to OpenAI-compatible JSON schemas, and managing a registry of available tools.

**Domain:** LLM adapters

**Key Responsibilities:**

* Providing a `ToolBase` class that serves as a base class for all tools, defining common attributes and methods.
* Implementing a `_limit_tool_output` function for safely truncating long tool outputs to avoid overflowing the LLM context.
* Offering a `ToolRegistry` class for managing and storing registered tools in a dictionary-like structure.
* Exporting tools from the registry to a format compatible with the OpenAI Responses API.
* Enabling validation of tool arguments using Pydantic (if available).
* Allowing for override of the `run` method by each specific tool.

### `intergrax\websearch\__init__.py`

DESCRIPTION: This module initializes the web search functionality within the Intergrax framework, providing a basic setup and import structure.

DOMAIN: Web Search

KEY RESPONSIBILITIES:
- Initializes the web search component
- Defines the entry point for the web search functionality
- Imports necessary modules and sub-modules
- Exposes the web search service through the application

### `intergrax\websearch\cache\__init__.py`

**Description:** This module provides a simple in-memory query cache with optional time-to-live (TTL) and maximum size for storing web search results. It allows efficient reuse of cached results for equivalent queries.

**Domain:** Cache/Query Storage

**Key Responsibilities:**

* Provides an immutable `QueryCacheKey` dataclass to describe unique web search configurations
* Offers a simple in-memory query cache with TTL and max size, suitable for single-process use cases (e.g., notebooks, local development)
* Allows storing and retrieving cached results using the `InMemoryQueryCache` class
* Implements basic eviction strategy (removing oldest entry) when maximum size is reached

### `intergrax\websearch\context\__init__.py`

DESCRIPTION: 
This module initializes and configures the context for web search functionality within Intergrax.

DOMAIN: RAG (Retrieval-Augmented Generation) logic

KEY RESPONSIBILITIES:
- Initializes web search context
- Configures relevant parameters for web search operations

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module, `websearch_context_builder`, is responsible for building LLM-ready textual context and chat messages from web search results.

**Domain:** RAG logic (Reinforced Active Learning with Graphs)

**Key Responsibilities:**

* Builds a textual context string from WebDocument objects or serialized dicts
* Constructs chat messages (system + user) for chat-style LLMs
* Allows customization of context building parameters, such as maximum number of documents and characters per document
* Provides strict system prompts enforcing web sources-only mode and no hallucinations

Note: There is a small typo in the provided content where `contex` should be `context`.

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: 
This module initializes and configures the web search fetcher for Intergrax, responsible for retrieving relevant data from various online sources.

DOMAIN: Web Search Fetcher Configuration

KEY RESPONSIBILITIES:
• Initializes the web search engine instance
• Configures search parameters (e.g., query filters, result limits)
• Sets up logging and monitoring for fetcher operations

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module contains HTML extraction functionality for web pages, providing both lightweight and advanced readability-based extraction capabilities.

**Domain:** Web Search/Fetcher/Extractor

**Key Responsibilities:**

* Perform lightweight HTML extraction on a PageContent instance (extract_basic):
	+ Extract <title>
	+ Extract meta description
	+ Extract <html lang> attribute
	+ Extract Open Graph meta tags (og:*)
	+ Produce a plain-text version of the page
* Perform advanced readability-based extraction on a PageContent instance (extract_advanced):
	+ Remove obvious boilerplate elements (scripts, styles, iFrames, navigation)
	+ Prefer trafilatura (when available) to extract primary readable content
	+ Fallback to BeautifulSoup plain-text extraction if trafilatura fails
	+ Normalize whitespace and reduce noise

**Notes:** The module appears to be a core component of the web search functionality within the Intergrax framework. It is well-structured, with clear responsibilities and use cases outlined in the code.

### `intergrax\websearch\fetcher\http_fetcher.py`

Description: This module provides an asynchronous HTTP client for fetching web pages, allowing Intergrax to retrieve and process content from external URLs.

Domain: Web Search Fetcher

Key Responsibilities:
- Performs HTTP GET requests with sane default headers
- Captures final URL, status code, raw HTML, and body size
- Handles transport-level failures and returns None in such cases
- Keeps higher-level concerns (robots, throttling, extraction) outside this module

### `intergrax\websearch\integration\__init__.py`

Description: The __init__.py file serves as the entry point for the web search integration within Intergrax, responsible for setting up and configuring the integration.

Domain: Integration

Key Responsibilities:
- Initializes the web search integration module.
- Sets up necessary dependencies and configurations.

### `intergrax\websearch\integration\langgraph_nodes.py`

Description: This module provides a web search node implementation using the Intergrax framework, allowing integration with various search engines like Google Custom Search and Bing Web.

Domain: RAG logic

Key Responsibilities:
- Provides a `WebSearchNode` class that wraps a `WebSearchExecutor` instance for encapsulating configuration and delegation of search tasks.
- Offers synchronous (`run`) and asynchronous (`run_async`) node methods for processing web search queries.
- Exposes default and functional API wrappers (`websearch_node` and `websearch_node_async`) for convenience and backward compatibility.
- Utilizes the `WebSearchExecutor` to perform actual search operations, with default parameters configurable via class initialization or node method calls.

### `intergrax\websearch\pipeline\__init__.py`

DESCRIPTION: This module initializes the pipeline for web search functionality within the Intergrax framework.

DOMAIN: Web Search Pipeline Configuration

KEY RESPONSIBILITIES:
• Initializes the web search pipeline components
• Configures pipeline processing flow 
• Sets up relevant data handlers and transforms

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This file implements a web search pipeline that allows for concurrent fetching and processing of search results from multiple providers. The pipeline is designed to be provider-agnostic and includes features such as rate limiting, deduplication, and quality scoring.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Orchestrates multi-provider web search, fetching, extraction, deduplication, and basic quality scoring
* Works with any WebSearchProvider instance
* Async fetching with rate limiting (TokenBucket)
* Simple deduplication via text-based dedupe key
* Minimal and testable design with no direct LLM coupling

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: This module initializes and registers web search providers for the Intergrax framework.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
- Initializes web search provider modules.
- Registers available providers with the framework.

### `intergrax\websearch\providers\base.py`

**Description:** This module defines the base interface for web search providers in the Integrax framework, specifying common responsibilities and a stable API.

**Domain:** Web Search Providers

**Key Responsibilities:**

* Accept provider-agnostic query specifications through the `QuerySpec` class.
* Return a ranked list of search results (SearchHit items) from executing a single search request.
* Expose minimal capabilities for feature negotiation, including language and freshness support.
* Implementations should honor top-k search limits, include provider-specific metadata in hits, and sanitize/validate URLs.

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search provider for the Intergrax framework, enabling integration with the Bing REST API.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a Bing Web Search (v7) provider class (`BingWebProvider`) that integrates with the Bing REST API
* Allows configuration of API key through environment variables or constructor arguments
* Supports filtering by language, region, and freshness
* Fetches search results from Bing and transforms them into `SearchHit` objects
* Closes the underlying HTTP session when finished

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** 
This module provides a provider class for Google Custom Search (CSE) within the Intergrax framework, allowing users to query and fetch search results using the CSE REST API.

**Domain:** Websearch Providers

**Key Responsibilities:**

* Provides a `GoogleCSEProvider` class that implements the `WebSearchProvider` interface
* Exposes environment variables for API key and CX (Search Engine ID)
* Handles parameters such as language, UI language, and content language filtering
* Caches search results to prevent excessive requests
* Supports freshness filtering through custom engines or site filters
* Returns search hits in a standardized format

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a web search provider for Google Places, allowing the retrieval of business data and integration with other Intergrax framework components.

**Domain:** LLM adapters > Web Search Providers

**Key Responsibilities:**

* Provides a web search interface for Google Places
* Supports text search by arbitrary query (name + city, category, etc.)
* Returns core business data:
	+ name, address, location (lat/lng)
	+ rating, user_ratings_total
	+ types (categories)
	+ website, international_phone_number, opening_hours
	+ Google Maps URL (via url or constructed maps link)
* Handles environment variables and API key configuration
* Supports pagination and details lookups
* Mapping helpers for converting Places data to SearchHit objects

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a Reddit search provider for the Intergrax framework, utilizing the official OAuth2 API. It enables full-featured searching with rich post metadata and optional comment fetching.

**Domain:** Websearch Providers

**Key Responsibilities:**

* Authenticates using application-only OAuth2 (client_credentials) to obtain an access token
* Uses the access token to perform searches on Reddit
* Fetches top-level comments for each search result, if enabled
* Returns a simplified list of comment dictionaries for each search result
* Maps search results to SearchHit objects
* Provides capabilities metadata, including supported language and freshness features
* Refreshes the access token if it expires or becomes invalid

### `intergrax\websearch\schemas\__init__.py`

Description: This module defines the schema and initialization for web search functionality in the Intergrax framework.
Domain: Web Search

Key Responsibilities:
• Defines the schema for web search queries
• Initializes the web search configuration and data ingestion mechanisms
• Registers necessary components for web search operations
• Optionally includes experimental or auxiliary code, but appears to be a key part of the web search domain.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a data class `PageContent` to represent and encapsulate the content of a web page, including its metadata and extracted information.

**Domain:** Web search/RAG logic

**Key Responsibilities:**

* Represents the fetched and optionally extracted content of a web page
* Encapsulates both raw HTML and derived metadata
* Allows post-processing stages (extraction, readability, deduplication) to work independently of the original HTTP layer
* Provides methods for checking if the page has content, generating a short summary, and calculating the approximate size of the content in kilobytes

**Note:** The file appears to be part of a production-grade framework, as indicated by its documentation and code quality.

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a dataclass `QuerySpec` for canonical search query specifications used by web search providers. It provides attributes and methods to standardize and normalize query parameters.

**Domain:** RAG (Recurrent Attention-based Generative) logic

**Key Responsibilities:**
- Defines a `QuerySpec` dataclass with attributes for query parameters
- Provides methods for normalizing the query string (`normalized_query`)
- Limits the top results count based on provider caps (`capped_top_k`)

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a dataclass `SearchHit` that encapsulates metadata for individual search result entries, providing provider-agnostic information about the hit.

**Domain:** Web Search Schema

**Key Responsibilities:**

* Defines the `SearchHit` dataclass with fields for common search engine result metadata
* Performs minimal safety checks on URL and rank values during initialization
* Provides methods for extracting domain (netloc) from URLs and creating a minimal, LLM-friendly representation of each hit.

### `intergrax\websearch\schemas\web_document.py`

**Description:** This module defines a unified structure for representing web documents, integrating search hits, page content, and analysis results.

**Domain:** Web Search

**Key Responsibilities:**
- Defines the `WebDocument` dataclass to hold relevant information about a fetched and processed web document.
- Connects original search hit metadata with extracted content and analysis results.
- Provides methods for validating document validity (`is_valid`), merging textual content (`merged_text`), and generating summary lines (`summary_line`).

### `intergrax\websearch\service\__init__.py`

Description: The `__init__.py` file serves as the entry point for the web search service, responsible for initializing and setting up the necessary components.

Domain: Web Search Service Configuration

Key Responsibilities:
- Initializes the service
- Sets up dependencies and configurations for the web search functionality
- Exposes the service to be used by other modules within the Intergrax framework.

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module implements a high-level helper for web search and answering questions, integrating with LLM adapters.

**Domain:** Web Search, LLM Integration

**Key Responsibilities:**

* Runs web search via `WebSearchExecutor`
* Builds LLM-ready context/messages from web documents using `WebSearchContextBuilder`
* Calls an `LLMAdapter` to generate a final answer
* Provides both async and sync interfaces for question answering (`answer_async` and `answer_sync`)

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module provides a high-level, configurable web search executor that constructs query specifications, executes search pipelines with chosen providers, and converts web documents into LLM-friendly dicts.

**Domain:** Web Search Executor

**Key Responsibilities:**

* Construct QuerySpec from raw queries and configuration
* Execute SearchAndReadPipeline with chosen providers
* Convert WebDocument objects into LLM-friendly dicts
* Provide methods for building query specifications and executing web search pipelines asynchronously
* Integrate with query caching mechanisms for optimized performance

### `intergrax\websearch\utils\__init__.py`

DESCRIPTION: This is an initialization module for web search-related utilities in the Intergrax framework.

DOMAIN: Web Search Utilities

KEY RESPONSIBILITIES:
* Initializes and sets up utility functions for web search operations
* Possibly imports or instantiates other modules for web search functionality 
* Might contain some configuration settings or constants related to web search

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides simple deduplication utilities, specifically designed for web search pipeline usage.

**Domain:** Data Ingestion/Preprocessing

**Key Responsibilities:**

* Normalize text inputs for deduplication by stripping whitespace, converting to lowercase, and collapsing internal whitespace sequences.
* Generate a stable SHA-256 based deduplication key for given text inputs.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module implements a simple token bucket rate limiter for asyncio-compatible applications, designed to prevent excessive resource consumption.

**Domain:** Rate limiting

**Key Responsibilities:**

* Provides a TokenBucket class for implementing rate limiting policies
* Allows setting a maximum refill rate (tokens per second) and capacity (maximum number of tokens stored)
* Offers two methods: `acquire` (waits until tokens are available before consuming them) and `try_acquire` (non-blocking attempt to consume tokens, returning True if successful or False otherwise)

Note: The code appears well-structured and complete.

### `main.py`

Description: This module serves as the entry point for the Intergrax framework, executing the primary function when run directly.

Domain: Configuration/Initialization

Key Responsibilities:
- Defines the main execution flow of the framework.
- Provides a print statement to verify framework initialization.

### `mcp\__init__.py`

**Intergrax Framework Documentation**

**File:** `mcp/__init__.py`

**Description:** This is the main entry point for the MCP (Module Container Provider) module, responsible for bootstrapping and initializing the Intergrax framework components.

**Domain:** Core Infrastructure

**Key Responsibilities:**
* Initializes core modules and services
* Sets up global configuration and constants
* Defines main application entry points

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

**Description:** This Jupyter notebook provides a basic sanity-check demo for the Drop-In Knowledge Mode runtime in the Intergrax framework. It verifies that the runtime can create or load a session, append user and assistant messages, build conversation history from SessionStore, and return a RuntimeAnswer object.

**Domain:** Drop-In Knowledge Mode runtime

**Key Responsibilities:**

* Verify the functionality of the Drop-In Knowledge Mode runtime
* Create or load a session using InMemorySessionStore
* Append user and assistant messages to the session
* Build conversation history from SessionStore
* Return a RuntimeAnswer object

This notebook appears to be a demonstration of the Intergrax framework's capabilities, specifically the Drop-In Knowledge Mode runtime. It does not contain any experimental or auxiliary code, and it is likely intended for educational purposes.

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

Description: This Jupyter notebook demonstrates the usage of Intergrax's Drop-In Knowledge Mode runtime, specifically for handling attachments and ingestion in a conversational AI setting.

Domain: LLM adapters

Key Responsibilities:
- Initializes an InMemorySessionStore for storing session data
- Sets up an Ollama + LangChain LLM adapter
- Configures embedding manager using Ollama embeddings and vector store manager using Chroma as the vector store
- Creates a RuntimeConfig instance with specified configurations (e.g., enabling RAG, websearch, tools, long-term memory, etc.)
- Initializes a DropInKnowledgeRuntime instance with the session store and configured components
- Prepares an AttachmentRef for a local project document to simulate attachment ingestion in a chat UI

### `notebooks\drop_in_knowledge_mode\03_rag_context_builder_demo.ipynb`

Description: This Jupyter Notebook demonstrates the usage of the ContextBuilder in Intergrax's Drop-In Knowledge Mode runtime. It provides a step-by-step guide to load a demo chat session, work with an existing attachment, and initialize the ContextBuilder.

Domain: RAG logic (Reactive Attention-based Generator)

Key Responsibilities:
- Load a demo chat session using SessionStore.
- Work with an existing attachment by ingesting it or preparing a single demo attachment.
- Initialize the ContextBuilder instance using RuntimeConfig and IntergraxVectorstoreManager.
- Build context for a single user question using the ContextBuilder.build_context method.
- Inspect the result of context building, including reduced chat history, retrieved document chunks, system prompt, and RAG debug information.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

Description: This Jupyter Notebook demonstrates an end-to-end hybrid multi-source RAG workflow that combines local files and web search results using Intergrax components with LangGraph.

Domain: LLM adapters, RAG logic, data ingestion

Key Responsibilities:
- Ingest content from multiple sources (local PDF/DOCX files, live web results).
- Build a unified RAG corpus by normalizing documents, attaching metadata, and splitting into chunks.
- Create an in-memory vector index using Intergrax embedding manager and vectorstore manager.
- Answer user questions with a RAG agent that loads, merges, indexes, retrieves, and answers.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

Description: This notebook demonstrates the integration between Intergrax and LangGraph, showcasing how to use an Intergrax LLM adapter as a node inside a LangGraph graph.

Domain: Integrax-LangGraph Integration

Key Responsibilities:
- Initializes an OpenAI API client with the provided key.
- Defines a simple state for the LLM QA example, holding chat messages and the final answer.
- Implements a LangGraph node that calls the Intergrax LLM adapter to generate answers.
- Builds a StateGraph with a single node `llm_answer_node`.
- Runs the graph on a sample user question.

Note: This notebook is experimental and intended as a starting point for further development. It serves to demonstrate the integration between Intergrax and LangGraph, but it may not be production-ready without additional testing and refinement.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

Description: This is a Jupyter notebook that demonstrates the usage of Intergrax framework components for building a practical web research agent. The notebook orchestrates the flow as a multi-step graph to power “no-hallucination” web-based Q&A.

Domain: RAG logic (Reinforcement Augmented Graph)

Key Responsibilities:
- Defines a WebResearchState data structure representing the graph state
- Implements nodes for normalizing user questions and running web searches using Intergrax components
- Demonstrates the usage of OpenAI's LLM adapter with Intergrax's WebSearch components
- Sets up environment variables for API keys and initializes required components
- Defines a node for generating a final answer with sources

Note: This notebook is likely intended as an example or proof-of-concept, given its focus on demonstration rather than production-readiness.

### `notebooks\openai\rag_openai_presentation.ipynb`

Description: This Jupyter Notebook provides a presentation of the RAG (Relational Atomic Graph) functionality using OpenAI, showcasing how to fill VectorStore and test queries.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Import necessary modules for OpenAI interaction and Intergrax RagOpenAI integration
- Load environment variables from .env file
- Set up OpenAI client and Rag instance with a specified vector store ID
- Fill the VectorStore by uploading a local folder to it
- Test queries using the Rag instance

### `notebooks\rag\chat_agent_presentation.ipynb`

**Description:** This Jupyter notebook defines a high-level hybrid agent that integrates RAG (Recurrent Attention Generative) logic, tools, and LLM (Large Language Model) chat capabilities.

**Domain:** RAG Logic & Hybrid Agents

**Key Responsibilities:**

* Creates an instance of the `IntergraxChatAgent` class, which integrates RAG components with tools and LLM chat functionality.
* Defines a RAG component (`rag_docs`) that responds to questions related to Mooff regulations, privacy policies, internal documentation, and compliance rules.
* Registers available tools (in this case, a demo weather tool) using the `ToolRegistry`.
* Configures vector store and RAG components, including embedding manager, retriever, reranker, and answerer.
* Demonstrates test questions and their routing decisions, highlighting the agent's ability to route queries to relevant tools or RAG components.

**Note:** This notebook appears to be a working example of the Intergrax framework in action, showcasing its hybrid agent capabilities.

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the usage and integration of various Intergrax components, including LLM adapters, conversational memory, tools agents, and RAG (Retrieval-Augmented Generator) logic. It provides a structured output format for the response using Pydantic models.

**Domain:** RAG Logic

**Key Responsibilities:**

* Demonstrates usage of Intergrax's LLM adapters (e.g., Ollama-backed model)
* Showcases integration with conversational memory
* Defines and registers tools agents (WeatherTool) with specific schema models (e.g., WeatherAnswer)
* Uses RAG components for structured output, including:
	+ Embedding manager
	+ Vector store manager
	+ Retrieval retriever
	+ ReRanker
	+ Answerer
* Provides example usage of the tools agent and RAG logic to generate a structured response

**Status:** This notebook appears to be part of a larger demonstration or tutorial, showcasing the capabilities of the Intergrax framework. It is not marked as experimental, auxiliary, legacy, or incomplete.

### `notebooks\rag\rag_custom_presentation.ipynb`

Description: This Jupyter notebook script loads and processes documents, splits them into chunks, generates vector embeddings using the Ollama model, and stores these embeddings in a Vectorstore database. It demonstrates a RAG (Retrieval-Augmented Generation) pipeline within the Integrax framework.

Domain: RAG logic

Key Responsibilities:
- Loads raw documents from a specified directory
- Splits loaded documents into smaller chunks for optimal embedding granularity
- Generates vector embeddings for each document chunk using an Ollama model
- Stores generated embeddings in a Vectorstore database
- Checks if a target corpus is already present in the Vectorstore before performing ingestion

### `notebooks\rag\rag_multimodal_presentation.ipynb`

Description: This notebook is a test script for the Intergrax framework's RAG (Retrieval-Augmented Generation) components, specifically focusing on multimodal presentation and ingestion into VectorStore.

Domain: RAG logic / Multimodal Presentation

Key Responsibilities:
- Loads documents from various sources (video, audio, image) using `IntergraxDocumentsLoader`.
- Splits and embeds the loaded documents using `IntergraxDocumentsSplitter` and `IntergraxEmbeddingManager`, respectively.
- Checks if corpus is present in VectorStore using `IntergraxVectorstoreManager` and determines if ingestion is needed.
- If ingestion is required, loads documents again, deduplicates IDs, and adds them to VectorStore using `IntergraxDocumentsLoader`, `dedup_batch`, and `IntergraxVectorstoreManager`.
- Tests retriever functionality using `IntergraxRagRetriever`.

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter Notebook script demonstrates the integration of multimedia processing capabilities within the Integrax framework, showcasing video and audio loading, transcription, frame extraction, and image description.

**Domain:** RAG logic / Multimedia Processing

**Key Responsibilities:**

* Loads videos from YouTube using `yt_download_video` function
* Transcribes videos to VTT format using `transcribe_to_vtt` function
* Extracts frames and metadatas from video using `extract_frames_and_metadata` function
* Translates audio using `translate_audio` function
* Uses ollama model to describe images in the extracted frames

**Note:** The script appears to be a demonstration or tutorial code, showcasing various multimedia processing capabilities within the Integrax framework. It does not appear to be experimental, auxiliary, legacy, or incomplete.

### `notebooks\rag\tool_agent_presentation.ipynb`

**Description:** This Jupyter notebook demonstrates the usage of the Integrax framework, specifically showcasing a tools agent that can execute various tasks such as fetching weather information and performing arithmetic calculations.

**Domain:** RAG logic (Reasoning-Augmented Generative) / Tools Agent

**Key Responsibilities:**
- Defines two custom tools: `WeatherTool` for fetching weather data and `CalcTool` for basic arithmetic calculations.
- Sets up a conversational memory shared across interactions using `IntergraxConversationalMemory`.
- Creates a tool registry holding available tools, registering the custom tools.
- Initializes an LLM (Large Language Model) adapter using Ollama, which serves as the planner/controller for tools.
- Orchestrates the agent's behavior by creating an instance of `IntergraxToolsAgent` with specified components (LLM, tools, memory, and config).
- Runs two test scenarios demonstrating tool selection and invocation:
  - Test 1: Queries about weather in Warsaw should select the `get_weather` tool.
  - Test 2: Arithmetic calculation questions should select the `calc_expression` tool.

### `notebooks\supervisor\supervisor_test.ipynb`

Description: This notebook defines a set of components for the Integrax framework, each responsible for performing specific tasks such as compliance checking, cost estimation, and financial audits.

Domain: Supervisor Components

Key Responsibilities:
- Compliance Checker:
  - Verifies whether proposed changes comply with privacy policies and terms of service (mock).
  - Returns findings on compliance, policy violations, and recommended actions.
- Cost Estimator:
  - Estimates the cost of changes based on UX audit reports (mock).
  - Calculates a mock formula-based estimate using base cost and per-issue costs.
- Final Summary Report:
  - Generates a final consolidated summary report using all collected artifacts.
  - Includes status pipeline, terminated by, terminate reason, PM decision, and other relevant information.
- Financial Audit:
  - Generates a mock financial report and VAT calculation (test data).
  - Calculates gross value from net, VAT amount, and other financial metrics.

Note: All components appear to be functional and production-ready.

### `notebooks\websearch\websearch_presentation.ipynb`

Description: This notebook demonstrates the usage of the Intergrax framework's web search capabilities, specifically with Google Custom Search and Bing Search providers.

Domain: WebSearch/RAG (Reformulated Augmented Generation)

Key Responsibilities:
- Loads environment variables for Google Custom Search API key and CX ID
- Defines a query specification using the `QuerySpec` class
- Creates an instance of the `GoogleCSEProvider` class to perform search queries
- Executes the search query and prints the results, including provider, rank, title, URL, snippet, domain, and published date for each hit.
