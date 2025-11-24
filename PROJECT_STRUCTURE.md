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

Description: This is the main entry point for the API module, responsible for importing and configuring sub-modules.

Domain: API

Key Responsibilities:
- Imports sub-modules within the api package.
- Configures API endpoints and settings.

### `api\chat\__init__.py`

Here is the documentation for the provided file:

**Description:** The `__init__.py` file in the `api/chat` directory serves as an initializer module, responsible for setting up and configuring the chat API.

**Domain:** LLM (Large Language Model) adapters/ integration

**Key Responsibilities:**
* Initializes the chat API environment
* Sets up dependencies for chat functionality
* Possibly configures other related modules or services

### `api\chat\main.py`

Description: This module is responsible for handling chat interactions within the Integrax framework. It provides endpoints for initiating conversations, uploading and indexing documents, listing available documents, and deleting documents.

Domain: RAG logic (Reactive Architecture with Generative capabilities)

Key Responsibilities:
- Handles incoming queries through the "/chat" endpoint
- Initializes and manages conversation sessions
- Utilizes models to generate answers based on user input
- Provides endpoints for document management (uploading, indexing, listing, and deleting)
- Integrates with Chroma for document indexing and deletion

### `api\chat\tools\__init__.py`

**FILE:** api/chat/tools/__init__.py

**Description:** This module serves as the entry point for chat tools within the API, importing and exposing utility functions for various chat-related operations.

**Domain:** Chat Utilities

**Key Responsibilities:**

* Imports and exposes chat tool utilities
* Provides interface for external access to these utilities

### `api\chat\tools\chroma_utils.py`

**Description:** This module provides tools for interacting with the Chroma vector store, including loading and splitting documents, indexing documents to Chroma, and deleting documents.

**Domain:** RAG logic / Vector Store Utilities

**Key Responsibilities:**
- Load and split documents from a file path into individual documents.
- Index documents to Chroma using a file path and ID.
- Delete documents from Chroma based on their file ID. 

Note: The `load_and_split_documents` function appears to be a duplicate import, possibly indicating an incomplete or refactored implementation.

### `api\chat\tools\db_utils.py`

**Description:** This module provides a set of utilities for interacting with a SQLite database used in the Integrax framework. It includes functions for creating and migrating the database schema, inserting and retrieving messages and documents, as well as ensuring session existence.

**Domain:** Data storage and retrieval (RAG logic)

**Key Responsibilities:**

* Creating and migrating the database schema
* Inserting and retrieving messages
	+ Ensuring session existence
	+ Inserting user and assistant messages
	+ Retrieving messages for a given session
* Inserting and retrieving documents
	+ Creating document records with metadata
	+ Deleting document records by ID
	+ Retrieving all documents in descending order of upload timestamp

Note: The file appears to be well-maintained, but some of the functions seem to be duplicated (e.g., `create_schema()` and `create_application_logs()`, which only create a new schema). Additionally, there are some commented-out code blocks that may indicate experimental or auxiliary functionality.

### `api\chat\tools\pydantic_models.py`

Description: This module defines data models using Pydantic for various API interactions, including query inputs and responses, document information, and deletion requests.

Domain: LLM adapters

Key Responsibilities:
- Defines enumerations for model names
- Provides input models for querying (QueryInput) and responding to queries (QueryResponse)
- Offers a model for storing document metadata (DocumentInfo)
- Specifies the structure of a request to delete files (DeleteFileRequest)

### `api\chat\tools\rag_pipeline.py`

**Description:** This module is responsible for managing and providing the Retrieval-Augmented Generator (RAG) pipeline, a critical component of the Integrax framework. It handles tasks such as vector store management, embedding generation, document retrieval, ranking, and answer generation.

**Domain:** RAG Logic

**Key Responsibilities:**

* Manages vector store using IntergraxVectorstoreManager
* Generates embeddings using IntergraxEmbeddingManager
* Retrieves relevant documents through IntergraxRagRetriever
* Ranks retrieved documents using IntergraxReRanker
* Provides answer generation functionality through IntergraxRagAnswerer
* Includes utility functions for managing LLM adapters and generating prompts

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides utility functions for interacting with the Integrax API, enabling features such as chatting, document uploading, listing, and deleting.

**Domain:** API utilities

**Key Responsibilities:**
- Provides a function to get API endpoint URL for specified operations (e.g., "chat", "upload-doc", "list-docs")
- Offers functionality for making POST requests to the Integrax API for various operations:
  - `get_api_response`: sends a POST request with question, model, and optional session ID
  - `upload_document`: uploads a file using a POST request
  - `delete_document`: deletes a document by ID using a POST request
- Handles exceptions and error handling for each operation
- Includes utility functions to interact with the Integrax API, but does not appear to be experimental or incomplete.

### `applications\chat_streamlit\chat_interface.py`

**Description:** This module provides a user interface for interacting with the chat system, utilizing Streamlit components for displaying and handling user input and responses.

**Domain:** Chat Interface

**Key Responsibilities:**

* Displays the chat history using `st.session_state.messages`
* Handles user input through `st.chat_input` and appends it to the chat history
* Retrieves API response from `get_api_response` function and updates session state with new ID
* Displays generated response in a formatted manner, including details about the model used and session ID

This file appears to be production-ready, providing a functional user interface for interacting with the chat system.

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides a Streamlit application's sidebar functionality, including model selection, document upload, listing uploaded documents, and deleting selected documents.

**Domain:** Chat Application Interface Components

**Key Responsibilities:**

* Displaying the model selection component in the sidebar
* Handling file uploads through the `upload_document` API call
* Listing uploaded documents with their IDs and timestamps
* Enabling deletion of selected documents through the `delete_document` API call

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module provides the core functionality for a Streamlit application that integrates with the Intergrax framework, enabling user interaction through a chat interface and sidebar.

**Domain:** Chat Application UI Components

**Key Responsibilities:**

* Initializes Streamlit state variables for storing messages and session IDs
* Displays an interactive sidebar using `display_sidebar` function
* Displays a chat interface using `display_chat_interface` function
* Sets up application title with "intergrax RAG Chatbot" label

### `applications\company_profile\__init__.py`

Description: This module serves as the entry point for the company profile application, responsible for initializing and configuring the necessary components.

Domain: Application Initialization/Configuration

Key Responsibilities:
• Initializes the company profile application
• Sets up required dependencies and services
• Configures application-level settings and constants

### `applications\figma_integration\__init__.py`

Description: This module serves as the entry point for Figma integration, handling initialization and setup.

Domain: Integration

Key Responsibilities:
• Initializes Figma connection and authentication
• Sets up event listeners for file changes and updates
• Exposes APIs for importing Figma designs into Intergrax applications

### `applications\ux_audit_agent\__init__.py`

DESCRIPTION: 
This module initializes and configures the UX audit agent, responsible for auditing user experiences within the Intergrax framework.

DOMAIN: UX Audit Agent Configuration

KEY RESPONSIBILITIES:
• Initializes the UX audit agent with necessary configuration
• Sets up logging and monitoring for audit events
• Defines default settings and behaviors for the audit process

### `applications\ux_audit_agent\components\__init__.py`

Description: This is the entry point for the UX audit agent's components, responsible for registering and initializing various component classes.

Domain: Agents

Key Responsibilities:
* Registers available component classes
* Initializes component instances
* Provides access to registered components

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module provides a compliance checker component for the Intergrax framework, which evaluates proposed changes against privacy policies and regulations.

**Domain:** Compliance Checker

**Key Responsibilities:**

* Checks whether proposed changes comply with privacy policy or regulatory rules (mocked validation)
* Returns findings on compliance, including potential policy violations and recommendations
* Can stop pipeline execution if non-compliant, pending corrections or DPO review
* Produces artifacts and logs for auditing purposes

### `applications\ux_audit_agent\components\cost_estimator.py`

Description: This module contains a cost estimation agent component that calculates the estimated cost of UX-related changes based on an audit report.

Domain: UX Audit Agent Components

Key Responsibilities:
- Provides a cost estimation service for UX updates derived from audits.
- Uses a mock pricing model to calculate the estimated cost.
- Returns a cost estimate, including currency and method used, along with metadata.

### `applications\ux_audit_agent\components\final_summary.py`

**Description:** This module defines a component that generates a final summary of the execution pipeline using collected artifacts.
It is designed to be executed at the end of the pipeline, providing a comprehensive report.

**Domain:** RAG (Results, Artifacts, and Gatherings) logic

**Key Responsibilities:**

* Collects artifacts from the execution pipeline
* Generates a summary based on the collected artifacts
* Returns a final report as a component result

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module defines a Financial Audit Agent component that generates mock financial reports and VAT calculations for testing purposes.

**Domain:** RAG logic (Reusable Aggregate components)

**Key Responsibilities:**

* Defines the Financial Agent component using the `@component` decorator
* Generates mock financial report data, including net values, VAT rates, and gross amounts
* Returns a ComponentResult object with the generated report and log messages

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component for the Intergrax system, answering questions about its structure, features, and configuration.

**Domain:** LLM adapters

**Key Responsibilities:**

* Answers general questions about the Intergrax system
* Includes information on modules, architecture, and documentation
* Returns mock data to simulate real-world knowledge responses

### `applications\ux_audit_agent\components\project_manager.py`

**applications\ux_audit_agent\components\project_manager.py**

Description: This module defines a pipeline component that simulates a project manager's decision on UX reports.

Domain: RAG logic (Risk Assessment and Governance)

Key Responsibilities:
- Simulate a project manager's decision based on a mock model
- Approve or reject UX proposals with comments (70% chance of approval)
- Stop pipeline execution if proposal is rejected
- Produce decision, notes, and relevant metadata

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module defines a UX audit component for the Integrax framework, responsible for analyzing UI/UX based on Figma mockups and generating sample reports with recommendations.

Domain: LLM adapters

Key Responsibilities:
- Performs UX audits on Figma mockups
- Returns sample reports with recommendations
- Supports WCAG AA compliance and visual style consistency checks
- Provides estimated cost for implementation

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This Jupyter Notebook file appears to be a test or demonstration of the UXAuditTest module within the Integrax framework. It defines multiple steps for performing UX audits on FIGMA mockups, verifying compliance with company policy, and preparing summary reports.

Domain: RAG (Rational Agent Goal) logic

Key Responsibilities:
- Define and execute UX audit steps
- Verify changes comply with company policy
- Prepare summary reports for Project Managers
- Evaluate financial impact of changes
- Final report preparation and synthesis

### `generate_project_overview.py`

**Description:** This module provides an automated way to generate a human-readable and LLM-friendly overview of the Intergrax framework's project structure.

**Domain:** Project Structure Documentation

**Key Responsibilities:**

* Recursively scan the project directory
* Collect relevant source files (Python, Jupyter Notebooks, configurable)
* Generate a structured summary for each file via an LLM adapter:
	+ Read the source code and extract content + metadata
	+ Create a prompt to summarize the file's purpose, domain, responsibilities, etc.
	+ Send the prompt to the LLM and retrieve a response
	+ Format the response into a concise but meaningful technical explanation

Note: This module appears to be complete and production-ready.

### `intergrax\__init__.py`

Description: The `__init__.py` file is the entry point and package initializer for the Intergrax framework, responsible for setting up the overall structure and organization of the project.

Domain: Framework core

Key Responsibilities:
- Initializes the Intergrax package and its internal dependencies
- Sets up the project's namespace and module hierarchy
- Provides a central location for importing common functions and modules across the framework

### `intergrax\chains\__init__.py`

Description: This module initializes and sets up the chaining functionality in Intergrax, providing a foundation for sequential processing of data.

Domain: Chains configuration

Key Responsibilities:
• Initializes the chain setup process
• Establishes the base class for chain configurations
• Sets up default chain behaviors and settings

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module defines a flexible QA chain implementation using LangChain, enabling the creation of question-answering pipelines with customizable stages and hooks.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Builds a QA chain with multiple stages:
	+ Retrieval (RAG → [rerank] → prompt)
	+ Prompt building with optional hooks
	+ LLm processing
	+ Output parsing and result packaging
* Provides a LangChain-style API for invoking the QA pipeline
* Allows customization through configuration options and hook functions

**Note:** This file appears to be a core component of the Intergrax framework, implementing a crucial part of its functionality. It does not appear to be experimental, auxiliary, legacy, or incomplete.

### `intergrax\chat_agent.py`

**Description:** This module implements a chat agent based on the Intergrax framework, providing a unified interface for various routes such as RAG (Retrieval-Augmented Generation), tools, and general LLM-based conversations.

**Domain:** Chat Agents / Routing Logic

**Key Responsibilities:**

* Provides a unified API for routing decisions via LLM
* Supports multiple routes: RAG, Tools, General
* Integrates with various components such as IntergraxConversationalMemory, LLMAdapter, and RagComponent
* Offers tool-based operations and vector store management
* Handles streaming and structured output

**Note:** The file appears to be a core component of the Intergrax framework, implementing critical functionality for chat agent routing and integration with various components.

### `intergrax\llm\__init__.py`

DESCRIPTION:
This module serves as the entry point for the LLM adapters, providing a centralized initialization and registration mechanism.

DOMAIN: LLM Adapters

KEY RESPONSIBILITIES:
- Registers available LLM adapter classes
- Provides a default LLM adapter instance
- Initializes and configures LLM adapter instances upon request

### `intergrax\llm\conversational_memory.py`

**Description:** This module provides a universal conversation memory system that stores chat messages in RAM and offers methods to save and load the history to/from files. It supports filtering messages based on their role and tool calls.

**Domain:** LLM (Large Language Model) adapters, conversational memory management

**Key Responsibilities:**

* Store chat messages in RAM with support for filtering by role and tool calls
* Save and load conversation history to/from JSON and NDJSON files
* Handle message overflow and provide methods for appending and extending the conversation history
* Support native tools (e.g., OpenAI) and planners (e.g., Ollama) by filtering out older 'tool' messages or returning the full history
* Provide a locking mechanism to ensure thread safety during concurrent access.

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** 
This module provides a set of utilities and adapters for LLM (Large Language Model) interactions, including OpenAI Chat Completions.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Providing an interface for LLM adapters to interact with various models
* Utility functions for converting between different formats (e.g., chat message → OpenAI schema)
* Specific adapters for interacting with OpenAI's Chat Completions API

Note:
- This file appears to be a part of the main codebase and not experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm_adapters\__init__.py`

**Description:** This module serves as the entry point for LLM adapters in the Intergrax framework, providing a centralized registry and utility functions for working with large language models.

**Domain:** LLM Adapters

**Key Responsibilities:**
- Registers LLM adapter implementations (e.g., OpenAI, Gemini, Ollama) with the LLMAdapterRegistry.
- Exports the registered adapters as part of the module's API.
- Provides utility functions for model registration and validation.

### `intergrax\llm_adapters\base.py`

**Description:** This module defines the core functionality for integrating Large Language Models (LLMs) with the Integrax framework. It provides a universal interface for LLM adapters, as well as tools for handling structured output and tool-based interactions.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Defines the `LLMAdapter` protocol for interacting with LLMs
* Provides methods for generating messages from LLMs (`generate_messages`, `stream_messages`)
* Offers optional support for tools (e.g., `supports_tools`, `generate_with_tools`, `stream_with_tools`)
* Enables structured output generation (`generate_structured`)
* Includes a helper function to map internal chat messages to OpenAI-compatible message dictionaries (`_map_messages_to_openai`)
* Defines an adapter registry (`LLMAdapterRegistry`) for easy registration and creation of LLM adapters

**Notes:** The code appears to be well-maintained, with clear documentation and adherence to best practices. However, there are some experimental features (e.g., tool support) that may not be fully fleshed out. Overall, this module provides a solid foundation for integrating LLMs with the Integrax framework.

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module provides a minimal Gemini chat adapter for the Integrax framework, focusing on simple chat usage without tool integration.

**Domain:** LLM adapters

**Key Responsibilities:**
- Initializes the Gemini chat adapter with a model and optional defaults
- Splits system messages from conversation history
- Generates responses to input messages using the underlying model
- Streams generated messages
- Provides placeholder methods for tool integration (not implemented)

### `intergrax\llm_adapters\ollama_adapter.py`

Description: This module implements an adapter class for integrating Ollama models with the Integrax framework using LangChain's ChatModel interface.

Domain: LLM adapters

Key Responsibilities:
- Adapts Ollama models to work within the Integrax framework
- Converts internal chat messages to LangChain message objects
- Supports temperature and max tokens adjustments for generation
- Provides a planner-style pattern for tool-calling (no native support)
- Offers structured output functionality via prompt + validation

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an OpenAI adapter for generating responses using the new Responses API. It supports plain chat, tools, and structured JSON output.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides a public interface compatible with the previous Chat Completions adapter
* Supports single-shot completion (non-streaming) and streaming completion using the Responses API
* Converts Chat Completion style messages to the Responses API "input items"
* Extracts assistant's output text from Responses API results
* Generates responses with potential function/tool calls
* Returns structured JSON output validated against a provided schema

Note: This module appears to be fully functional, well-documented, and properly maintained.

### `intergrax\logging.py`

Description: This module provides the configuration and setup for logging in the Integrax framework, establishing global settings for the logger and specifying its behavior.

Domain: Logging Configuration

Key Responsibilities:
- Configures basic logging with specified level and format.
- Sets up logging to display INFO and higher levels of severity.
- Specifies a custom log format including timestamp and log level.

### `intergrax\multimedia\__init__.py`

Description: The `__init__.py` file is responsible for initializing the multimedia module within the Intergrax framework, defining its namespace and importing necessary components.

Domain: Multimedia Processing

Key Responsibilities:
- Initializes the multimedia module
- Defines the module's namespace
- Imports necessary classes and functions from other modules

### `intergrax\multimedia\audio_loader.py`

**Description:** This module provides functionality for downloading and translating audio files from YouTube URLs using the yt_dlp library and Whisper speech recognition model.

**Domain:** Multimedia (specifically, Audio Loading)

**Key Responsibilities:**

* Downloads audio files from specified YouTube URLs in various formats (default is MP3)
* Creates a directory structure to store downloaded audio files
* Extracts audio from videos using FFmpegExtractAudio postprocessor
* Translates audio text into the desired language using Whisper model

This module appears to be complete and production-ready.

### `intergrax\multimedia\images_loader.py`

Description: This module provides a utility for transcribing images using LLM adapters, specifically the ollama library.

Domain: Image Transcription via LLM Adapters

Key Responsibilities:
- Provides an `transcribe_image` function to extract text from images
- Utilizes the ollama library and its 'llava-llama3' model for transcription
- Allows for customization of the model via the `model` parameter

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, including audio, images, and videos, within Jupyter notebooks.

**Domain:** Multimedia Display Utilities

**Key Responsibilities:**

* `display_audio_at_data`: Displays an audio file at a specified position.
* `_is_image_ext`: Checks if a given path is an image extension.
* `display_image`: Displays an image file or a path to an image.
* `_serve_path`: Serves a file by copying it to a temporary directory and returning the relative path.
* `display_video_jump`: Displays a video with a specified start position, poster frame, and other optional parameters.

Note: The code appears to be well-structured and maintained, but I couldn't identify any specific flags that indicate experimental, auxiliary, legacy, or incomplete status.

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides functionality for video processing, including downloading videos from YouTube, transcribing audio to text, and extracting frames with associated metadata.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloading videos from YouTube using `yt_dlp` library
* Transcribing audio to text using `whisper` library
* Extracting frames from video with associated metadata using OpenCV
* Saving extracted frames and metadata to specified paths

### `intergrax\openai\__init__.py`

Description: This package serves as the entry point for OpenAI integration within Intergrax, initializing and setting up necessary components.

Domain: LLM adapters

Key Responsibilities:
* Initializes the OpenAI API client
* Sets default parameters for API interactions
* Imports and registers available OpenAI models and functions

### `intergrax\openai\rag\__init__.py`

Description: The __init__.py file is a module that serves as an entry point for the RAG (ReAdGeR) logic within the Intergrax framework, providing import functionality and potentially other initialization-related tasks.

Domain: RAG logic

Key Responsibilities:
- Exposes the main interface for interacting with the RAG components
- Initializes the necessary dependencies for RAG operations

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides an implementation of the RAG (Relevance, Accuracy, and Generality) protocol for OpenAI clients within the Integrax framework. It enables interaction with vector stores and file storage.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Initializes a client connection to OpenAI
* Manages interaction with vector stores, including retrieving and clearing data
* Provides methods for uploading folders to vector stores
* Generates RAG prompts based on the provided protocol guidelines

### `intergrax\rag\__init__.py`

DESCRIPTION:
This module initializes and sets up the RAG (Retrieval-Augmented Generation) logic within the Intergrax framework.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes RAG components
- Sets up RAG configuration
- Imports required modules for RAG functionality

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader for various file formats, including text, images, PDFs, and Excel spreadsheets. It allows injecting metadata, includes safety guards, and enables flexible configuration.

**Domain:** Document Loaders

**Key Responsibilities:**

* Loading documents from various file formats (e.g., .txt, .docx, .pdf, .xlsx)
* Injecting metadata into loaded documents
* Handling images, including OCR and EXIF data extraction
* Supporting video loading through multimedia adapters
* Configurable document loader with adjustable settings for verbosity, file patterns, extensions, and more

The code appears to be well-structured and comprehensive, covering various aspects of document loading. It is likely a core component of the Intergrax framework.

### `intergrax\rag\documents_splitter.py`

**Description:** This module provides a high-quality text splitter for RAG pipelines, implementing the 'semantic atom' policy. It generates stable chunk ids and rich metadata.

**Domain:** RAG logic

**Key Responsibilities:**

*   Splits documents into chunks based on pre-defined separators
*   Generates stable chunk ids using available anchors (para_ix/row_ix/page_index) or falls back to index + hash of content
*   Adds metadata to each chunk, including parent id, source name, and page index if present
*   Merges tiny tails with previous chunks per document
*   Applies optional hard cap on the number of chunks per document

Note: The provided file appears to be a well-maintained and functional module within the Intergrax framework.

### `intergrax\rag\dual_index_builder.py`

**Description:** This module provides functionality for building dual indexes using vector embeddings. It splits documents into primary and auxiliary collections based on specific criteria.

**Domain:** RAG (Retrieval Augmented Generation) logic

**Key Responsibilities:**

*   Builds two vector indexes:
    *   Primary collection (`CHUNKS`): all chunks/documents after splitting
    *   Auxiliary collection (`TOC`): only DOCX headings within specified levels
*   Embeds documents using an embedding manager
*   Adds embedded documents to the primary and auxiliary collections in batches
*   Provides logging and error handling for the process

The code appears to be production-ready, with proper error handling and logging mechanisms. However, it does use some proprietary classes and functions (e.g., `IntergraxVectorstoreManager`, `IntergraxEmbeddingManager`), which may require additional context or documentation for external users.

### `intergrax\rag\dual_retriever.py`

**Description:** The `dual_retriever.py` module provides a class for retrieving relevant chunks from the Intergrax vector store, utilizing both direct querying and contextual expansion via table of contents (TOC). It enables searching across different sections or sources based on their parent IDs.

**Domain:** RAG logic / Retrieval-Augmented Generation

**Key Responsibilities:**

* Initializes a dual retriever with vector store managers for chunks and TOC
* Performs base retrieval from the chunk vector store using embeddings
* Expands context via TOC to search locally by parent ID (propagates filter conditions)
* Merges, deduplicates, and sorts hits by similarity score
* Provides a `retrieve` method to execute the dual retrieval process

**Note:** This module appears to be part of the Intergrax framework's Retrieval-Augmented Generation logic, designed to efficiently search and retrieve relevant information from the vector store. The code is well-structured and follows good practices; there are no clear indications of experimental, auxiliary, legacy, or incomplete status.

### `intergrax\rag\embedding_manager.py`

**Description:** The `intergrax\rag\embedding_manager.py` file is a unified embedding manager that enables working with various types of embeddings, including Hugging Face (SentenceTransformer), Ollama, and OpenAI embeddings.

**Domain:** RAG logic / Embedding management

**Key Responsibilities:**

* Unified loading and configuration of different embedding models
* Provider switch between "hg", "ollama", and "openai"
* Reasonable defaults for model names if not provided
* Batch and single-text embedding capabilities with optional L2 normalization
* Embedding support for LangChain Documents (returns np.ndarray + aligned docs)
* Cosine similarity utilities and top-K retrieval
* Robust logging, shape validation, and light retry for transient errors

### `intergrax\rag\rag_answerer.py`

**Description:** This module implements an answerer component for the Integrax framework, utilizing Retrieve-Augment-Generate (RAG) logic to retrieve relevant context fragments and generate answers with the assistance of a Large Language Model (LLM).

**Domain:** RAG logic / LLM adapters

**Key Responsibilities:**

*   Retrieval:
    *   Retrieves top-k most similar context fragments based on similarity scores.
    *   Applies local similarity threshold filter if configured.
*   Re-ranking (optional):
    *   Re-ranks retrieved hits using a reranker component if provided.
*   Context building:
    *   Concatenates relevant context text from used hits, considering character limits.
    *   Creates citations for sources.
*   Message construction:
    *   Builds system and user messages based on the question, context, and user instruction (if available).
*   Answer generation:
    *   Passes constructed messages to an LLM adapter for answer generation.
    *   Supports streaming answer generation.

**Note:** The code appears well-structured, but some functionality is commented out or marked as optional. Additionally, there are some minor issues with formatting and documentation style (e.g., inconsistent spacing around functions and classes). Overall, this module seems to be a key component of the Integrax framework, enabling RAG logic for answer generation with LLM assistance.

### `intergrax\rag\rag_retriever.py`

**Description:** 
This module provides a scalable, provider-agnostic RAG (Relevance Aware Generator) retriever for the Intergrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Normalizes `where` filters for Chroma
* Normalizes query vector shape to [[D]]
* Unified similarity scoring for different providers
* Deduplication by ID and per-parent result limiting (diversification)
* Optional MMR diversification when embeddings are returned
* Batch retrieval for multiple queries
* Optional reranker hook (e.g., cross-encoder, re-ranking model)

Note: The file appears to be complete and is a core component of the Intergrax framework.

### `intergrax\rag\re_ranker.py`

**Description:** The `re_ranker.py` file is a module within the Integrax framework that provides a re-ranking mechanism for candidates based on cosine similarity to a given query. It allows users to input a query and a list of candidates, and returns the top-ranked candidates.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Accepts hits from the intergraxRagRetriever (dict) or raw LangChain Documents
* Embeds texts in batches using intergraxEmbeddingManager
* Optional score fusion with original retriever similarity
* Preserves schema of hits; only adds 'rerank_score', optional 'fusion_score', and 'rank_reranked'
* Re-ranks candidates by cosine similarity to the query
* Supports various input formats, including positional arguments and keyword arguments

Note: The code appears to be well-structured and comprehensive, with clear documentation and a straightforward implementation. There is no indication that it is experimental, auxiliary, legacy, or incomplete.

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module, `vectorstore_manager.py`, serves as the unified vector store manager for Intergrax, supporting ChromaDB, Qdrant, and Pinecone. It provides a consistent interface to initialize target stores, create collections or indexes (if needed), upsert documents with embeddings, query similar vectors, count vectors, and delete by IDs.

**Domain:** Vector Store Management

**Key Responsibilities:**

* Initialize vector store client based on configuration
* Create collection or index (lazy for Qdrant/Pinecone)
* Upsert documents with embeddings (with batching)
* Query top-K similar vectors
* Count vectors
* Delete vectors by IDs
* Handle ChromaDB, Qdrant, and Pinecone-specific configurations and APIs

**Note:** The module appears to be a critical component of the Intergrax framework, providing essential functionality for managing vector stores.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a windowed answerer, which is a layer on top of the base Answerer. It enables processing large amounts of context by breaking it down into smaller windows and generating partial answers for each window before synthesizing a final answer.

**Domain:** RAG logic

**Key Responsibilities:**

* Process large amounts of context by dividing it into smaller windows
* Generate partial answers for each window using the LLM
* Synthesize a final answer from the partial answers
* Deduplicate sources and maintain memory-awareness throughout the process

### `intergrax\runtime\__init__.py`

Description: The __init__.py file serves as the entry point and package initializer for the Intergrax runtime, responsible for setting up essential components and modules.

Domain: Runtime Initialization

Key Responsibilities:
- Initializes the package namespace
- Defines the root module for the runtime
- Sets up default configurations and dependencies

### `intergrax\runtime\drop_in_knowledge_mode\__init__.py`

DESCRIPTION: The `__init__.py` file is the entry point for the drop-in knowledge mode functionality in the Intergrax runtime. It initializes and configures the necessary components for this feature.

DOMAIN: Drop-In Knowledge Mode Logic

Key Responsibilities:
- Initializes the drop-in knowledge mode environment
- Configures required parameters and settings
- Exposes APIs for enabling and disabling the feature

### `intergrax\runtime\drop_in_knowledge_mode\attachments.py`

**Description:** This module provides utilities for resolving attachments in the Intergrax framework's Drop-In Knowledge Mode.

**Domain:** RAG (Reasoning and Generation) pipeline utility modules

**Key Responsibilities:**

* Defines an `AttachmentResolver` protocol that abstracts attachment resolution
* Provides a concrete implementation, `FileSystemAttachmentResolver`, for local filesystem-based URIs
* Allows decoupling of attachment storage mechanisms from the RAG pipeline's consumption of attachments

### `intergrax\runtime\drop_in_knowledge_mode\config.py`

**Description:** This module provides configuration objects for the Drop-In Knowledge Mode runtime in the Intergrax framework.

**Domain:** RAG logic/configuration

**Key Responsibilities:**

* Define main knobs for controlling the runtime's behavior
* Integrate with existing Intergrax components (LLM adapters, embedding and vector store managers)
* Enable or disable specific features (RAG, web search, tools, long-term memory)
* Configure token budgets for context construction
* Provide a strongly typed configuration object for the Drop-In Knowledge Mode runtime

### `intergrax\runtime\drop_in_knowledge_mode\engine.py`

**Description:** 
This module provides the core runtime engine for Drop-In Knowledge Mode, responsible for handling chat sessions, user input, and generating responses.

**Domain:** RAG (Relevant Answer Generator) logic

**Key Responsibilities:**

* Load or create a chat session via SessionStore
* Append user messages to the session
* Build conversation history for the LLM
* Call the configured LLM adapter to generate responses
* Produce a RuntimeAnswer object with final answer text and metadata

Note: The engine is currently in its early stages, with some responsibilities (e.g., RAG integration) not yet implemented.

### `intergrax\runtime\drop_in_knowledge_mode\ingestion.py`

**Description:** This module provides an ingestion pipeline for attachments in the context of the Drop-In Knowledge Mode, leveraging the Intergrax RAG components to load, split, and embed documents.

**Domain:** Data Ingestion / Vector Database Management

**Key Responsibilities:**

* Resolve AttachmentRef objects into filesystem Paths using AttachmentResolver
* Load documents using IntergraxDocumentsLoader.load_document(...)
* Split documents into chunks using IntergraxDocumentsSplitter.split_documents(...)
* Embed chunks via IntergraxEmbeddingManager
* Store vectors via IntergraxVectorstoreManager
* Return a structured IngestionResult per attachment

This file appears to be part of the mainline codebase, with no obvious signs of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\runtime\drop_in_knowledge_mode\response_schema.py`

**Description:** This module provides data models for the Intergrax framework's Drop-In Knowledge Mode runtime, defining request and response structures for applications interacting with it.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defining high-level contract between applications (e.g., FastAPI, Streamlit) and the DropInKnowledgeRuntime
* Exposing citations, routing information, tool calls, and basic statistics
* Providing dataclasses for request (`RuntimeRequest`) and response (`RuntimeAnswer`) structures
* Including utility classes for:
	+ `Citation`: representing a single citation/reference used in the final answer
	+ `RouteInfo`: describing how the runtime decided to answer the question
	+ `ToolCallInfo`: describing a single tool call executed during the runtime request
	+ `RuntimeStats`: basic statistics about a runtime call

Note: The file appears to be well-documented, stable, and used in production (given its position within the `intergrax` project structure).

### `intergrax\runtime\drop_in_knowledge_mode\session_store.py`

**Description:** This module provides a session storage abstraction for the Intergrax framework's Drop-In Knowledge Mode runtime.

**Domain:** Session Storage Abstractions

**Key Responsibilities:**

* Defines data classes for chat sessions and messages
* Establishes a `SessionStore` protocol for different backend implementations (in-memory, SQLite, PostgreSQL, etc.)
* Provides an in-memory implementation (`InMemorySessionStore`) for quick experiments and notebooks

### `intergrax\supervisor\__init__.py`

DESCRIPTION: 
This is the entry point for Intergrax's supervisor module, responsible for initializing and managing the overall workflow of the framework.

DOMAIN: Supervisor/Manager

KEY RESPONSIBILITIES:
- Initializes and sets up the framework components
- Manages the execution flow and sequence of tasks
- Provides a centralized control mechanism

### `intergrax\supervisor\supervisor.py`

**Description:** This module provides the core functionality of the Integrax supervisor, which is responsible for planning and executing tasks using a combination of machine learning models and domain-specific knowledge.

**Domain:** Task Supervisor

**Key Responsibilities:**

* Planning tasks based on user queries and metadata
* Decomposing tasks into individual steps with associated components and methods
* Assigning components to each step based on the task's requirements
* Executing tasks using the assigned components and machine learning models
* Analyzing the results of each task and providing feedback for improvement

**Notes:** The file appears to be well-maintained and up-to-date, with clear documentation and organization. However, some comments suggest that certain features are experimental or legacy, which may require further investigation.

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides core components and utilities for building and managing tasks within the Integrax framework.

**Domain:** Supervision and Task Management

**Key Responsibilities:**

* Defines data structures for pipeline state and component results.
* Provides classes for defining task components with metadata and implementation functions.
* Offers a decorator for quick registration of new components.
* Implements basic logic for running components and handling errors.

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides default prompt templates for the Intergrax Unified Supervisor, which are used to guide the planning process and ensure correctness.

**Domain:** RAG logic / Planning framework

**Key Responsibilities:**

* Define the planning system's rules and principles (e.g., decomposition-first mandate, primary principles)
* Provide a universal plan format (JSON object) for the supervisor to return
* Specify the plan validation checklist to ensure correctness
* Offer default prompt templates for the intergrax unified Supervisor (plan_system and plan_user_template)

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module provides utilities for creating and managing the LangGraph pipeline within the Integrax framework. It handles state management, node creation, and graph construction.

**Domain:** RAG logic (Reasoning and Action Graphs)

**Key Responsibilities:**

* Manages global state traveling through the LangGraph pipeline
* Provides functions for resolving inputs, persisting outputs, and appending logs to the state
* Creates unique node names from plan steps and their titles
* Generates a topological ordering of plan steps based on dependencies
* Builds a LangGraph pipeline from a Plan instance, including node creation and connection in a stable order

### `intergrax\system_prompts.py`

Description: This module defines a default RAG system instruction template for the Integrax framework.

Domain: LLM adapters

Key Responsibilities:
- Provides a strict RAG (Role and Guidelines) system instruction for knowledge assistants.
- Outlines procedures for understanding questions, searching for context, verifying consistency, answering, citing sources, and handling ambiguity or uncertainty.
- Offers guidelines on style, format, and precision in response generation.

### `intergrax\tools\__init__.py`

Description: This is the main entry point for the Intergrax tools, responsible for initializing and setting up various utility modules.

Domain: Configuration

Key Responsibilities:
* Initializes Intergrax tooling
* Sets up utility module dependencies
* Defines API entry points for tools functionality

### `intergrax\tools\tools_agent.py`

**Description:** The `tools_agent.py` module provides a class-based implementation of the Intergrax Tools Agent, which enables the integration of external tools with the LLM (Large Language Model) adapters.

**Domain:** LLM adapters

**Key Responsibilities:**

* Instantiates and configures the Tools Agent instance
* Provides methods for tool invocation and result handling
* Supports both native tools (OpenAI) and JSON planners (Ollama)
* Implements pruning of tool messages in OpenAI mode
* Builds output structure based on tool traces and answer text

Note that this file appears to be part of a larger framework, and its functionality is closely tied to other components within the Intergrax project.

### `intergrax\tools\tools_base.py`

**Description:** This module provides base classes and utilities for tool development within the Integrax framework.

**Domain:** Configuration/Utility Modules

**Key Responsibilities:**

* Provides a `ToolBase` class that serves as a base for all tools, defining common attributes and methods.
* Offers utility functions for safely truncating long tool outputs to avoid overflowing LLM context (`_limit_tool_output`).
* Implements a `ToolRegistry` class for registering, managing, and exporting tools in a format compatible with the OpenAI Responses API.
* Allows tools to be validated using Pydantic schema models (optional).
* Enables tools to be easily converted into OpenAI-compatible JSON objects.

### `intergrax\websearch\__init__.py`

DESCRIPTION: This module serves as the entry point for the web search functionality within the Intergrax framework, defining the interface and initialization for related components.

DOMAIN: Web Search API

KEY RESPONSIBILITIES:
- Initializes the web search component
- Defines API endpoints for web search functionality
- Sets up dependencies and configurations for web search operations

### `intergrax\websearch\cache\__init__.py`

**Description:** This module implements a simple in-memory query cache with optional time-to-live (TTL) and maximum size for web search results.

**Domain:** Web Search Cache

**Key Responsibilities:**

* Provides an `InMemoryQueryCache` class to store and retrieve cached web search results
* Utilizes dataclasses (`QueryCacheKey`, `QueryCacheEntry`) to represent cache keys and entries
* Offers methods to set and get cached values, with optional TTL and maximum size control

### `intergrax\websearch\context\__init__.py`

Description: This module serves as the entry point for the web search context within the Intergrax framework, managing initialization and configuration for search-related operations.

Domain: Web Search Context Configuration

Key Responsibilities:
- Initializes the web search context with default settings.
- Defines configurable parameters for web search functionality.
- Provides a central location for integrating web search modules.

### `intergrax\websearch\context\websearch_context_builder.py`

Description: This module provides a builder for creating textual context and chat messages from web search results, specifically tailored for Large Language Models (LLMs).

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Builds a textual context string from WebDocument objects or serialized dicts.
- Constructs system and user prompts for chat-style LLMs.
- Allows configuration of maximum documents, characters per document, snippet inclusion, URL inclusion, and source label prefix.

### `intergrax\websearch\fetcher\__init__.py`

Description: This module initializes and configures the web search fetcher, responsible for retrieving data from external sources.

Domain: Web Search Fetcher Utilities

Key Responsibilities:
* Initializes the fetcher instance with configuration settings
* Sets up logging and error handling mechanisms
* Defines API endpoints for fetching data from external sources

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module contains two functions, `extract_basic` and `extract_advanced`, which are responsible for extracting metadata and content from web pages. The `extract_basic` function performs lightweight HTML extraction, while the `extract_advanced` function uses readability-based extraction.

**Domain:** Web Search - Fetcher

**Key Responsibilities:**

- Perform lightweight HTML extraction (`extract_basic`)
  - Extract title
  - Extract meta description
  - Extract language attribute from `<html>` tag
  - Extract Open Graph metadata tags
  - Produce a plain-text version of the page

- Perform advanced readability-based extraction (`extract_advanced`)
  - Remove boilerplate elements (scripts, styles, iFrames, navigation)
  - Try to extract primary readable content using trafilatura (if installed)
  - Fallback to BeautifulSoup plain-text extraction if trafilatura fails
  - Normalize whitespace and reduce noise
  - Optionally overwrite existing text

The `extract_basic` function is intentionally conservative and does not perform advanced readability or boilerplate removal. The `extract_advanced` function is designed to be synchronous, allowing it to be used in both synchronous and asynchronous pipelines.

Note: This file appears to be a part of the Intergrax framework's web search functionality and seems to be well-maintained and complete.

### `intergrax\websearch\fetcher\http_fetcher.py`

Description: This module provides an HTTP fetcher for web pages, allowing retrieval of page content and metadata.

Domain: Web Search

Key Responsibilities:
- Performs asynchronous HTTP GET requests to fetch web pages
- Returns a PageContent instance containing metadata (status code, headers, etc.) and raw HTML content
- Includes default headers and allows user-provided custom headers

### `intergrax\websearch\integration\__init__.py`

Description: This module is the entry point for the web search integration in Intergrax, responsible for initializing and configuring the integration.

Domain: Web Search Integration

Key Responsibilities:
• Initializes the web search integration
• Configures the integration with external services
• Sets up necessary connections and APIs

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a web search functionality as a LangGraph-compatible node wrapper. It delegates to an externally configured `WebSearchExecutor` instance for actual search logic.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

*   Encapsulates configuration of `WebSearchExecutor`
*   Implements sync and async node methods operating on `WebSearchState`
*   Delegates search logic to the provided `WebSearchExecutor` instance
*   Provides a default, module-level node instance for convenience and backward compatibility

### `intergrax\websearch\pipeline\__init__.py`

Description: This module defines the entry point for the web search pipeline, responsible for setting up and running the search process.

Domain: Web Search Pipeline Setup

Key Responsibilities:
* Initializes the search engine
* Sets up query processing pipeline
* Establishes connection to search data store

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module implements a pipeline for web search and data retrieval, handling multiple providers, fetching, extraction, deduplication, and quality scoring.

**Domain:** Websearch Pipeline

**Key Responsibilities:**

* Orchestrates multi-provider web search with rate limiting and deduplication
* Fetches and extracts SearchHit objects into WebDocument objects
* Performs basic quality scoring and dedupe key computation
* Sorts results by quality score (descending) and source rank (ascending)
* Allows for synchronous execution through a convenience wrapper

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: 
This module initializes and manages the web search providers for Intergrax.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
* Initializes the web search provider instances
* Registers available web search providers with the framework
* Provides a unified interface for accessing different web search services

### `intergrax\websearch\providers\base.py`

**Description:** This module defines the base interface for web search providers in the Integrax framework. It abstracts away provider-specific details and provides a standardized way to execute searches.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an interface for executing searches with a QuerySpec
* Returns a ranked list of SearchHit items
* Exposes minimal capabilities for feature negotiation (language, freshness)
* Allows providers to honor top_k, sanitize/validate URLs, and include provider-specific fields in hits

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module implements a Bing Web Search provider for the Intergrax framework, utilizing the Bing REST API (v7). It enables querying and retrieving search results with various filtering options.

**Domain:** LLM adapters

**Key Responsibilities:**

*   Initializes the Bing Web Search provider with an optional API key, session, and timeout
*   Retrieves capabilities of the provider (supports language, freshness, and max page size)
*   Builds request headers for authentication with the API key
*   Constructs query parameters based on the input QuerySpec object
*   Performs a search using the constructed parameters and returns a list of SearchHit objects
*   Closes the session after use

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module provides a web search provider for the Intergrax framework, specifically integrating with Google Custom Search (CSE) via their REST API.

**Domain:** WebSearch Providers

**Key Responsibilities:**

* Initializes the provider with required environment variables and parameters
* Supports language filtering through 'lr' and 'hl' parameters
* Handles freshness by ignoring it due to lack of native support in CSE
* Validates URLs and extracts necessary metadata from search results
* Converts items into SearchHit objects for further processing

Note: This module appears to be a standard implementation of the web search provider, with no indications of being experimental, auxiliary, legacy, or incomplete.

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a Google Places API provider for the Intergrax web search functionality, allowing text searches and retrieving business details.

**Domain:** Web Search Providers (LLM adapters)

**Key Responsibilities:**

* Initializes the Google Places API with an optional API key
* Supports text searching using the `QuerySpec` query as input
* Retrieves business details for a given place ID
* Maps search results to `SearchHit` objects
* Provides capabilities such as language support and maximum page size

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a Reddit API provider for the Intergrax framework, enabling full-featured search and metadata retrieval.

**Domain:** Websearch Providers

**Key Responsibilities:**

* Implements authentication with Reddit's OAuth2 API using client credentials
* Provides capabilities such as language filtering (currently disabled) and freshness support
* Fetches search results from Reddit's official API
* Optionally fetches top-level comments for each post
* Maps response data to SearchHit objects

### `intergrax\websearch\schemas\__init__.py`

Description: This module initializes the schemas for web search functionality within the Intergrax framework.

Domain: Web Search Schemas

Key Responsibilities:
- Imports and defines schema modules
- Initializes schema registry
- Exports schema utilities for use in web search components.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass `PageContent` to represent the content of a web page, encapsulating raw HTML and derived metadata.

**Domain:** Web Search

**Key Responsibilities:**

* Represents fetched and optionally extracted web page content
* Encapsulates both raw HTML and derived metadata for post-processing stages
* Includes fields for common metadata such as URL, status code, title, description, language, etc.
* Provides methods for filtering empty fetches (`has_content`), generating a truncated text snippet (`short_summary`), and estimating content size in kilobytes (`content_length_kb`)

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a dataclass representing a canonical search query specification for web search providers.

**Domain:** Query Schemas

**Key Responsibilities:**
- Defines the `QuerySpec` dataclass with frozen=True.
- Specifies the structure and default values of the query specification model.
- Provides methods to normalize the query string (`normalized_query`) and cap the top results count (`capped_top_k`).

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a data class `SearchHit` that represents a single search result entry, providing metadata for various search providers.

**Domain:** Search schemas

**Key Responsibilities:**

* Defines a frozen data class `SearchHit` with attributes for provider-agnostic search result metadata.
* Provides validation checks in the `__post_init__` method to ensure valid rank and URL values.
* Offers utility methods:
	+ `domain`: returns the netloc part of the URL for quick grouping or scoring
	+ `to_minimal_dict`: generates a minimal, LLM-friendly representation of the hit

This file appears to be production-ready and is used within the Intergrax framework.

### `intergrax\websearch\schemas\web_document.py`

**intergrax\websearch\schemas\web_document.py**

Description: This module defines a dataclass `WebDocument` representing a unified structure for processed web documents, integrating search hits with extracted content and analysis results.

Domain: Web Search / RAG logic

Key Responsibilities:
- Provides a dataclass `WebDocument` to encapsulate fetched and processed web document metadata.
- Offers methods for validation (`is_valid`) and text extraction (`merged_text`, `summary_line`).
- Integrates search hit (provider metadata) with extracted content (PageContent) and analysis results.

### `intergrax\websearch\service\__init__.py`

DESCRIPTION: This module initializes the web search service, providing a foundation for subsequent functionality.

DOMAIN: Web Search Service

KEY RESPONSIBILITIES:
• Initializes and configures the web search service.
• Establishes dependencies for further web search-related operations.

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level helper class `WebSearchAnswerer` for performing web searches, building context/messages from search results, and generating answers via Large Language Models (LLMs).

**Domain:** Web Search/LLM Integration

**Key Responsibilities:**

* Runs web search using `WebSearchExecutor`
* Builds LLM-ready context/messages from search results
* Calls an `LLMAdapter` to generate a final answer
* Supports both asynchronous and synchronous operation modes
* Provides optional overrides for system prompt and answer language

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module implements a high-level web search executor that provides configurable and cacheable web search functionality.

**Domain:** Web Search Framework

**Key Responsibilities:**

* Construct QuerySpec from raw query and configuration parameters
* Execute SearchAndReadPipeline with chosen providers
* Convert WebDocument objects into LLM-friendly dicts for serialization
* Manage query cache for serialized results
* Provide a main entry point for web search orchestration in notebooks, LangGraph nodes, and other code

### `intergrax\websearch\utils\__init__.py`

Description: This is the entry point for the web search utility module, responsible for initializing and providing core functionality.

Domain: Web Search Utilities

Key Responsibilities:
* Initializes web search utilities
* Registers available web search engines
* Provides a consistent interface for interacting with web search results

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides utilities for deduplicating web search results by normalizing text and generating stable hash keys.

**Domain:** Web Search Utilities

**Key Responsibilities:**

* Normalizes text before deduplication (removes whitespace, converts to lower case)
* Generates a stable SHA-256 based hash key from normalized text
* Provides simple deduplication functionality for web search pipeline

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module implements a simple token bucket rate limiter for asynchronous tasks, ensuring that the average request rate does not exceed a specified limit.

**Domain:** RAG logic (Rate Adjustment & Governance)

**Key Responsibilities:**

* Provides an asyncio-compatible token bucket rate limiter
* Allows configuration of refill rate and capacity
* Supports both blocking (`acquire`) and non-blocking (`try_acquire`) consumption of tokens
* Designed for use in concurrent coroutines within a single process

Note: The code appears to be well-maintained, complete, and production-ready.

### `main.py`

**Description:** This module serves as the entry point for the Integrax framework, responsible for initiating its execution and displaying a welcome message.

**Domain:** Configuration/Initialization

**Key Responsibilities:**
• Provides an entry point for the Intergrax framework.
• Initializes and runs the main application logic.
 
Note: There's no indication of experimental, auxiliary, legacy, or incomplete functionality in this file.

### `mcp\__init__.py`

DESCRIPTION:
This file serves as the main entry point for the Intergrax framework, initializing and setting up necessary modules.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:
* Initializes core components
* Sets up module dependencies
* Configures global settings 

Note: This is a standard __init__.py file, expected to be part of any Python package, hence marked as "not experimental", "auxiliary" or "legacy".

### `notebooks\drop_in_knowledge_mode\01_basic_memory_demo.ipynb`

**Description:** This is a Jupyter notebook used for testing the Intergrax framework, specifically the Drop-In Knowledge Mode runtime. It provides a basic sanity check to verify that the runtime can create or load a session, append user and assistant messages, build conversation history from SessionStore, and return a RuntimeAnswer object.

**Domain:** Agents

**Key Responsibilities:**

* Verifying the functionality of the Drop-In Knowledge Mode runtime
* Creating or loading a session using InMemorySessionStore
* Appending user and assistant messages to the session
* Building conversation history from SessionStore
* Returning a RuntimeAnswer object

### `notebooks\drop_in_knowledge_mode\02_attachments_ingestion_demo.ipynb`

**Description:** This notebook demonstrates the Intergrax framework's Drop-In Knowledge Mode runtime capabilities, specifically its ability to ingest attachments and store them in a vector store. It showcases how to initialize the runtime, create an attachment reference for a local file, and let the runtime handle ingestion.

**Domain:** RAG logic

**Key Responsibilities:**

- Initializes the Drop-In Knowledge Mode runtime with an in-memory session store, LLM adapter, embedding manager, and vector store manager.
- Demonstrates how to prepare an AttachmentRef for a local project document and simulate its ingestion by the runtime.
- Highlights the use of Intergrax's RAG components (embedding manager and vector store manager) and its ability to handle ingestion without performing RAG retrieval at this stage.

### `notebooks\langgraph\hybrid_multi_source_rag_langgraph.ipynb`

**Description:** This Jupyter notebook demonstrates an end-to-end RAG workflow using the Intergrax framework and LangGraph, combining multiple knowledge sources into a single in-memory vector index.

**Domain:** Hybrid Multi-Source RAG with Intergrax + LangGraph

**Key Responsibilities:**

* Ingest content from multiple sources (local PDF files, local DOCX/Word files, live web results using the Intergrax `WebSearchExecutor`)
* Build a unified RAG corpus by normalizing documents into a common internal format and attaching basic metadata
* Create an in-memory vector index using an Intergrax embedding manager (e.g., OpenAI or Ollama)
* Answer user questions with a RAG agent using LangGraph to orchestrate the flow and generate a structured report

**Note:** This file appears to be a demonstration notebook for a specific use case, and its primary purpose is educational.

### `notebooks\langgraph\simple_llm_langgraph.ipynb`

**Description:** This notebook demonstrates the integration between Intergrax and LangGraph, specifically a simple LLM QA example where an Intergrax LLM adapter is used as a node inside a LangGraph graph.

**Domain:** LLM adapters, RAG logic

**Key Responsibilities:**

* Initialize an OpenAIChatResponsesAdapter instance using the client and model
* Define a State (SimpleLLMState) that holds messages and answer fields
* Implement a LangGraph node (llm_answer_node) that calls the Intergrax LLM adapter to generate answers
* Build a StateGraph with a single node llm_answer_node
* Run the graph on a sample input to demonstrate the integration

**Note:** The code provided is a notebook output, and it seems like some parts are missing or truncated. However, based on the content, it appears that this notebook is designed to showcase the interaction between Intergrax and LangGraph for LLM QA tasks.

### `notebooks\langgraph\simple_web_research_langgraph.ipynb`

**Description:** This Jupyter Notebook demonstrates a Web Research Agent built from Intergrax components and LangGraph. It showcases how to power "no-hallucination" web-based Q&A inside a graph-based agent.

**Domain:** LLM adapters, RAG logic, data ingestion, agents

**Key Responsibilities:**

* Initializes LLM adapter (OpenAIChatResponsesAdapter) and WebSearch components
* Defines the graph state for the Web Research Agent (WebResearchState)
* Implements nodes for normalizing user questions and running web search using Intergrax WebSearchExecutor
* Demonstrates a practical example of how Intergrax can power "no-hallucination" web-based Q&A inside a graph-based agent

**Notes:** This notebook is part of the Intergrax framework and appears to be a working example, not experimental or auxiliary.

### `notebooks\openai\rag_openai_presentation.ipynb`

**Description:** This notebook demonstrates the usage of the Integrax framework's RAG (ReasoR-Augmented Generation) OpenAI implementation, specifically for filling a VectorStore with data from a local folder and querying it.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Fills a VectorStore with data from a local folder
* Uses the OpenAI API to interact with the VectorStore
* Demonstrates query capabilities on the filled VectorStore

### `notebooks\rag\chat_agent_presentation.ipynb`

Description: This notebook demonstrates the configuration and usage of various components within the Integrax framework, including LLM adapters, conversational memory, vector stores, re-ranking logic, RAG answerers, tools, and a high-level hybrid chat agent.

Domain: LLM adapters, RAG logic, data ingestion, agents, configuration

Key Responsibilities:
- Initializes LLM adapter for Ollama model
- Configures conversational memory
- Demonstrates usage of weather tool
- Sets up vector store and re-ranking configuration
- Creates RAG answerer with specified configuration
- Defines high-level hybrid chat agent combining RAG, tools, and LLM capabilities
- Provides test questions to evaluate the agent's performance

Note: This notebook appears to be a comprehensive example of integrating various components within the Integrax framework. It covers multiple domains and responsibilities in detail.

### `notebooks\rag\output_structure_presentation.ipynb`

**Description:** This Jupyter Notebook demonstrates an example usage of the Integrax framework, showcasing its features in generating structured responses to user queries. It integrates with various modules such as LLM adapters, conversational memory, and tool registries.

**Domain:** RAG (Recurrent Attentional Graph) logic

**Key Responsibilities:**

* Defines a schema for structured output using Pydantic (`WeatherAnswer` and `ExecSummary`)
* Implements a simple weather tool (`WeatherTool`) that returns demo data
* Sets up an IntegraxToolsAgent, which orchestrates LLM reasoning, tool selection, and invocation
* Demonstrates usage of the agent with a natural-language question and requests structured output

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This notebook demonstrates the usage of various components within the Integrax framework for RAG (Retrieval-Augmented Generation) pipelines, specifically focusing on document loading, splitting, embedding, and vector storage management.

**Domain:** RAG logic

**Key Responsibilities:**

* Demonstrates how to load documents from a directory using `IntergraxDocumentsLoader` with various configuration options.
* Shows the process of splitting loaded documents into smaller chunks using `IntergraxDocumentsSplitter`.
* Illustrates the creation and usage of embeddings for document chunks through `IntergraxEmbeddingManager`.
* Provides an example of interacting with a vector store using `IntergraxVectorstoreManager`.

**Note:** This notebook appears to be part of the official Integrax documentation or demonstration material, showcasing various features and functionalities of the framework.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter notebook appears to be a test and demonstration environment for the Integrax framework's multimodal retrieval capabilities. It loads documents from various sources (video, audio, image), splits and embeds them, and then uses this data to perform vector store operations and retrievals.

**Domain:** RAG (Retriever-Reader-Generator) logic

**Key Responsibilities:**

* Loads documents from video, audio, and image files using the `IntergraxDocumentsLoader`
* Splits loaded documents into chunks
* Embeds documents using the `IntergraxEmbeddingManager` with an Ollama adapter
* Performs vector store operations (insertion and querying) using the `IntergraxVectorstoreManager`
* Demonstrates retrieval capabilities using the `IntergraxRagRetriever`

**Notes:**

* This file is not experimental, auxiliary, or incomplete. It appears to be a fully functional test environment for the Integrax framework.
* Some code blocks are commented out (e.g., the "RETRIEVER" section), suggesting that this notebook might have been used as a proof-of-concept or demonstration tool at some point.

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter Notebook demonstrates how to utilize the Integrax framework for multimedia processing, including downloading and transcribing video and audio files from YouTube, as well as extracting frames and metadata. It also showcases the use of the Ollama model for describing images.

**Domain:** Multimedia Processing (RAG)

**Key Responsibilities:**

* Downloading video and audio files from YouTube using `yt_download_video` and `yt_download_audio`
* Transcribing video to VTT format using `transcribe_to_vtt`
* Extracting frames and metadata from videos using `extract_frames_and_metadata`
* Translating audio files using `translate_audio`
* Using the Ollama model to describe images in `transcribe_image`

Note: This notebook appears to be a demonstration of the Integrax framework's capabilities, rather than an experimental or auxiliary file.

### `notebooks\rag\tool_agent_presentation.ipynb`

**Description:** This Jupyter notebook defines and configures an Intergrax tools agent, which integrates a Language Model (LLM) with a set of task-specific tools to perform various actions. The agent is demonstrated with two example queries: one for the current weather and another for arithmetic calculation.

**Domain:** RAG logic / Agents

**Key Responsibilities:**

*   Defines a conversational memory for context tracking
*   Registers and configures tools (e.g., WeatherTool, CalcTool)
*   Sets up an LLM adapter (in this case, Ollama-backed) for reasoning and tool selection
*   Initializes a tools agent that orchestrates LLM reasoning, tool invocation, and conversational memory updates
*   Demonstrates the agent's functionality with two example queries

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This notebook contains implementations of various components for the Intergrax framework, including compliance checker, cost estimator, final summary generator, and financial audit. These components are designed to be used within a pipeline and provide functionalities such as policy compliance verification, UX-driven change cost estimation, consolidated summary generation, and mock financial report creation.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Compliance checker:
	+ Verifies whether proposed changes comply with privacy policies and terms of service
	+ Returns a result indicating non-compliance or compliance
* Cost estimator:
	+ Estimates the cost of UX-driven changes based on the UX audit report
	+ Provides a mock formula for calculation
* Final summary generator:
	+ Consolidates all collected artifacts into a final summary report
	+ Includes status pipeline, terminated by, terminate reason, and other relevant information
* Financial audit:
	+ Generates a mock financial report with VAT calculation (test data)
	+ Supports both Polish and English keys for interoperability

**Notes:** All components are well-documented with clear descriptions, examples of use cases, and parameters. They are also annotated with `@component` decorator from the Intergrax framework, which suggests that these components are designed to be reusable and configurable within a pipeline. Overall, this notebook appears to be a part of a larger system for automated workflow management and decision-making in a business context.

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This notebook demonstrates the usage of the Intergrax framework's WebSearchExecutor, specifically with Google Custom Search and Bing Search providers. It showcases how to execute web search queries, retrieve results, and format them in a human-readable manner.

**Domain:** LLM adapters (LangChain/LangGraph)

**Key Responsibilities:**

* Load environment variables for Google CSE API key and CX
* Create a QuerySpec object with specific parameters (query, top_k, locale, region, language, safe_search)
* Initialize a GoogleCSEProvider instance
* Execute the search query using the provider's `search` method
* Print the number of results found
* Iterate over the search hits and print their details (provider, rank, title, URL, snippet, domain, published date)
