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
- `notebooks\openai\rag_openai_presentation.ipynb`
- `notebooks\rag\rag_custom_presentation.ipynb`
- `notebooks\rag\rag_multimodal_presentation.ipynb`
- `notebooks\rag\rag_video_audio_presentation.ipynb`
- `notebooks\supervisor\supervisor_test.ipynb`
- `notebooks\websearch\websearch_presentation.ipynb`

## Detailed File Documentation

### `api\__init__.py`

DESCRIPTION: 
This module serves as the entry point for the API, responsible for initializing and configuring the application's routing and services.

DOMAIN: Configuration

KEY RESPONSIBILITIES:
- Initializes API routes
- Configures service bindings
- Sets up error handling mechanisms

### `api\chat\__init__.py`

**Description:** 
This is the entry point for the chat API, responsible for setting up and configuring the application.

**Domain:** API Infrastructure

**Key Responsibilities:**
* Initializes the chat API module
* Sets up necessary dependencies and configurations
* Defines the API's main functionality and endpoints

### `api\chat\main.py`

**Description:** This is the main entry point for the chat API in the Integrax framework. It handles incoming requests and orchestrates interactions with various components, including answerers, databases, and document indexing pipelines.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Handles POST /chat requests, processing user queries and returning AI-generated responses
* Manages session IDs and logs application interactions in the database
* Integrates with Chroma document indexing pipeline for uploading and deleting documents
* Provides endpoints for listing all documents and deleting individual documents by ID
* Utilizes various utility modules (e.g., db_utils, chroma_utils) to perform database operations and interact with external services.

### `api\chat\tools\__init__.py`

Description: This is the initialization file for the chat API tools, responsible for setting up and configuring other modules.

Domain: LLM adapters

Key Responsibilities:
* Initializes tool sub-modules
* Registers available tools with the main application
* Provides entry points for tool-related functionality 

Note: The content appears to be minimal, suggesting this file may primarily serve as a container or interface for other modules. However, without more context, it's difficult to assess its overall significance or completeness.

### `api\chat\tools\chroma_utils.py`

**Description:** This module provides utility functions for managing documents in the Chroma vector store, including loading and splitting documents, indexing them to Chroma, and deleting indexed documents.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* `load_and_split_documents(file_path: str) -> List[Document]`: Loads a document from a file path and splits it into individual documents.
* `index_document_to_chroma(file_path: str, file_id: int) -> bool`: Indexes a document to Chroma with the given file ID.
* `delete_doc_from_chroma(file_id: int) -> bool`: Deletes an indexed document from Chroma by its file ID.

Note: The module appears to be functional and used in the context of the Integrax framework.

### `api\chat\tools\db_utils.py`

**Description:** This module provides database utilities for the Integrax framework, including schema creation, data insertion, and retrieval. It uses SQLite as the underlying database management system.

**Domain:** Database Utilities

**Key Responsibilities:**

* Provides functions for creating and managing the database schema
* Offers tools for inserting and retrieving data from the database tables (sessions, messages, documents)
* Supports backward-compatibility entry points for legacy applications
* Includes public API endpoints for interacting with the database

The file appears to be stable and well-maintained.

### `api\chat\tools\pydantic_models.py`

**Description:** This module defines Pydantic models for handling chat API queries, responses, and document information within the Integrax framework.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Defines an enum `ModelName` for specifying supported large language model names.
* Implements a `QueryInput` model for structuring user queries with session ID and optional model override.
* Defines a `QueryResponse` model to standardize the format of answers provided by LLMs.
* Specifies a `DocumentInfo` model for encapsulating metadata about uploaded documents, including their IDs, filenames, and timestamps.
* Provides a `DeleteFileRequest` model for handling file deletion requests through the chat API.

### `api\chat\tools\rag_pipeline.py`

**Description:** This module provides utilities for building and managing a Retrieval-Augmented Generator (RAG) pipeline, specifically designed for the Integrax framework. It handles vector store management, embedding manager, retriever, reranker, and LLM adapter creation.

**Domain:** RAG logic

**Key Responsibilities:**

* Manages vector stores and embeddings through singleton instances
* Creates LLM adapters based on specified models
* Provides default user prompts for the RAG pipeline
* Offers a template system prompt with instructions for the asystent's behavior

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides API utilities for interacting with the Integrax framework, including functionality for sending requests to a chat endpoint and managing document uploads, listings, and deletions.

**Domain:** RAG logic / API Utilities

**Key Responsibilities:**

* Send POST requests to a chat endpoint with user questions and receive responses
* Upload files to an API endpoint
* List available documents on the server-side
* Delete documents by ID
* Handle errors and exceptions using Streamlit's `st.error` function for error reporting

### `applications\chat_streamlit\chat_interface.py`

**Description:** This module provides a user interface for a chat application using Streamlit, integrating with an external API for generating responses.

**Domain:** Chat Interface

**Key Responsibilities:**
- Displays a chat interface with input field and generated response
- Handles user input and sends it to the API for processing
- Receives and displays API-generated responses in the chat window
- Provides additional details about the generated answer, model used, and session ID

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides a Streamlit sidebar for managing uploaded documents and interacting with the Integrax framework's API.

**Domain:** Configuration/Utility Modules

**Key Responsibilities:**
- Provides model selection component in the sidebar
- Enables uploading documents to the server using the `upload_document` function from `api_utils`
- Lists uploaded documents and allows refreshing the list
- Allows deleting selected documents using the `delete_document` function from `api_utils`
- Displays document metadata (filename, ID, upload timestamp) for each listed document

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This module provides a user interface for the Intergrax RAG Chatbot using Streamlit, including a sidebar and chat interface.

**Domain:** Configuration/Chat Interface

**Key Responsibilities:**
- Initializes the Streamlit application with a title.
- Checks and initializes session state variables (messages and session_id).
- Displays the sidebar and chat interface.

### `applications\company_profile\__init__.py`

Description: The company profile application initialization module sets up the framework and establishes connections for data retrieval and storage.

Domain: Application Framework

Key Responsibilities:
- Initializes the application environment
- Configures database connections
- Sets up data retrieval pipelines
- Establishes API endpoints for external interactions

### `applications\figma_integration\__init__.py`

DESCRIPTION: The Figma integration module initializes and sets up the integration with Figma, enabling Intergrax to interact with the design platform.

DOMAIN: Integration adapters

KEY RESPONSIBILITIES:
• Initializes the Figma API connection
• Sets up event listeners for design updates
• Exposes functions for syncing designs between Figma and Intergrax

### `applications\ux_audit_agent\__init__.py`

Description: This module serves as the entry point for the UX audit agent application, responsible for bootstrapping and initializing necessary components.

Domain: Agents

Key Responsibilities:
• Initializes the UX audit agent application
• Sets up required dependencies and services
• Defines entry points for the application

### `applications\ux_audit_agent\components\__init__.py`

Description: The `__init__.py` file initializes and configures the UX audit agent components, setting up the necessary modules and resources for the application.

Domain: Agents

Key Responsibilities:
* Initializes component registry
* Configures agent settings and dependencies
* Sets up event listeners and hooks for auditing events

### `applications\ux_audit_agent\components\compliance_checker.py`

Description: This module implements a compliance checker component for the Integrax framework, which verifies if changes are in accordance with privacy policies and regulations.

Domain: UX Audit Agent - Compliance Checker

Key Responsibilities:
- Checks if a list of changes is compliant with privacy policy and regulations (mock implementation).
- Returns findings on compliance status, including potential policy violations.
- Generates meta-data for stopping the pipeline if non-compliance is detected.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a cost estimator component for the Integrax framework, which calculates an estimated cost of changes based on a UX report. The component is designed to be used in audit processes.

**Domain:** RAG logic

**Key Responsibilities:**

* Calculates an estimated cost of changes based on a UX report
* Returns a ComponentResult object with the estimated cost and meta information
* Uses a simple formula to calculate the cost, based on a base value and additional cost per issue

### `applications\ux_audit_agent\components\final_summary.py`

Description: This module defines a component that generates a final summary of the process from all artifacts.

Domain: UX Audit Agent components

Key Responsibilities:
- Generates a complete process summary with all artifacts.
- Returns a ComponentResult object containing the final report and logs.

### `applications\ux_audit_agent\components\financial_audit.py`

Description: This module defines a financial audit component for the Intergrax framework, generating a sample financial report and VAT calculations.

Domain: Financial Audit Agent

Key Responsibilities:
- Provides a "Financial Audit" component with example data
- Calculates example financial reports (e.g., gross income, VAT)
- Returns a ComponentResult object containing the generated report

### `applications\ux_audit_agent\components\general_knowledge.py`

**Description:** This module provides a general knowledge component for the Intergrax framework, answering questions about its system structure and modules.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a component for answering general questions about the Intergrax system
* Includes example use cases and descriptions for the component
* Returns a response with relevant information and citations (fake documents)
* Logs a message indicating that a sample response was returned

### `applications\ux_audit_agent\components\project_manager.py`

Description: This module defines a project manager component that verifies UX reports and randomly accepts or rejects them based on a predefined probability.

Domain: UX Audit Agent Components

Key Responsibilities:
- Verify UX reports
- Randomly accept or reject changes with 70% chance of acceptance
- Generate decision and notes based on the outcome
- Produce meta-data for logging and decision-making purposes
- Optionally stop the pipeline if rejected by the project manager

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module provides a UX audit component for the Integrax framework, responsible for analyzing Figma designs and returning an example report with recommendations.

Domain: LLM adapters

Key Responsibilities:
- Performs UX analysis on Figma designs
- Returns an example report with recommendations and estimated cost
- Utilizes the `component` decorator from `intergrax.supervisor.supervisor_components`

### `applications\ux_audit_agent\UXAuditTest.ipynb`

Description: This notebook appears to be a test script for the UX audit agent, executing various tasks such as user role simulation, change list verification, report preparation, and cost estimation.

Domain: UX Audit Agent

Key Responsibilities:
- Execute user role simulation (UX Audytora)
- Verify change lists using RAG logic
- Prepare reports using specialized tools
- Estimate costs of changes using domain operation
- Synthesize results from previous steps

### `generate_project_overview.py`

**Description:** This module provides an automated documentation generator for the Intergrax framework project structure. It scans the project directory, collects relevant files, generates a summary for each file using an LLM adapter, and creates a Markdown document containing a navigable overview of the project layout.

**Domain:** Project Structure Documentation

**Key Responsibilities:**

* Recursively scan the project directory to collect all relevant source files (Python, Jupyter Notebooks, configurable)
* For each file:
	+ Read the source code
	+ Send content + metadata to an LLM adapter
	+ Generate a structured summary: purpose, domain, responsibilities
* Create a Markdown document (`PROJECT_STRUCTURE.md`) containing a clear, navigable, human-readable, and LLM-friendly description of the entire project layout

Note: This module appears to be well-structured, complete, and production-ready. There are no obvious signs of experimental or legacy code.

### `intergrax\__init__.py`

DESCRIPTION: The `__init__.py` file serves as the entry point and main container for the Intergrax framework, responsible for initializing its core components.

DOMAIN: Framework Initialization

KEY RESPONSIBILITIES:

* Initializes global configuration settings
* Sets up framework-specific dependencies and imports
* Registers core modules and their dependencies
* Exposes a main entry point for framework execution

### `intergrax\chains\__init__.py`

Description: This module serves as the entry point and container for Intergrax's chain-based architecture.

Domain: Chain Architecture

Key Responsibilities:
- Initializes chain-related modules and imports.
- Defines interfaces and abstract classes for chains.
- Provides utility functions for building and managing chains.

### `intergrax\chains\langchain_qa_chain.py`

**Description:** This module implements a flexible QA chain (RAG → [rerank] → prompt → LLM) based on the LangChain framework, allowing for customization through hooks and configuration.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Builds a QA chain with customizable hooks and configuration
* Supports various input formats (e.g., question-only, question with context)
* Retrieves relevant documents using the IntergraxRagRetriever
* Optionally reranks retrieved documents using the IntergraxReRanker
* Constructs prompts for LLMs based on retrieved documents and user-provided configurations
* Interfaces with LangChain's LLM adapters (e.g., ChatOllama, ChatOpenAI)
* Parses output from LLMs using StrOutputParser

Note: The file appears to be a part of the Intergrax framework, which is proprietary and confidential.

### `intergrax\chat_agent.py`

**Description:** This module provides the core functionality of the Intergrax chat agent, enabling routing decisions between RAG (Reactive Aggregation of Graphs), tools, and general LLM-based conversations.

**Domain:** Chat Agent

**Key Responsibilities:**

* Routing decisions via LLM (with descriptions and tools_enabled flag)
* Execution of RAG, tool-based, or general LLM-based conversations
* Handling memory, streaming, structured output, and result statistics
* Return a stable result with answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component

### `intergrax\llm\__init__.py`

DESCRIPTION: This is the entry point of the LLM adapters module, responsible for importing and initializing various language model interfaces.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
* Imports and initializes various LLM adapter classes
* Provides a centralized registry for accessing different LLM models
* Sets up default configuration and initialization hooks for adapter instances

### `intergrax\llm\conversational_memory.py`

**Description:** This module provides a universal conversation memory system that supports chat message storage, filtering, and loading/saving to various formats (JSON, NDJSON, SQLite). It is designed to work independently of LLM/adapters.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Stores chat messages in RAM with optional message expiration
* Provides methods for adding, extending, getting all, and clearing messages
* Supports filtering messages based on model type (native tools or planners)
* Saves entire history to a single JSON file or an NDJSON file
* Loads history from a JSON or NDJSON file
* Stores messages in SQLite database with schema management

Note: The code appears to be well-structured, and the functionality is clearly defined. There are no signs of incompleteness, legacy, or experimental status.

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This module defines a unified interface for LLM adapters and provides implementation details for specific adapters like OpenAI Chat Completions.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines the `LLMAdapter` protocol with methods for generating messages, streaming messages, and supporting tools
* Implements the `OpenAIChatCompletionsAdapter` class, which uses the OpenAI API to generate completions
* Provides utility functions like `_map_messages_to_openai`, `_strip_code_fences`, and `_extract_json_object`
* Supports structured output with Pydantic v2, v1, and a fallback **dict constructor** using the `_validate_with_model` function.

### `intergrax\llm_adapters\__init__.py`

**intergrax/llm_adapters/__init__.py**

Description: This module serves as the entry point for LLM adapters, registering and providing access to various language model adapter implementations.

Domain: LLM adapters

Key Responsibilities:
- Registers and provides access to multiple LLM adapters (OpenAI, Gemini, LangChain Ollama) through a registry.
- Exposes base classes and utilities for adapter implementation and validation.

### `intergrax\llm_adapters\base.py`

**Description:** This module provides base functionality for LLM (Large Language Model) adapters, including a universal interface and helper functions for structured output.

**Domain:** LLM adapters

**Key Responsibilities:**

* Defines the `LLMAdapter` protocol with required methods for generating messages (`generate_messages`, `stream_messages`) and optional tools (`supports_tools`, `generate_with_tools`, `stream_with_tools`)
* Provides helper functions for extracting JSON objects from text, validating model instances from JSON strings, and converting ChatMessage objects to OpenAI-compatible message dictionaries
* Implements an adapter registry (`LLMAdapterRegistry`) for registering and creating LLM adapters

### `intergrax\llm_adapters\gemini_adapter.py`

**Description:** This module implements a minimal adapter for the Gemini large language model (LLM) to integrate with the Intergrax framework.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a basic interface for interacting with the Gemini LLM
* Offers methods for generating and streaming chat messages
* Currently does not support tools or manual wiring, focusing on simple chat usage

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides an adapter for using Ollama models via the LangChain's ChatModel interface.

**Domain:** LLM adapters

**Key Responsibilities:**

- Provides a class `LangChainOllamaAdapter` to interact with Ollama models
- Supports planner-style pattern for tool calls (no native support)
- Converts internal `ChatMessage` list into LangChain message objects
- Applies temperature and max tokens options for generation parameters
- Invokes chat model using the adapter's defaults or user-provided kwargs
- Returns generated content as a string or streams it in chunks
- Validates structured output against a JSON schema

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an adapter for interacting with the OpenAI Responses API, allowing the Intergrax framework to utilize its capabilities. The adapter offers various methods for generating and streaming chat responses, as well as supporting tools integration.

**Domain:** LLM adapters

**Key Responsibilities:**

* Providing a bridge between the Intergrax framework and the OpenAI Responses API
* Supporting single-shot and streaming completion requests using the Responses API
* Integrating tool support for higher-level orchestration
* Generating structured JSON output validated against user-defined schemas

### `intergrax\logging.py`

Description: This module provides basic logging configuration for the Integrax framework, setting up global logging settings and output format.

Domain: Logging Configuration

Key Responsibilities:
- Sets up global logging level to INFO with DEBUG showing more details
- Specifies logging format including timestamp, log level, and message
- Forces new configurations over any existing ones

### `intergrax\multimedia\__init__.py`

Description: This module serves as the entry point for Intergrax's multimedia functionality, providing a framework for importing and utilizing various multimedia-related components.

Domain: Multimedia Framework

Key Responsibilities:
• Importing necessary modules for multimedia processing
• Setting up environment and dependencies for multimedia operations
• Exposing API for accessing multimedia functionalities

### `intergrax\multimedia\audio_loader.py`

Description: This module provides functionality for downloading audio from YouTube URLs and translating the downloaded audio using a Whisper model.

Domain: Multimedia Processing

Key Responsibilities:
- Downloads audio files from specified YouTube URLs to a designated output directory.
- Extracts audio from videos using FFmpeg.
- Translates downloaded audio into another language using the Whisper model.

### `intergrax\multimedia\images_loader.py`

Description: This module enables the integration of image loading capabilities within the Intergrax framework, allowing for image-based text generation using LLaMA models.

Domain: Multimedia, LLM adapters

Key Responsibilities:
- Loads and transcribes images using ollama's chat functionality
- Utilizes LLaMA model "llava-llama3:latest" for image-text generation
- Expects an input prompt, image path, and optional model name as arguments

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides utilities for displaying multimedia content, such as audio, images, and videos, within Jupyter notebooks.

**Domain:** Multimedia display

**Key Responsibilities:**

* Displaying audio files with optional autoplay and label support
* Displaying images from local file paths or URLs
* Serving video files to allow for playback within the notebook, including jump-to-time functionality
* Utilities for serving files to a temporary directory

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides functionalities for video loading, transcription to VTT format, and extraction of frames along with their metadata from a given video. It utilizes various libraries such as `yt_dlp` for YouTube video download, `whisper` for speech-to-text, and `cv2` for image processing.

**Domain:** Multimedia Processing

**Key Responsibilities:**

*   Downloading videos from YouTube using `yt_dlp`
*   Transcribing audio content to VTT format with customizable model size and language
*   Extracting frames from a video with optional metadata (transcript, timestamp) at specified intervals or based on the transcript's segments
*   Resizing extracted images while maintaining aspect ratio for efficient storage

**Notes:** The code appears well-structured and utilizes various best practices. However, there are some minor points that could be improved upon:

*   Some magic numbers (e.g., 350) could be defined as constants to improve readability.
*   Error handling in the `extract_frames_from_video` function could be more comprehensive.

Overall, this module is a useful addition to any multimedia processing pipeline.

### `intergrax\openai\__init__.py`

Description: This is the entry point for Intergrax's OpenAI integration, responsible for loading and initializing various components.

Domain: LLM adapters

Key Responsibilities:
- Initializes the OpenAI API client
- Sets up default configuration options
- Loads available models and model-specific components

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: The `__init__.py` file is the entry point for Intergrax's RAG (Retrieval Augmented Generation) logic integration with OpenAI adapters.

DOMAIN: LLM Adapters

KEY RESPONSIBILITIES:
- Initializes the OpenAI adapter
- Sets up default configuration for RAG operations

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides a class for interacting with OpenAI's vector store and implementing the Retrieval-Augmented Generator (RAG) logic within the Integrax framework.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Initializes an IntergraxRAGOpenAI object with an OpenAI client and a vector store ID
* Retrieves the vector store by its ID
* Clears all files loaded into the vector store
* Uploads a folder to the vector store, including optional file patterns
* Provides a prompt for generating RAG outputs, enforcing specific rules and guidelines

### `intergrax\rag\__init__.py`

DESCRIPTION: This is the main entry point for Intergrax's RAG (Reactor-Augmented Graph) logic, responsible for initializing and managing RAG instances.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
- Initializes and configures RAG instances
- Provides interface for creating and managing Reactor-Augmented Graphs
- Manages dependencies and imports for the RAG module

### `intergrax\rag\documents_loader.py`

**Description:** This module, `DocumentsLoader`, provides a robust and extensible way to load documents from various file formats while injecting metadata and implementing safety guards.

**Domain:** RAG logic (document loading and processing)

**Key Responsibilities:**

* Loading documents from various file formats (e.g., .docx, .pdf, .xlsx, .csv)
* Injecting metadata into loaded documents
* Implementing safety guards for document loading and processing
* Supporting OCR (Optical Character Recognition) for image and PDF files
* Supporting captioning for images using a framework adapter
* Providing customizable options for document loading and processing (e.g., file patterns, extensions map, exclude globs)
* Integrating with other modules in the Intergrax framework (e.g., LLM adapters, multimedia loaders)

Note: The code appears to be well-structured and comprehensive, indicating that it is a stable and integral part of the Intergrax framework.

### `intergrax\rag\documents_splitter.py`

**Description:** This module implements a high-quality text splitter for RAG pipelines, providing stable chunk IDs and rich metadata. It uses a "semantic atom" policy to decide when to split documents.

**Domain:** RAG (Relational-Augmented-Generative) Logic

**Key Responsibilities:**

*   Splits documents into chunks based on user-configurable rules
*   Assigns stable, human-readable chunk IDs using available anchors (para_ix/row_ix/page_index)
*   Provides rich metadata for each chunk, including parent ID, source name/path, and page index
*   Merges tiny tails and applies optional hard caps on the number of chunks per document
*   Optionally calls a custom metadata function to enrich chunk metadata

Note: The module appears to be well-maintained and production-ready. There are no obvious signs of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\rag\dual_index_builder.py`

**Description:** This module builds two vector indexes: primary (CHUNKS) and auxiliary (TOC), using embeddings for efficient search.

**Domain:** RAG logic (Retrieval-Augmented Generation)

**Key Responsibilities:**
- Builds primary index (CHUNKS) from all chunks/documents after splitting
- Builds auxiliary index (TOC) only from DOCX headings within specified levels
- Uses embeddings to compute similarity between documents
- Supports batching for efficient insertion into vector stores
- Provides logging and verbosity control

### `intergrax\rag\dual_retriever.py`

**Description:** This module provides a dual retriever functionality for retrieving relevant chunks of text from the Intergrax vector store.

**Domain:** RAG logic (Relevance Aggregation)

**Key Responsibilities:**

* Provides a dual retriever class (`IntergraxDualRetriever`) that combines two retrieval steps:
	+ First, it queries the TOC (table of contents) to identify relevant sections.
	+ Then, it searches locally within those identified sections to retrieve specific chunks.
* Offers several helper methods for normalizing query results and merging where conditions with parent IDs.
* The `retrieve` method aggregates results from both retrieval steps and returns a sorted list of relevant chunks.

**Note:** This file appears to be part of the Intergrax framework, specifically designed for Retrieval-Augmented Generation (RAG) logic.

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for various text embeddings from HuggingFace, Ollama, and OpenAI. It allows switching between providers, handling model loading, and offers features like batch/single text embedding, L2 normalization, and cosine similarity utilities.

**Domain:** RAG (Retrieval Augmented Generation) logic

**Key Responsibilities:**

* Unified embedding manager for HuggingFace, Ollama, and OpenAI
* Provider switch between "hg", "ollama", and "openai"
* Reasonable defaults if model_name is None
* Batch/single text embedding; optional L2 normalization
* Embedding for LangChain Documents (returns np.ndarray + aligned docs)
* Cosine similarity utilities and top-K retrieval
* Robust logging, shape validation, light retry for transient errors

Note: The code appears to be well-structured, and the documentation is clear. However, it's worth noting that this module seems to be part of a larger framework (Intergrax), which might make it harder to use standalone without the surrounding context.

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides an answerer component for the Integrax framework, responsible for retrieving and ranking context fragments, generating answers using a language model, and constructing relevant citations.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Retrieval of context fragments from a retriever instance
* Optional re-ranking of retrieved hits using a reranker instance
* Building context text by selecting relevant hits and concatenating their content
* Generating citations for the used hits
* Constructing system and user messages to be passed to the language model
* Invoking the language model to generate an answer or stream responses
* Handling output structure generation (optional)

### `intergrax\rag\rag_retriever.py`

**Description:** The `rag_retriever.py` file provides a scalable, provider-agnostic retriever for the Intergrax framework, supporting various use cases such as text search and retrieval.

**Domain:** RAG (Re Retrieve-Aggregate-Generate) logic

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

Note: This module appears to be a key component of the Intergrax framework's RAG logic, providing essential functionality for text search and retrieval tasks.

### `intergrax\rag\re_ranker.py`

**Description:** The intergrax ReRanker is a fast and scalable cosine re-ranker that embeds texts in batches using the `intergraxEmbeddingManager` and computes similarities between query and document vectors.

**Domain:** RAG logic (Reinforced Attention-based Generators)

**Key Responsibilities:**

* Embeds texts in batches using `intergraxEmbeddingManager`
* Computes cosine similarities between query and document vectors
* Supports score fusion with original retriever similarity
* Preserves schema of hits; adds 'rerank_score', optional 'fusion_score', and 'rank_reranked' fields
* Can be used as a standalone component or integrated with other components in the Intergrax framework

Note: The code appears to be well-structured, well-documented, and production-ready. No indication of experimental, auxiliary, legacy, or incomplete status.

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module manages vector stores for the Integrax framework, supporting ChromaDB, Qdrant, and Pinecone. It provides unified functionality for initializing stores, upserting documents with embeddings, querying similarities, counting vectors, and deleting by IDs.

**Domain:** RAG logic (vector store management)

**Key Responsibilities:**

* Initialize target vector store and create collection/index lazily
* Upsert documents + embeddings with batching
* Query top-K similar vectors by cosine/dot/euclidean similarity
* Count vectors
* Delete vectors by IDs

Note: The code appears to be well-structured, comprehensive, and production-ready.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a Windowed Answerer, which is a layer on top of the base Answerer in the Integrax framework. It provides an efficient way to retrieve and process large amounts of context by dividing it into smaller windows.

**Domain:** RAG logic

**Key Responsibilities:**

* Divides context into smaller windows for more efficient processing
* Builds messages with memory-awareness, including injecting context as a separate system message if the answerer has a memory store
* Retrieves top-k candidates using the retriever layer
* Synthesizes final answers from partials using the LLM adapter
* Deduplicates sources to provide relevant information without redundancy

This implementation appears to be part of the main functionality and is not marked as experimental, auxiliary, or legacy.

### `intergrax\supervisor\__init__.py`

DESCRIPTION: The __init__.py file serves as the entry point and initializes the supervisor component within the Intergrax framework.

DOMAIN: Supervisor logic

KEY RESPONSIBILITIES:
- Defines the main API for interacting with the supervisor.
- Initializes the necessary dependencies and modules.
- Sets up event handling and communication protocols.

### `intergrax\supervisor\supervisor.py`

**Description:** This module contains the core functionality of the Intergrax framework's supervisor component, responsible for planning and executing tasks.

**Domain:** Task Management / Planning

**Key Responsibilities:**

*   Planning and execution of tasks through a two-stage process:
    *   Decomposition (stage 1): Break down queries into individual steps
    *   Assignment (stage 2): Assign components to each step based on the plan
*   Handling user input and generating plans using LLMs (Large Language Models)
*   Managing component registrations, listing available components, and checking their availability
*   Providing utilities for extracting text from various adapter return types and safely parsing JSON strings

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides a framework for implementing and managing components in the Integrax system, enabling developers to define and execute reusable workflow steps.

**Domain:** Supervisor Components

**Key Responsibilities:**

* Defining component structure and behavior through the `Component` class
* Providing context and pipeline state to components through the `ComponentContext` and `PipelineState` dataclasses
* Offering a decorator (`component`) for registering new components with minimal boilerplate code
* Executing components and handling their output, errors, and logging through the `run` method

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module defines default prompt templates for the Intergrax framework's Supervisor, providing a structured format for planning and execution.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Defines the structure of a plan for hybrid RAG + Tools + General Reasoning engine
* Specifies rules for decomposition-first approach, including task decomposition, component selection policy, and output→method guards
* Validates plans against strict rules (resource flags, DAG, gates, synthesis, reliability, and decomposition coverage)
* Provides default prompt templates for the Supervisor (plan_system and plan_user_template) as a dataclass (`SupervisorPromptPack`)

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module is responsible for creating and managing the state graph of a LangGraph pipeline from a given plan. It includes utilities for parsing node inputs and outputs, resolving dependencies, and building the final graph structure.

**Domain:** RAG logic ( Reasoning and Action Graphs)

**Key Responsibilities:**

*   Creating nodes from plan steps with associated components
*   Resolving node inputs using artifact and state values
*   Handling node execution and output persistence
*   Building a stable topological order of nodes based on dependencies
*   Constructing the final LangGraph pipeline from the plan

This module appears to be part of a larger framework for executing complex tasks through a graph-based pipeline. It seems well-structured, with clear separation of concerns and reasonable naming conventions. There are no obvious signs of incompleteness or experimental code.

### `intergrax\system_prompts.py`

Description: This module defines the default RAG system instruction, which outlines the rules and procedures for a strict RAG (Rola i Zasady Pracy) setup.

Domain: RAG logic

Key Responsibilities:
- Defines the structure and format of responses
- Outlines the importance of accuracy, precision, and completeness in answers
- Specifies how to handle conflicting information from different sources
- Emphasizes the need for clear references to source documents
- Provides guidelines for citing and referencing documents
- Addresses issues such as uncertainty and lack of data

### `intergrax\tools\__init__.py`

DESCRIPTION:
This module serves as the entry point for the Intergrax tools package, providing an interface to access various utility functions.

DOMAIN: Tools Package Initialization

KEY RESPONSIBILITIES:
• Initializes the Intergrax tools package.
• Exposes the tool sub-modules.

### `intergrax\tools\tools_agent.py`

**Description:** This module defines the `IntergraxToolsAgent` class, which is a tool agent that interacts with tools to generate answers. It also contains various utility functions for processing user input, handling tool traces, and building output structures.

**Domain:** LLM adapters / Agents

**Key Responsibilities:**

* Initializes an instance of the `IntergraxToolsAgent` class with an LLM adapter, tools registry, memory, configuration, and verbosity level
* Provides methods to run the tool agent with user input, context, and optional arguments for tool choice and output model
* Processes user input and generates answers using tools in a conversational manner
* Manages tool traces and builds output structures based on the results of each tool call

### `intergrax\tools\tools_base.py`

**Description:** This module provides basic tool utilities and classes for managing tools within the Integrax framework.

**Domain:** Tools, Utility Modules

**Key Responsibilities:**

* Defines a `ToolBase` class that serves as a base for all tools, providing common attributes (name, description) and methods (get_parameters, run, validate_args, to_openai_schema).
* Implements a `ToolRegistry` class that manages registered tools, allowing registration, retrieval, and listing of tools.
* Provides a helper function `_limit_tool_output` for safely truncating long tool outputs.

### `intergrax\websearch\__init__.py`

Description: The `websearch` module is responsible for initializing the search functionality within Intergrax.

Domain: Search module setup

Key Responsibilities:
* Initializes the search API
* Sets up default search parameters and configurations
* Registers search-related services and dependencies

### `intergrax\websearch\context\__init__.py`

DESCRIPTION: This module sets up the context for web search functionality within Intergrax.

DOMAIN: Web Search Context

KEY RESPONSIBILITIES:
- Initializes the web search context
- Configures default settings and behaviors
- Exposes necessary interfaces for downstream components

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides utilities for building LLM-ready textual context and chat messages from web search results, including support for raw WebDocument objects and serialized dictionaries.

**Domain:** Web Search

**Key Responsibilities:**

* Builds a textual context string from WebDocument objects or serialized dictionaries
* Supports customization of maximum document count, character limit per document, inclusion of snippet and URL, and source label prefix
* Provides strict system prompts enforcing web sources usage only, no hallucinations, single answer, and explicit handling of missing information
* Builds user-facing prompts wrapping web sources, questions, and tasks for chat-style LLMs

**Status:** None - appears to be a fully functional module.

### `intergrax\websearch\fetcher\__init__.py`

DESCRIPTION: The `__init__.py` file serves as the entry point for the web search fetcher module, responsible for initializing and configuring its components.

DOMAIN: Web Search Fetcher

Key Responsibilities:
- Initializes the web search fetcher instance
- Configures connection settings for web search APIs
- Registers available web search adapters
- Provides access to fetcher-related utilities

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module provides an extractor function for lightweight HTML extraction, retrieving essential metadata and plain-text content from a web page.

**Domain:** Web Search

**Key Responsibilities:**
- Extracts <title> tag
- Extracts meta description
- Extracts <html lang> attribute
- Extracts Open Graph meta tags (og:)
- Produces a plain-text version of the page

### `intergrax\websearch\fetcher\http_fetcher.py`

**Description:** This module provides a basic HTTP fetcher for web pages, handling requests and returning parsed content in the form of a `PageContent` instance.

**Domain:** Web search fetching agents

**Key Responsibilities:**

* Perform an asynchronous HTTP GET request with customizable headers
* Capture final URL, status code, raw HTML, and body size
* Return a `PageContent` instance on success or None on transport-level failure

### `intergrax\websearch\integration\__init__.py`

Description: This module initializes the integration module for web search functionality within the Intergrax framework.

Domain: RAG logic

Key Responsibilities:
- Initializes and sets up integration module components.
- Defines and registers necessary integration functions.
- Imports and configures required modules for web search integration.

### `intergrax\websearch\integration\langgraph_nodes.py`

Description: This module provides a LangGraph-compatible web search node wrapper that encapsulates configuration of a WebSearchExecutor instance, which is used to perform web searches.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Encapsulates configuration of a WebSearchExecutor instance.
- Provides synchronous and asynchronous node methods for performing web searches.
- Implements a default WebSearchNode instance for convenience and backward compatibility.
- Offers functional wrappers around the default WebSearchNode for simple integrations.

### `intergrax\websearch\pipeline\__init__.py`

Description: This module serves as the entry point for the web search pipeline, responsible for initializing and configuring its components.

Domain: RAG logic (Reinforcement Learning Augmented with human feedback) Pipeline Setup

Key Responsibilities:
- Initializes the pipeline components
- Configures the pipeline settings
- Sets up the data flow between components

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module orchestrates multi-provider web search, fetching, extraction, deduplication, and basic quality scoring. It provides a pipeline for executing queries against multiple providers, fetching and processing results, and returning WebDocument objects ready for consumption by LLM adapters or embedding pipelines.

**Domain:** Web Search Pipeline

**Key Responsibilities:**

* Executes query against all configured providers
* Fetches and extracts search hits into WebDocument objects
* Performs simple deduplication via text-based dedupe key
* Computes basic quality score based on content length and title presence
* Returns List[WebDocument] ready for consumption by LLM adapters or embedding pipelines

**Notes:** This module appears to be a critical component of the Intergrax framework's web search capabilities, providing a flexible and modular pipeline for executing complex web search tasks.

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: This module serves as the entry point for web search providers in Intergrax, handling the registration and initialization of various provider classes.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
- Registers provider classes with the main framework.
- Initializes provider instances upon request.

### `intergrax\websearch\providers\base.py`

**Description:** This module defines the base interface for web search providers in the Integrax framework, ensuring consistency across different search engines.

**Domain:** LLM adapters / Web Search Providers

**Key Responsibilities:**
- Accept a provider-agnostic QuerySpec and execute a single search request.
- Return a ranked list of SearchHit items with 'rank' ordering.
- Expose minimal capabilities for feature negotiation (language, freshness).
- Provide optional resource cleanup through the close method.
- Honor provider-side caps for top_k in QuerySpec.
- Include 'provider' and 'query_issued' fields in hits.
- Sanitize/validate URLs.

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search provider for the Intergrax framework, enabling search functionality via the Bing REST API.

**Domain:** LLM adapters > WebSearch providers

**Key Responsibilities:**

* Initializes the Bing Web Search provider with an API key and optional session timeout
* Supports language and region filtering
* Enables freshness filtering (Day, Week, Month) and safe search settings
* Provides query capabilities and generates parameters for the Bing API request
* Handles HTTP requests to the Bing API, parsing responses and extracting relevant data
* Transforms extracted data into SearchHit objects, which are then returned in a list

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module provides an implementation of the Google Custom Search (CSE) provider for web search queries.

**Domain:** Websearch providers

**Key Responsibilities:**

* Establishes a connection to the Google CSE REST API
* Handles environment variables for API key and search engine ID
* Builds query parameters based on the provided QuerySpec
* Supports language filtering via 'lr' (content language) and 'hl' (UI language)
* Does not natively support freshness; ignores spec.freshness
* Extracts relevant metadata from search results, including title, snippet, display link, published time, and source type
* Returns a list of SearchHit objects representing the search results

This file appears to be production-ready.

### `intergrax\websearch\schemas\__init__.py`

Description: The __init__.py file in the intergrax\websearch\schemas directory initializes and exports schema-related functionality for web search.

Domain: Schema Initialization and Exportation

Key Responsibilities:
- Initializes schema modules.
- Exports schema-related functions and classes.
- Sets up schema imports.

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module represents the fetched and optionally extracted content of a web page, providing utilities for post-processing stages.

**Domain:** Web Search / Page Content Representation

**Key Responsibilities:**
* Represents the fetched web page content with its metadata
* Encapsulates raw HTML and derived metadata for post-processing stages
* Provides methods to filter out failed or empty fetches
* Offers a truncated text snippet for logging and debugging purposes
* Calculates the approximate size of the content in kilobytes

### `intergrax\websearch\schemas\query_spec.py`

Description: This module defines the canonical search query specification used by web search providers in the Integrax framework.

Domain: Query schema / Websearch provider interface

Key Responsibilities:
- Provides a dataclass `QuerySpec` to represent a search query with various attributes (query, top_k, locale, region, language, freshness, site_filter, safe_search).
- Offers two utility methods for query normalization and top-k capping based on provider-specific limitations.
 
Note: This file appears to be a production-ready module within the Intergrax framework.

### `intergrax\websearch\schemas\search_hit.py`

Description: This module defines a dataclass `SearchHit` for provider-agnostic metadata of a single search result entry.

Domain: Search Result Schemas

Key Responsibilities:
* Defines the `SearchHit` dataclass with various attributes (provider, query_issued, rank, title, url, etc.) to store metadata about a search result.
* Provides methods (`__post_init__`, `domain`, and `to_minimal_dict`) for validation, domain extraction, and minimal representation of the hit.

### `intergrax\websearch\schemas\web_document.py`

**intergrax\websearch\schemas\web_document.py**

Description: This module defines a unified structure for representing fetched and processed web documents, providing essential fields and methods for analysis and logging.

Domain: Web Search Schema

Key Responsibilities:
- Defines the `WebDocument` class to store provider metadata (SearchHit), extracted content (PageContent), deduplication key, quality score, and source rank.
- Provides methods for document validation (`is_valid`), text merging (`merged_text`), and summary generation (`summary_line`).

### `intergrax\websearch\service\__init__.py`

Description: The web search service is initialized and configured in this file, providing a foundation for subsequent operations.

Domain: Web Search Service Configuration

Key Responsibilities:
- Initializes the web search service instance
- Configures search query processing and result handling
- Sets up database connections for storing search results (if applicable)

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module, `websearch_answerer.py`, provides a high-level helper class for answering user questions by combining web search results with Large Language Model (LLM) capabilities.

**Domain:** Web Search Integration & LLM Adapters

**Key Responsibilities:**

* Runs web search via the `WebSearchExecutor` instance
* Builds LLM-ready context/messages from web documents using the `WebSearchContextBuilder`
* Calls an LLMAdapter to generate a final answer
* Provides both async and sync interfaces for answering questions (`answer_async` and `answer_sync`)
* Supports optional system prompt overriding at the instance or per-call level

### `intergrax\websearch\service\websearch_executor.py`

**Description:** This module provides a high-level, configurable web search executor that enables the construction of query specifications, execution of web search pipelines, and serialization of web documents into LLM-friendly dictionaries.

**Domain:** Web Search Execution

**Key Responsibilities:**

* Construct QuerySpec objects from raw queries and configuration
* Execute SearchAndReadPipeline with chosen providers
* Convert WebDocument objects into LLM-friendly dictionaries
* Provide synchronous and asynchronous interfaces for web search execution
* Support customization of provider selection, query specification, and result serialization

### `intergrax\websearch\utils\__init__.py`

Description: This module serves as the initializer for Intergrax's web search utilities, encapsulating essential functionality for the web search engine.

Domain: Web Search Utilities

Key Responsibilities:
* Initializes and sets up web search utility components
* Defines import paths and dependencies for web search utilities 
* Provides entry points for web search-related functionality

### `intergrax\websearch\utils\dedupe.py`

Description: This module provides utility functions for deduplicating web search results, normalizing text to ensure identical documents are detected.

Domain: Web Search Utilities

Key Responsibilities:
- Normalizes text before deduplication using the `normalize_for_dedupe` function.
- Produces a stable SHA-256 based deduplication key for given text using the `simple_dedupe_key` function.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module implements a simple asyncio-compatible token bucket rate limiter, allowing for concurrent access and limiting the request rate.

**Domain:** RAG logic (Rate Adjustment and Governing)

**Key Responsibilities:**

* Provides a `TokenBucket` class that limits the rate of token acquisition based on a specified rate per second and capacity.
* Offers two methods to acquire tokens: `acquire()` which waits until at least n tokens are available, and `try_acquire()` which attempts to consume tokens without waiting.
* Ensures thread safety through the use of asyncio locks.

Note: The code appears well-structured and complete. There is no indication that it is experimental, auxiliary, legacy, or incomplete.

### `main.py`

**FILE PATH:** main.py
**CONTENT:**
"""
# ... (truncated)
"""

Description: This module serves as the entry point for the Intergrax framework, providing a basic execution environment.

Domain: Configuration/Initialization

Key Responsibilities:
* Defines the `main` function that prints a hello message when executed.
* Serves as the entry point for the application.

### `mcp\__init__.py`

**File:** mcp/__init__.py

**Description:** The MCP init file serves as the entry point for the Intergrax framework, handling imports and setup for other modules.

**Domain:** Configuration

**Key Responsibilities:**

* Initializes the Intergrax framework by importing core components
* Sets up module paths and namespace
* Provides a common interface for importing and using framework modules

### `notebooks\openai\rag_openai_presentation.ipynb`

**Description:** This Jupyter Notebook script demonstrates the usage of the Intergrax framework with OpenAI's RAG (Retrieval-Augmented Generation) model to interact with a Vector Store. It showcases how to upload a local folder to the Vector Store, ensure its existence, clear any existing data, and run queries to retrieve answers.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Uploads a local folder to the OpenAI Vector Store
* Ensures the Vector Store's existence and clears any existing data
* Configures OpenAI API keys and loads environment variables from a .env file
* Initializes an IntergraxRagOpenAI instance with the OpenAI client and Vector Store ID
* Runs queries using the RAG model to retrieve answers

Note: This script appears to be a demonstration or example code, rather than production-ready implementation.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter Notebook provides a demonstration of the Intergrax framework's functionality for loading, splitting, and embedding documents. It showcases various components such as `IntergraxDocumentsLoader`, `IntergraxDocumentsSplitter`, and `IntergraxEmbeddingManager`.

**Domain:** RAG (Recurrent Attention Graph) logic

**Key Responsibilities:**

* Load documents from a directory using `IntergraxDocumentsLoader`
* Split loaded documents into chunks using `IntergraxDocumentsSplitter`
* Embed documents using `IntergraxEmbeddingManager` with various providers and models
* Interact with a vector store (Chroma) to check for corpus presence and perform ingest operations

**Notes:**

The file appears to be a demonstration or test code rather than a production-ready implementation. The content is organized into sections for loading, splitting, embedding, and interacting with the vector store. It includes example usage of various Intergrax components and demonstrates how they can be used together to perform tasks such as document ingestion and embedding.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

**Description:** This Jupyter Notebook is a multimodal presentation of Intergrax framework capabilities, showcasing loading, splitting, and embedding documents, as well as interacting with VectorStore.

**Domain:** RAG (Retrieval-Augmented Generation) logic & VectorStore management

**Key Responsibilities:**

* Load documents from different sources (video, audio, images)
* Split documents into chunks
* Embed documents using various models
* Interact with VectorStore to add documents and retrieve information
* Perform retriever tests

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This notebook appears to be a demonstration or tutorial for utilizing the Integrax framework's multimedia processing capabilities, specifically for video and audio data.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloads video from YouTube using `yt_download_video`
* Extracts frames and metadatas from videos using `extract_frames_and_metadata`
* Transcribes video to VTT format using `transcribe_to_vtt`
* Downloads audio from YouTube using `yt_download_audio`
* Translates audio using `translate_audio`
* Uses the ollama model to describe images using `transcribe_image`
* Extracts frames from videos and transcribes images using `extract_frames_from_video`

### `notebooks\supervisor\supervisor_test.ipynb`

Description: This notebook provides a collection of components for the Integrax framework, including compliance checker, cost estimator, final summary, and financial audit.

Domain: RAG (Reason-Action-Gap) logic

Key Responsibilities:
* Compliance Checker:
	+ Verifies whether proposed changes comply with privacy policies and terms of service
	+ Returns findings on policy violations and recommends review with DPO if non-compliant
* Cost Estimator:
	+ Estimates the cost of UX-driven changes based on the UX audit report (mock)
	+ Uses a formula to calculate the estimated cost: base + per-issue * number_of_issues
* Final Summary:
	+ Generates the final consolidated summary using all collected artifacts
	+ Includes status pipeline, terminated by, terminate reason, PM decision, and more
* Financial Audit:
	+ Generates a mock financial report and VAT calculation (test data)
	+ Includes net, VAT rate, VAT amount, gross, currency, and budget last quarter

Note: This notebook appears to be part of the Integrax framework's RAG logic implementation, focusing on compliance, cost estimation, summarization, and financial simulations. The code is well-structured and uses the @component decorator from supervisor_components to define each component.

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This Jupyter notebook script provides a presentation of the web search functionality within the Integrax framework, focusing on Google Custom Search and Bing Search integration. It demonstrates how to execute web searches using these providers.

**Domain:** WebSearch (RAG logic)

**Key Responsibilities:**

* Importing necessary modules for environment setup and dotenv loading
* Defining query specifications and environment variables for search providers
* Creating instances of the Google CSE provider and executing a web search with specified parameters
* Printing search results in a structured format
* Providing an example use case for WebSearchExecutor with Google and Bing Search
