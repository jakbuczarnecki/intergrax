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
- `notebooks\openai\rag_openai_presentation.ipynb`
- `notebooks\rag\rag_custom_presentation.ipynb`
- `notebooks\rag\rag_multimodal_presentation.ipynb`
- `notebooks\rag\rag_video_audio_presentation.ipynb`
- `notebooks\supervisor\supervisor_test.ipynb`
- `notebooks\websearch\websearch_presentation.ipynb`

## Detailed File Documentation

### `api\__init__.py`

DESCRIPTION: The `__init__.py` file serves as the entry point for the Intergrax API, responsible for initializing and exposing its functionality to other modules.

DOMAIN: API initialization

KEY RESPONSIBILITIES:
* Imports necessary components from the Intergrax framework
* Initializes and sets up the API's main configuration and context
* Exposes key API endpoints and services for use by other parts of the application

### `api\chat\__init__.py`

Description: This is the entry point for the chat API module, responsible for initializing and configuring the chat functionality.

Domain: API/Chat Initialization

Key Responsibilities:
* Initializes the chat API module
* Sets up necessary configurations and dependencies
* Exposes APIs for interacting with the chat service 

Note: The file appears to be a standard __init__.py file in a Python package, indicating it's part of the project's structure rather than experimental or auxiliary.

### `api\chat\main.py`

**Description:** This module provides the main entry point for the Integrax chat API, handling user queries and document management.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Handles incoming user queries through the `/chat` endpoint
	+ Retrieves history from database
	+ Passes query to answerer (LLM)
	+ Logs result in database
* Manages document upload, indexing, and deletion
	+ Handles file upload and validation
	+ Indexes documents into Chroma using custom pipeline
	+ Deletes documents from Chroma and database

**Note:** This module appears to be part of the main API logic, with clear responsibilities for handling user queries and managing documents.

### `api\chat\tools\__init__.py`

Description: This module initializes the tools package within the API chat module, providing a entry point for tool-related functionality.

Domain: Utility Modules

Key Responsibilities:
* Initializes the tools package and its submodules
* Provides a registry for registering new tools
* Sets up tool-related configuration and dependencies

### `api\chat\tools\chroma_utils.py`

**Description:** This module provides utility functions for interacting with Chroma, a vector store used in the Integrax framework for RAG (Retrieval-Augmented Generation) logic.

**Domain:** RAG Logic

**Key Responsibilities:**
- Loads and splits documents from a given file path.
- Indexes a document to Chroma using a provided file path and ID.
- Deletes a document from Chroma by its file ID.

**Notes:** The functions in this module appear to be part of the main framework codebase, suggesting they are not experimental or auxiliary.

### `api\chat\tools\db_utils.py`

**Description:** This module provides utilities for interacting with a SQLite database, specifically for managing chat sessions and documents. It includes functions for creating the database schema, migrating data from legacy formats, and inserting/retrieving messages and document records.

**Domain:** Data Ingestion & Storage

**Key Responsibilities:**

* Creating and populating the SQLite database schema
* Migrating data from legacy application logs to the new message format
* Inserting and retrieving messages (with support for multiple roles and models)
* Inserting and retrieving document records, with optional metadata
* Providing backward-compatible entry points for legacy code
* Maintaining database connection management and transactional integrity

### `api\chat\tools\pydantic_models.py`

**Description:** This module provides data models for API requests and responses using the Pydantic library.

**Domain:** API/Chat Tools

**Key Responsibilities:**
- Defines model classes for querying input, response, document information, and deleting files
- Utilizes Pydantic's BaseModel and Field features for robust data modeling and validation

### `api\chat\tools\rag_pipeline.py`

**Description:** This file contains utility functions for the RAG pipeline in the Integrax framework, including lazy-loading singletons and retrieval of vectorstore and embedding managers.

**Domain:** RAG logic

**Key Responsibilities:**

* Lazy-load singleton instances of IntergraxVectorstoreManager and IntergraxEmbeddingManager
* Retrieve instances of these managers using lazy-loading functions (_get_vectorstore(), _get_embedder())
* Define default user and system prompts for the RAG pipeline
* Define utility functions for building LLM adapters and retrieving vectorstore/embedding managers

**Notes:** The file appears to be a utility module, providing essential functions for the RAG pipeline in the Integrax framework. It does not seem experimental or auxiliary.

### `applications\chat_streamlit\api_utils.py`

**Description:** This module provides Streamlit application utility functions for interacting with the Intergrax framework's API, including making HTTP requests to manage documents and chats.

**Domain:** Chat Application API Utilities

**Key Responsibilities:**
* Making POST requests to send chat messages to the API
* Handling API responses and errors
* Uploading files to the API
* Listing available documents in the API
* Deleting documents from the API

### `applications\chat_streamlit\chat_interface.py`

Description: This module provides a Streamlit-based interface for interacting with the chatbot, including displaying messages and generating responses.

Domain: Chat Interface

Key Responsibilities:
- Displays user and assistant messages in a chat interface
- Handles user input through a text input field
- Sends API requests to generate responses
- Updates session state with received response data

### `applications\chat_streamlit\sidebar.py`

**Description:** This module provides a Streamlit-based user interface for interacting with the Integrax framework, including model selection, document upload, listing uploaded documents, and deletion of selected documents.

**Domain:** Chat Interface

**Key Responsibilities:**

* Provides a Streamlit sidebar for model selection
* Enables document upload via file uploader and upload button
* Displays list of uploaded documents and allows refreshing the list
* Allows deleting selected documents from the uploaded documents list

### `applications\chat_streamlit\streamlit_app.py`

**Description:** This file implements the Streamlit application for the Intergrax RAG Chatbot, providing a user interface with a sidebar and chat interface.

**Domain:** Frontend/Chat Interface

**Key Responsibilities:**
- Initializes the Streamlit session state
- Displays the sidebar using the `display_sidebar` function from `sidebar.py`
- Displays the chat interface using the `display_chat_interface` function from `chat_interface.py`
- Sets up default values for session state variables if not already initialized

### `applications\company_profile\__init__.py`

Description: This module initializes and configures the company profile application.
Domain: Application Configuration

Key Responsibilities:
• Initializes the company profile application context
• Configures application-wide settings
• Establishes database connections for company data storage

### `applications\figma_integration\__init__.py`

Description: This module serves as the entry point for the Figma integration, facilitating interactions between Intergrax and Figma.

Domain: Integration modules

Key Responsibilities:
- Initializes the Figma integration module
- Provides access to Figma APIs
- Manages authentication with Figma services

### `applications\ux_audit_agent\__init__.py`

Description: This module initializes the UX Audit Agent application, defining its entry point and setting up necessary dependencies.

Domain: Application Initialization / Agents

Key Responsibilities:
* Initializes the UX Audit Agent application
* Sets up required dependencies for the agent's functionality
* Defines the main entry point for the application

### `applications\ux_audit_agent\components\__init__.py`

Description: This module serves as the entry point for the UX audit agent's components, responsible for initializing and registering necessary modules.

Domain: Agents

Key Responsibilities:
• Initializes component modules
• Registers components with the agent framework

### `applications\ux_audit_agent\components\compliance_checker.py`

**Description:** This module implements a compliance checker component for the Integrax framework, responsible for verifying if changes are in line with privacy policies and regulations.

**Domain:** RAG (Risk-Assessment-Governance) logic

**Key Responsibilities:**

* Verifies compliance of changes with privacy policies and regulations
* Returns findings on policy violations and requires DPO review
* Can be used to assess the conformity of UX changes with policies and regulations (mock implementation)
* Produces a ComponentResult object containing compliance findings, logs, and meta information

**Note:** This file appears to be a part of the Integrax framework's core functionality.

### `applications\ux_audit_agent\components\cost_estimator.py`

**Description:** This module provides a cost estimation component for the UX audit process, utilizing mock data to calculate an estimated cost based on identified issues.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a cost estimation function using mock data
* Utilizes random import (although not used in this specific code snippet)
* Registers the cost estimator component with the supervisor using the `@component` decorator
* Returns a ComponentResult object containing the estimated cost, currency, and calculation method

### `applications\ux_audit_agent\components\final_summary.py`

**applications\ux_audit_agent\components\final_summary.py**

Description: This module defines a final summary component for the Integrax framework, responsible for generating a comprehensive report at the end of a process.

Domain: RAG logic (Report, Action, Gap)

Key Responsibilities:
- Generates a final report by collecting all relevant artifacts.
- Provides a structured summary including pipeline status, termination reason, and related reports.
- Returns a ComponentResult with the generated summary as output.

### `applications\ux_audit_agent\components\financial_audit.py`

**Description:** This module defines a component for generating financial reports and VAT calculations as part of the Integrax framework's audit agent.

**Domain:** Financial Audit Agent

**Key Responsibilities:**
- Defines a 'Finansowy Agent' component with example usage and test data.
- Generates a mock financial report with calculated values (netto, VAT, brutto) and budget data.
- Exposes this data through the `ComponentResult` object.

### `applications\ux_audit_agent\components\general_knowledge.py`

Description: This module provides a component for handling general knowledge questions about the Intergrax system and its architecture.

Domain: RAG logic

Key Responsibilities:
- Provides an answer to general knowledge questions about the Intergrax system.
- Returns citations in the form of fake documentation pages.

### `applications\ux_audit_agent\components\project_manager.py`

Here is the documentation for the provided file:

**Description:** This module implements a project management component that randomly accepts or rejects UX reports based on certain conditions.

**Domain:** RAG logic (Risk, Action, and Governance)

**Key Responsibilities:**

* Randomly accept or reject UX reports
* Generate notes for accepted or rejected decisions
* Produce a decision and corresponding notes as output
* Log the decision and reason for rejection if applicable

### `applications\ux_audit_agent\components\ux_audit.py`

Description: This module provides a UX audit component for the Integrax framework, which analyzes user interface designs and generates a sample report with recommendations.

Domain: LLM adapters / RAG logic

Key Responsibilities:
- Performs UX analysis on Figma mocks or prototypes.
- Generates a sample report with summary, problems, recommendations, and estimated cost.
- Includes example use cases and descriptions for the audit component.

### `applications\ux_audit_agent\UXAuditTest.ipynb`

**Description:** This Jupyter Notebook file, UXAuditTest.ipynb, appears to be a test script for the Integrax framework's UX Audit Agent. It contains various steps and logic for auditing user experiences.

**Domain:** RAG logic (Retrieval-Aggregation-Generation)

**Key Responsibilities:**

* Executing UX audit tasks using specialized tools and components
* Verifying changes in a list of modifications
* Preparing reports
* Estimating costs of changes
* Synthesizing results

Note: This file seems to be part of the Integrax framework's development or testing process, as it includes experimental warnings and notes about missing inputs. The content suggests that this is an auxiliary or test script rather than a production-ready module.

### `generate_project_overview.py`

**Description:** This module generates a Markdown file summarizing the project structure of the Intergrax framework.

**Domain:** Configuration and Documentation Generation

**Key Responsibilities:**

- Recursively scans the project directory for relevant files (Python, Jupyter Notebooks, configurable).
- Collects metadata and content from each file.
- Sends file content + metadata to an LLM adapter (LangChainOllamaAdapter) for summarization.
- Generates a structured summary for each file, including purpose, domain, responsibilities.
- Assembles the summaries into a Markdown document describing the entire project layout.
- Writes the generated document to disk at a specified output path.

Note: The module appears to be well-documented and functional. It does not exhibit characteristics of experimental, auxiliary, legacy, or incomplete code.

### `intergrax\__init__.py`

DESCRIPTION: 
This is the main entry point and package initializer for the Intergrax framework.

DOMAIN: Package Initialization

KEY RESPONSIBILITIES:
• Initializes the Intergrax package with necessary imports and setup.
• Defines package-wide constants and configuration.

### `intergrax\chains\__init__.py`

DESCRIPTION: This is the entry point for Intergrax's chain modules, handling imports and setup.

DOMAIN: Chain Management

KEY RESPONSIBILITIES:
• Initializes chain module imports
• Sets up chain-specific configurations and environments
• Provides a central registry for chains to register their functionality

### `intergrax\chains\langchain_qa_chain.py`

**Description:** 
This file implements a flexible QA chain based on the LangChain framework, allowing for hooks to modify data at various stages of processing.

**Domain:** RAG logic (Retrieval-Augmented Generation)

**Key Responsibilities:**

* Builds a QA chain with hooks for modifying data at different stages
* Supports retrieval, re-ranking, and generation using LangChain
* Configurable via `IntergraxChainConfig` object
* Provides input/output formats for questions and answers

Note: There is no indication that this file is experimental, auxiliary, legacy, or incomplete.

### `intergrax\chat_agent.py`

**Description:** The `intergrax.chat_agent` module provides a chat agent that routes user questions to various LLM-based components, including RAG endpoints and tools. It handles memory management, streaming, and structured output.

**Domain:** LLM (Large Language Model) adapters and routing logic

**Key Responsibilities:**

* Routing user questions to LLM-based components using an LLM router
* Managing conversational memory and storing user questions
* Executing RAG endpoints and tools based on the routed component
* Handling streaming and structured output
* Providing a stable result shape with answer, tool traces, sources, summary, messages, output structure, stats, route, and rag component

### `intergrax\llm\__init__.py`

DESCRIPTION: The `__init__.py` file is the entry point for the LLM adapters in the Intergrax framework, responsible for initializing and exposing adapter functionality.

DOMAIN: LLM adapters

KEY RESPONSIBILITIES:
- Initializes LLM adapter modules
- Exposes adapter interfaces for external use

### `intergrax\llm\conversational_memory.py`

**Description:** This module provides a universal conversation memory that keeps track of chat messages in RAM and allows saving/loading to files (JSON/NDJSON) or SQLite. It supports filtering messages for sending to LLMs/adapters.

**Domain:** Conversation Memory

**Key Responsibilities:**

* Keeping track of chat messages in RAM
* Saving/loading to files (JSON/NDJSON)
* Saving/loading to SQLite
* Filtering messages for sending to LLMs/adapters
* Providing methods for adding, extending, and clearing the conversation memory
* Converting conversation memory to a format compatible with OpenAI Responses API / ChatCompletions

### `intergrax\llm\llm_adapters_legacy.py`

**Description:** This module provides a universal interface for LLM (Large Language Model) adapters and implementations of specific LLM adapters, including OpenAI Chat Completions.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Defines the `LLMAdapter` protocol with methods for generating and streaming messages
* Provides concrete adapter implementations, such as `OpenAIChatCompletionsAdapter`
* Includes helper functions for mapping chat messages to OpenAI schema and converting between different structured output formats
* Supports tools (optional) and provides a way to validate structured output with Pydantic models

### `intergrax\llm_adapters\__init__.py`

Description: This module serves as the entry point for LLM adapters in Intergrax, providing a registry and default adapter registrations.

Domain: LLM adapters

Key Responsibilities:
- Exposes the `LLMAdapter` class for adapter implementation
- Offers `LLMAdapterRegistry` for managing adapter registrations
- Includes base classes for model representation (`BaseModel`) and JSON schema handling (_model_json_schema)
- Provides utility functions for working with OpenAI responses and JSON extraction
- Registries default adapters (OpenAI, Gemini, Ollama) through the `register()` method

### `intergrax\llm_adapters\base.py`

**Description:** This module provides a base implementation for LLM adapters, including utilities for structured output and a universal interface for interacting with language models.

**Domain:** LLM Adapters

**Key Responsibilities:**

* Provides a universal interface (`LLMAdapter` protocol) for interacting with language models
* Offers tools for extracting JSON objects from text, validating model instances, and creating schema for models
* Includes an adapter registry (`LLMAdapterRegistry`) for registering and creating adapters
* Supports structured output generation and streaming for various LLMs

### `intergrax\llm_adapters\gemini_adapter.py`

Description: This module provides a minimal Gemini chat adapter for the Intergrax framework.
Domain: LLM adapters
Key Responsibilities:
- Initializes Gemini chat model with user-provided defaults
- Splits system messages from conversation and prepends them manually if needed
- Generates responses based on input messages using the underlying model
- Supports streaming of generated responses

Note: This implementation is a skeleton, intentionally not wiring tools for Gemini.

### `intergrax\llm_adapters\ollama_adapter.py`

**Description:** This module provides an adapter for Ollama models used via LangChain's ChatModel interface. It allows agents to interact with Ollama models in a planner-style pattern.

**Domain:** LLM adapters

**Key Responsibilities:**

* Converts internal chat messages to LangChain message objects
* Injects tool result as contextual system message when no native tools are supported
* Maps generation parameters from base_kwargs to options dictionary for Ollama models
* Invokes the chat model with adapted messages and returns the generated content
* Supports structured output via prompt and validation

**Note:** This adapter appears to be part of a larger framework, specifically designed for LLM adapters. It is not marked as experimental, auxiliary, legacy, or incomplete.

### `intergrax\llm_adapters\openai_responses_adapter.py`

**Description:** This module provides an adapter for interacting with OpenAI's Responses API, allowing for chat completion and tool-based interactions.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides an interface to the OpenAI Responses API
* Offers methods for generating messages and streaming responses
* Supports tools and function calls
* Generates structured JSON output based on a provided model

### `intergrax\logging.py`

**Description:** This module configures and standardizes logging within the Intergrax framework.

**Domain:** Logging Configuration

**Key Responsibilities:**
- Sets global logging settings using `logging.basicConfig()`.
- Configures log level to display INFO and higher severity messages.
- Formats log output with timestamp, severity level, and message.

### `intergrax\multimedia\__init__.py`

Description: The multimedia module serves as an entry point for interacting with multimedia-related components within the Intergrax framework.

Domain: Multimedia integration

Key Responsibilities:
* Imports and configures necessary multimedia libraries
* Establishes interfaces for handling multimedia data (e.g., images, videos) 
* Sets up default settings for multimedia operations

### `intergrax\multimedia\audio_loader.py`

Description: This module provides functionality for downloading and translating audio files from YouTube URLs.
Domain: Multimedia

Key Responsibilities:
- Downloads audio from a given YouTube URL using `yt_dlp`.
- Extracts audio to a specified output directory in a chosen format (default is MP3).
- Translates the downloaded audio using the Whisper model.

### `intergrax\multimedia\images_loader.py`

**intergrax\multimedia\images_loader.py**

Description: This module contains a function to transcribe images using an external AI model, providing a way to generate text descriptions from visual content.

Domain: LLM adapters

Key Responsibilities:
- Provides a function `transcribe_image` to generate text description of an image
- Utilizes the ollama library for interfacing with the AI model
- Supports multiple models, defaulting to "llava-llama3:latest" if none specified

### `intergrax\multimedia\ipynb_display.py`

**Description:** This module provides functionality for displaying multimedia content, including audio, images, and videos with custom playback settings. It uses IPython's display capabilities to render the media.

**Domain:** Multimedia

**Key Responsibilities:**

* Displaying audio files at a specific time position
* Displaying image files (including support for various formats)
* Serving video files with custom playback settings (e.g., start time, poster frame, autoplay, muted, and playback rate)
* Utility functions for serving paths and resolving file extensions

Note: The code appears to be well-maintained and production-ready.

### `intergrax\multimedia\video_loader.py`

**Description:** This module provides functionality for working with multimedia content, specifically video files. It includes utilities for downloading videos from YouTube, transcribing audio to text using the Whisper model, and extracting frames and metadata from videos.

**Domain:** Multimedia Processing

**Key Responsibilities:**

* Downloading videos from YouTube using `yt_dl`
* Transcribing audio to text using the Whisper model
* Extracting frames and metadata from video files
	+ Maintaining aspect ratio for resized images
	+ Saving extracted frames as JPEG images
	+ Generating metadata for each frame, including transcript, start/end times, and video segment ID
* Saving extracted metadata as a JSON file

Note: The file appears to be well-structured and complete.

### `intergrax\openai\__init__.py`

Description: This package serves as the entry point for OpenAI integrations within Intergrax, facilitating interactions with the OpenAI API.

Domain: LLM adapters

Key Responsibilities:
* Initializes OpenAI API client instance
* Sets up default configuration options
* Provides a basic structure for integrating custom OpenAI models and functionality

### `intergrax\openai\rag\__init__.py`

DESCRIPTION: This is the main entry point for the RAG (Retrieval-Augmented Generation) logic within Intergrax, responsible for setting up and configuring the retrieval components.

DOMAIN: LLM adapters / RAG logic

Key Responsibilities:
- Initializes the RAG components
- Configures the retrieval models and databases

### `intergrax\openai\rag\rag_openai.py`

**Description:** This module provides the IntergraxRagOpenAI class, which enables the integration of OpenAI's Rag logic with the Integrax framework. The class allows for interacting with an OpenAI client and a vector store to perform tasks such as retrieving documents, verifying consistency, answering questions, and uploading files.

**Domain:** RAG (Relevant And Grounded) Logic

**Key Responsibilities:**

* Initializes the IntergraxRagOpenAI class with an OpenAI client and vector store ID
* Provides a prompt for Rag logic implementation following specific guidelines
* Ensures the existence of a vector store by its ID
* Clears all files loaded into the vector store
* Uploads a folder to the vector store, allowing for file manipulation and management

### `intergrax\rag\__init__.py`

DESCRIPTION: This is the main entry point for the RAG (Relational Attention-based Graph) logic in Intergrax.

DOMAIN: RAG logic

KEY RESPONSIBILITIES:
• Initializes and sets up the RAG module.
• Defines and exposes API endpoints for interacting with the RAG model.

### `intergrax\rag\documents_loader.py`

**Description:** This module provides a robust and extensible document loader with metadata injection and safety guards. It supports various file formats, including text files, Word documents, HTML files, PDFs, Excel spreadsheets, CSV files, images, and videos.

**Domain:** RAG logic (Documents Loader)

**Key Responsibilities:**

* Loads documents from various file formats
* Injects metadata into the loaded documents
* Provides safety guards to prevent errors during loading
* Supports multiple document formats, including text, Word, HTML, PDF, Excel, CSV, images, and videos
* Offers customization options for loading documents, such as specifying file patterns, extensions map, and exclusions
* Integrates with other modules in the Intergrax framework, such as LLM adapters and multimedia loaders

### `intergrax\rag\documents_splitter.py`

**Description:** This module implements a high-quality text splitter for RAG pipelines, generating stable chunk IDs with rich metadata.

**Domain:** RAG Logic

**Key Responsibilities:**
- Infers page index from common loader keys.
- Ensures source fields are present in the document's metadata.
- Decides whether a document is a semantic atom based on its type and content.
- Builds stable, human-readable chunk IDs using available anchors (para_ix/row_ix/page_index).
- Finalizes chunks by adding indexing information, parent ID, source name/path, and page index if present.
- Optionally merges tiny tails and applies hard caps to the number of chunks per document.

### `intergrax\rag\dual_index_builder.py`

**Description:** This module builds and maintains two vector indexes: a primary (CHUNKS) index for all documents and an auxiliary (TOC) index for specific document headings.

**Domain:** RAG logic

**Key Responsibilities:**

* Builds two vector indexes: primary (CHUNKS) and auxiliary (TOC)
* CHUNKS index contains all chunks/documents after splitting
* TOC index contains only DOCX headings within specified levels [toc_min_level, toc_max_level]
* Embeds documents using the IntergraxEmbeddingManager
* Adds documents to the vector stores in batches for efficiency
* Provides logging and verbosity control

**Note:** This module appears to be well-structured and well-documented. There are no obvious red flags or indications of experimental or auxiliary code.

### `intergrax\rag\dual_retriever.py`

**Description:** This module implements a dual retriever for the Intergrax framework, which combines retrieval from both the TOC (Table of Contents) and local chunks. It uses vector store managers to retrieve relevant documents based on user queries.

**Domain:** RAG logic (Relevance Aware Generators)

**Key Responsibilities:**

*   Initializes the dual retriever with vector store managers for chunks and TOC, as well as an optional embedding manager.
*   Provides methods for querying the vector stores, normalizing hits, merging where conditions, expanding context via TOC, and retrieving final results.
*   The main `retrieve` method fetches base hits from chunks, expands context via TOC, merges and deduplicates results, and returns top-k relevant documents based on similarity scores.

Note: The module appears to be well-structured and functional.

### `intergrax\rag\embedding_manager.py`

**Description:** This module provides a unified embedding manager for HuggingFace, Ollama, and OpenAI embeddings.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Unified management of different embedding providers (HuggingFace, Ollama, OpenAI)
* Automatic model loading with reasonable defaults
* Batch/single text embedding with optional L2 normalization
* Embedding for LangChain Documents (returns np.ndarray + aligned docs)
* Cosine similarity utilities and top-K retrieval
* Robust logging and transient error handling

### `intergrax\rag\rag_answerer.py`

**Description:** This module provides a Rag answerer, responsible for retrieving relevant context fragments, re-ranking them (if configured), building the context and citations, generating answers using LLM adapters, and handling user instructions.

**Domain:** RAG Logic

**Key Responsibilities:**

* Retrieval of relevant context fragments
* Optional re-ranking of retrieved hits
* Building context from retrieved hits
* Generating citations for used context fragments
* Handling user instructions and system prompts
* Integrating with LLM adapters for answer generation
* Supporting streaming or non-streaming output modes

### `intergrax\rag\rag_retriever.py`

**Description:** This module provides a scalable, provider-agnostic RAG retriever for the Intergrax framework. It handles various tasks such as normalization of filters and embeddings, deduplication, batch retrieval, and optional reranking.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**
- Normalizes `where` filters for Chroma
- Normalizes query vector shape (1D/2D → [[D]])
- Unified similarity scoring for various providers
- Deduplication by ID + per-parent result limiting
- Optional MMR diversification when embeddings are returned
- Batch retrieval for multiple queries
- Optional reranking hook (e.g., cross-encoder, re-ranking model)

### `intergrax\rag\re_ranker.py`

**Description:** This module provides a Re-Ranker component for the Intergrax framework, allowing fast and scalable cosine re-ranking of candidate chunks.

**Domain:** RAG logic

**Key Responsibilities:**

* Re-rank candidates based on cosine similarity to a query
* Embed texts in batches using `intergraxEmbeddingManager`
* Optional score fusion with original retriever similarity
* Preserves schema of hits; adds 'rerank_score', 'fusion_score' (if enabled), and 'rank_reranked'
* Supports various input types, including hits from the retriever (dict) or raw LangChain Documents
* Configurable parameters for score fusion, normalization, cache query embeddings, and batch sizes

### `intergrax\rag\vectorstore_manager.py`

**Description:** This module manages vector stores for the Intergrax framework, supporting ChromaDB, Qdrant, and Pinecone.

**Domain:** Vector Store Management

**Key Responsibilities:**
- Initializes target store with configurable settings.
- Creates collection/index in target store (lazy creation for Qdrant/Pinecone).
- Upserts documents + embeddings (with batching) to the vector store.
- Queries top-K by cosine/dot/euclidean similarity in the vector store.
- Counts vectors in the vector store.
- Deletes vectors by their IDs.

**Notes:** The code appears well-maintained and clearly documented. It handles different vector stores, including ChromaDB, Qdrant, and Pinecone, providing a unified interface for vector management within the Intergrax framework.

### `intergrax\rag\windowed_answerer.py`

**Description:** This module implements a Windowed Answerer layer on top of the base Answerer in the Integrax framework. It provides functionality for memory-aware question answering with windowing.

**Domain:** RAG logic

**Key Responsibilities:**

*   Initializes the Windowed Answerer with an Answerer, retriever, and optional verbose mode.
*   Performs broad retrieval and windowing to process multiple context windows.
*   Builds messages with memory awareness (if available) and generates partial answers using LLM.
*   Synthesizes a final answer from partials and deduplicates sources.
*   Appends the final answer and optional summary to the memory store (if available).

### `intergrax\supervisor\__init__.py`

DESCRIPTION: The __init__.py file serves as the entry point and initializer for the supervisor module in Intergrax, responsible for setting up its internal structure.

DOMAIN: Supervision

KEY RESPONSIBILITIES:
- Initializes the supervisor module's dependencies and components.
- Defines the interface for external interactions with the supervisor.

### `intergrax\supervisor\supervisor.py`

**Description:** This module provides the core functionality of the Intergrax framework's supervisor component. It includes planning and assignment logic, LLM adapters, component management, and configuration handling.

**Domain:** Supervisor / Planner

**Key Responsibilities:**

* Planning and assignment logic using two-stage decomposition (decompose + per-step assign)
* LLM adapter integration for text extraction and JSON parsing
* Component registration and management
* Configuration handling and validation
* Robust planning fallbacks (heuristic, minimal) when LLM fails
* Two-stage planner knobs for customization
* Semantic fallback for keyword-less component assignment

### `intergrax\supervisor\supervisor_components.py`

**Description:** This module provides a framework for building and managing components in the Integrax supervisor, allowing for reusable step implementations with their own logic and context.

**Domain:** Supervisor components

**Key Responsibilities:**

* Defines the `Component` class, which represents a single step or operation that can be executed within the pipeline.
* Provides a `run` method for executing a component, taking in a `PipelineState` and `ComponentContext`.
* Offers a decorator (`component`) for registering new components with minimal boilerplate code.
* Includes data classes for representing component results (`ComponentResult`) and context (`ComponentContext`).

### `intergrax\supervisor\supervisor_prompts.py`

**Description:** This module provides a set of default prompt templates for the Intergrax Supervisor, outlining the planning process and constraints for creating auditable DAG plans.

**Domain:** RAG logic / Unified Planning Framework

**Key Responsibilities:**

* Defines the universal prompts for planning, focusing on decomposition-first approach
* Specifies primary principles for planning, including decomposition, assignment, data-flow, and component selection
* Outlines strict rules for resource flags, DAG structure, gates, synthesis, reliability, and decomposition coverage
* Provides a validation checklist to ensure plan correctness before returning
* Defines the return schema for the unified Supervisor's response

### `intergrax\supervisor\supervisor_to_state_graph.py`

**Description:** This module provides utilities for transforming a Plan into a runnable LangGraph pipeline. It includes functions for ensuring state defaults, appending logs, resolving inputs, persisting outputs, creating node names, and building the graph.

**Domain:** Supervisor to State Graph Utilities

**Key Responsibilities:**

* Ensuring state defaults with `_ensure_state_defaults`
* Appending logs with `_append_log`
* Resolving inputs with `_resolve_inputs`
* Persisting outputs with `_persist_outputs`
* Creating readable node names with `_slugify` and `_make_node_name`
* Building the graph with `topo_order` and `build_langgraph_from_plan`

This file appears to be a crucial part of the Integrax framework, handling the transformation of plans into executable LangGraph pipelines. It is likely used by the supervisor components to create and manage the pipeline graphs. The code is well-structured, readable, and seems to be production-ready.

### `intergrax\system_prompts.py`

Here is the documentation of the provided file:

**Description:** This module defines a RAG system instruction template for strict RAG (Rola i Zasady Pracy) operation in the Integrax framework.

**Domain:** RAG logic

**Key Responsibilities:**

* Defines a prompt for the RAG system instruction
* Specifies the rules and guidelines for the RAG operation, including:
	+ No use of general knowledge or adding facts not present in documents
	+ Responses must be based on document content only
	+ Citing sources is required
	+ Handling ambiguity and uncertainty
	+ Precise terminology and formatting requirements

**Note:** The file appears to be a part of the Integrax framework's proprietary and confidential codebase, indicating that it may not be publicly available or modifiable without permission.

### `intergrax\tools\__init__.py`

DESCRIPTION: This package serves as the entry point for various tools within the Intergrax framework, facilitating access to utility functions and modules.

DOMAIN: Configuration/Utilities

KEY RESPONSIBILITIES:
- Provides a centralized entry point for tool-related functionality.
- Imports and initializes various utility modules.

### `intergrax\tools\tools_agent.py`

**Description:** 
This module provides the implementation of an Intergrax tools agent, which is a key component in managing and utilizing external tools for answering questions. The tools agent interacts with language models (LLMs) and manages tool outputs, including processing and formatting responses.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**
- Manages interaction between the Intergrax framework and external tools
- Processes tool outputs, extracting relevant information and formatting responses for the user
- Utilizes language models (LLMs) to generate answers or guide the tool-calling process
- Maintains a conversational memory to track interactions and context throughout the conversation
- Supports both native tool-calling in OpenAI LLMs and JSON-based planners like Ollama

### `intergrax\tools\tools_base.py`

**Description:** This module provides a foundation for building tools within the Integrax framework. It offers base classes and functionality for defining, registering, and exporting tools in a format compatible with the OpenAI Responses API.

**Domain:** Tooling/Utilities

**Key Responsibilities:**

* Defines `ToolBase` class for creating custom tools
	+ Provides attributes for name, description, schema model, and strict validation
	+ Offers methods for getting parameters, running the tool (must be overridden), validating arguments, and converting to OpenAI schema
* Introduces `ToolRegistry` class for managing registered tools
	+ Allows registering tools with unique names and retrieving them by name
	+ Exposes tools in a format compatible with the OpenAI Responses API

This module appears to be stable, well-structured, and part of the main Integrax framework.

### `intergrax\websearch\__init__.py`

DESCRIPTION: This module serves as the entry point for web search functionality within the Intergrax framework, responsible for initializing and configuring related components.

DOMAIN: Web Search Module

KEY RESPONSIBILITIES:
* Initializes web search-related services
* Configures web search settings and parameters
* Exposes necessary interfaces for integrating web search capabilities

### `intergrax\websearch\cache\__init__.py`

**Description:** This module provides a simple in-memory query cache for web search results with optional time-to-live (TTL) and maximum size.

**Domain:** Data Ingestion/Caching

**Key Responsibilities:**
- Stores cached web documents for given queries
- Provides `get` method to retrieve cached documents by query key
- Offers `set` method to store new documents in cache with query key
- Supports TTL and max size configuration
- Uses a dictionary-based data structure for storage

### `intergrax\websearch\context\__init__.py`

Description: This is the initialization file for the web search context, responsible for setting up the necessary components and configurations.

Domain: Web Search Context

Key Responsibilities:
* Initializes the web search context with required dependencies
* Sets up any default configuration or constants
* Defines the entry point for the web search context module

### `intergrax\websearch\context\websearch_context_builder.py`

**Description:** This module provides a WebSearchContextBuilder class, which builds LLM-ready textual context and chat messages from web search results. It can work with raw WebDocument objects or serialized dicts.

**Domain:** RAG logic (Relevance Aware Generation)

**Key Responsibilities:**

* Builds textual context string from WebDocument objects or serialized dicts
* Constructs chat messages for chat-style LLMs, enforcing strict "sources-only" mode
* Allows customization of context building and message formatting through constructor parameters and override methods

Note: The code is well-structured, readable, and includes clear documentation. It appears to be a stable and functional part of the Intergrax framework.

### `intergrax\websearch\fetcher\__init__.py`

Description: Initializes the web search fetcher component, responsible for retrieving data from various online sources.

Domain: Web Search Fetcher

Key Responsibilities:
• Registers available web search providers
• Configures fetcher settings
• Defines default fetcher behavior
• Sets up logging and error handling 

Note: This file appears to be a standard initialization module for the web search fetcher component.

### `intergrax\websearch\fetcher\extractor.py`

**Description:** This module provides a basic HTML extractor for web search results, extracting essential metadata and text content from pages.

**Domain:** Web Search Extractor

**Key Responsibilities:**
- Extracts <title> from the page's HTML.
- Extracts meta description from the page's HTML.
- Extracts <html lang> attribute from the page's HTML.
- Extracts Open Graph meta tags (og:*).
- Produces a plain-text version of the page.

### `intergrax\websearch\fetcher\http_fetcher.py`

Description: This module provides an asynchronous HTTP fetcher for web pages.

Domain: Web Search Fetchers

Key Responsibilities:
- Performs HTTP GET requests with customizable headers.
- Captures final URL, status code, raw HTML, and body size.
- Keeps higher-level concerns (robots, throttling, extraction) outside.

### `intergrax\websearch\integration\__init__.py`

Description: This module serves as the entry point for web search integration within the Intergrax framework, importing necessary dependencies and configurations.

Domain: Web Search Integration

Key Responsibilities:
• Imports required modules and libraries for web search functionality
• Sets up configuration options for web search integration
• Initializes the web search pipeline

### `intergrax\websearch\integration\langgraph_nodes.py`

**Description:** This module provides a LangGraph-compatible web search node wrapper, encapsulating configuration of a WebSearchExecutor instance and implementing sync and async node methods operating on a minimal state contract for web search nodes.

**Domain:** LLM adapters/RAG logic

**Key Responsibilities:**

* Encapsulates the configuration of a WebSearchExecutor instance
* Provides sync and async node methods operating on a minimal state contract for web search nodes
* Implements the _extract_query method to extract the search query from the node state
* Runs synchronous and asynchronous searches using the provided executor instance

**Note:** This file appears to be part of the main Intergrax framework, with clear documentation and well-structured code. No signs of experimental, auxiliary, legacy, or incomplete features are observed.

### `intergrax\websearch\pipeline\__init__.py`

**Description:** This module initializes and configures the pipeline for web search functionality within Intergrax.
**Domain:** Web Search Pipeline Configuration
**Key Responsibilities:**
* Initializes pipeline components and dependencies
* Configures pipeline settings and parameters
* Exposes API to register pipeline steps and tasks

### `intergrax\websearch\pipeline\search_and_read.py`

**Description:** This module defines the SearchAndReadPipeline class, which orchestrates multi-provider web search, fetching, extraction, deduplication, and basic quality scoring. It provides a high-level flow for executing queries against configured providers.

**Domain:** Websearch pipeline logic

**Key Responsibilities:**

- Orchestrates multi-provider web search
- Fetches and extracts SearchHit objects into WebDocument objects
- Performs deduplication based on simple text-based keys
- Applies quality scoring heuristics
- Provides an async interface for fetching multiple pages concurrently
- Offers a synchronous convenience wrapper for environments without an external event loop

The module appears to be stable and production-ready, with clear design goals and a well-structured implementation.

### `intergrax\websearch\providers\__init__.py`

DESCRIPTION: 
This module initializes the web search providers for the Intergrax framework, setting up the necessary infrastructure for interacting with various online resources.

DOMAIN: Web Search Providers

KEY RESPONSIBILITIES:
- Initializes and configures web search provider instances
- Sets up connection parameters for each provider
- Defines the interface for web search operations

### `intergrax\websearch\providers\base.py`

**Description:** This module defines the base interface for web search providers in the Integrax framework, specifying a contract for executing searches and exposing minimal capabilities.

**Domain:** Web Search Providers

**Key Responsibilities:**
- Execute a single search request with a provider-agnostic QuerySpec
- Return a ranked list of SearchHit items
- Expose minimal capabilities for feature negotiation (language, freshness)
- Support resource cleanup through the close method (optional)

### `intergrax\websearch\providers\bing_provider.py`

**Description:** This module provides a Bing Web Search (v7) provider for the Intergrax framework, enabling users to perform web searches using the Bing API.

**Domain:** LLM adapters

**Key Responsibilities:**

* Initializes and configures the Bing Web Search provider with an optional API key, session, and timeout.
* Supports filtering by language and region through environment variables or query specifications.
* Offers freshness filtering for search results (Day, Week, Month).
* Implements safeSearch functionality to filter explicit content.
* Provides a `search` method for executing web searches based on user-specified queries.
* Converts API responses into SearchHit objects for processing.

Note: This module appears complete and ready for use.

### `intergrax\websearch\providers\google_cse_provider.py`

**Description:** This module implements a provider for the Google Custom Search (CSE) service, which is used to perform web searches.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides a class `GoogleCSEProvider` that inherits from `WebSearchProvider`
* Handles API key and search engine ID retrieval from environment variables
* Defines methods for building query parameters and making requests to the CSE API
* Parses response data to extract search hits, including title, URL, snippet, and other metadata
* Supports language filtering using 'lr' (content language) and 'hl' (UI language)
* Handles pagination with a cap of 10 results per request

### `intergrax\websearch\providers\google_places_provider.py`

**Description:** This module provides a WebSearchProvider implementation for Google Places, allowing text searches and fetching of business details.

**Domain:** LLM adapters (WebSearchProvider)

**Key Responsibilities:**

* Provides an interface to the Google Places API
* Supports text search by arbitrary query
* Fetches business details for searched places
* Maps results to SearchHit objects
* Handles environment variables and API key management
* Implements capabilities and parameter builders for text search and details fetching
* Performs requests and handles JSON responses from the Google Places API

### `intergrax\websearch\providers\reddit_search_provider.py`

**Description:** This module provides a Reddit search provider using the official OAuth2 API, allowing for full-featured search and rich post metadata retrieval.

**Domain:** LLM adapters

**Key Responsibilities:**

* Provides authentication with Reddit's OAuth2 API
* Supports language and freshness filtering in search queries
* Retrieves top-level comments for each post (optional)
* Maps search results to the SearchHit schema
* Handles token refresh and caching for improved performance

### `intergrax\websearch\schemas\__init__.py`

Description: This module initializes the web search schema configurations and utility functions.

Domain: Web Search Schema Configuration

Key Responsibilities:
* Registers schema definitions for web search functionality
* Initializes configuration settings for web search
* Provides utility functions for schema management

### `intergrax\websearch\schemas\page_content.py`

**Description:** This module defines a dataclass representing the content of a web page, encapsulating both raw HTML and extracted metadata.

**Domain:** Web search (RAG logic)

**Key Responsibilities:**

* Represents web page content as a dataclass with various attributes
* Includes methods for filtering empty fetches (`has_content`), generating text summaries (`short_summary`), and calculating content size in kilobytes (`content_length_kb`)
* Supports optional metadata extraction, such as Open Graph tags and schema.org markup

### `intergrax\websearch\schemas\query_spec.py`

**Description:** This module defines a data class for canonical search query specifications used by web search providers.

**Domain:** Web Search Schemas

**Key Responsibilities:**

* Defines the `QuerySpec` data class with attributes for raw user query, top results per provider, locale, region, language, freshness constraint, site restriction, and safe search filtering.
* Provides methods to normalize the query string with an applied site filter (`normalized_query`) and cap the number of top results based on a provider's cap (`capped_top_k`).

### `intergrax\websearch\schemas\search_hit.py`

**Description:** This module defines a `SearchHit` data class to represent provider-agnostic metadata for a single search result entry, including fields such as provider identifier, query string, rank, title, URL, and more.

**Domain:** RAG (Retrieval-Augmented Generation) logic

**Key Responsibilities:**

* Defines the `SearchHit` data class with various attributes for storing search result metadata
* Enforces minimal safety checks in the `__post_init__` method to ensure validity of rank and URL fields
* Provides methods for extracting domain from the URL (`domain`), and converting the hit to a minimal dictionary representation suitable for LLMs (`to_minimal_dict`)

### `intergrax\websearch\schemas\web_document.py`

Description: This module defines the WebDocument data class, which serves as a unified structure for representing and processing fetched web documents.

Domain: Web Search Schemas

Key Responsibilities:
- Provides a standardized representation of web documents with source metadata, content, and analysis results.
- Offers methods for determining document validity, merging textual content, and generating summary lines.

### `intergrax\websearch\service\__init__.py`

Description: This service module initializes and sets up the web search functionality for Intergrax.

Domain: Web Search Service

Key Responsibilities:
- Initializes the web search service instance
- Sets up event listeners for relevant events
- Exposes interface for other modules to interact with the web search service

### `intergrax\websearch\service\websearch_answerer.py`

**Description:** This module provides a high-level helper class `WebSearchAnswerer` that integrates web search results with Large Language Model (LLM) adapters to generate answers.

**Domain:** Web Search Service

**Key Responsibilities:**
- Runs web searches using the `WebSearchExecutor`
- Builds LLM-ready context/messages from web documents
- Calls an LLMAdapter to generate a final answer
- Supports both async and sync execution modes

### `intergrax\websearch\service\websearch_executor.py`

Description: This module provides a high-level, configurable web search executor that constructs QuerySpec from raw queries and configuration, executes SearchAndReadPipeline with chosen providers, and converts WebDocument objects into LLM-friendly dictionaries.

Domain: Web search execution and caching

Key Responsibilities:
- Construct QuerySpec from raw query and configuration
- Execute SearchAndReadPipeline with chosen providers
- Convert WebDocument objects into LLM-friendly dictionaries
- Build a simple, deterministic signature of the provider configuration for cache key generation
- Serialize WebDocument objects into dicts suitable for LLM prompts and logging
- Attempt to return cached serialized results when available and valid

### `intergrax\websearch\utils\__init__.py`

Description: This module provides a set of utility functions for web search functionality within the Intergrax framework.

Domain: Web Search Utilities

Key Responsibilities:
- Initializes and sets up necessary libraries and dependencies.
- Provides common utility methods for web search operations.
- Possibly contains legacy or experimental code. 

Note: The lack of specific content in the file makes it difficult to provide a more detailed analysis, but based on its location within the project structure, it appears to be an initialization module for web search utilities.

### `intergrax\websearch\utils\dedupe.py`

**Description:** This module provides deduplication utilities for web search, normalizing and hashing input text to detect near-identical documents.

**Domain:** Websearch Utilities

**Key Responsibilities:**
* Normalizes text before deduplication by stripping whitespace, converting to lower case, and collapsing internal whitespace sequences.
* Produces a stable SHA-256 based deduplication key for the given text using the normalized input.

### `intergrax\websearch\utils\rate_limit.py`

**Description:** This module implements a token bucket rate limiter, allowing for asynchronous and concurrent usage while preventing exceeding the specified average request rate.

**Domain:** RAG logic

**Key Responsibilities:**

* Provides a simple asyncio-compatible token bucket rate limiter
* Ensures that the total consumption of tokens never exceeds capacity
* Guarantees that the average request rate does not exceed the specified rate
* Offers two methods for acquiring tokens: `acquire` (waits until tokens are available) and `try_acquire` (non-blocking, attempts to consume tokens)
* Allows for concurrent usage across multiple coroutines in a single process

### `main.py`

**Description:** This module serves as the entry point for the Intergrax framework, responsible for initiating its execution and providing a basic greeting message.

**Domain:** Framework Initialization

**Key Responsibilities:**
- Serves as the primary entry point for the Integrax framework.
- Initializes the framework's execution and prints a greeting message.

### `mcp\__init__.py`

**Description:** The `__init__.py` file serves as the entry point for the Intergrax framework, initializing and configuring its core components.

**Domain:** Framework initialization

**Key Responsibilities:**

* Initializes the Intergrax framework
* Configures core components and settings
* Sets up default imports and dependencies

### `notebooks\openai\rag_openai_presentation.ipynb`

**Description:** This Jupyter notebook provides a presentation of the RAG (Retrieve, Answer, Generate) OpenAI integration within the Integrax framework. It showcases how to set up and use this feature for filling a VectorStore with data from a local folder and testing queries.

**Domain:** LLM adapters / RAG logic

**Key Responsibilities:**

* Import necessary libraries and modules
* Set up environment variables for API keys and VectorStore ID
* Create an OpenAI client and IntergraxRagOpenAI instance
* Ensure the VectorStore exists, clear it if necessary, and upload a local folder to it
* Test queries by running RAG operations on specific questions

**Note:** This file appears to be production-ready.

### `notebooks\rag\rag_custom_presentation.ipynb`

**Description:** This Jupyter Notebook appears to be a script for loading and processing documents within the Integrax framework. It demonstrates various tasks such as document loading, splitting, embedding, and vector store management.

**Domain:** RAG (Recurrent Attention Graph) logic / Documents Management

**Key Responsibilities:**

* Load documents from file paths using `IntergraxDocumentsLoader`
* Split loaded documents into chunks using `IntergraxDocumentsSplitter`
* Embed documents using `IntergraxEmbeddingManager` and store the embeddings in a vector store
* Manage vector store configuration and query documents for existence
* Optionally ingest new documents into the vector store if they do not already exist

Note: This script appears to be self-contained, providing all necessary imports and module calls within itself. However, it relies on external modules (`intergrax.rag.documents_loader`, `intergrax.rag.documents_splitter`, etc.) which may need to be reviewed for their individual documentation.

### `notebooks\rag\rag_multimodal_presentation.ipynb`

Description: This Jupyter notebook script provides an example of loading multimodal documents (video, audio, image), splitting and embedding them, and then ingesting the embeddings into a VectorStore. Additionally, it demonstrates how to query the VectorStore using a retriever.

Domain: RAG (Retriever-Indexer-Generator) logic

Key Responsibilities:
* Load multimodal documents from file paths
* Split and embed documents
* Ingest embedded documents into a VectorStore
* Query the VectorStore using a retriever

### `notebooks\rag\rag_video_audio_presentation.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of various multimedia processing functions within the Intergrax framework, including video and audio loading, transcription, frame extraction, and image description.

**Domain:** Multimedia processing

**Key Responsibilities:**

* Load videos from YouTube using `yt_download_video`
* Download and save videos to a specified directory
* Transcribe videos to VTT format using `transcribe_to_vtt`
* Extract frames and metadata from videos using `extract_frames_and_metadata`
* Translate audio files using `translate_audio`
* Use the Ollama model to describe images using `transcribe_image`

### `notebooks\supervisor\supervisor_test.ipynb`

**Description:** This notebook contains a collection of components for the Integrax framework, which are used to build a data pipeline. The components provided include compliance checkers, cost estimators, and final summary generators.

**Domain:** RAG (Reasoning And Generation) logic

**Key Responsibilities:**

* Compliance checker:
	+ Verifies whether proposed changes comply with privacy policies and terms of service
	+ Returns a mock compliance decision with an 80% chance of success
* Cost estimator:
	+ Estimates the cost of changes based on the UX audit report (mock)
	+ Uses a formula to calculate the base cost and additional cost per issue
* Final summary generator:
	+ Generates the final consolidated summary using all collected artifacts
	+ Includes status pipeline, terminated by, terminate reason, PM decision, and other relevant information
* Financial auditor:
	+ Generates a mock financial report and VAT calculation (test data)
	+ Returns a report with net value, VAT rate, VAT amount, gross value, currency, and last quarter budget

### `notebooks\websearch\websearch_presentation.ipynb`

**Description:** This Jupyter Notebook demonstrates the usage of the Intergrax framework's web search capabilities, specifically with Google Custom Search and Bing Search providers.

**Domain:** WebSearch

**Key Responsibilities:**

* Demonstrates how to configure and use the Google Custom Search provider
* Provides an example query specification for searching on a specific topic
* Executes a search using the specified provider and prints out results
* Shows how to handle search hits, including retrieving information about each hit
* Includes examples of handling multiple search results and printing out relevant information.
