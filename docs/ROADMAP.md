# Intergrax Roadmap

**Last updated:** 2026-01-12

This is a living engineering roadmap / TODO list.  
It reflects current development priorities and may change frequently.

---

[DONE] Logging — global settings and global contracts — simplify logging  
[DONE] Diagnostics and tracing — in runtime and RuntimeState — replace dictionaries with typed structures  

[] Prompting — global settings and global contracts — move all prompts into a new structured location  
[] Tests — create production-grade unit tests — start by converting notebooks into unit tests  
[] LLM — create adapter for Grok  
[] Cloud — create mechanisms for cloud computing integrations (Azure, AWS, etc.)  
[] Runtime — create lifecycle events to notify users about reasoning and pipelines, and allow interruption when needed  
[] Orchestration + CoT — create a flexible and customizable architecture for orchestration and delegation  
[] MCP — create foundations for MCP configurations that allow building backend services  
[] API / FastAPI — create foundations for API / FastAPI configurations to allow building backend services  
[] Agent — design an IT headhunter agent  
[] Agent — design a company profile agent  
[] Agent — create a virtual company team with a supervisor and design communication between them  
[] Pipelines — refactor pipeline architecture for customization — instead of predefined pipelines, allow using e.g. LangGraph  
    It is important to create an architecture that allows implementing custom reasoning blocks.  
[] Integrations — Google Docs, Google Drive, Google Sheets  
[] Voice agent — create an example voice chatbot  
[] Agent / Tool — Text-to-SQL  
[] Integrations — Pinecone  
[] Human-in-the-loop — implement specialized handlers for human interventions  
[] Integrations — Firebase  
[] Budget control — create an architecture that allows defining budget policies  
[] Agent — implement an app similar to Google NotebookLM to demonstrate all Intergrax mechanisms  
[] Agent — implement an agent that searches project directories and creates summaries and comments  
[] Large data handling — handle large data sources such as source-code repositories by creating scalable reasoning chains  
[] Critics in CoT — implement self-awareness and auto-correction modules in the chain of thought  
[] Human-in-the-loop — further focus on HITL components  
[] Tooling — create a base set of tools for tool-agents, starting with integrations (Google, weather, calculator, stocks, websites, etc.)  
[] Integrations — SerpAPI  
[] Integrations — DuckDuckGo  
[] Integrations — other useful and well-known APIs  
[] Sessions — implement production storage adapters (databases) and architecture for reading and writing user profiles and sessions  
[] LLM Adapters — change generate_messages to return a custom object instead of a raw string  
[] LLM Adapters — implement full-usage stream_messages  
[] EnginePlan — replace debug dictionaries with strongly typed structures  
[] Guardrails — implement a new guardrails layer for the engine  
[] Memory improvement — implement mechanisms for improving reasoning while history profiles grow  
[] Organization profiles — implement proper organization profile support in the runtime environment (similar to user profiles and memory)  
[] Skills — implement a skill mechanism similar to Claude  
[] Runtime loop — handle long user questions by splitting them into manageable parts  
[] Runtime loop — remove the strategy flag and replace it with a configuration-based pipeline selection mechanism  
[] Logging — attach logger to other system components  
[] Trace events — implement persistent trace event storage (databases, files, system logs)
