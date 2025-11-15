# Voice Agent & Real-Time Speech Interaction Component

**Status:** Planned  
**Owner:** Artur Czarnecki  
**Start Date:** _(add date)_  
**Target Version:** v1.x  
**Last Updated:** _(add date)_  

---

## Goal

Extend the intergrax framework with a **Voice Interaction Component** enabling agents to communicate using **real-time, bidirectional voice processing** – including speech recognition, reasoning, and speech synthesis.

The component should serve as a bridge between the intergrax ecosystem (Agents, RAG, Supervisor, Memory) and audio channels to deliver natural voice-driven experiences.

---

## Description

This milestone introduces the `intergrax-voice` module — a standardized layer for **voice-based interaction** between the system and the user.

The module will enable intergrax agents to:

- Receive streaming or static audio from a microphone, file upload, or WebRTC source.
- Perform transcription via Speech-to-Text (Whisper / realtime APIs).
- Detect intent and link voice input into the agent state machine.
- Generate spoken output using Text-to-Speech (TTS) pipelines.
- Support both command-style (push-to-talk) and natural conversational streaming UX.

Agents can call this tool to:

- Enable hands-free conversational assistants.
- Provide voice-accessible RAG-enhanced knowledge sessions.
- Support AI copilots embedded in mobile, web, automotive, or IoT environments.
- Execute reasoning workflows where text-typing is inefficient.

---

## Key Components

| Component | Role |
|----------|------|
| **`intergraxVoiceListener`** | Receives live or recorded audio input (microphone, WebRTC stream, uploads). |
| **`intergraxSpeechToText`** | Converts raw audio to transcription using STT providers (Whisper, Deepgram, OpenAI realtime audio). |
| **`intergraxIntentExtractor`** | Maps transcription to structured intent for agent routing. |
| **`intergraxVoiceAgent`** | Central orchestrator for voice session handling, context, and memory. |
| **`intergraxTextToSpeech`** | Generates speech audio with configurable voice models and providers. |
| **`intergraxVoiceTool`** *(MCP-compatible)* | Exposes voice capabilities to the intergrax Supervisor and chain-executing agents. |
| **`AudioStreamRouter`** | Handles real-time transport for duplex audio streams (WebSocket/WebRTC). |

Supported providers (initial phase):

- **STT:** Whisper (local), OpenAI, Deepgram  
- **TTS:** OpenAI Realtime Voice, Azure Neural Voice, ElevenLabs  

---

## Implementation Plan

| Phase | Description | Status |
|-------|------------|--------|
| 1 | Define module architecture and audio abstraction layer. | ☐ |
| 2 | Implement STT provider abstraction with first provider. | ☐ |
| 3 | Implement TTS provider abstraction and first output channel. | ☐ |
| 4 | Build `intergraxVoiceAgent` for push-to-talk conversational workflows. | ☐ |
| 5 | Add realtime streaming (WebRTC/WebSockets). | ☐ |
| 6 | Expose functionality as an MCP agent tool (`intergraxVoiceTool`). | ☐ |
| 7 | Integrate with RAG and memory for multi-turn voice conversations. | ☐ |
| 8 | Benchmark performance, latency, UX behaviour (barge-in, silence detection). | ☐ |

---

## Progress Journal

| Date | Commit / Ref | Summary |
|------|---------------|---------|
| YYYY-MM-DD |  | Designed module architecture and integration boundaries. |
| YYYY-MM-DD |  | Added prototype STT processing using Whisper. |
| YYYY-MM-DD |  | Implemented prototype TTS with basic streaming audio output. |
| YYYY-MM-DD |  | Integrated MCP interface and agent routing logic. |
| YYYY-MM-DD |  |  |

---

## Notes & Dependencies

- **Latency critical:** Streaming must operate in near-real-time (< 1s perceived delay).  
- **Security:** Voice data may contain personal context — storage and transmission must be controlled.  
- **Offline mode** planned as a future enhancement (local Whisper + local TTS engine).  
- **UX considerations:** diarization, noise filtering, endpointing, and fallback to text mode.  
- **Client support:** UI/SDK examples for CLI, Web, Flutter, and headless integrations recommended.  
- **Logging:** Debug metadata should include: audio chunk sequence, transcription timestamps, provider used, failure events.

---

## Related Documents

- `01-web-research-agent.md`  
- `conversation_memory.md`  
- `realtime_agent.md`  

---

**Maintainer:** Artur Czarnecki  
**Repository:** https://github.com/jakbuczarnecki/intergrax  
