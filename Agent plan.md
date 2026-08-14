This phased roadmap structures your continuous learning agentic harness according to your 11 steps. It addresses your Phase 2 context retrieval question and outlines the engineering specs for each phase.

### **Solution to Phase 2 Open Question: Attending Chat History Without Context Inflation**

To query long historical conversations without filling up the context window:

> 1. **Two-Tier Storage:**  
   * **Active Working Context (In-Memory Ring Buffer):** Holds the last $N$ raw message turns.  
   * **Persisted Canonical Log (Database):** Every incoming/outgoing raw message is recorded in SQLite or PostgreSQL with timestamps and metadata.  
> 2. **Task Summarizer \+ Active Reflection:**  
   * The Ring Buffer feeds a **Task Summarizer** prompt that compresses the ongoing dialogue into a structured TASK.md prompt ($\\Phi$).  
   * If the user references past information not present in the current $N$ turns (e.g., *"Use the database credential I gave you last week"*), the Task Summarizer identifies missing information and generates a historical retrieval query.  
> 3. **Hybrid Search Tool (search\_history):**  
   * Do **not** inject all chat history into every request. Instead, treat chat history as an **external tool available to the Task Summarizer and Agent**:

     * **Full-Text Search (FTS):** SQLite FTS5 or Postgres pg\_trgm for exact keyword matches (API keys, URLs, usernames, file names).  
     * **Vector Embeddings (Dense Search):** Local embedding model (e.g., bge-small-en-v1.5) for semantic similarity searches over past turn chunks.  
   * When needed, the agent executes search\_history(query="database credential", limit=3) to retrieve *only* the top-k relevant turns into its active context.

## **Technical Roadmap**

\+-----------------------------------------------------------------------------------+  
|                               PHASED ARCHITECTURE                                 |  
\+-----------------------------------------------------------------------------------+  
| \[Phases 1-3\] Core Bot, Besedka API, Task Buffer, Retrieval, Tool Engine          |  
| \[Phases 4-6\] Docker Sandbox, Trajectory Exporter, Durable FSM Scheduler          |  
| \[Phases 7-10\] Open-Weights Inference, PESO Python Pipeline, Validation, Reloading  |  
| \[Phase 11\]  Autonomous "Dreaming", Game Sandboxes, RPE Curiosity Engine       |  
\+-----------------------------------------------------------------------------------+

### **Phase 1: Simple Request/Response Agent & Besedka Gateway**

* **Besedka Ingress (Go):** Implement a client in Go that connects to the Besedka Bot API endpoint (/api/v1/...). Listen for message events, filter incoming payloads, and react **only** when the message explicitly directly mentions the bot (@botusername).  
* **OpenAI API Client Engine (Go):** Wrap standard HTTP requests targeting OpenAI-compatible REST endpoints. Initially configure base URL and API keys to route to Google Gemini's OpenAI-compatible API (\[https://generativelanguage.googleapis.com/v1beta/openai/\](https://generativelanguage.googleapis.com/v1beta/openai/)).  
* **Basic Execution Pipeline:**  
  1. Receive mention event from Besedka (websocket with rest API fallback).  
  2. Strip bot handle and extract user input $x$.  
  3. Send request to Gemini endpoint.  
  4. Post final response back via Besedka API.

### **Phase 2: Ring Buffer, Task Summarizer, & Hybrid History Retrieval**

* **Ring Buffer (Go):** Maintain a thread-safe circular queue holding the last $N$ message turns per chat/user.  
* **Task Summarizer Engine:**  
  * Before generating a final answer, pass the Ring Buffer contents to a lightweight LLM summarization call with system prompt:*"Compress the conversation into a single canonical task description (TASK.md). Preserve explicit constraints, code snippets, and active goals. If crucial information is referenced but missing, output \<SEARCH\_REQUIRED: 'search query'\>."*  
  * Subsequent user turns trigger an update pass to mutate or replace the active TASK.md.  
* **Database Persister:** Write every turn to SQLite/PostgreSQL. Implement search\_history(query, method="hybrid") combining FTS5 lexical matching and local dense embedding retrieval.

### **Phase 3: Tool Execution Framework & Multimodal Attachments**

* **Tool Calling Interface:** Implement Go JSON-Schema parser for OpenAI tool-calling primitives (tools and tool\_calls).  
* **Multimodal Attachments:**  
  * Parse incoming image/file attachments from Besedka message payloads.  
  * Implement an image generation tool (e.g., calling Imagen / FLUX / DALL-E endpoints) returning generated media back to Besedka via multipart file uploads.  
  * Implement file extraction tools (parsing .txt, .pdf, .json, .csv) and mounting them into working memory.

### **Phase 4: Pluggable Environment Sandboxes (Docker Driver)**

* **Environment Interface:** Create the Go Environment interface abstraction (Init, Exec, InjectFile, ReadFile, Close).  
* **Docker Environment Driver:**  
  * Implement using the official \[github.com/docker/docker/client\](https://github.com/docker/docker/client) Go SDK.  
  * On task execution, spawn a temporary container (e.g., alpine:latest or ubuntu:22.04) with constrained limits (Memory=512MB, CPU=1.0, Network=Disabled or whitelist proxy, ReadOnlyRootfs=false with restricted tmpfs).  
  * Expose tools to the agent: bash\_exec, read\_file, write\_file.  
  * Ensure teardown and volume cleanup on session completion or timeout.

### **Phase 5: Trajectory Logging & Fast-Slow Data Collection**

* **Trajectory Logger:** Record full interaction sessions in Go as structured JSON objects:

  $$\\tau \= \\big(\\text{session\\\_id}, x, \\Phi\_{\\text{task}}, z, a, o, y, R\_{\\text{eval}}\\big)$$  
* **Reward Engine:**  
  * Calculate scalar scores $R\_{\\text{eval}}$ based on tool execution exit codes (0 vs non-zero), user explicit feedback (thumbs up/down or text corrections), and guardrail checks.

* **Storage Exporter:** Asynchronously push JSON trajectory blobs to object storage (S3/MinIO) to serve as training samples for the future Python pipeline.

### **Phase 6: Durable FSM & Task Scheduler**

* **Durable State Machine:** Implement a persistent Finite State Machine (FSM) in Go (or integrate a lightweight workflow engine like Temporal / River / SQLite-backed queue).  
* **Background & Scheduled Tasks:**  
  * Allow the agent to register background jobs: schedule\_task(cron\_expr, instruction).  
  * The scheduler executes tasks out-of-band inside Docker environments without blocking the main chat interface, posting completion notifications back to the user's Besedka chat.  
* **State Recovery:** Ensure running agent tasks survive binary restarts by checkpointing active execution nodes to disk.

### **Phase 7: Local Open-Weights Inference Backend**

* **Inference Engine Setup:** Deploy an open-weights inference server (such as vLLM or SGLang) hosting a foundation model (e.g., Qwen3.6-35B-A3B or gemma-4-26B-A4B).  
* **OpenAI Proxy Compatibility:** Ensure the server exposes standard /v1/chat/completions endpoints with support for dynamic LoRA adapter mounting (/v1/load\_lora\_adapter).  
* **Go Harness Switch:** Update the Go binary config to target the local inference backend instead of Gemini. Can still use external models for more demanding tasks to generate higher quality traces. Not sure if gemini will fit here as it does not expose full reasoning AFAIK. Maybe some top open weights models like Qwen-3.8, Kimi K3 accessed via API (openrouter) would be a better option.

### **Phase 8: Offline FSL & PESO Training Pipeline**

* **Python Training Service:** Build a standalone Python service using PyTorch, Hugging Face peft, and the **PESO** algorithm repository.

* **Dataset Ingestion:** Filter trajectories from Phase 5 to extract high-reward execution steps and user correction patterns.

* **PESO Optimization Loop:**  
  * Train a single evolving LoRA adapter using the proximal regularizer loss:

    $$\\mathcal{L}\_{\\text{PESO}}(\\theta) \= \\mathcal{L}\_{\\text{task}}(\\theta) \+ \\lambda \\mathcal{R}\_{\\text{prox}}(\\theta, \\theta\_{\\text{prev}})$$  
  * The regularizer anchors parameters to $\\theta\_{\\text{prev}}$, preserving core model capabilities while updating task directions.

### **Phase 9: Automated LoRA Validation & Benchmarking**

* **Evaluation Pipeline:** Before any newly trained LoRA adapter is approved, the Python service runs an automated test suite:  
  1. **Task Benchmark:** Evaluates execution success on a held-out set of command execution and tool-routing tasks.  
  2. **Catastrophic Forgetting Check:** Measures performance on general reasoning and code generation to ensure no policy collapse occurred.  
  3. **Safety Alignment Probe:** Runs prompt-injection and guardrail test suites to verify safety boundaries have not drifted.  
* **Gatekeeper:** Output a binary PASS/FAIL report alongside quality metrics.

### **Phase 10: Dynamic LoRA Deployment & Hot-Reloading**

* **Deployment Orchestrator:**  
  * Upon a PASS validation score, upload the new LoRA weights to the inference server storage.  
  * Send an API request to vLLM/SGLang to load/hot-reload the adapter.  
* **Harness Model Router:** Update the Go harness active model configuration key dynamically, routing subsequent chat requests through the updated LoRA adapter without restarting the Go binary.

### **Phase 11: "Dream" Self-Training & Autonomous Curiosity**

* **Toy / Game Environment Drivers:**  
  * Implement game drivers satisfying the Go Environment interface (e.g., Gym/PettingZoo wrappers, text adventure engines, or custom strategy gridworlds).  
* **Autonomous Offline Rollouts ("Dreaming"):**  
  * During idle hours, the harness spawns autonomous agent sessions inside game/toy sandboxes without user interaction.  
* **Reward Prediction Error (RPE) Curiosity Engine:**  
  * Implement an internal world-model critic that measures prediction error between expected tool outcomes and actual sandbox state changes.  
  * High-RPE (unexpected/surprising) trajectories are flagged as valuable learning moments and prioritized for offline PESO training passes, encouraging holistic skill development.

### **Phase 12: Skills support**

Support loading skills based on the current task

### **Phase 13: RAG for knowledge**

Accumulate facts/knowledge/skills and provide RAG search tool to the agent.  
