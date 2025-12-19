# Language Model Interaction for Tiledesk

This project provides an API to interact with language models (LLMs) via API calls.

## Prerequisites

* **Python 3.13:** Ensure you have Python 3.13 installed on your system.
* **Virtual Environment:** It is recommended to use a virtual environment to isolate project dependencies.
* **Poetry:** Use Poetry for dependency management.

### Prerequisites Installation

1.  **Create a virtual environment:**

    ```bash
    python3.13 -m venv llms
    source llms/bin/activate # For Unix/macOS systems
    # venv\Scripts\activate # For Windows
    ```

2.  **Install Poetry:**

    ```bash
    curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
    ```
    or
    ```bash
    pip install poetry
    ``` 

3.  **Install project dependencies:**

    ```bash
    poetry install
    ```

## Installation

### Production Installation

After activating the virtual environment, run:

```bash
pip install .
```

or for development environment:

```bash
pip install -e .
```

# Launch

Start the service with Gunicorn. Configure worker settings via environment variables:

```bash
# Set worker configuration (adjust based on your CPU cores)
export WORKERS=3                     # Recommended: 2 * CPU cores + 1
export TIMEOUT=180                   # Worker timeout in seconds
export MAXREQUESTS=1200              # Max requests per worker before restart
export MAXRJITTER=5                  # Jitter added to max_requests
export GRACEFULTIMEOUT=30            # Graceful restart timeout

# Start the service
tilelite
```

The service will be available at `http://localhost:8000`. Use `curl` to test:

```bash
curl http://localhost:8000/
```

# Docker

```bash
sudo docker build -t tilelite .
```


```bash
sudo docker run -d -p 8000:8000 \
--env WORKERS=3 \
--env TIMEOUT=180 \
--env MAXREQUESTS=1200 \
--env MAXRJITTER=5 \
--env GRACEFULTIMEOUT=30 \
--name tilelite tilelite

### Environment Variables
The following environment variables can be set for the Docker container:

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | Number of Gunicorn workers (2*CPU+1 recommended) | 3 |
| `TIMEOUT` | Worker timeout in seconds | 180 |
| `MAXREQUESTS` | Max requests per worker before restart | 1200 |
| `MAXRJITTER` | Jitter added to max_requests | 5 |
| `GRACEFULTIMEOUT` | Graceful restart timeout | 30 |

**Note**: API keys are passed directly in the request body (`llm_key` field) rather than via environment variables.

### Docker Compose Example
```yaml
version: '3.8'
services:
  tilelite:
    build: .
    ports:
      - "8000:8000"
    environment:
      WORKERS: 3
      TIMEOUT: 180
      MAXREQUESTS: 1200
      MAXRJITTER: 5
      GRACEFULTIMEOUT: 30
    restart: unless-stopped
```

## API Endpoints

The service provides three main endpoints for interacting with LLMs:

### POST `/api/ask`
Standard LLM interaction with support for streaming and chat history.

**Request Body (QuestionToLLM):**
```json
{
  "question": "Your question here",
  "llm_key": "your-api-key",
  "llm": "openai",
  "model": "gpt-3.5-turbo",
  "temperature": 0.0,
  "max_tokens": 128,
  "top_p": 1.0,
  "stream": false,
  "system_context": "You are a helpful AI bot.",
  "chat_history_dict": null
}
```

**Parameters:**
- `question`: The user's question (required)
- `llm_key`: API key for the LLM provider (required)
- `llm`: Provider name (`openai`, `anthropic`, `cohere`, `google`, `groq`, `deepseek`, `ollama`, `vllm`) (required)
- `model`: Model identifier (see Models section) (required)
- `temperature`: Sampling temperature (0.0-1.0, default 0.0)
- `max_tokens`: Maximum tokens to generate (50-132000, default 128)
- `top_p`: Nucleus sampling parameter (0.0-1.0, default 1.0)
- `stream`: Enable streaming response (default false)
- `system_context`: System prompt (default helpful bot)
- `chat_history_dict`: Dictionary of previous chat turns (optional)
- `structured_output`: Enable structured output with Pydantic schema (default false)
- `output_schema`: Pydantic model for structured output (optional)

**Response (SimpleAnswer):**
```json
{
  "answer": "LLM response",
  "chat_history_dict": {"0": {"question": "...", "answer": "..."}},
  "prompt_token_info": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "llm_key": "sk-...",
    "llm": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
    "max_tokens": 128,
    "stream": false
  }'
```

---

### POST `/api/thinking`
LLM interaction with reasoning capabilities (thinking models). Supports OpenAI GPT-5, Anthropic Claude, Google Gemini 2.5/3.0, DeepSeek.

**Request Body (QuestionToLLM with thinking config):**
```json
{
  "question": "Complex reasoning question",
  "llm_key": "your-api-key",
  "llm": "anthropic",
  "model": "claude-3-7-sonnet-20250219",
  "temperature": 0.0,
  "max_tokens": 1024,
  "thinking": {
    "show_thinking_stream": true,
    "type": "enabled",
    "budget_tokens": 1000
  },
  "stream": false
}
```

**Thinking Configuration (ReasoningConfig):**
- `show_thinking_stream`: Whether to show thinking content in stream (default true)
- Provider-specific parameters:
  - OpenAI GPT-5: `reasoning_effort` (`low`, `medium`, `high`), `reasoning_summary` (`auto`, `always`, `never`)
  - Anthropic Claude: `type` (`enabled`, `disabled`), `budget_tokens` (0-100000)
  - Google Gemini 2.5: `thinkingBudget` (-1=dynamic, 0=disabled, max 32000)
  - Google Gemini 3.0: `thinkingLevel` (`low`, `medium`, `high`)

**Response (ReasoningAnswer):**
```json
{
  "answer": "Final answer",
  "reasoning_content": "Model's reasoning process",
  "chat_history_dict": {...},
  "prompt_token_info": {...}
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/thinking \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Solve the equation: 2x + 5 = 15",
    "llm_key": "sk-...",
    "llm": "anthropic",
    "model": "claude-3-7-sonnet-20250219",
    "temperature": 0.0,
    "max_tokens": 1024,
    "thinking": {
      "show_thinking_stream": true,
      "type": "enabled",
      "budget_tokens": 1000
    }
  }'
```

---

### POST `/api/mcp-agent`
LLM agent with Model Context Protocol (MCP) tools. Connects to MCP servers for tool-enhanced interactions.

**Request Body (QuestionToMCPAgent):**
```json
{
  "question": "What's the weather in Tokyo?",
  "llm_key": "your-api-key",
  "llm": "openai",
  "model": "gpt-3.5-turbo",
  "temperature": 0.0,
  "max_tokens": 1024,
  "system_context": "You are a helpful assistant with access to tools.",
  "servers": {
    "weather": {
      "transport": "sse",
      "url": "http://weather-mcp-server:8000/sse"
    }
  }
}
```

**MCP Server Configuration:**
- `transport`: Connection type (`sse`, `stdio`, `websocket`)
- `url`: Server URL (for SSE/WebSocket)
- `command`/`args`: Command to execute (for stdio)
- `api_key`: Optional API key for the server
- `parameters`: Additional parameters

**Response (SimpleAnswer):**
```json
{
  "answer": "The weather in Tokyo is sunny, 22°C.",
  "tools_log": ["Tool call results..."],
  "chat_history_dict": {},
  "prompt_token_info": {...}
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/mcp-agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What'\''s the weather in Tokyo?",
    "llm_key": "sk-...",
    "llm": "openai",
    "model": "gpt-3.5-turbo",
    "temperature": 0.0,
    "max_tokens": 1024,
    "servers": {
      "weather": {
        "transport": "sse",
        "url": "http://weather-mcp-server:8000/sse"
      }
    }
  }'
```

---

## Models
Models for /api/ask

### OpenAI - engine: openai
**Frontier Models:**
- gpt-5.2 (latest flagship model)
- gpt-5.2-pro (enhanced version)
- gpt-5-mini (cost-efficient)
- gpt-5-nano (fastest, most cost-efficient)
- gpt-5.1 (previous generation)

**Production Models:**
- gpt-4.1 (smartest non-reasoning model)
- gpt-4.1-mini (smaller, faster version)
- gpt-4.1-nano (fastest version)

**Legacy Models:**
- gpt-4o (fast, intelligent model)
- gpt-4o-mini (affordable small model)
- gpt-4-turbo (older high-intelligence)
- gpt-3.5-turbo (legacy, cheap)

### Google - engine: google
**Gemini 3.0 Series:**
- gemini-3.0-pro (latest flagship)
- gemini-3.0-flash (fast, efficient)
- gemini-3.0-flash-thinking (with reasoning)

**Gemini 2.5 Series:**
- gemini-2.5-pro (previous generation)
- gemini-2.5-flash (fast version)

**Gemini 2.0 Series:**
- gemini-2.0-flash (efficient)

### Anthropic - engine: anthropic
**Claude 4.5 Series:**
- claude-sonnet-4-5-20250929
- claude-haiku-4-5-20251001
- claude-opus-4-5-20251101

**Claude 4.0 Series:**
- claude-sonnet-4-20250514
- claude-opus-4-20250514
- claude-opus-4-1-20250805

**Claude 3.7 Series:**
- claude-3-7-sonnet-20250219 (latest flagship)

### Groq - engine: groq
**Production Models:**
- llama-3.3-70b-versatile (latest flagship)
- llama-3.1-8b-instant (fast, efficient)
- openai/gpt-oss-120b (OpenAI open-weight)
- openai/gpt-oss-20b (smaller open-weight)

**Preview Models:**
- meta-llama/llama-4-maverick-17b-128e-instruct
- meta-llama/llama-4-scout-17b-16e-instruct
- qwen/qwen3-32b

### Cohere - engine: cohere
**Command Series:**
- command-r8 (latest flagship)
- command-r7b (previous generation)
- command-a (alternative)

### DeepSeek - engine: deepseek
- deepseek-chat (general purpose)
- deepseek-coder (code specialized)

### Ollama - engine: ollama
Ollama supports local models running via Ollama server. Use the `LocalModel` type with Ollama server URL:

**LocalModel Configuration:**
```json
{
  "name": "llama3.2",
  "url": "http://localhost:11434",
  "dimension": 1024
}
```

**Supported Models:**
- Any model available in Ollama library (llama3.2, mistral, codellama, etc.)
- Custom models imported into Ollama
- Specify Ollama server URL in the `url` field

### vLLM - engine: vllm
vLLM supports any local model with OpenAI-compatible API. Use the `LocalModel` type with custom configuration:

**LocalModel Configuration:**
```json
{
  "name": "codellama/CodeLlama-7b-Instruct-hf",
  "url": "http://localhost:8000/v1",
  "dimension": 1024
}
```

**Supported Models:**
- Any model hosted with vLLM server (Llama, Mistral, CodeLlama, etc.)
- Custom models with OpenAI-compatible API
- Specify vLLM server URL in the `url` field



