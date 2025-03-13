# Language Model Interaction for Tiledesk

This project provides an API to interact with language models (LLMs) via API calls.

## Prerequisites

* **Python 3.12:** Ensure you have Python 3.12 installed on your system.
* **Virtual Environment:** It is recommended to use a virtual environment to isolate project dependencies.
* **Poetry:** Use Poetry for dependency management.

### Prerequisites Installation

1.  **Create a virtual environment:**

    ```bash
    python3.12 -m venv llms
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

```bash

export WORKERS=INT number of workers 2*CPU+1
export TIMEOUT=INT seconds of timeout default=180
export MAXREQUESTS=INT The maximum number of requests a worker will process before restarting. deafult=1200
export MAXRJITTER=INT The maximum jitter to add to the max_requests setting default=5
export GRACEFULTIMEOUT=INT Timeout for graceful workers restart default=30 
tilelite
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
```


## Models
Models for /api/ask

### OpenAI - engine: openai
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo
- got-4o
- got-4o-mini

### Cohere - engine: cohere
- command-r
- command-r-plus

### Google - engine: google
- gemini-pro
- gemini-1.5-flash
- gemini-2-0-flash

### Anthropic - engine: anthropic
- claude-3-5-sonnet-20240620
- claude-3-5-sonnet-20241022
- claude-3-7-sonnet-20250219

### Groq - engine: groq
- llama3-70b-8192
- llama3-8b-8192
- llama-3.1-8b-instant
- llama-3.1-70b-versatile
- Gemma-7b-It
- ...



