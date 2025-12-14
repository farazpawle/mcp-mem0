<h1 align="center">MCP-Mem0: Long-Term Memory for AI Agents</h1>

<p align="center">
  <img src="public/Mem0AndMCP.png" alt="Mem0 and MCP Integration" width="600">
</p>

A template implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server integrated with [Mem0](https://mem0.ai) for providing AI agents with persistent memory capabilities.

Use this as a reference point to build your MCP servers yourself, or give this as an example to an AI coding assistant and tell it to follow this example for structure and code correctness!

## Overview

This project demonstrates how to build an MCP server that enables AI agents to store, retrieve, and search memories using semantic search. It serves as a practical template for creating your own MCP servers, simply using Mem0 and a practical example.

The implementation follows the best practices laid out by Anthropic for building MCP servers, allowing seamless integration with any MCP-compatible client.

## Features

The server provides memory tools designed for both traditional “store/search” and for low-friction **vibe coding**.

### Core tools

1. **`save_memory`**: Store information in long-term memory with semantic indexing (optionally with `file_path` / `tags`)
2. **`get_all_memories`**: Retrieve all stored memories for comprehensive context
3. **`search_memories`**: Find relevant memories using semantic search
4. **`delete_memory`**: Remove an outdated or incorrect memory by ID

### Vibe coding tools (recommended)

These tools make memory more reliable by keeping it structured, scoped, and clean:

1. **`save_vibe_memory`**: Save a structured memory as one of `CONVENTION`, `DECISION`, or `GOTCHA` (with deduplication)
2. **`get_session_briefing`**: “Start-of-session” compact briefing (top relevant conventions/decisions/gotchas)
3. **`seed_project_from_files`**: One-shot onboarding by reading repo files (defaults: `pyproject.toml,README.md`) and saving starter memories
4. **`correct_memory`**: Mark a wrong/stale memory as refuted and save a replacement (prevents repeated agent mistakes)
5. **`get_vibe_status`**: Markdown dashboard of current `CONVENTION`/`DECISION`/`GOTCHA` plus “needs review” / “expired”

## Vibe Coding: How to use this server

### 1) Scope your memories (avoid cross-project contamination)

Memories are scoped by `project_name` (internally used as Mem0 `user_id`).

- Use a unique `project_name` per repo/workspace (example: `mycompany-website`, `mcp-mem0-server`, etc.)
- If you reuse the same `project_name` across different repos, memories will mix

### 2) Use only 3 memory types

To keep memory clean and searchable, vibe memories are stored in a mini-schema:

```
TYPE: CONVENTION | DECISION | GOTCHA
SCOPE: repo/module/feature
TRIGGER: when to recall this
CONTENT: the actual rule/decision/gotcha
LAST_VALIDATED: YYYY-MM-DD
REVIEW_AFTER: YYYY-MM-DD (optional)
EXPIRES: YYYY-MM-DD (optional)
```

### 3) Recommended workflow

1. **First time in a repo**: run `seed_project_from_files` using that repo’s `project_name`
2. **At the start of every session**: call `get_session_briefing`
3. **After agreeing on an approach**: call `save_vibe_memory` as a `DECISION`
4. **After a fix lands**: call `save_vibe_memory` as a `GOTCHA` (root cause + fix)
5. **If the agent gets something wrong**: call `correct_memory` once, and move on

### 4) Safety: “no secrets” policy

Do **not** store secrets in memory (API keys, tokens, passwords, private keys, full connection strings). Store procedures instead.

Example:
- ✅ “Credentials live in `.env` under `LLM_API_KEY`”
- ❌ “My OpenRouter key is …”

## Prerequisites

- Python 3.12+
- Supabase or any PostgreSQL database (for vector storage of memories)
- API keys for your chosen LLM provider (OpenAI, OpenRouter, or Ollama)
- Docker if running the MCP server as a container (recommended)

## Installation

### Using uv

1. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/coleam00/mcp-mem0.git
   cd mcp-mem0
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

5. Configure your environment variables in the `.env` file (see Configuration section)

### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t mcp/mem0 --build-arg PORT=8050 .
   ```

2. Create a `.env` file based on `.env.example` and configure your environment variables

## Configuration

The following environment variables can be configured in your `.env` file:

| Variable | Description | Example |
|----------|-------------|----------|
| `TRANSPORT` | Transport protocol (sse or stdio) | `sse` |
| `HOST` | Host to bind to when using SSE transport | `0.0.0.0` |
| `PORT` | Port to listen on when using SSE transport | `8050` |
| `LLM_PROVIDER` | LLM provider (openai, openrouter, or ollama) | `openai` |
| `LLM_BASE_URL` | Base URL for the LLM API | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for the LLM provider | `sk-...` |
| `LLM_CHOICE` | LLM model to use | `gpt-4o-mini` |
| `EMBEDDING_MODEL_CHOICE` | Embedding model to use | `text-embedding-3-small` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:port/db` |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
uv run src/main.py
```

The MCP server will essentially be run as an API endpoint that you can then connect to with config shown below.

#### Stdio Transport

With stdio, the MCP client iself can spin up the MCP server, so nothing to run at this point.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p:8050:8050 mcp/mem0
```

The MCP server will essentially be run as an API endpoint within the container that you can then connect to with config shown below.

#### Stdio Transport

With stdio, the MCP client iself can spin up the MCP server container, so nothing to run at this point.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "mem0": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "mem0": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

> **Note for n8n users**: Use host.docker.internal instead of localhost since n8n has to reach outside of it's own container to the host machine:
> 
> So the full URL in the MCP node would be: http://host.docker.internal:8050/sse

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "mem0": {
      "command": "your/path/to/mcp-mem0/.venv/Scripts/python.exe",
      "args": ["your/path/to/mcp-mem0/src/main.py"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "LLM_CHOICE": "gpt-4o-mini",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small",
        "DATABASE_URL": "YOUR-DATABASE-URL"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "mem0": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT", 
               "-e", "LLM_PROVIDER", 
               "-e", "LLM_BASE_URL", 
               "-e", "LLM_API_KEY", 
               "-e", "LLM_CHOICE", 
               "-e", "EMBEDDING_MODEL_CHOICE", 
               "-e", "DATABASE_URL", 
               "mcp/mem0"],
      "env": {
        "TRANSPORT": "stdio",
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": "https://api.openai.com/v1",
        "LLM_API_KEY": "YOUR-API-KEY",
        "LLM_CHOICE": "gpt-4o-mini",
        "EMBEDDING_MODEL_CHOICE": "text-embedding-3-small",
        "DATABASE_URL": "YOUR-DATABASE-URL"
      }
    }
  }
}
```

## Building Your Own Server

This template provides a foundation for building more complex MCP servers. To build your own:

1. Add your own tools by creating methods with the `@mcp.tool()` decorator
2. Create your own lifespan function to add your own dependencies (clients, database connections, etc.)
3. Modify the `utils.py` file for any helper functions you need for your MCP server
4. Feel free to add prompts and resources as well  with `@mcp.resource()` and `@mcp.prompt()`
