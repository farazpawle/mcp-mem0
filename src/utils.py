import os
import sys
import logging

# CRITICAL: Configure logging BEFORE importing mem0 to prevent stdout pollution
# MCP uses stdout for JSON-RPC, so any log output there corrupts the protocol
logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format='%(name)s - %(levelname)s - %(message)s'
)
# Suppress noisy loggers that would otherwise corrupt the MCP protocol
logging.getLogger('mem0').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

from mem0 import Memory

# Custom instructions for memory processing
# Optimized for "Vibe Coding" - keeping memory clean, structured, and actionable.
CUSTOM_INSTRUCTIONS = """
You are a memory extraction system for a developer's coding assistant.
Your job is to store only durable, high-signal information that prevents future mistakes.

Only store memories in one of these 3 types:
- CONVENTION: coding standards, patterns, preferred tools/commands
- DECISION: architectural choices and the reasons behind them
- GOTCHA: sharp edges, failure modes, and fixes

Output each memory in this plain-text mini-schema and keep it concise:
TYPE: CONVENTION | DECISION | GOTCHA
SCOPE: repo/module/feature
TRIGGER: when to recall this
CONTENT: the actual rule/decision/gotcha
LAST_VALIDATED: YYYY-MM-DD
REVIEW_AFTER: YYYY-MM-DD (optional)
EXPIRES: YYYY-MM-DD (optional)

Hard rules:
- Do not store diary-like chatter or generic facts.
- Do not store secrets (API keys, tokens, passwords, connection strings, private keys). Store procedures instead.
- Prefer updating/replacing overlapping memories rather than adding duplicates.
"""

def get_mem0_client():
    # Get LLM provider and configuration
    llm_provider = os.getenv('LLM_PROVIDER')
    llm_api_key = os.getenv('LLM_API_KEY')
    llm_model = os.getenv('LLM_CHOICE')
    embedding_model = os.getenv('EMBEDDING_MODEL_CHOICE')
    
    # Initialize config dictionary
    config = {}
    
    # Configure LLM based on provider
    if llm_provider == 'openai' or llm_provider == 'openrouter':
        # For OpenRouter, set the specific API key
        if llm_provider == 'openrouter' and llm_api_key:
            os.environ["OPENROUTER_API_KEY"] = llm_api_key

        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        if llm_provider == 'openrouter':
             config["llm"]["config"]["openai_base_url"] = "https://openrouter.ai/api/v1"
             config["llm"]["config"]["api_key"] = llm_api_key
        
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key
            
    elif llm_provider == 'ollama':
        config["llm"] = {
            "provider": "ollama",
            "config": {
                "model": llm_model,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
        }
        
        # Set base URL for Ollama if provided
        llm_base_url = os.getenv('LLM_BASE_URL')
        if llm_base_url:
            config["llm"]["config"]["ollama_base_url"] = llm_base_url
    
    # Configure embedder based on provider
    # Determine dimensions based on model name
    if "multilingual-e5-large" in (embedding_model or ""):
        dims = 1024
    elif "nomic-embed-text" in (embedding_model or ""):
        dims = 768
    elif "text-embedding-3" in (embedding_model or ""):
        dims = 1536
    else:
        # Fallback logic
        dims = 1536 if llm_provider == "openai" or llm_provider == "openrouter" else 768

    if llm_provider == 'openai':
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": dims
            }
        }
        
        # Set API key in environment if not already set
        if llm_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = llm_api_key

    elif llm_provider == 'openrouter':
        # OpenRouter uses OpenAI-compatible API for embeddings
        config["embedder"] = {
            "provider": "openai",
            "config": {
                "model": embedding_model or "text-embedding-3-small",
                "embedding_dims": dims,
                "openai_base_url": "https://openrouter.ai/api/v1",
                "api_key": llm_api_key
            }
        }
    
    elif llm_provider == 'ollama':
        config["embedder"] = {
            "provider": "ollama",
            "config": {
                "model": embedding_model or "nomic-embed-text",
                "embedding_dims": dims
            }
        }
        
        # Set base URL for Ollama if provided
        embedding_base_url = os.getenv('LLM_BASE_URL')
        if embedding_base_url:
            config["embedder"]["config"]["ollama_base_url"] = embedding_base_url
    
    # Configure Supabase vector store
    # print(f"DEBUG: Model={embedding_model}, Dims={dims}")
    config["vector_store"] = {
        "provider": "supabase",
        "config": {
            "connection_string": os.environ.get('DATABASE_URL', ''),
            "collection_name": "mem0_memories_main",
            "embedding_model_dims": dims
        }
    }

    config["custom_prompt"] = CUSTOM_INSTRUCTIONS
    
    # Create and return the Memory client
    return Memory.from_config(config)