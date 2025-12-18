import os
import sys
import logging

# CRITICAL: Hijack stdout IMMEDIATELY to prevent any library from polluting it.
# MCP protocol requires stdout to be pure JSON-RPC.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

# Configure logging to go to stderr
logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, os.getenv("LOG_LEVEL", "WARNING").upper(), logging.WARNING),
    format='%(name)s - %(levelname)s - %(message)s'
)

# Suppress all known noisy loggers
for logger_name in [
    'mem0', 'mem0.vector_stores', 'mem0.vector_stores.supabase',
    'httpx', 'httpcore', 'openai', 'anthropic', 'google.generativeai'
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Now import other modules after logging is configured
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dotenv import load_dotenv
from mem0 import Memory
import asyncio
import json
import datetime as _dt
import pathlib
import tomllib

from vibe_memory import (
    build_vibe_memory,
    build_vibe_metadata,
    detect_potential_secret,
    format_vibe_text,
    parse_vibe_text,
    today_yyyy_mm_dd,
)

from utils import get_mem0_client

load_dotenv()

# Default user ID for memory operations
DEFAULT_USER_ID = "user"

# Create a dataclass for our application context
@dataclass
class Mem0Context:
    """Context for the Mem0 MCP server."""
    mem0_client: Memory

@asynccontextmanager
async def mem0_lifespan(server: FastMCP) -> AsyncIterator[Mem0Context]:
    """
    Manages the Mem0 client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Mem0Context: The context containing the Mem0 client
    """
    # Create and return the Memory client with the helper function in utils.py
    mem0_client = get_mem0_client()
    
    try:
        yield Mem0Context(mem0_client=mem0_client)
    finally:
        # No explicit cleanup needed for the Mem0 client
        pass

# Initialize FastMCP server
# We must pass the original stdout to the server so it can communicate.
mcp = FastMCP(
    "mcp-mem0",
    lifespan=mem0_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)

# Note: FastMCP handles its own stdio transport, but we've redirected sys.stdout
# globally. We need to ensure the underlying transport uses the real stdout.
# Since we are using FastMCP, it typically handles this, but it's safer to 
# ensure the global sys.stdout is restored just before the server starts.


def _extract_results(maybe_dict_or_list):
    if isinstance(maybe_dict_or_list, dict) and "results" in maybe_dict_or_list:
        return maybe_dict_or_list.get("results") or []
    return maybe_dict_or_list or []


def _find_memory_by_id(*, mem0_client: Memory, project_name: str, memory_id: str) -> dict | None:
    for item in _extract_results(mem0_client.get_all(user_id=project_name)):
        if isinstance(item, dict) and item.get("id") == memory_id:
            return item
    return None


def _parse_date(value: str | None) -> _dt.date | None:
    if not value:
        return None
    try:
        return _dt.date.fromisoformat(value.strip())
    except Exception:
        return None


def _read_text_file(path: pathlib.Path, *, max_bytes: int = 250_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _safe_path(repo_root: str, relative_or_abs: str) -> pathlib.Path:
    root = pathlib.Path(repo_root).resolve()
    p = pathlib.Path(relative_or_abs)
    if not p.is_absolute():
        p = (root / p).resolve()
    # Basic containment check (prevents reading outside repo root accidentally)
    try:
        p.relative_to(root)
    except Exception:
        raise ValueError(f"Refusing to read outside repo_root: {p}")
    return p


def _best_effort_replace_memory(
    *,
    mem0_client: Memory,
    project_name: str,
    existing_memory_id: str,
    new_text: str,
    new_metadata: dict,
) -> tuple[str, str | None]:
    """Replace/update a memory while trying to preserve metadata.

    Returns:
        (action, new_memory_id)
    """
    # Prefer private update that can carry metadata when available.
    try:
        update_fn = getattr(mem0_client, "_update_memory", None)
        embedding_model = getattr(mem0_client, "embedding_model", None)
        if callable(update_fn) and embedding_model is not None:
            existing_embeddings = {new_text: embedding_model.embed(new_text, "update")}
            update_fn(existing_memory_id, new_text, existing_embeddings, metadata=new_metadata)
            return ("updated", existing_memory_id)
    except Exception as exc:
        logger.warning("Failed metadata update for memory_id=%s: %s", existing_memory_id, exc)

    # Fall back to replace so metadata stays accurate.
    try:
        mem0_client.delete(existing_memory_id)
    except Exception as exc:
        logger.warning("Failed to delete memory_id=%s before replace: %s", existing_memory_id, exc)

    try:
        result = mem0_client.add(
            [{"role": "user", "content": new_text}],
            user_id=project_name,
            metadata={**new_metadata, "replaces_memory_id": existing_memory_id},
        )
        # mem0 may return dict/list; the ID is not consistently exposed.
        new_id = None
        if isinstance(result, dict):
            new_id = result.get("id")
        elif isinstance(result, list) and result:
            new_id = result[0].get("id") if isinstance(result[0], dict) else None
        return ("replaced", new_id)
    except Exception as exc:
        logger.exception("Failed to replace memory_id=%s: %s", existing_memory_id, exc)
        raise


def _select_similar_candidate(
    *,
    mem0_client: Memory,
    project_name: str,
    memory_type: str,
    scope: str,
    content: str,
    limit: int,
    score_threshold: float,
) -> str | None:
    """Find a likely duplicate memory id using semantic search + schema checks."""
    query = f"TYPE: {memory_type} SCOPE: {scope} {content}"
    try:
        search_results = mem0_client.search(query, user_id=project_name, limit=limit)
    except Exception as exc:
        logger.warning("Search failed during dedupe: %s", exc)
        return None

    for item in _extract_results(search_results):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if score is not None:
            try:
                if float(score) < score_threshold:
                    continue
            except Exception:
                # If score isn't numeric, ignore thresholding.
                pass

        text = item.get("memory", "")
        fields = parse_vibe_text(text)
        if fields.get("TYPE") == memory_type and fields.get("SCOPE") == scope:
            return item.get("id")

        meta = item.get("metadata") or {}
        if meta.get("memory_type") == memory_type and meta.get("scope") == scope:
            return item.get("id")

    return None


def _save_vibe_memory_impl(
    *,
    mem0_client: Memory,
    project_name: str,
    memory_type: str,
    scope: str,
    trigger: str | None,
    content: str,
    file_path: str | None,
    tags: str | None,
    last_validated: str | None,
    review_after: str | None,
    expires: str | None,
    dedupe: bool,
    dedupe_mode: str,
    dedupe_limit: int,
    dedupe_score_threshold: float,
) -> str:
    secret_reason = detect_potential_secret(content)
    if secret_reason:
        return (
            "Refusing to store potential secret. "
            f"Detected: {secret_reason}. Store the env var name/procedure, not the value."
        )

    vibe_memory = build_vibe_memory(
        memory_type=memory_type,
        scope=scope,
        trigger=trigger,
        content=content,
        last_validated=last_validated,
        review_after=review_after,
        expires=expires,
    )
    vibe_text = format_vibe_text(vibe_memory)
    metadata = build_vibe_metadata(memory=vibe_memory, file_path=file_path, tags=tags)

    if dedupe:
        existing = _extract_results(mem0_client.get_all(user_id=project_name))
        vibe_key = metadata.get("vibe_key")

        for item in existing:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata") or {}
            if vibe_key and meta.get("vibe_key") == vibe_key:
                existing_id = item.get("id")
                if not existing_id:
                    break
                if dedupe_mode.lower() == "skip":
                    return f"Memory already exists (dedupe=skip). id={existing_id}"
                action, new_id = _best_effort_replace_memory(
                    mem0_client=mem0_client,
                    project_name=project_name,
                    existing_memory_id=existing_id,
                    new_text=vibe_text,
                    new_metadata=metadata,
                )
                if action == "updated":
                    return f"Updated existing vibe memory. id={existing_id}"
                return (
                    f"Replaced existing vibe memory. old_id={existing_id}"
                    + (f" new_id={new_id}" if new_id else "")
                )

        if dedupe_mode.lower() in {"replace_similar", "update_similar", "replace"}:
            candidate_id = _select_similar_candidate(
                mem0_client=mem0_client,
                project_name=project_name,
                memory_type=vibe_memory.memory_type,
                scope=vibe_memory.scope,
                content=vibe_memory.content,
                limit=dedupe_limit,
                score_threshold=dedupe_score_threshold,
            )
            if candidate_id:
                action, new_id = _best_effort_replace_memory(
                    mem0_client=mem0_client,
                    project_name=project_name,
                    existing_memory_id=candidate_id,
                    new_text=vibe_text,
                    new_metadata=metadata,
                )
                if action == "updated":
                    return f"Updated similar vibe memory. id={candidate_id}"
                return (
                    f"Replaced similar vibe memory. old_id={candidate_id}"
                    + (f" new_id={new_id}" if new_id else "")
                )

    mem0_client.add(
        [{"role": "user", "content": vibe_text}],
        user_id=project_name,
        metadata=metadata,
    )

    # Best-effort canonicalization: Mem0 may rewrite the memory text during extraction.
    # We re-find the stored record by vibe_key and update its text/metadata to our schema.
    try:
        vibe_key = metadata.get("vibe_key")
        if vibe_key:
            candidates = []
            for item in _extract_results(mem0_client.get_all(user_id=project_name)):
                if not isinstance(item, dict):
                    continue
                meta = item.get("metadata") or {}
                if meta.get("vibe_key") == vibe_key:
                    candidates.append(item)

            if candidates:
                # Pick the most recent record if possible.
                def _sort_key(i: dict):
                    return str(i.get("created_at") or "")

                candidates.sort(key=_sort_key, reverse=True)
                canonical_id = candidates[0].get("id")
                if canonical_id:
                    _best_effort_replace_memory(
                        mem0_client=mem0_client,
                        project_name=project_name,
                        existing_memory_id=canonical_id,
                        new_text=vibe_text,
                        new_metadata=metadata,
                    )
    except Exception as exc:
        logger.warning("Failed to canonicalize vibe memory: %s", exc)

    return f"Saved vibe memory ({vibe_memory.memory_type}) to project '{project_name}'."


def _seed_from_pyproject(*, content: str) -> list[tuple[str, str, str]]:
    """Return (memory_type, scope, content) items from a pyproject.toml string."""
    try:
        data = tomllib.loads(content)
    except Exception:
        return []

    project = data.get("project") or {}
    name = project.get("name")
    requires_python = project.get("requires-python")
    deps = project.get("dependencies") or []
    if not isinstance(deps, list):
        deps = []

    items: list[tuple[str, str, str]] = []
    if name or requires_python or deps:
        content_lines = []
        if name:
            content_lines.append(f"Project name: {name}")
        if requires_python:
            content_lines.append(f"Python requirement: {requires_python}")
        if deps:
            content_lines.append("Dependencies: " + ", ".join(deps[:25]))
        items.append(("DECISION", "repo", "Stack baseline from pyproject.toml. " + " ".join(content_lines)))

    return items


def _seed_from_readme(*, content: str) -> list[tuple[str, str, str]]:
    """Return (memory_type, scope, content) items from README text.

    This is intentionally conservative: we only store stack-level facts explicitly
    named in the README, not inferred architecture.
    """
    lowered = content.lower()
    stack_bits: list[str] = []
    if "model context protocol" in lowered or "mcp" in lowered:
        stack_bits.append("MCP server")
    if "mem0" in lowered:
        stack_bits.append("Mem0")
    if "supabase" in lowered or "postgres" in lowered:
        stack_bits.append("Supabase/Postgres vector store")
    if "openrouter" in lowered:
        stack_bits.append("OpenRouter")
    if "ollama" in lowered:
        stack_bits.append("Ollama (optional)")

    items: list[tuple[str, str, str]] = []
    if stack_bits:
        items.append(("DECISION", "repo", "Stack baseline from README: " + ", ".join(sorted(set(stack_bits)))))
    return items

@mcp.tool()
async def save_memory(
    ctx: Context,
    text: str,
    project_name: str = "default_project",
    file_path: str = None,
    tags: str = None,
    memory_type: str = None,
    scope: str = None,
    trigger: str = None,
    last_validated: str = None,
    review_after: str = None,
    expires: str = None,
    dedupe: bool = True,
    dedupe_mode: str = "replace_similar",
    dedupe_limit: int = 10,
    dedupe_score_threshold: float = 0.78,
) -> str:
    """Save information to your long-term memory.

    This tool is designed to store any type of information that might be useful in the future.
    The content will be processed and indexed for later retrieval through semantic search.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        text: The content to store in memory, including any relevant details and context
        project_name: The name of the project to associate this memory with (acts as user_id)
        file_path: Optional path to the file related to this memory
        tags: Optional comma-separated tags for categorization
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        # If caller provides a structured memory_type, save as a vibe memory.
        if memory_type:
            return _save_vibe_memory_impl(
                mem0_client=mem0_client,
                project_name=project_name,
                memory_type=memory_type,
                scope=scope or "repo",
                trigger=trigger,
                content=text,
                file_path=file_path,
                tags=tags,
                last_validated=last_validated,
                review_after=review_after,
                expires=expires,
                dedupe=dedupe,
                dedupe_mode=dedupe_mode,
                dedupe_limit=dedupe_limit,
                dedupe_score_threshold=dedupe_score_threshold,
            )

        # Generic memory (backwards compatible)
        secret_reason = detect_potential_secret(text)
        if secret_reason:
            return (
                "Refusing to store potential secret. "
                f"Detected: {secret_reason}. Store the env var name/procedure, not the value."
            )

        messages = [{"role": "user", "content": text}]
        metadata = {}
        if file_path:
            metadata["file_path"] = file_path
        if tags:
            metadata["tags"] = tags

        mem0_client.add(messages, user_id=project_name, metadata=metadata)
        return (
            f"Successfully saved memory to project '{project_name}': {text[:100]}..."
            if len(text) > 100
            else f"Successfully saved memory to project '{project_name}': {text}"
        )
    except Exception as e:
        return f"Error saving memory: {str(e)}"


@mcp.tool()
async def save_vibe_memory(
    ctx: Context,
    memory_type: str,
    content: str,
    project_name: str = "default_project",
    scope: str = "repo",
    trigger: str = None,
    file_path: str = None,
    tags: str = None,
    last_validated: str = None,
    review_after: str = None,
    expires: str = None,
    dedupe: bool = True,
    dedupe_mode: str = "replace_similar",
    dedupe_limit: int = 10,
    dedupe_score_threshold: float = 0.78,
) -> str:
    """Save a structured, low-noise "vibe" memory.

    Use this when you want predictable retrieval and hygiene.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        return _save_vibe_memory_impl(
            mem0_client=mem0_client,
            project_name=project_name,
            memory_type=memory_type,
            scope=scope,
            trigger=trigger,
            content=content,
            file_path=file_path,
            tags=tags,
            last_validated=last_validated,
            review_after=review_after,
            expires=expires,
            dedupe=dedupe,
            dedupe_mode=dedupe_mode,
            dedupe_limit=dedupe_limit,
            dedupe_score_threshold=dedupe_score_threshold,
        )
    except Exception as e:
        return f"Error saving vibe memory: {str(e)}"

@mcp.tool()
async def get_all_memories(ctx: Context, project_name: str = "default_project") -> str:
    """Get all stored memories for the user/project.
    
    Call this tool when you need complete context of all previously memories.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        project_name: The name of the project to retrieve memories for

    Returns a JSON formatted list of all stored memories, including IDs, creation time,
    content, and metadata. Results are paginated with a default of 50 items per page.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=project_name)
        
        results = []
        source_list = memories["results"] if isinstance(memories, dict) and "results" in memories else memories
        
        # If source_list is None or empty, return empty list
        if not source_list:
            return json.dumps([], indent=2)
            
        for memory in source_list:
            item = {
                "id": memory.get("id"),
                "text": memory.get("memory"),
                "created_at": memory.get("created_at")
            }
            if memory.get("metadata"):
                item["metadata"] = memory.get("metadata")
            results.append(item)
            
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

@mcp.tool()
async def search_memories(ctx: Context, query: str, project_name: str = "default_project", limit: int = 3) -> str:
    """Search memories using semantic search.

    This tool should be called to find relevant information from your memory. Results are ranked by relevance.
    Always search your memories before making decisions to ensure you leverage your existing knowledge.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        query: Search query string describing what you're looking for. Can be natural language.
        project_name: The name of the project to search memories in
        limit: Maximum number of results to return (default: 3)
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.search(query, user_id=project_name, limit=limit)
        
        results = []
        source_list = memories["results"] if isinstance(memories, dict) and "results" in memories else memories
        
        # If source_list is None or empty, return empty list
        if not source_list:
            return json.dumps([], indent=2)

        for memory in source_list:
            item = {
                "id": memory.get("id"),
                "text": memory.get("memory"),
                "score": memory.get("score"),
                "created_at": memory.get("created_at")
            }
            if memory.get("metadata"):
                item["metadata"] = memory.get("metadata")
            results.append(item)
            
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error searching memories: {str(e)}"

@mcp.tool()
async def delete_memory(ctx: Context, memory_id: str) -> str:
    """Delete a specific memory by its ID.

    Use this tool to remove outdated, incorrect, or irrelevant memories.
    You should first search or list memories to get the correct ID.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        memory_id: The unique identifier of the memory to delete
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        mem0_client.delete(memory_id)
        return f"Successfully deleted memory with ID: {memory_id}"
    except Exception as e:
        return f"Error deleting memory: {str(e)}"

@mcp.tool()
async def get_project_context(ctx: Context, project_name: str = "default_project") -> str:
    """Retrieve a comprehensive context briefing for the project.
    
    Use this tool at the start of a session to "vibe check" the project.
    It returns a summarized view of all architectural decisions, preferences,
    and constraints stored in memory.

    Args:
        ctx: The MCP server provided context which includes the Mem0 client
        project_name: The name of the project to retrieve context for
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        memories = mem0_client.get_all(user_id=project_name)
        
        source_list = memories["results"] if isinstance(memories, dict) and "results" in memories else memories
        
        if not source_list:
            return f"No context found for project '{project_name}'. Start by saving some memories about architecture and preferences."

        grouped: dict[str, list[dict]] = {"CONVENTION": [], "DECISION": [], "GOTCHA": [], "OTHER": []}
        for mem in source_list:
            text = mem.get("memory", "")
            meta = mem.get("metadata", {}) or {}
            fields = parse_vibe_text(text)

            # Prefer metadata because we control it; Mem0 may rewrite text.
            mem_type = ((meta.get("memory_type") or "") or fields.get("TYPE") or "").strip().upper()
            if mem_type in grouped:
                grouped[mem_type].append(mem)
            else:
                grouped["OTHER"].append(mem)

        context_lines = [f"# Project Context: {project_name}\n"]
        for section in ["CONVENTION", "DECISION", "GOTCHA", "OTHER"]:
            if not grouped[section]:
                continue
            context_lines.append(f"## {section}")
            for mem in grouped[section]:
                text = mem.get("memory", "")
                meta = mem.get("metadata", {}) or {}
                item_str = f"- {text}"
                meta_parts = []
                if meta.get("file_path"):
                    meta_parts.append(f"File: {meta.get('file_path')}")
                if meta.get("tags"):
                    meta_parts.append(f"Tags: {meta.get('tags')}")
                if meta.get("last_validated"):
                    meta_parts.append(f"LastValidated: {meta.get('last_validated')}")
                if meta_parts:
                    item_str += f" ({', '.join(meta_parts)})"
                context_lines.append(item_str)

            context_lines.append("")

        return "\n".join(context_lines).rstrip()
    except Exception as e:
        return f"Error retrieving project context: {str(e)}"


@mcp.tool()
async def get_session_briefing(
    ctx: Context,
    project_name: str = "default_project",
    scope: str = None,
    search_limit: int = 10,
    max_items: int = 3,
) -> str:
    """Return a compact "start-of-session" briefing.

    This is designed to reduce context blunders: it fetches the most relevant
    conventions/decisions/gotchas plus any "current goal" style notes.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        scope_hint = f" SCOPE: {scope}" if scope else ""
        queries = [
            f"TYPE: CONVENTION{scope_hint}",
            f"TYPE: DECISION{scope_hint}",
            f"TYPE: GOTCHA{scope_hint}",
            f"stack{scope_hint}",
            f"current goal{scope_hint}",
        ]

        candidates: dict[str, dict] = {}
        for q in queries:
            results = _extract_results(mem0_client.search(q, user_id=project_name, limit=search_limit))
            for item in results:
                if not isinstance(item, dict):
                    continue
                memory_id = item.get("id")
                if not memory_id:
                    continue
                # Keep best score per id.
                if memory_id not in candidates:
                    candidates[memory_id] = item
                else:
                    try:
                        if float(item.get("score") or 0) > float(candidates[memory_id].get("score") or 0):
                            candidates[memory_id] = item
                    except Exception:
                        pass

        ordered = sorted(
            candidates.values(),
            key=lambda x: float(x.get("score") or 0),
            reverse=True,
        )
        picked = ordered[: max_items]

        response = {
            "project_name": project_name,
            "scope": scope,
            "generated_at": today_yyyy_mm_dd(),
            "items": [
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "text": m.get("memory"),
                    "created_at": m.get("created_at"),
                    "metadata": m.get("metadata") or {},
                }
                for m in picked
            ],
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error generating session briefing: {str(e)}"


@mcp.tool()
async def seed_project_from_files(
    ctx: Context,
    project_name: str = "default_project",
    repo_root: str = ".",
    file_paths: str = "pyproject.toml,README.md",
    scope: str = "repo",
    trigger: str = "at start of session / when joining repo",
    tags: str = "seed, stack",
    dedupe: bool = True,
) -> str:
    """Seed initial vibe memories from repo files (low-friction onboarding).

    `file_paths` is a comma-separated list relative to `repo_root`.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        paths = [p.strip() for p in (file_paths or "").split(",") if p.strip()]
        if not paths:
            return "No file_paths provided."

        saved: list[dict] = []
        skipped: list[dict] = []
        errors: list[dict] = []

        for p in paths:
            try:
                abs_path = _safe_path(repo_root, p)
                if not abs_path.exists() or not abs_path.is_file():
                    skipped.append({"path": p, "reason": "not found"})
                    continue
                text = _read_text_file(abs_path)
                derived: list[tuple[str, str, str]] = []
                if abs_path.name.lower() == "pyproject.toml":
                    derived = _seed_from_pyproject(content=text)
                elif abs_path.name.lower() in {"readme.md", "readme"}:
                    derived = _seed_from_readme(content=text)
                else:
                    skipped.append({"path": p, "reason": "unsupported file type"})
                    continue

                for mem_type, mem_scope, mem_content in derived:
                    res = _save_vibe_memory_impl(
                        mem0_client=mem0_client,
                        project_name=project_name,
                        memory_type=mem_type,
                        scope=mem_scope or scope,
                        trigger=trigger,
                        content=mem_content,
                        file_path=str(abs_path),
                        tags=tags,
                        last_validated=today_yyyy_mm_dd(),
                        review_after=None,
                        expires=None,
                        dedupe=dedupe,
                        dedupe_mode="replace_similar",
                        dedupe_limit=10,
                        dedupe_score_threshold=0.78,
                    )
                    saved.append({"path": p, "type": mem_type, "result": res})
            except Exception as exc:
                errors.append({"path": p, "error": str(exc)})

        return json.dumps(
            {"project_name": project_name, "saved": saved, "skipped": skipped, "errors": errors},
            indent=2,
        )
    except Exception as e:
        return f"Error seeding project from files: {str(e)}"


@mcp.tool()
async def correct_memory(
    ctx: Context,
    project_name: str,
    old_memory_id: str,
    replacement_type: str,
    replacement_content: str,
    scope: str = "repo",
    trigger: str = "when similar issue appears",
    tags: str = "correction",
    file_path: str = None,
) -> str:
    """Correct a bad/stale memory: mark the old one as refuted and save a replacement.

    This is the "one-and-done" fix for repeated agent mistakes.
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client

        existing = _find_memory_by_id(mem0_client=mem0_client, project_name=project_name, memory_id=old_memory_id)
        existing_text = (existing or {}).get("memory") or ""
        existing_meta = (existing or {}).get("metadata") or {}

        refuted_at = today_yyyy_mm_dd()
        refuted_meta = dict(existing_meta)
        refuted_meta.update(
            {
                "status": "refuted",
                "refuted_at": refuted_at,
                "refuted_by": "correct_memory",
            }
        )
        refuted_text = (
            "STATUS: REFUTED\n"
            f"REFUTED_AT: {refuted_at}\n"
            f"REFUTED_REASON: Replaced by new {replacement_type} memory\n\n"
            + (existing_text.strip() or "(original content unavailable)")
        )

        # Mark old memory as refuted.
        action, _ = _best_effort_replace_memory(
            mem0_client=mem0_client,
            project_name=project_name,
            existing_memory_id=old_memory_id,
            new_text=refuted_text,
            new_metadata=refuted_meta,
        )

        # Save replacement.
        replacement_result = _save_vibe_memory_impl(
            mem0_client=mem0_client,
            project_name=project_name,
            memory_type=replacement_type,
            scope=scope,
            trigger=trigger,
            content=replacement_content,
            file_path=file_path,
            tags=tags,
            last_validated=today_yyyy_mm_dd(),
            review_after=None,
            expires=None,
            dedupe=True,
            dedupe_mode="replace_similar",
            dedupe_limit=10,
            dedupe_score_threshold=0.78,
        )

        return json.dumps(
            {
                "project_name": project_name,
                "old_memory_id": old_memory_id,
                "old_memory_action": action,
                "replacement_result": replacement_result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error correcting memory: {str(e)}"


@mcp.tool()
async def get_vibe_status(
    ctx: Context,
    project_name: str = "default_project",
    scope: str = None,
    include_other: bool = False,
) -> str:
    """Return a readable dashboard of the current vibe memories.

    Shows active CONVENTION/DECISION/GOTCHA and highlights anything stale
    (REVIEW_AFTER reached) or expired (EXPIRES reached).
    """
    try:
        mem0_client = ctx.request_context.lifespan_context.mem0_client
        all_items = _extract_results(mem0_client.get_all(user_id=project_name))
        today = _dt.date.today()

        buckets: dict[str, list[dict]] = {"CONVENTION": [], "DECISION": [], "GOTCHA": [], "OTHER": []}
        stale: list[dict] = []
        expired: list[dict] = []

        for item in all_items:
            if not isinstance(item, dict):
                continue
            text = item.get("memory", "")
            meta = item.get("metadata") or {}
            fields = parse_vibe_text(text)
            # Prefer metadata because we control it; Mem0 may rewrite text.
            mem_type = ((meta.get("memory_type") or "") or fields.get("TYPE") or "OTHER").strip().upper()

            item_scope = (fields.get("SCOPE") or meta.get("scope") or "").strip()
            if scope and item_scope and item_scope != scope:
                continue

            review_after = _parse_date(fields.get("REVIEW_AFTER") or meta.get("review_after"))
            expires = _parse_date(fields.get("EXPIRES") or meta.get("expires"))

            if expires and expires <= today:
                expired.append(item)
            elif review_after and review_after <= today:
                stale.append(item)

            if mem_type in buckets:
                buckets[mem_type].append(item)
            else:
                buckets["OTHER"].append(item)

        def _fmt_item(i: dict) -> str:
            text = (i.get("memory") or "").strip().replace("\n", " ")
            if len(text) > 180:
                text = text[:177] + "..."
            meta = i.get("metadata") or {}
            mt = (meta.get("memory_type") or "").strip()
            sid = i.get("id")
            return f"- {text}" + (f"  (id: {sid}, type: {mt})" if sid else "")

        lines: list[str] = []
        lines.append(f"# Vibe Status: {project_name}")
        if scope:
            lines.append(f"Scope filter: {scope}")
        lines.append("")

        for section in ["CONVENTION", "DECISION", "GOTCHA"]:
            lines.append(f"## {section}")
            if not buckets[section]:
                lines.append("- (none)")
            else:
                for i in buckets[section][:25]:
                    lines.append(_fmt_item(i))
            lines.append("")

        lines.append("## Needs Review")
        if not stale:
            lines.append("- (none)")
        else:
            for i in stale[:25]:
                lines.append(_fmt_item(i))
        lines.append("")

        lines.append("## Expired")
        if not expired:
            lines.append("- (none)")
        else:
            for i in expired[:25]:
                lines.append(_fmt_item(i))
        lines.append("")

        if include_other:
            lines.append("## Other")
            if not buckets["OTHER"]:
                lines.append("- (none)")
            else:
                for i in buckets["OTHER"][:25]:
                    lines.append(_fmt_item(i))
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"
    except Exception as e:
        return f"Error generating vibe status: {str(e)}"

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    # The server needs a clean environment. We already redirected sys.stdout to sys.stderr at the top.
    # Now we run the server, and we'll ensure the transport uses the original real stdout.
    
    transport_name = os.getenv("TRANSPORT", "stdio").lower()
    if transport_name == "stdio":
        # For stdio, we explicitly use the original stdout buffer for the server
        # and keep the global sys.stdout redirected to stderr to catch any pollution.
        # However, FastMCP doesn't give us easy access to the transport initialization.
        # So we temporarily restore sys.stdout for the duration of the server run.
        try:
            sys.stdout = _real_stdout
            asyncio.run(main())
        finally:
            sys.stdout = sys.stderr
    else:
        asyncio.run(main())
