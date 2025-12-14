import asyncio
import json
import pathlib
import sys
import time

from dotenv import load_dotenv

# Ensure project root is importable (so `src` is a package).
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_mem0_client
from src.main import (
    correct_memory,
    get_session_briefing,
    get_vibe_status,
    save_vibe_memory,
    seed_project_from_files,
)

load_dotenv()

# Mock Context for testing
class MockContext:
    def __init__(self, client):
        self.request_context = type('obj', (object,), {
            'lifespan_context': type('obj', (object,), {
                'mem0_client': client
            })
        })

async def test_vibe_integration():
    print("Initializing Mem0 client for integration test...")
    try:
        client = get_mem0_client()
        ctx = MockContext(client)
        print("Client initialized.")
    except Exception as e:
        print(f"Skipping integration test (no DB/Client): {e}")
        return

    project_id = f"test_vibe_integration_v1_{int(time.time())}"
    
    print(f"\n1. Saving a CONVENTION memory...")
    res1 = await save_vibe_memory(
        ctx,
        memory_type="CONVENTION",
        content="Always use snake_case for variables.",
        project_name=project_id,
        scope="repo",
        trigger="naming variables",
        tags="style, python"
    )
    print(f"Result: {res1}")

    print(f"\n2. Saving a DECISION memory...")
    res2 = await save_vibe_memory(
        ctx,
        memory_type="DECISION",
        content="Use Supabase for the database layer.",
        project_name=project_id,
        scope="backend",
        tags="architecture, db"
    )
    print(f"Result: {res2}")

    print(f"\n3. Testing Deduplication (Saving same CONVENTION again)...")
    res3 = await save_vibe_memory(
        ctx,
        memory_type="CONVENTION",
        content="Always use snake_case for variables.",
        project_name=project_id,
        scope="repo",
        trigger="naming variables",
        tags="style, python"
    )
    print(f"Result (should be update/skip): {res3}")

    print(f"\n4. Getting Session Briefing...")
    briefing_json = await get_session_briefing(ctx, project_name=project_id, max_items=5)
    briefing = json.loads(briefing_json)
    
    print(f"Briefing Items Found: {len(briefing.get('items', []))}")
    for item in briefing.get('items', []):
        print(f" - [{item.get('metadata', {}).get('memory_type')}] {item.get('text')}")

    print("\n5. Seeding from repo files (pyproject + README)...")
    seed_result = await seed_project_from_files(
        ctx,
        project_name=project_id,
        repo_root=str(PROJECT_ROOT),
        file_paths="pyproject.toml,README.md",
    )
    print(seed_result)

    print("\n6. Saving a GOTCHA then correcting it...")
    gotcha_res = await save_vibe_memory(
        ctx,
        memory_type="GOTCHA",
        content="We should store secrets directly in memories.",
        project_name=project_id,
        scope="repo",
        trigger="handling credentials",
        tags="security",
        dedupe=False,
    )
    print(f"GOTCHA save result: {gotcha_res}")

    # Find that GOTCHA memory ID by searching for a distinctive substring.
    all_memories = client.get_all(user_id=project_id)
    results = all_memories.get("results") if isinstance(all_memories, dict) else all_memories
    bad_id = None
    for mem in results or []:
        if "store secrets directly" in (mem.get("memory") or "").lower():
            bad_id = mem.get("id")
            break

    if bad_id:
        correction = await correct_memory(
            ctx,
            project_name=project_id,
            old_memory_id=bad_id,
            replacement_type="GOTCHA",
            replacement_content="Never store secrets (API keys/tokens/passwords/connection strings) in memories; store procedures and env var names only.",
            scope="repo",
            trigger="handling credentials",
            tags="security, policy",
        )
        print(correction)
    else:
        print("Could not locate the GOTCHA memory to correct (unexpected).")

    print("\n7. Vibe Status dashboard...")
    status_md = await get_vibe_status(ctx, project_name=project_id, include_other=False)
    print(status_md)

    # Cleanup (Optional, but good for repeated tests)
    # print("\nCleaning up test memories...")
    # for item in briefing.get('items', []):
    #     client.delete(item['id'])

if __name__ == "__main__":
    asyncio.run(test_vibe_integration())
