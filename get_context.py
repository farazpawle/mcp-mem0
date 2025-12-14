import asyncio
import os
from dotenv import load_dotenv
from src.utils import get_mem0_client

load_dotenv()

def get_context():
    client = get_mem0_client()
    project_id = "test_project_v1" # Using the ID we used for testing
    
    print(f"Fetching context for {project_id}...")
    memories = client.get_all(user_id=project_id)
    source_list = memories["results"] if isinstance(memories, dict) and "results" in memories else memories
    
    if not source_list:
        print("No memories found.")
        return

    print(f"\n# Project Context: {project_id}\n")
    for mem in source_list:
        text = mem.get("memory", "")
        meta = mem.get("metadata", {})
        
        item_str = f"- {text}"
        meta_parts = []
        if meta and meta.get("file_path"):
            meta_parts.append(f"File: {meta.get('file_path')}")
        if meta and meta.get("tags"):
            meta_parts.append(f"Tags: {meta.get('tags')}")
            
        if meta_parts:
            item_str += f" ({', '.join(meta_parts)})"
            
        print(item_str)

if __name__ == "__main__":
    get_context()