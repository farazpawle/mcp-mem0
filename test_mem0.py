import asyncio
import os
from dotenv import load_dotenv
from src.utils import get_mem0_client

load_dotenv()

def test_mem0():
    print("Initializing Mem0 client...")
    try:
        client = get_mem0_client()
        print("Client initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize client: {e}")
        return

    project_id = "test_project_v1"
    # text = "This is a test memory for the MCP server performance check."
    text = "I strictly follow PEP 8 guidelines for Python code. Code readability is paramount."

    print(f"\n1. Saving memory to project '{project_id}'...")
    try:
        # mem0 add returns a dictionary or list of dictionaries
        # Adding metadata test
        metadata = {"file_path": "docs/style_guide.md", "tags": "python, style"}
        result = client.add([{"role": "user", "content": text}], user_id=project_id, metadata=metadata)
        print(f"Save result: {result}")
    except Exception as e:
        print(f"Failed to save memory: {e}")
        return

    print(f"\n2. Searching memories in project '{project_id}'...")
    try:
        search_results = client.search("PEP 8 guidelines", user_id=project_id)
        print(f"Search results: {search_results}")
    except Exception as e:
        print(f"Failed to search memories: {e}")

    print(f"\n3. Getting all memories for project '{project_id}'...")
    try:
        all_memories = client.get_all(user_id=project_id)
        print(f"All memories: {all_memories}")
        
        # Extract ID for deletion
        memory_id = None
        if isinstance(all_memories, dict) and "results" in all_memories:
            results = all_memories["results"]
        else:
            results = all_memories
            
        if results:
            # Find our test memory
            for mem in results:
                if text in mem.get("memory", ""):
                    memory_id = mem.get("id")
                    break
            
            if memory_id:
                print(f"\n4. Deleting memory ID: {memory_id}...")
                # client.delete(memory_id)
                print("Deletion requested (SKIPPED for visualization).")
                
                # Verify deletion
                # print("Verifying deletion...")
                # check = client.get_all(user_id=project_id)
                # print(f"Memories after deletion: {check}")
            else:
                print("Could not find the test memory ID to delete.")
        else:
            print("No memories found to delete.")

    except Exception as e:
        print(f"Failed to get/delete memories: {e}")

if __name__ == "__main__":
    test_mem0()
