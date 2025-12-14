import os
import sys
from dotenv import load_dotenv
from src.utils import get_mem0_client
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def print_memories(project_name="test_project_v1"):
    print(f"\n{'='*60}")
    print(f"Viewing Memories for Project: {project_name}")
    print(f"{'='*60}\n")

    try:
        m = get_mem0_client()
        
        # Get all memories
        print("Fetching memories...")
        memories = m.get_all(user_id=project_name)
        
        results = memories.get("results", []) if isinstance(memories, dict) else memories
        
        if not results:
            print("No memories found.")
            print("\nWould you like to add a sample memory? (y/n)")
            # We can't take input in this environment easily, so we'll just add one if empty for demo
            # But wait, I can't interact.
            # I'll just print a message.
            print("(Run 'python test_mem0.py' to add a sample memory if needed)")
            return

        print(f"Found {len(results)} memories:\n")
        
        # Header
        print(f"{'ID':<36} | {'Date':<16} | {'Meta':<20} | {'Memory'}")
        print("-" * 120)
        
        for mem in results:
            mem_id = mem.get('id', 'N/A')
            date_str = mem.get('created_at', 'N/A')
            # Try to parse date if it's a timestamp string
            try:
                if 'T' in str(date_str):
                    dt = datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
                
            text = mem.get('memory', '')
            meta = mem.get('metadata', {})
            meta_str = ""
            if meta:
                # Compact metadata string
                parts = []
                if meta.get('tags'): parts.append(f"[{meta['tags']}]")
                if meta.get('file_path'): parts.append(f"({os.path.basename(meta['file_path'])})")
                meta_str = " ".join(parts)
            
            # Truncate text if too long
            display_text = (text[:60] + '...') if len(text) > 60 else text
            display_meta = (meta_str[:20] + '..') if len(meta_str) > 20 else meta_str
            
            print(f"{mem_id:<36} | {date_str:<16} | {display_meta:<20} | {display_text}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_memories()
