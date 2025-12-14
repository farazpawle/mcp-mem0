import vecs
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")
vx = vecs.create_client(DB_URL)

collection_name = "mem0_memories_main"
try:
    print("Listing all collections:")
    print(vx.list_collections())

    docs = vx.get_collection(collection_name)
    print(f"Connected to collection: {collection_name}")
    
    # Count items
    # vecs doesn't have a direct count method easily accessible without query, 
    # but we can try to fetch some
    # Actually, we can use the underlying adapter to execute SQL if needed, 
    # but let's just try to peek.
    
    # We can't easily count without a query vector in vecs unless we use the private adapter
    # But we can try to query with a dummy vector if we had one.
    
    # Let's just print that it exists.
    print(f"Collection '{collection_name}' exists in schema 'vecs'.")
    print("To view in Supabase: Go to Table Editor -> Switch Schema (top left) from 'public' to 'vecs'.")
    
except Exception as e:
    print(f"Error: {e}")
