import vecs
import os
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    print("No DATABASE_URL found")
    exit(1)

print(f"Connecting to {DB_URL.split('@')[-1]}") # Print host only for safety

try:
    vx = vecs.create_client(DB_URL)
except Exception as e:
    print(f"Failed to create vecs client: {e}")
    exit(1)

try:
    collections = vx.list_collections()
    print("Collections found:", collections)
except Exception as e:
    print(f"Failed to list collections: {e}")
    collections = []

# Try to create a new one with 1024
TEST_NAME = "test_vecs_1024_direct"
print(f"\nAttempting to create collection '{TEST_NAME}' with dimension 1024...")

# Ensure it's clean
try:
    vx.delete_collection(TEST_NAME)
    print(f"Deleted existing '{TEST_NAME}'")
except:
    pass

# DELETE mem0migrations to fix dimension mismatch
try:
    vx.delete_collection("mem0migrations")
    print("Deleted 'mem0migrations' collection")
except Exception as e:
    print(f"Failed to delete 'mem0migrations': {e}")

try:
    c = vx.create_collection(name=TEST_NAME, dimension=1024)
    print(f"SUCCESS: Created '{TEST_NAME}' with 1024 dims")
except Exception as e:
    print(f"FAILURE: Failed to create '{TEST_NAME}' with 1024 dims: {e}")
