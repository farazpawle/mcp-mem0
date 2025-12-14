import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.environ.get("DATABASE_URL")

if not DB_URL:
    print("Error: DATABASE_URL not found in .env")
    exit(1)

print(f"Connecting to database...")
try:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        # Create the view
        # Note: We use double quotes for the table name because it's in a different schema and might be case sensitive
        sql = """
        CREATE OR REPLACE VIEW public.memories_view AS
        SELECT * FROM vecs."mem0_memories_main";
        """
        conn.execute(text(sql))
        conn.commit()
        print("SUCCESS: View 'public.memories_view' created successfully.")
        print("You can now see 'memories_view' in the Supabase Table Editor under the 'public' schema.")
except Exception as e:
    print(f"Error creating view: {e}")
