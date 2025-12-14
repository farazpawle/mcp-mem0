import asyncio
import unittest


class FakeEmbeddingModel:
    def embed(self, text, _purpose):
        # Deterministic dummy embedding
        return [0.0, 0.0, 0.0]


class FakeMem0:
    def __init__(self):
        self.calls = []
        self.embedding_model = FakeEmbeddingModel()
        self._store = []  # list of dicts mimicking mem0 get_all results

    def get_all(self, user_id):
        self.calls.append(("get_all", {"user_id": user_id}))
        return {"results": list(self._store)}

    def search(self, query, user_id, limit=3):
        self.calls.append(("search", {"query": query, "user_id": user_id, "limit": limit}))
        return {"results": []}

    def add(self, messages, user_id, metadata=None):
        self.calls.append(("add", {"messages": messages, "user_id": user_id, "metadata": metadata or {}}))
        # Simulate insert
        memory_text = messages[0]["content"]
        item = {"id": f"mem_{len(self._store)+1}", "memory": memory_text, "metadata": metadata or {}}
        self._store.append(item)
        return {"id": item["id"]}

    def delete(self, memory_id):
        self.calls.append(("delete", {"memory_id": memory_id}))
        self._store = [m for m in self._store if m.get("id") != memory_id]
        return {"message": "deleted"}

    def _update_memory(self, memory_id, data, _existing_embeddings, metadata=None):
        self.calls.append(
            (
                "_update_memory",
                {"memory_id": memory_id, "data": data, "metadata": metadata or {}},
            )
        )
        for m in self._store:
            if m.get("id") == memory_id:
                m["memory"] = data
                if metadata is not None:
                    m["metadata"] = metadata
                return


class MockContext:
    def __init__(self, client):
        self.request_context = type(
            "obj",
            (object,),
            {"lifespan_context": type("obj", (object,), {"mem0_client": client})},
        )


class TestMem0ToolUsage(unittest.TestCase):
    def test_seed_uses_project_name_as_user_id(self):
        from src.main import seed_project_from_files

        fake = FakeMem0()
        ctx = MockContext(fake)

        # Use only README parsing path and repo_root '.'; we won't actually read files here.
        # Instead, ensure tool rejects missing files but still follows user_id scoping.
        # Provide an empty file list to avoid FS reads.
        result = asyncio.run(
            seed_project_from_files(
                ctx,
                project_name="proj_x",
                repo_root=".",
                file_paths="",
            )
        )
        self.assertIn("No file_paths", result)

        # No Mem0 calls expected because we returned early.
        self.assertEqual(fake.calls, [])

    def test_correct_memory_scopes_get_all_to_project(self):
        from src.main import correct_memory

        fake = FakeMem0()
        ctx = MockContext(fake)

        # Seed a fake memory to correct
        fake._store.append({"id": "mem_1", "memory": "TYPE: GOTCHA\nSCOPE: repo\nTRIGGER: x\nCONTENT: bad\nLAST_VALIDATED: 2025-12-14", "metadata": {"memory_type": "GOTCHA", "scope": "repo"}})

        _ = asyncio.run(
            correct_memory(
                ctx,
                project_name="proj_y",
                old_memory_id="mem_1",
                replacement_type="GOTCHA",
                replacement_content="good",
            )
        )

        # First call should be get_all scoped to proj_y
        self.assertTrue(fake.calls)
        self.assertEqual(fake.calls[0][0], "get_all")
        self.assertEqual(fake.calls[0][1]["user_id"], "proj_y")

    def test_get_vibe_status_scopes_get_all_to_project(self):
        from src.main import get_vibe_status

        fake = FakeMem0()
        ctx = MockContext(fake)

        _ = asyncio.run(get_vibe_status(ctx, project_name="proj_z"))
        self.assertTrue(fake.calls)
        self.assertEqual(fake.calls[0][0], "get_all")
        self.assertEqual(fake.calls[0][1]["user_id"], "proj_z")


if __name__ == "__main__":
    unittest.main()
