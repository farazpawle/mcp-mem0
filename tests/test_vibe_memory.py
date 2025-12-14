import unittest

from src.vibe_memory import (
    build_vibe_memory,
    build_vibe_metadata,
    detect_potential_secret,
    format_vibe_text,
    make_vibe_key,
    normalize_csv_tags,
    parse_vibe_text,
)


class TestVibeMemory(unittest.TestCase):
    def test_detect_potential_secret_flags_openai_key(self):
        reason = detect_potential_secret("please store sk-1234567890abcdefghijklmnopqrstuvwxyz")
        self.assertIsNotNone(reason)

    def test_detect_potential_secret_allows_normal_text(self):
        self.assertIsNone(detect_potential_secret("Use PEP 8 and type hints."))

    def test_normalize_csv_tags(self):
        self.assertEqual(normalize_csv_tags("python, style,python"), "python, style")

    def test_format_and_parse_roundtrip(self):
        mem = build_vibe_memory(
            memory_type="convention",
            scope="repo",
            trigger="when writing python",
            content="Follow PEP 8.",
            last_validated="2025-12-14",
            review_after="2026-01-14",
            expires=None,
        )
        text = format_vibe_text(mem)
        fields = parse_vibe_text(text)
        self.assertEqual(fields["TYPE"], "CONVENTION")
        self.assertEqual(fields["SCOPE"], "repo")
        self.assertEqual(fields["TRIGGER"], "when writing python")
        self.assertEqual(fields["CONTENT"], "Follow PEP 8.")
        self.assertEqual(fields["LAST_VALIDATED"], "2025-12-14")
        self.assertEqual(fields["REVIEW_AFTER"], "2026-01-14")

    def test_vibe_key_stable(self):
        key1 = make_vibe_key(
            memory_type="CONVENTION",
            scope="Repo",
            trigger="when coding",
            content="Use black formatting",
        )
        key2 = make_vibe_key(
            memory_type="convention",
            scope="repo",
            trigger="WHEN CODING",
            content="Use   black   formatting\n",
        )
        self.assertEqual(key1, key2)

    def test_metadata_contains_vibe_key(self):
        mem = build_vibe_memory(
            memory_type="DECISION",
            scope="repo",
            trigger=None,
            content="Use Supabase for vector storage.",
            last_validated="2025-12-14",
            review_after=None,
            expires=None,
        )
        meta = build_vibe_metadata(memory=mem, file_path=None, tags="db, supabase")
        self.assertIn("vibe_key", meta)
        self.assertEqual(meta["memory_type"], "DECISION")


if __name__ == "__main__":
    unittest.main()
