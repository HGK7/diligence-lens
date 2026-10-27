"""Helpers for labeled retrieval / boundary cases."""

from __future__ import annotations


def expects_retrieval_hit(case: dict) -> bool:
    """True when Recall/MRR should score this case against expected_sources."""
    if case.get("expect_hit") is False:
        return False
    return bool(case.get("expected_sources"))


def case_category(case: dict) -> str:
    return str(case.get("category") or "unspecified")
