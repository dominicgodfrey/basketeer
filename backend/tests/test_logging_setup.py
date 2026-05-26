import logging

import pytest

from app.logging_setup import configure_logging, get_logger
from app.llm.providers import FakeProvider
from app.llm.router import Task, model_for
from app.primitives import FindSimilarRequest, WriteContext, find_similar, write
from app.similarity import InMemoryVectorStore, VectorRecord


def test_configure_logging_sets_root_level() -> None:
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    configure_logging("INFO")
    assert logging.getLogger().level == logging.INFO


def test_configure_logging_idempotent_handler_count() -> None:
    configure_logging("INFO")
    handler_count_before = len(logging.getLogger().handlers)
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == handler_count_before


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("app.test.module")
    assert logger.name == "app.test.module"


def test_write_primitive_emits_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.primitives.write")
    write(
        WriteContext(question="anything", findings=["a", "b"]),
        FakeProvider("text"),
        model_for(Task.NARRATIVE_WRITE),
    )
    messages = [r.message for r in caplog.records]
    assert any("write.invoke" in m for m in messages)
    assert any("write.complete" in m for m in messages)


def test_find_similar_emits_info_log(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.primitives.find_similar")
    store = InMemoryVectorStore()
    store.upsert([VectorRecord(id="a", vector=[1.0, 0.0])])
    find_similar(FindSimilarRequest(vector=[1.0, 0.0], top_k=1), store)
    assert any("find_similar source=vector" in r.message for r in caplog.records)
