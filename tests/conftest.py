import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from ragagent.config import Settings
from ragagent.demo import build_engine
from ragagent.demo_data import GLOSSARY, demo_documents, golden_set
from ragagent.store import DocumentStore
from ragagent.tools import ToolRegistry


@pytest.fixture()
def settings():
    return Settings()


@pytest.fixture()
def store():
    store = DocumentStore(dim=128)
    store.add_documents(demo_documents())
    return store


@pytest.fixture()
def tools():
    return ToolRegistry(reference_date="2026-11-17", glossary=GLOSSARY)


@pytest.fixture()
def engine(settings):
    return build_engine(settings)


@pytest.fixture()
def cases():
    return golden_set()
