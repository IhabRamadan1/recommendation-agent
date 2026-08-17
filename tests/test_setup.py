"""Smoke tests that dependencies and package layout are importable."""


def test_core_imports() -> None:
    import fastapi
    import langchain
    import langgraph
    import pydantic

    assert langgraph.__name__ == "langgraph"
    assert langchain.__name__ == "langchain"
    assert pydantic.__name__ == "pydantic"
    assert fastapi.__name__ == "fastapi"


def test_package_layout_imports() -> None:
    import agentic_service
    import config
    import recommendation_agent
    import recommendation_graph

    assert recommendation_agent.__name__ == "recommendation_agent"
    assert recommendation_graph.__name__ == "recommendation_graph"
    assert agentic_service.__name__ == "agentic_service"
    assert config.__name__ == "config"
