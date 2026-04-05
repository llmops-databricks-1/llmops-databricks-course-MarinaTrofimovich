import mlflow
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent

config = ModelConfig(
    development_config={
        "catalog": "llmops_dev",
        "schema": "arxiv",
        "genie_space_id": "01f10b2087aa1a9f96c84e2b42ae37ba",
        "system_prompt": (
            "You are a helpful AI assistant that helps users find and "
            "understand research papers."
        ),
        "llm_endpoint": "databricks-llama-4-maverick",
    }
)

agent = ArxivAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
)
mlflow.models.set_model(agent)
