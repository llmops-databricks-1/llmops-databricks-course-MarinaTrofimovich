import mlflow
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent

config = ModelConfig(
    development_config={
        "catalog": "llmops_dev",
        "schema": "arxiv",
        "genie_space_id": "01f10b2087aa1a9f96c84e2b42ae37ba",
        "system_prompt": (
            "You are a LinkedIn content creation assistant specialized in generating "
            "engaging posts about AI and machine learning research. Search for "
            "relevant papers, synthesize 2-3 strong matches into one narrative, and "
            "return a single polished LinkedIn post only. Start with a strong hook, "
            "mention paper titles and arXiv IDs inline, and end with 2-3 hashtags on "
            "the final line. Do not return bullet lists, paper inventories, or preambles "
            "such as 'Here are a few papers'."
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
