# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Agent to Serving Endpoint
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Gets the model version from the previous task
# MAGIC 2. Deploys it to a serving endpoint
# MAGIC 3. Configures inference tables and environment variables

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
from loguru import logger

from arxiv_curator.config import load_config
from arxiv_curator.serving import serve_model
from arxiv_curator.utils.common import get_widget

# Get model_version from previous task or widget
try:
    model_version = dbutils.jobs.taskValues.get(
        taskKey="log_register_agent",
        key="model_version",
    )
    logger.info(f"Got model version from previous task: {model_version}")
except Exception:
    model_version = get_widget("model_version", "1")
    logger.info(f"Got model version from widget: {model_version}")

git_sha = get_widget("git_sha", "local")
env = get_widget("env", "dev")
secret_scope = get_widget("secret_scope", "arxiv-agent-scope")

# Load configuration
cfg = load_config("project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = f"arxiv-agent-{env}"
workspace = WorkspaceClient()

logger.info(f"Environment: {env}")
logger.info(f"Model: {model_name}@{model_version}")
logger.info(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy to Serving Endpoint

# COMMAND ----------

serve_model(
    entity_name=model_name,
    entity_version=str(model_version),
    endpoint_name=endpoint_name,
    catalog_name=cfg.catalog,
    schema_name=cfg.schema,
    table_name_prefix=f"arxiv_agent_{env}",
    tags={"project_name": "arxiv_curator", "env": env},
    env_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": str(model_version),
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "DATABRICKS_HOST": workspace.config.host,
    },
)

logger.info("\n✓ Deployment complete!")
logger.info(f"  Endpoint: {endpoint_name}")
logger.info(f"  Model: {model_name}@{model_version}")
