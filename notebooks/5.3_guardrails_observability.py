# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.3: Guardrails & Observability
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - MLflow tracing for observability
# MAGIC - System tables for monitoring
# MAGIC - Guardrails and safety checks
# MAGIC - Performance monitoring
# MAGIC - Cost tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### Install Custom Package
# MAGIC
# MAGIC The wheel file includes all required dependencies from pyproject.toml

# COMMAND ----------

# MAGIC %pip install ../dist/arxiv_curator-0.1.0-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# COMMAND ----------
# Setup
import os

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

w = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

endpoint_name = f"arxiv-agent-{env}"

logger.info(f"Environment: {env}")
logger.info(f"Endpoint: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. MLflow Tracing
# MAGIC
# MAGIC **MLflow Tracing** provides:
# MAGIC - End-to-end request tracking
# MAGIC - Tool call visibility
# MAGIC - Latency breakdown
# MAGIC - Token usage tracking
# MAGIC - Error debugging

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query Traces from System Tables

# COMMAND ----------

# Query recent traces
traces_query = f"""
SELECT
    request_id,
    timestamp,
    status,
    execution_duration_ms,
    request_metadata,
    tags
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 1 HOUR
ORDER BY timestamp DESC
LIMIT 20
"""

traces_df = spark.sql(traces_query)
display(traces_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Performance Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ### Latency Analysis

# COMMAND ----------

latency_stats = spark.sql(f"""
SELECT
    DATE_TRUNC('minute', timestamp) as minute,
    COUNT(*) as request_count,
    PERCENTILE(execution_duration_ms, 0.5) as p50_latency_ms,
    PERCENTILE(execution_duration_ms, 0.95) as p95_latency_ms,
    PERCENTILE(execution_duration_ms, 0.99) as p99_latency_ms,
    MAX(execution_duration_ms) as max_latency_ms
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
GROUP BY minute
ORDER BY minute DESC
""")

display(latency_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Error Rate Monitoring

# COMMAND ----------

error_stats = spark.sql(f"""
SELECT
    DATE_TRUNC('hour', timestamp) as hour,
    status,
    COUNT(*) as count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY DATE_TRUNC('hour', timestamp)) as percentage
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 24 HOURS
GROUP BY hour, status
ORDER BY hour DESC, count DESC
""")

display(error_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Cost Tracking

# COMMAND ----------

# MAGIC %md
# MAGIC ### Token Usage Analysis

# COMMAND ----------

token_usage = spark.sql(f"""
SELECT
    DATE_TRUNC('day', timestamp) as day,
    SUM(total_tokens) as total_tokens,
    AVG(total_tokens) as avg_tokens_per_request,
    SUM(prompt_tokens) as total_prompt_tokens,
    SUM(completion_tokens) as total_completion_tokens
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 7 DAYS
GROUP BY day
ORDER BY day DESC
""")

display(token_usage)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Guardrails Implementation
# MAGIC
# MAGIC **Guardrails** protect against:
# MAGIC - Inappropriate content
# MAGIC - Excessive costs
# MAGIC - Security risks
# MAGIC - Performance issues

# COMMAND ----------

# MAGIC %md
# MAGIC ### Content Safety Check

# COMMAND ----------


def check_content_safety(text: str) -> dict:
    """Check if content is safe using simple heuristics."""

    # Blocked keywords (example)
    blocked_keywords = ["hack", "exploit", "malicious"]

    # Check for blocked content
    text_lower = text.lower()
    for keyword in blocked_keywords:
        if keyword in text_lower:
            return {"safe": False, "reason": f"Contains blocked keyword: {keyword}"}

    # Check length
    if len(text) > 10000:
        return {"safe": False, "reason": "Content too long"}

    return {"safe": True, "reason": None}


# Test
test_input = "How do I build a recommendation system?"
result = check_content_safety(test_input)
logger.info(f"Safety check: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rate Limiting

# COMMAND ----------

# Query request rate
request_rate = spark.sql(f"""
SELECT
    DATE_TRUNC('minute', timestamp) as minute,
    COUNT(*) as requests_per_minute
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 1 HOUR
GROUP BY minute
ORDER BY minute DESC
LIMIT 10
""")

display(request_rate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alerting Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Alert Conditions

# COMMAND ----------

# Alert thresholds
LATENCY_THRESHOLD_MS = 5000  # 5 seconds
ERROR_RATE_THRESHOLD = 0.05  # 5%
TOKEN_BUDGET_DAILY = 1000000  # 1M tokens per day

logger.info("Alert Thresholds:")
logger.info(f"  Max Latency: {LATENCY_THRESHOLD_MS}ms")
logger.info(f"  Max Error Rate: {ERROR_RATE_THRESHOLD * 100}%")
logger.info(f"  Daily Token Budget: {TOKEN_BUDGET_DAILY:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for Anomalies

# COMMAND ----------

# Check recent performance
recent_stats = spark.sql(f"""
SELECT
    COUNT(*) as total_requests,
    AVG(execution_duration_ms) as avg_latency_ms,
    MAX(execution_duration_ms) as max_latency_ms,
    SUM(CASE WHEN status != 200 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as error_rate
FROM system.ai.serving_endpoint_request_logs
WHERE endpoint_name = '{endpoint_name}'
    AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL 1 HOUR
""").collect()[0]

logger.info("Recent Performance (Last Hour):")
logger.info(f"  Total Requests: {recent_stats['total_requests']}")
logger.info(f"  Avg Latency: {recent_stats['avg_latency_ms']:.2f}ms")
logger.info(f"  Max Latency: {recent_stats['max_latency_ms']:.2f}ms")
logger.info(f"  Error Rate: {recent_stats['error_rate'] * 100:.2f}%")

# Check thresholds
if recent_stats["max_latency_ms"] > LATENCY_THRESHOLD_MS:
    logger.info("\n⚠️  ALERT: Latency exceeded threshold!")

if recent_stats["error_rate"] > ERROR_RATE_THRESHOLD:
    logger.info("\n⚠️  ALERT: Error rate exceeded threshold!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learned:
# MAGIC
# MAGIC 1. ✅ How to use MLflow tracing for observability
# MAGIC 2. ✅ How to query system tables for monitoring
# MAGIC 3. ✅ How to track performance metrics
# MAGIC 4. ✅ How to implement guardrails
# MAGIC 5. ✅ How to set up alerting
# MAGIC 6. ✅ How to monitor costs
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Set up automated evaluation pipelines
# MAGIC - Implement A/B testing
# MAGIC - Create dashboards for monitoring
# MAGIC - Integrate with incident management systems
