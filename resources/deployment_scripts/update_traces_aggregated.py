# Databricks notebook source
import json as _json

import mlflow
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig
from arxiv_curator.evaluation import (
    hook_in_post_guideline,
    polite_tone_guideline,
    word_count_check,
)
from arxiv_curator.utils.common import get_widget

env = get_widget("env", "dev")
cfg = ProjectConfig.from_yaml("../../project_config.yml", env=env)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.schema
aggregated_table = f"{catalog}.{schema}.arxiv_traces_aggregated"
payload_table = f"{catalog}.{schema}.arxiv_agent_{env}_payload"

# COMMAND ----------
# Read serving endpoint traces from the inference (payload) table

payload_sdf = spark.read.table(payload_table)
payload_pdf = payload_sdf.toPandas()
logger.info(f"Inference table rows found: {len(payload_pdf)}")

# Keep only successful, deduplicated requests
payload_pdf = payload_pdf[payload_pdf["status_code"] == 200]
payload_pdf = payload_pdf.drop_duplicates(subset=["databricks_request_id"], keep="first")
logger.info(f"After filtering status_code=200 and dedup: {len(payload_pdf)}")

# COMMAND ----------
# Parse request/response from JSON strings


def _safe_json_loads(val: object) -> dict:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if not isinstance(val, str):
        return {}
    try:
        return _json.loads(val)
    except Exception:
        return {}


def extract_request_text(request: dict) -> str:
    """Extract user query text from the parsed request dict."""
    # Unwrap nested {'request': {...}} wrapper if present
    if "request" in request and isinstance(request["request"], dict):
        request = request["request"]
    # OpenAI-style: {"messages"/"input": [{"role": "user", "content": "..."}]}
    for key in ("messages", "input"):
        msgs = request.get(key, [])
        if not isinstance(msgs, list):
            continue
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    return " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict)
                    )
                return str(content)
    return ""


def extract_response_text(response: dict) -> str:
    """Extract assistant response text from the parsed response dict."""
    if not response:
        return ""
    outputs = response.get("output", [])
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
    return str(response)[:500]


# Build a clean dataframe of traces
rows = []
for _, row in payload_pdf.iterrows():
    request = _safe_json_loads(row.get("request"))
    response = _safe_json_loads(row.get("response"))
    req_text = extract_request_text(request)
    resp_text = extract_response_text(response)
    if not req_text or not resp_text:
        continue
    rows.append(
        {
            "trace_id": str(row["databricks_request_id"]),
            "request_time": str(row.get("request_time", "")),
            "request_preview": req_text,
            "response_text": resp_text,
            "latency_seconds": (float(row.get("execution_duration_ms") or 0) / 1000.0),
        }
    )
traces_pdf = pd.DataFrame(rows)
logger.info(f"Valid traces with request+response: {len(traces_pdf)}")

# COMMAND ----------
# Check which traces were already evaluated in a previous run

try:
    existing_agg = spark.read.table(aggregated_table).toPandas()
    already_evaluated = set(
        existing_agg.loc[existing_agg["word_count_check"] != 0, "trace_id"]
    )
    logger.info(f"Already evaluated traces: {len(already_evaluated)}")
except Exception:
    already_evaluated = set()
    existing_agg = pd.DataFrame()
    logger.info("No existing aggregated table found, evaluating all")

unevaluated = traces_pdf[~traces_pdf["trace_id"].isin(already_evaluated)]
logger.info(f"New traces to evaluate: {len(unevaluated)}")

# COMMAND ----------
# Build eval input for unevaluated traces

eval_pdf = pd.DataFrame(
    {
        "trace_id": unevaluated["trace_id"].values,
        "inputs": unevaluated["request_preview"].apply(lambda x: {"query": x}).values,
        "outputs": unevaluated["response_text"].values,
    }
)

skip_evaluation = len(eval_pdf) == 0
if skip_evaluation:
    logger.info("No new traces to evaluate — skipping evaluation.")

# COMMAND ----------
# Run word_count_check on unevaluated traces

eval_scores: dict[str, dict] = {}

if not skip_evaluation:
    wc_result = mlflow.genai.evaluate(
        data=eval_pdf[["inputs", "outputs"]],
        scorers=[word_count_check],
    )
    for tid, assessments in zip(
        eval_pdf["trace_id"],
        wc_result.result_df["assessments"],
        strict=True,
    ):
        val = assessments[0]["feedback"]["value"]
        eval_scores.setdefault(tid, {})["word_count_check"] = val

    logger.info(f"Evaluated word_count_check for {len(eval_pdf)} traces")

# COMMAND ----------
# Run LLM-judge scorers on a 10% sample

if not skip_evaluation:
    sample_size = max(1, int(len(eval_pdf) * 0.1))
    sampled_pdf = eval_pdf.sample(n=sample_size)
    logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge eval")

    llm_result = mlflow.genai.evaluate(
        data=sampled_pdf[["inputs", "outputs"]],
        scorers=[polite_tone_guideline, hook_in_post_guideline],
    )
    for tid, assessments in zip(
        sampled_pdf["trace_id"],
        llm_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            eval_scores.setdefault(tid, {})[a["assessment_name"]] = a["feedback"]["value"]

    logger.info(f"Evaluated polite_tone/hook_in_post for {len(sampled_pdf)}")

# COMMAND ----------
# Build the aggregated table — merge new scores with previous results


def score_val(val: object, true_values: tuple) -> int:
    """Convert a scorer value to 1/0."""
    if val is None:
        return 0
    return 1 if str(val).lower() in true_values else 0


agg_rows = []
for _, row in traces_pdf.iterrows():
    tid = row["trace_id"]
    scores = eval_scores.get(tid, {})
    agg_rows.append(
        {
            "trace_id": tid,
            "request_time": row["request_time"],
            "request_preview": row["request_preview"],
            "response_text": row["response_text"],
            "latency_seconds": row["latency_seconds"],
            "word_count_check": score_val(scores.get("word_count_check"), ("true",)),
            "polite_tone": score_val(scores.get("polite_tone"), ("yes", "true", "pass")),
            "hook_in_post": score_val(
                scores.get("hook_in_post"), ("yes", "true", "pass")
            ),
        }
    )
agg_pdf = pd.DataFrame(agg_rows)

# Preserve scores from previous runs for already-evaluated traces
if len(already_evaluated) > 0 and len(existing_agg) > 0:
    prev_map = (
        existing_agg[existing_agg["trace_id"].isin(already_evaluated)]
        .set_index("trace_id")[["word_count_check", "polite_tone", "hook_in_post"]]
        .to_dict("index")
    )
    for i, row in agg_pdf.iterrows():
        if row["trace_id"] in prev_map:
            for col in ("word_count_check", "polite_tone", "hook_in_post"):
                agg_pdf.at[i, col] = prev_map[row["trace_id"]][col]

# Ensure string columns are clean
for col in [
    "trace_id",
    "request_time",
    "request_preview",
    "response_text",
]:
    agg_pdf[col] = agg_pdf[col].fillna("").astype(str)

logger.info(f"Aggregated rows: {len(agg_pdf)}")

# Write to Delta table
spark.sql(f"DROP TABLE IF EXISTS {aggregated_table}")
agg_sdf = spark.createDataFrame(agg_pdf)
agg_sdf.write.mode("overwrite").saveAsTable(aggregated_table)

logger.info(f"Written {aggregated_table} with {len(agg_pdf)} rows")

# COMMAND ----------
