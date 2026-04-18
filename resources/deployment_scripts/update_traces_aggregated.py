# Databricks notebook source
import ast as _ast
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
mlflow.set_experiment(cfg.experiment_name)

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.schema
aggregated_table = f"{catalog}.{schema}.arxiv_traces_aggregated"
endpoint_name = f"arxiv-agent-{env}"
payload_table = f"{catalog}.{schema}.arxiv_agent_{env}_payload"

# COMMAND ----------
# Fetch traces from MLflow experiment using search_traces()

experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
experiment_traces_pdf = mlflow.search_traces(
    locations=[experiment.experiment_id],
    max_results=1000,
)
experiment_traces_pdf = experiment_traces_pdf.reset_index()
print(f"Experiment traces found: {len(experiment_traces_pdf)}")

# COMMAND ----------
# Fetch traces from the serving endpoint inference (payload) table

try:
    payload_sdf = spark.read.table(payload_table)
    payload_pdf = payload_sdf.toPandas()
    logger.info(f"Inference table rows found: {len(payload_pdf)}")
    print(f"Inference table columns: {list(payload_pdf.columns)}")
    if len(payload_pdf) > 0:
        print(f"Sample row keys: {list(payload_pdf.iloc[0].index)}")
except Exception as e:
    logger.warning(f"Could not read inference table {payload_table}: {e}")
    payload_pdf = pd.DataFrame()

# COMMAND ----------
# Normalize inference table rows into the same shape as experiment traces


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


def normalize_payload_row(row: pd.Series) -> dict:
    """Convert an inference table row into the trace-like dict used downstream."""
    request = _safe_json_loads(row.get("request"))
    response = _safe_json_loads(row.get("response"))
    trace_id = str(row.get("databricks_request_id", ""))
    request_time = str(row.get("request_time", ""))
    return {
        "trace_id": trace_id,
        "request": request,
        "response": response,
        "request_time": request_time,
        "execution_duration": row.get("execution_duration_ms", 0),
        "assessments": [],
        "spans": [],
        "source": "inference_table",
    }


if len(payload_pdf) > 0:
    # Only keep successful requests and deduplicate
    payload_pdf = payload_pdf[payload_pdf["status_code"] == 200]
    payload_pdf = payload_pdf.drop_duplicates(
        subset=["databricks_request_id"], keep="first"
    )
    payload_rows = [normalize_payload_row(r) for _, r in payload_pdf.iterrows()]
    payload_traces_pdf = pd.DataFrame(payload_rows)
    logger.info(f"Normalized {len(payload_traces_pdf)} inference table traces")
else:
    payload_traces_pdf = pd.DataFrame()

# COMMAND ----------
# Merge experiment traces and inference table traces, deduplicate by trace_id

experiment_traces_pdf["source"] = "experiment"
if "assessments" not in experiment_traces_pdf.columns:
    experiment_traces_pdf["assessments"] = None
if "spans" not in experiment_traces_pdf.columns:
    experiment_traces_pdf["spans"] = None

all_sources = [experiment_traces_pdf]
if len(payload_traces_pdf) > 0:
    all_sources.append(payload_traces_pdf)

traces_pdf = pd.concat(all_sources, ignore_index=True)
traces_pdf = traces_pdf.drop_duplicates(subset=["trace_id"], keep="first")
print(f"Total merged traces: {len(traces_pdf)}")

# COMMAND ----------
# Extract request/response text and filter unevaluated traces


def _parse_column(value: object) -> dict:
    """Parse a column value that may be a JSON string, Python repr, or dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    # Try JSON first
    try:
        return _json.loads(value)
    except (ValueError, _json.JSONDecodeError):
        pass
    # Try Python literal (handles None, True, False, single quotes)
    try:
        return _ast.literal_eval(value)
    except Exception:
        return {}


def extract_response_text(trace_row: pd.Series) -> str:
    """Extract assistant response text from the 'response' column."""
    try:
        resp = _parse_column(trace_row.get("response", ""))
        if not resp:
            return ""
        # Look for output items with type=message
        outputs = resp.get("output", []) if isinstance(resp, dict) else []
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list):
                    return content[0].get("text", "")
        return str(resp)[:500] if resp else ""
    except Exception:
        return str(trace_row.get("response", ""))[:500]


def extract_request_text(trace_row: pd.Series) -> str:
    """Extract user query from the 'request' column."""
    try:
        req = _parse_column(trace_row.get("request", ""))
        if not req:
            return ""
        # Unwrap nested {'request': {...}} wrapper if present
        if "request" in req and isinstance(req["request"], dict):
            req = req["request"]
        # OpenAI-style: {"messages": [{"role": "user", "content": "..."}]}
        for key in ("messages", "input"):
            msgs = req.get(key, []) if isinstance(req, dict) else []
            for msg in reversed(msgs):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        return " ".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    return str(content)
        return str(req)[:500] if req else ""
    except Exception:
        return str(trace_row.get("request", ""))[:500]


# Filter to ALL traces without assessments (not yet evaluated)
unevaluated = traces_pdf[
    traces_pdf["assessments"].apply(lambda x: x is None or len(x) == 0)
].copy()

# Track which trace IDs are from experiments (can use mlflow.log_feedback)
experiment_trace_ids = set(traces_pdf[traces_pdf["source"] == "experiment"]["trace_id"])

# Debug: test extraction on first trace
if len(traces_pdf) > 0:
    first = traces_pdf.iloc[0]
    raw_req = first.get("request", "")
    parsed_req = _parse_column(raw_req)
    keys = list(parsed_req.keys()) if isinstance(parsed_req, dict) else "N/A"
    print(f"DEBUG request type={type(raw_req).__name__} keys={keys}")
    print(f"DEBUG extracted request_text={extract_request_text(first)[:100]}")
    print(f"DEBUG extracted response_text={extract_response_text(first)[:100]}")

unevaluated["response_text"] = unevaluated.apply(extract_response_text, axis=1)
unevaluated["request_text"] = unevaluated.apply(extract_request_text, axis=1)
unevaluated = unevaluated.dropna(subset=["response_text"])

logger.info(f"Unevaluated traces: {len(unevaluated)}")

# COMMAND ----------
# Build eval input

eval_pdf = pd.DataFrame(
    {
        "trace_id": unevaluated["trace_id"].values,
        "inputs": unevaluated["request_text"].apply(lambda x: {"query": x}).values,
        "outputs": unevaluated["response_text"].values,
    }
)

skip_evaluation = len(eval_pdf) == 0
if skip_evaluation:
    logger.info(
        "No new traces to evaluate. Skipping evaluation, building aggregated table only."
    )

# COMMAND ----------
# Run word_count_check on all traces and log feedback

# Dict to store evaluation results for inference table traces
# (can't use mlflow.log_feedback for them — no MLflow trace ID)
eval_results: dict[str, dict] = {}

if not skip_evaluation:
    wc_result = mlflow.genai.evaluate(
        data=eval_pdf[["inputs", "outputs"]],
        scorers=[word_count_check],
    )

    for trace_id, assessments in zip(
        eval_pdf["trace_id"],
        wc_result.result_df["assessments"],
        strict=True,
    ):
        val = assessments[0]["feedback"]["value"]
        if trace_id in experiment_trace_ids:
            mlflow.log_feedback(
                trace_id=trace_id,
                name="word_count_check",
                value=val,
            )
        eval_results.setdefault(trace_id, {})["word_count_check"] = val

    logger.info(f"Evaluated word_count_check for {len(eval_pdf)} traces")

# COMMAND ----------
# Run LLM-judge scorers on a 10% sample and log feedback

if not skip_evaluation:
    sample_size = max(1, int(len(eval_pdf) * 0.1))
    sampled_pdf = eval_pdf.sample(n=sample_size)
    logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge evaluation")

    llm_result = mlflow.genai.evaluate(
        data=sampled_pdf[["inputs", "outputs"]],
        scorers=[polite_tone_guideline, hook_in_post_guideline],
    )

    for trace_id, assessments in zip(
        sampled_pdf["trace_id"],
        llm_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            name = a["assessment_name"]
            val = a["feedback"]["value"]
            if trace_id in experiment_trace_ids:
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name=name,
                    value=val,
                )
            eval_results.setdefault(trace_id, {})[name] = val

    logger.info(f"Evaluated polite_tone/hook_in_post for {len(sampled_pdf)} traces")

# COMMAND ----------
# Build aggregated table from all traces (experiment + inference)

# Re-fetch experiment traces to pick up newly logged assessments
traces_pdf_exp = mlflow.search_traces(
    locations=[experiment.experiment_id],
    max_results=1000,
)
traces_pdf_exp = traces_pdf_exp.reset_index(drop=True)
traces_pdf_exp["source"] = "experiment"

# Merge with inference table traces (they have no assessments)
all_final = [traces_pdf_exp]
if len(payload_traces_pdf) > 0:
    all_final.append(payload_traces_pdf)
traces_pdf_all = pd.concat(all_final, ignore_index=True)
traces_pdf_all = traces_pdf_all.drop_duplicates(subset=["trace_id"], keep="first")


def build_aggregated_row(row: pd.Series) -> dict:
    """Extract metrics from a trace for the aggregated table."""
    source = row.get("source", "experiment")
    result = {
        "trace_id": row.get("trace_id"),
        "source": source,
        "request_time": str(row.get("request_time", "")),
        "request_preview": extract_request_text(row),
        "response_text": extract_response_text(row),
        "latency_seconds": float(row.get("execution_duration") or 0) / 1000.0,
        "call_llm_exec_count": 0,
        "tool_call_count": 0,
        "total_tokens_used": 0,
    }
    # Use the spans column (list of span objects)
    spans = row.get("spans", []) or []
    for span in spans:
        span_name = span.name if hasattr(span, "name") else span.get("name", "")
        if span_name == "call_llm":
            result["call_llm_exec_count"] += 1
            outputs = (
                span.outputs if hasattr(span, "outputs") else span.get("outputs", {})
            )
            if isinstance(outputs, dict):
                usage = outputs.get("usage", {})
                if isinstance(usage, dict):
                    result["total_tokens_used"] += usage.get("total_tokens", 0)
        elif span_name == "execute_tool":
            result["tool_call_count"] += 1

    # Extract assessments - handle object, dict, or string formats
    assessments = row.get("assessments", []) or []
    assessment_map = {}
    for a in assessments:
        name, value = None, None
        if hasattr(a, "name") and hasattr(a, "value"):
            name, value = a.name, a.value
        elif isinstance(a, dict):
            name = a.get("name", "")
            value = a.get("value", "")
        elif isinstance(a, str):
            # Try parsing as JSON or Python literal
            parsed = _parse_column(a)
            if isinstance(parsed, dict):
                name = parsed.get("name", "")
                value = parsed.get(
                    "value",
                    parsed.get("feedback", {}).get("value", "")
                    if isinstance(parsed.get("feedback"), dict)
                    else "",
                )
        if name:
            assessment_map[name] = value

    # word_count_check is a bool scorer → value is True/False
    wc_val = assessment_map.get("word_count_check")
    # Guidelines scorers → value is "yes"/"no"
    pt_val = assessment_map.get("polite_tone")
    hp_val = assessment_map.get("hook_in_post")

    # Overlay eval_results for inference table traces (no MLflow assessments)
    trace_id = row.get("trace_id")
    if trace_id in eval_results:
        er = eval_results[trace_id]
        if "word_count_check" in er:
            wc_val = er["word_count_check"]
        if "polite_tone" in er:
            pt_val = er["polite_tone"]
        if "hook_in_post" in er:
            hp_val = er["hook_in_post"]

    result["word_count_check"] = (
        1 if wc_val is True or str(wc_val).lower() == "true" else 0
    )
    result["polite_tone"] = 1 if str(pt_val).lower() in ("yes", "true", "pass") else 0
    result["hook_in_post"] = 1 if str(hp_val).lower() in ("yes", "true", "pass") else 0

    return result


agg_rows = []
for i, (_, row) in enumerate(traces_pdf_all.iterrows()):
    agg_row = build_aggregated_row(row)
    if i < 2:
        # Debug: show assessment parsing result for first 2 traces
        wc = agg_row["word_count_check"]
        pt = agg_row["polite_tone"]
        hp = agg_row["hook_in_post"]
        req = agg_row["request_preview"][:60]
        print(
            f"DEBUG agg trace={agg_row['trace_id'][:20]}..."
            f" wc={wc} pt={pt} hp={hp} req={req}"
        )
        # Also show raw assessment
        a_list = row.get("assessments", []) or []
        if a_list:
            a0 = a_list[0]
            print(f"  raw assessment[0] type={type(a0).__name__} repr={repr(a0)[:200]}")
    agg_rows.append(agg_row)
agg_pdf = pd.DataFrame(agg_rows)

# Ensure string columns are never None (avoids missing parquet columns)
for col in ["trace_id", "request_time", "request_preview", "response_text", "source"]:
    agg_pdf[col] = agg_pdf[col].fillna("").astype(str)

# Drop and recreate to avoid schema mismatch
spark.sql(f"DROP TABLE IF EXISTS {aggregated_table}")
agg_sdf = spark.createDataFrame(agg_pdf)
agg_sdf.write.mode("overwrite").saveAsTable(aggregated_table)

logger.info(f"Aggregated table {aggregated_table} written with {len(agg_pdf)} rows")

# COMMAND ----------
