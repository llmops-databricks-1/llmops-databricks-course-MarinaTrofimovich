# course-code-hub

This repository contains course notebooks and a Databricks Asset Bundle for running an arXiv ingestion workflow in Databricks.

## What This Repo Does

The main runnable workflow in this repo is the ingestion job defined in [resources/arxiv_data_ingestion_job.yml](resources/arxiv_data_ingestion_job.yml), which runs the notebook [notebooks/1.3_arxiv_data_ingestion.py](notebooks/1.3_arxiv_data_ingestion.py).

That notebook:

1. Starts a Spark session in Databricks.
2. Loads environment-specific settings from [project_config.yml](project_config.yml).
3. Creates the configured Unity Catalog schema if it does not already exist.
4. Calls the arXiv API and fetches recent papers for `cs.AI` and `cs.LG`.
5. Converts the results into a Spark DataFrame.
6. Writes the data to a Unity Catalog Delta table named `arxiv_papers`.
7. Reads the table back and prints basic verification and summary statistics.

The table written by the notebook is:

`<catalog>.<schema>.arxiv_papers`

where `catalog` and `schema` come from [project_config.yml](project_config.yml).

## Notebook Classification

Not all notebooks in this repository should be treated the same way. Some are primarily for interactive learning, one is currently packaged as a deployable workflow, and some can create billable resources or depend on external credentials.

### Demo

#### [notebooks/1.1_foundation_models_overview.py](notebooks/1.1_foundation_models_overview.py)

Use this notebook interactively.

What it does:

1. Explains the difference between Foundation Model APIs, provisioned throughput, and external models.
2. Lists available Databricks serving endpoints in the workspace.
3. Sends a sample request to a Databricks-hosted model.
4. Compares pricing concepts.

Why it is classified as `demo`:

1. It is primarily educational.
2. It does not define a production workflow.
3. It is most useful for exploration inside a notebook session.

Recommendation:

Run it manually in Databricks if you want to explore model availability and understand the concepts. Do not deploy it as a scheduled job.

### Deployable

#### [notebooks/1.3_arxiv_data_ingestion.py](notebooks/1.3_arxiv_data_ingestion.py)

This is the main deployable workflow in the repo.

What it does:

1. Loads the selected environment from [project_config.yml](project_config.yml).
2. Creates the target schema if it does not already exist.
3. Fetches arXiv metadata.
4. Writes the results to the Delta table `arxiv_papers`.

Why it is classified as `deployable`:

1. It has a job definition in [resources/arxiv_data_ingestion_job.yml](resources/arxiv_data_ingestion_job.yml).
2. It is already wired into the Databricks Asset Bundle.
3. It performs a repeatable data-ingestion task with a clear output table.

Recommendation:

This is the notebook to validate, deploy, and run through the Databricks bundle workflow.

### Costly/Risky

#### [notebooks/1.2_provisioned_throughput_deployment.py](notebooks/1.2_provisioned_throughput_deployment.py)

What it does:

1. Creates a provisioned throughput serving endpoint.
2. Waits for the endpoint to become ready.
3. Sends a sample request to the endpoint.
4. Shows monitoring information and cost estimates.
5. Includes cleanup guidance.

Why it is classified as `costly/risky`:

1. It can create a billable Databricks model serving endpoint.
2. It contains hardcoded demo values such as endpoint name, model name, catalog, and schema.
3. Running it without adapting those values can fail or create resources you do not actually want.

Recommendation:

Do not deploy this by default. Only run it intentionally, after reviewing costs, model availability, permissions, and cleanup steps.

#### [notebooks/1.4_external_models_custom_provider.py](notebooks/1.4_external_models_custom_provider.py)

What it does:

1. Creates an external model endpoint backed by OpenAI.
2. Uses a Databricks secret reference for the provider API key.
3. Calls the endpoint to generate images.

Why it is classified as `costly/risky`:

1. It depends on a secret scope and secret key already existing in the workspace.
2. It can incur external provider charges.
3. It contains hardcoded demo endpoint and secret names.

Recommendation:

Do not deploy this by default. Only run it if you explicitly want external-model integration and have configured your own secret scope, secret key, and endpoint naming.

## Recommended Usage By Notebook

1. Use [notebooks/1.1_foundation_models_overview.py](notebooks/1.1_foundation_models_overview.py) as an interactive learning notebook.
2. Use [notebooks/1.3_arxiv_data_ingestion.py](notebooks/1.3_arxiv_data_ingestion.py) as the main deployable workflow.
3. Treat [notebooks/1.2_provisioned_throughput_deployment.py](notebooks/1.2_provisioned_throughput_deployment.py) as an intentional infrastructure exercise with potential cost.
4. Treat [notebooks/1.4_external_models_custom_provider.py](notebooks/1.4_external_models_custom_provider.py) as an intentional external-integration exercise with credential and cost implications.

## General Run Flow

The normal workflow for this repo is:

1. Prepare the local Python environment.
2. Authenticate the Databricks CLI to the target workspace.
3. Validate the Databricks bundle configuration.
4. Deploy the bundle to the Databricks workspace.
5. Run the deployed Databricks job.

In short, local commands prepare and package the code, while Databricks executes the actual workload.

## Prerequisites

Before running the workflow, make sure you have:

1. Python 3.12 installed.
2. `uv` installed.
3. The Databricks CLI installed.
4. Access to a Databricks workspace.
5. A Databricks catalog that you are allowed to use.
6. Compute in Databricks with Unity Catalog access.

For lecture 1.3 specifically, the required config values are mostly:

1. `catalog`
2. `schema`

The following values exist in [project_config.yml](project_config.yml), but are not required to run the 1.3 ingestion notebook itself:

1. `warehouse_id`
2. `llm_endpoint`
3. `embedding_endpoint`
4. `vector_search_endpoint`
5. `genie_space_id`

## Configuration Files

### [databricks.yml](databricks.yml)

This file defines the Databricks Asset Bundle.

It tells Databricks:

1. The bundle name.
2. How to build the Python artifact.
3. Which resource files to include.
4. Which workspace host to deploy to.
5. Where in the workspace the bundle should be stored.

### [project_config.yml](project_config.yml)

This file contains environment-specific values used by the notebooks.

For example, the `dev` section supplies:

1. The Unity Catalog name.
2. The schema name.
3. Other workspace resource identifiers used by later notebooks.

### [resources/arxiv_data_ingestion_job.yml](resources/arxiv_data_ingestion_job.yml)

This file defines the Databricks job.

It tells Databricks:

1. The job name.
2. The notebook to run.
3. The wheel dependency to attach.
4. The `env` parameter passed to the notebook.

## Commands To Run

Use the following commands from the repository root.

### 1. Sync the local Python environment

```bash
uv sync --extra dev
```

Why we need this:

1. It creates or updates the local virtual environment.
2. It installs the dependencies declared in [pyproject.toml](pyproject.toml).
3. It includes the `dev` extras needed for local development.

What it does:

1. Resolves dependency versions.
2. Creates `.venv` if needed.
3. Installs the package for this repo and its dependencies.

When to run it:

1. The first time you use the repo.
2. Any time dependencies change.

### 2. Log in to Databricks CLI

```bash
databricks auth login --host https://adb-1065629792770920.0.azuredatabricks.net
```

Why we need this:

1. The Databricks CLI must know which workspace to talk to.
2. The CLI must have credentials before it can validate, deploy, or run the bundle.

What it does:

1. Opens an authentication flow.
2. Saves credentials to a local Databricks CLI profile.
3. Associates that profile with the provided workspace host.

Important note:

If you already have multiple local Databricks CLI profiles for the same host, bundle commands may fail with an ambiguous profile error. In that case, explicitly pass `--profile <profile-name>` to bundle commands.

### 3. Validate the Databricks bundle

```bash
databricks bundle validate -t dev --profile adb-1065629792770920
```

Why we need this:

1. It checks the bundle before deployment.
2. It catches configuration problems early.

What it does:

1. Reads [databricks.yml](databricks.yml).
2. Loads included job resources such as [resources/arxiv_data_ingestion_job.yml](resources/arxiv_data_ingestion_job.yml).
3. Resolves variables and target-specific settings for `dev`.
4. Verifies that the configuration is structurally valid.

What `-t dev` means:

It selects the `dev` target from [databricks.yml](databricks.yml).

What `--profile adb-1065629792770920` means:

It tells the CLI exactly which saved local Databricks credentials to use.

### 4. Deploy the Databricks bundle

```bash
databricks bundle deploy -t dev --profile adb-1065629792770920
```

Why we need this:

1. Validation only checks the config.
2. Deployment actually publishes the code and job definition into the Databricks workspace.

What it does:

1. Builds the wheel artifact using the `build: uv build` instruction in [databricks.yml](databricks.yml).
2. Uploads bundle files to the workspace path for the selected target.
3. Creates or updates the Databricks job resources defined under [resources/arxiv_data_ingestion_job.yml](resources/arxiv_data_ingestion_job.yml).

After this step, the Databricks job exists in the workspace and is ready to run.

### 5. Run the Databricks job

```bash
databricks bundle run arxiv_data_ingestion_job -t dev --profile adb-1065629792770920
```

Why we need this:

1. This is the command that actually executes the workflow.
2. Everything before this step only prepares the environment and deployment.

What it does:

1. Starts the Databricks job named `arxiv_data_ingestion_job`.
2. Runs [notebooks/1.3_arxiv_data_ingestion.py](notebooks/1.3_arxiv_data_ingestion.py) in Databricks.
3. Passes `env=dev` as a base parameter.
4. Causes the notebook to load the `dev` block from [project_config.yml](project_config.yml).
5. Fetches data from arXiv and writes the `arxiv_papers` Delta table.

## End-To-End Example

```bash
uv sync --extra dev
databricks auth login --host https://adb-1065629792770920.0.azuredatabricks.net
databricks bundle validate -t dev --profile adb-1065629792770920
databricks bundle deploy -t dev --profile adb-1065629792770920
databricks bundle run arxiv_data_ingestion_job -t dev --profile adb-1065629792770920
```

## Common Points Of Confusion

### Bundle target vs CLI profile

These are different things:

1. `-t dev` selects the deployment target from [databricks.yml](databricks.yml).
2. `--profile adb-1065629792770920` selects the local saved Databricks CLI credentials.

### Do we need a SQL warehouse?

No, not for [notebooks/1.3_arxiv_data_ingestion.py](notebooks/1.3_arxiv_data_ingestion.py).

This notebook uses Spark compute with Unity Catalog access, not a SQL warehouse.

### Does the catalog need to exist first?

Yes, the catalog should already exist in Databricks.

The schema can be created by the notebook if you have permission, but the catalog is typically a pre-existing workspace resource.

## Expected Result

If the run succeeds, Databricks will create or update a Delta table:

`<catalog>.<schema>.arxiv_papers`

The table contains recent arXiv paper metadata including:

1. `arxiv_id`
2. `title`
3. `authors`
4. `summary`
5. `published`
6. `pdf_url`
7. `primary_category`

## Troubleshooting

### Multiple Databricks profiles matched

Use:

```bash
databricks bundle validate -t dev --profile adb-1065629792770920
```

or set the environment variable for the current shell:

```powershell
$env:DATABRICKS_CONFIG_PROFILE = "adb-1065629792770920"
```

### Local disk space errors during `uv sync`

This usually means the local disk is full while extracting Python packages.

Typical fixes:

1. Free disk space.
2. Remove `.venv` and retry.
3. Clear or relocate the `uv` cache.

## Summary

In general, we run this repo in two phases:

1. Local setup and packaging with `uv` and the Databricks CLI.
2. Remote execution in Databricks through the Asset Bundle job.

The most important operational command is:

```bash
databricks bundle run arxiv_data_ingestion_job -t dev --profile adb-1065629792770920
```

because that is the step that actually runs the notebook and loads data into Databricks.