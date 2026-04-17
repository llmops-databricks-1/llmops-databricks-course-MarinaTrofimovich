"""Common utilities for Databricks workflows."""

from pathlib import Path

from databricks.sdk.runtime import dbutils


def find_project_file(filename: str, max_levels: int = 5) -> str:
    """Find a file by searching the current directory and up to max_levels parents.

    Args:
        filename: Name of the file to find
        max_levels: Maximum number of parent directories to search

    Returns:
        Absolute path to the found file

    Raises:
        FileNotFoundError: If the file is not found
    """
    current = Path.cwd()
    for _ in range(max_levels):
        candidate = current / filename
        if candidate.exists():
            return str(candidate)
        current = current.parent
    raise FileNotFoundError(f"Could not find '{filename}' searching up from {Path.cwd()}")


def get_widget(name: str, default: str = "") -> str:
    """Get a Databricks widget value, falling back to a default.

    Args:
        name: Widget name
        default: Default value if widget is not set

    Returns:
        Widget value or default
    """
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default
