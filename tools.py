# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------

import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import re

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import pandas as pd  # pandas is a required dependency
import requests
from fredapi import Fred
from openai import OpenAI

# ---------------------------------------------------------------------------
# Local package imports
# ---------------------------------------------------------------------------

from agents import function_tool
from utils import outputs_dir, output_file, global_tracer

# ---------------------------------------------------------------------------
# Repository paths & globals
# ---------------------------------------------------------------------------

OUTPUT_DIR = outputs_dir()
PROMPT_PATH = Path(__file__).parent / "prompts" / "code_interpreter.md"
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    CODE_INTERPRETER_INSTRUCTIONS = f.read()

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def code_interpreter_error_handler(ctx, error):
    """
    Custom error handler for run_code_interpreter. Returns a clear message to the LLM about what went wrong and how to fix it.
    """
    return (
        "Error running code interpreter. "
        "You must provide BOTH a clear natural language analysis request and a non-empty list of input_files (relative to outputs/). "
        f"Details: {str(error)}"
    )

@function_tool(failure_error_function=code_interpreter_error_handler)
@global_tracer.trace(span_type="tool")
async def run_code_interpreter(request: str, input_files: list[str]) -> str:
    """
    Executes a quantitative analysis request using a local Python environment.
    Note: In this Groq-compatible version, we use local execution instead of OpenAI's cloud sandbox.

    Args:
        request (str): A clear, quantitative analysis request describing the specific computation.
        input_files (list[str]): List of file paths (relative to outputs/) containing data.

    Returns:
        str: JSON string with the analysis results.
    """
    # 1. Initialize client using environment variables (Groq-compatible)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    )
    
    # 2. Build the detailed request for the code generator
    system_prompt = CODE_INTERPRETER_INSTRUCTIONS
    user_prompt = f"Request: {request}\nInput Files: {input_files}\n\nRemember: All files are in ./outputs/"
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", # Attempting 8b instead to avoid rate limits
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        code_response = completion.choices[0].message.content
    except Exception as e:
        # Fallback to 8b if rate limited or other error
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        code_response = completion.choices[0].message.content

    # 3. Extract the Python code from the response
    import re
    code_match = re.search(r"```python\n(.*?)```", code_response, re.DOTALL)
    code_to_run = code_match.group(1) if code_match else code_response
    
    # Clean up potential markdown formatting if extract failed
    code_to_run = code_to_run.replace("```python", "").replace("```", "").strip()

    # 4. Monitor the outputs directory for new files
    files_before = set(os.listdir(OUTPUT_DIR))

    # 5. Execute the code in a subprocess
    import subprocess
    import sys
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", code_to_run],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        execution_out = result.stdout
        execution_err = result.stderr
        
    except subprocess.TimeoutExpired:
        execution_out = ""
        execution_err = "Error: Code execution timed out after 120 seconds."
    except Exception as e:
        execution_out = ""
        execution_err = f"Error during subprocess execution: {str(e)}"

    # 6. Identify new files generated
    files_after = set(os.listdir(OUTPUT_DIR))
    new_files = list(files_after - files_before)

    return json.dumps({
        "analysis_output": execution_out,
        "execution_errors": execution_err,
        "files": new_files,
        "thinking": "Generated and executed local python analysis."
    })

@function_tool
@global_tracer.trace(span_type="tool")
async def write_markdown(filename: str, content: str) -> str:
    """Write `content` to `outputs/filename` and return confirmation JSON."""
    if not filename.endswith(".md"):
        filename += ".md"
    path = output_file(filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return json.dumps({
            "status": "success",
            "file": filename,
            "message": "SUCCESS: The file has been written successfully. You have completed your task. Do NOT call this tool again. Output the word 'Finished' to the user and stop."
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to write file: {str(e)}"
        })

@function_tool
@global_tracer.trace(span_type="tool")
async def read_file(filename: str, n_rows: int = 10) -> str:
    """
    Read and preview the contents of a file from the outputs directory.

    Supports reading CSV, Markdown (.md), and plain text (.txt) files. For CSV files, returns a preview of the last `n_rows` as a Markdown table. For Markdown and text files, returns the full text content. For unsupported file types, returns an error message.

    Args:
        filename: The name of the file to read, relative to the outputs directory. Supported extensions: .csv, .md, .txt.
        n_rows: The number of rows to preview for CSV files (default: 10).

    Returns:
        str: A JSON string containing either:
            - For CSV: {"file": filename, "preview_markdown": "<markdown table>"}
            - For Markdown/Text: {"file": filename, "content": "<text content>"}
            - For errors: {"error": "<error message>", "file": filename}
    """
    path = output_file(filename, make_parents=False)
    if not path.exists():
        return json.dumps({"error": "file not found", "file": filename})

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path).tail(n_rows)
            table_md = df.to_markdown(index=False)
            return json.dumps({"file": filename, "preview_markdown": table_md})
        except Exception as e:
            return json.dumps({"error": str(e), "file": filename})
    elif suffix == ".md" or suffix == ".txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            return json.dumps({"file": filename, "content": content})
        except Exception as e:
            return json.dumps({"error": str(e), "file": filename})
    else:
        return json.dumps({"error": f"Unsupported file type: {suffix}", "file": filename})

@function_tool
@global_tracer.trace(span_type="tool")
async def get_fred_series(series_id: str, start_date: str, end_date: str, download_csv: bool = False) -> str:
    """Fetches a FRED economic time-series and returns simple summary statistics.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. "GDP" or "UNRATE".
    start_date : str
        ISO date string (YYYY-MM-DD).
    end_date : str
        ISO date string (YYYY-MM-DD).

    Returns
    -------
    str
        JSON string with basic statistics (mean, latest value, etc.). Falls back to a
        placeholder if fredapi is not available or an error occurs.
    """
    # Treat empty strings as unspecified
    start_date = start_date or None  # type: ignore
    end_date = end_date or None  # type: ignore

    if Fred is None:
        return json.dumps({"error": "fredapi not installed. returning stub result", "series_id": series_id})

    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        fred = Fred(api_key=fred_api_key)
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if data is None or data.empty:
            return json.dumps({"error": "Series not found or empty", "series_id": series_id})

        summary = {
            "series_id": series_id,
            "observations": len(data),
            "start": str(data.index.min().date()),
            "end": str(data.index.max().date()),
            "latest": float(data.iloc[-1]),
            "mean": float(data.mean()),
        }

        # ------------------------------------------------------------------
        # Optional CSV download
        # ------------------------------------------------------------------
        if download_csv:
            # Reset index to turn the DatetimeIndex into a column for CSV output
            df = data.reset_index()
            df.columns = ["Date", series_id]  # Capital D to match Yahoo Finance

            # Build date_range string for filename (YYYYMMDD-YYYYMMDD).
            start_str = start_date if start_date else str(df["Date"].min().date())
            end_str = end_date if end_date else str(df["Date"].max().date())
            date_range = f"{start_str}_{end_str}".replace("-", "")
            file_name = f"{series_id}_{date_range}.csv"

            # Save under outputs/
            csv_path = output_file(file_name)
            df.to_csv(csv_path, index=False)

            # Add file metadata to summary
            summary["file"] = file_name
            summary["schema"] = ["Date", series_id]

        return json.dumps(summary)
    except Exception as e:
        return json.dumps({"error": str(e), "series_id": series_id})

@function_tool
@global_tracer.trace(span_type="tool")
async def list_output_files(extension: str = None) -> str:
    """
    List all files in the outputs directory. Optionally filter by file extension (e.g., 'png', 'csv', 'md').
    Returns a JSON list of filenames.
    """
    out_dir = outputs_dir()
    if extension:
        files = [f.name for f in out_dir.glob(f'*.{extension}') if f.is_file()]
    else:
        files = [f.name for f in out_dir.iterdir() if f.is_file()]
    return json.dumps({"files": files})

# Public interface -----------------------------------------------------------

__all__ = [
    "run_code_interpreter",
    "write_markdown",
    "get_fred_series",
    "list_output_files",
    "read_file",
    "search_web",
] 
@function_tool
@global_tracer.trace(span_type="tool")
async def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web for up-to-date news and information using DuckDuckGo.
    """
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            if not results:
                return f"No results found for '{query}'."
            output = []
            for r in results:
                output.append(f"Title: {r.get('title')}\\nSource: {r.get('href')}\\nSnippet: {r.get('body')}\\n")
            return "\\n".join(output)
    except Exception as e:
        return f"Web search failed: {str(e)}"
