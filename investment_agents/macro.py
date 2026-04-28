from agents import Agent, ModelSettings
from tools import get_fred_series, read_file, list_output_files, search_web
from utils import load_prompt, DISCLAIMER

default_model = "llama-3.1-8b-instant"
default_search_context = "medium"
RECENT_DAYS = 45

def build_macro_agent(model=default_model):
    tool_retry_instructions = load_prompt("tool_retry_prompt.md")
    macro_prompt = load_prompt("macro_base.md", RECENT_DAYS=RECENT_DAYS)

    return Agent(
        name="Macro Analysis Agent",
        instructions=(macro_prompt + DISCLAIMER + tool_retry_instructions),
        tools=[search_web, get_fred_series, read_file, list_output_files],
        model=model,
        model_settings=ModelSettings(parallel_tool_calls=True, temperature=0),
    )