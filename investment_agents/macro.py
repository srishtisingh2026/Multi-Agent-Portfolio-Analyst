from agents import Agent, WebSearchTool, ModelSettings, function_tool
from tools import get_fred_series
from utils import load_prompt, DISCLAIMER

default_model = "llama-3.1-8b-instant"
default_search_context = "medium"
RECENT_DAYS = 45

def build_macro_agent():
    tool_retry_instructions = load_prompt("tool_retry_prompt.md")
    macro_prompt = load_prompt("macro_base.md", RECENT_DAYS=RECENT_DAYS)
    @function_tool
    def web_search(query: str) -> str:
        """Search the web for up-to-date information."""
        return f"Search result for '{query}': Due to Groq migration, real-time web search is currently using simulated results. [Search: {query}]"

    return Agent(
        name="Macro Analysis Agent",
        instructions=(macro_prompt + DISCLAIMER + tool_retry_instructions),
        tools=[web_search, get_fred_series],
        model=default_model,
        model_settings=ModelSettings(parallel_tool_calls=True, temperature=0),
    ) 