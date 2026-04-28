from agents import Agent, ModelSettings, function_tool, Runner
from utils import load_prompt, DISCLAIMER
from dataclasses import dataclass
from pydantic import BaseModel
import json
import asyncio


class SpecialistRequestInput(BaseModel):
    section: str  # e.g., 'fundamental', 'macro', 'quant', or 'pm'
    user_question: str
    guidance: str


# Specialist analysis function with flat arguments for better LLM compatibility
async def specialist_analysis_func(agent, section: str, user_question: str, guidance: str):
    input_data = {
        "section": section,
        "user_question": user_question,
        "guidance": guidance
    }
    result = await Runner.run(
        starting_agent=agent,
        input=json.dumps(input_data),
        max_turns=75,
    )
    return result.final_output

async def run_all_specialists_parallel(
    fundamental, macro, quant,
    fundamental_q: str, macro_q: str, quant_q: str
):
    # Simplified inputs for parallel run
    results = await asyncio.gather(
        specialist_analysis_func(fundamental, "fundamental", fundamental_q, "Analyze NVDA fundamentals."),
        specialist_analysis_func(macro, "macro", macro_q, "Analyze macro impact."),
        specialist_analysis_func(quant, "quant", quant_q, "Perform quant analysis.")
    )
    return {
        "fundamental": results[0],
        "macro": results[1],
        "quant": results[2]
    }

def build_head_pm_agent(fundamental, macro, quant, memo_edit_tool, model="llama-3.3-70b-versatile"):
    def make_agent_tool(agent, name, description):
        @function_tool(name_override=name, description_override=description)
        async def agent_tool(section: str, user_question: str, guidance: str):
            """
            Analyze a specific section of the investment report.
            """
            return await specialist_analysis_func(agent, section, user_question, guidance)
        return agent_tool

    # PM Tools
    pm_tools = [
        make_agent_tool(fundamental, "fundamental_analysis", "Analyze company financials, earnings, and valuation."),
        make_agent_tool(macro, "macro_analysis", "Analyze economic indicators, rates, and geopolitical impact."),
        make_agent_tool(quant, "quantitative_analysis", "Perform statistical modeling and data-driven price targets."),
        memo_edit_tool
    ]

    # Batch coordination tool
    @function_tool
    async def run_all_specialists_parallel_tool(fundamental_q: str, macro_q: str, quant_q: str):
        """
        Coordinate all specialist analysts in parallel for a comprehensive review.
        """
        return await run_all_specialists_parallel(fundamental, macro, quant, fundamental_q, macro_q, quant_q)

    pm_tools.append(run_all_specialists_parallel_tool)

    tool_retry_instructions = load_prompt("tool_retry_prompt.md")

    return Agent(
        name="Head Portfolio Manager Agent",
        instructions=(
            load_prompt("pm_base.md") + DISCLAIMER + tool_retry_instructions
        ),
        model=model,
        tools=pm_tools,
        model_settings=ModelSettings(parallel_tool_calls=True, tool_choice="auto", temperature=0)
    )