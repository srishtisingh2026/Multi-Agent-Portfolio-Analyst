from agents import Agent, ModelSettings, function_tool, Runner
from utils import load_prompt, DISCLAIMER, global_tracer
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
    @global_tracer.trace(span_type="agent", name=f"Agent: {agent.name}")
    async def _run_agent():
        return await Runner.run(
            starting_agent=agent,
            input=json.dumps(input_data),
            max_turns=75,
        )
    
    result = await _run_agent()
    return result.final_output

@global_tracer.trace(span_type="chain", name="Parallel Specialist Orchestration")
async def run_all_specialists_parallel(
    fundamental, macro, quant,
    fundamental_q: str, macro_q: str, quant_q: str
):
    # Running sequentially with a mandated wait to avoid 6,000 Tokens/Min Groq Rate Limits
    res_fund = await specialist_analysis_func(fundamental, "fundamental", fundamental_q, "Analyze NVDA fundamentals.")
    await asyncio.sleep(30)  # Wait 30s to drain token bucket
    
    res_macro = await specialist_analysis_func(macro, "macro", macro_q, "Analyze macro impact.")
    await asyncio.sleep(30)
    
    res_quant = await specialist_analysis_func(quant, "quant", quant_q, "Perform quant analysis.")
    
    return {
        "fundamental": res_fund,
        "macro": res_macro,
        "quant": res_quant
    }

def build_head_pm_agent(fundamental, macro, quant, memo_edit_tool, model="llama-3.3-70b-versatile"):
    def make_agent_tool(agent, name, description):
        @function_tool(name_override=name, description_override=description)
        @global_tracer.trace(span_type="chain", name=f"Delegate to: {name}")
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
    @global_tracer.trace(span_type="chain", name="Parallel Coordination Tool")
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
        model_settings=ModelSettings(parallel_tool_calls=False, tool_choice="auto", temperature=0)
    )