from agents import Agent, ModelSettings, function_tool, Runner, RunContextWrapper
from tools import write_markdown, read_file, list_output_files
from utils import load_prompt, DISCLAIMER, global_tracer
from pydantic import BaseModel
import json

default_model = "llama-3.1-8b-instant"

class MemoEditorInput(BaseModel):
    fundamental: str
    macro: str
    quant: str
    pm: str = ""
    files: list[str] = []

def build_editor_agent(model="llama-3.3-70b-versatile"):
    editor_prompt = load_prompt("editor_base.md")
    tool_retry_instructions = load_prompt("tool_retry_prompt.md")

    return Agent(
        name="Investment Editor",
        instructions=(editor_prompt + DISCLAIMER + tool_retry_instructions),
        tools=[write_markdown, read_file, list_output_files],
        model=model,
        model_settings=ModelSettings(temperature=0),
    )

def build_memo_edit_tool(editor):
    @function_tool(
        name_override="memo_editor",
        description_override="Stitch analysis sections into a Markdown memo and save it. This is the ONLY way to generate and save the final investment report. All memos must be finalized through this tool.",
    )
    @global_tracer.trace(span_type="chain", name="Delegate to: Memo Editor")
    async def memo_edit_tool(
        ctx: RunContextWrapper,
        fundamental: str,
        macro: str,
        quant: str,
        pm: str | None = "",
        files: list[str] | None = None
    ) -> str:
        if files is None: files = []
        if pm is None: pm = ""
        input_data = {
            "fundamental": fundamental,
            "macro": macro,
            "quant": quant,
            "pm": pm,
            "files": files
        }

        @global_tracer.trace(span_type="agent", name=f"Agent: {editor.name}")
        async def _run_agent():
            return await Runner.run(
                starting_agent=editor,
                input=json.dumps(input_data),
                context=ctx.context,
                max_turns=40,
            )
        
        result = await _run_agent()
        return result.final_output
    return memo_edit_tool 