import datetime
import json
import os
import asyncio
from pathlib import Path
from contextlib import AsyncExitStack
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import openai
# Configure for Groq if GROQ_API_KEY is provided
if os.environ.get("GROQ_API_KEY"):
    openai.api_key = os.environ["GROQ_API_KEY"]
    openai.base_url = "https://api.groq.com/openai/v1"
    print("Using Groq API with llama-3.1-8b-instant...")
elif os.environ.get("OPENAI_API_KEY"):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    # Check if a custom base URL is provided
    if os.environ.get("OPENAI_BASE_URL"):
        openai.base_url = os.environ["OPENAI_BASE_URL"]

# OpenAI Agents SDK imports
from agents import Runner, add_trace_processor, trace
from agents.tracing.processors import BatchTraceProcessor


# Local imports from the portfolio example
from utils import FileSpanExporter, output_file
from investment_agents.config import build_investment_agents

# Setting up basic tracing provided by the SDK example - DISABLED for Groq migration
# add_trace_processor(BatchTraceProcessor(FileSpanExporter()))

async def run_portfolio_analysis(question: str):
    """
    Runs the multi-agent portfolio analysis workflow.
    """
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY not set in environment variables.")
        return

    # Initialize the agent bundle (Head PM, Macro, Fundamental, Quant)
    bundle = build_investment_agents()

    async with AsyncExitStack() as stack:
        # Connect to any necessary MCP servers (e.g., Yahoo Finance)
        for agent in [getattr(bundle, "fundamental", None), getattr(bundle, "quant", None)]:
            if agent is None:
                continue
            for server in getattr(agent, "mcp_servers", []):
                await server.connect()
                await stack.enter_async_context(server)

        print("-" * 50)
        print("🚀 Starting Multi-Agent Portfolio Analysis")
        print(f"Question: {question}")
        print("-" * 50)

        # Tracing disabled to avoid OpenAI telemetry errors
        # with trace(
        #     "Investment Research Workflow",
        #     metadata={"question": question[:512]}
        # ) as workflow_trace:
        #     print(f"🔗 Trace ID: {workflow_trace.trace_id}")
            
        try:
            # Run the orchestrator agent (Head PM)
            response = await asyncio.wait_for(
                Runner.run(bundle.head_pm, question, max_turns=40),
                timeout=1200
            )
            
            # Check for output file
            report_path = None
            if hasattr(response, 'final_output'):
                output = response.final_output
                if isinstance(output, str):
                    try:
                        data = json.loads(output)
                        if isinstance(data, dict) and 'file' in data:
                            report_path = output_file(data['file'])
                    except json.JSONDecodeError:
                        pass

            print("\n✅ Workflow Completed")
            print(f"Final Report Path: {report_path if report_path else 'Generated internally'}")
            print("-" * 50)
            
            
        except asyncio.TimeoutError:
            print("\n❌ Workflow timed out after 20 minutes.")
        except Exception as e:
            print(f"\n❌ Error during execution: {e}")

if __name__ == "__main__":
    today_str = datetime.date.today().strftime("%B %d, %Y")
    default_question = (
        f"Today is {today_str}. "
        "Analyze the impact of a potential 25bps interest rate cut on tech stocks, "
        "specifically focusing on NVIDIA (NVDA). What is a realistic price target?"
    )
    
    asyncio.run(run_portfolio_analysis(default_question))
