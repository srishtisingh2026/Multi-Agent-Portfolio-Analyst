import streamlit as st
import asyncio
import os
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the core logic from main and agents
from investment_agents.config import build_investment_agents
from agents import Runner
from agents.tracing import set_tracing_disabled
from contextlib import AsyncExitStack
from utils import output_file, repo_path, load_prompt, DISCLAIMER, global_tracer

# Disables standard OpenAI remote telemetry/tracing to avoid 401 errors
set_tracing_disabled(True)

# Set up Page Config
st.set_page_config(
    page_title="Multi-Agent Portfolio Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .agent-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        background-color: #161b22;
        margin-bottom: 10px;
    }
    .report-card {
        padding: 20px;
        border-radius: 15px;
        background: #0d1117;
        border: 1px solid #4b6cb7;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- PROFESSIONAL LIGHT THEME STYLING ---
st.markdown("""
<style>
    /* CLEAN WHITE THEME */
    .stApp {
        background-color: #ffffff !important;
        color: #1e293b !important;
    }
    p, span, label, li, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #1e293b !important;
    }
    /* BUTTONS: WHITE TEXT ON DARK BACKGROUND */
    .stButton > button {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100%;
        transition: all 0.2s ease !important;
    }
    /* FORCE TEXT COLOR TO WHITE */
    .stButton > button p, .stButton > button div, .stButton > button span {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    /* Cards and Info */
    .custom-info {
        background: #f1f5f9;
        border-left: 5px solid #3b82f6;
        padding: 10px;
        color: #1e293b !important;
    }
    .custom-success {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 10px;
        color: #1e293b !important;
    }
    .report-card, .custom-card {
        background: #ffffff;
        padding: 20px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper for custom info
def custom_info(text):
    st.markdown(f'<div class="custom-info">{text}</div>', unsafe_allow_html=True)

def custom_success(text):
    st.markdown(f'<div class="custom-success">{text}</div>', unsafe_allow_html=True)

def clean_output(text: str) -> str:
    """Aggressively remove all technical noise, XML/JSON tags, and log prefixes."""
    import re
    if not text:
        return ""
    # (Removed aggressive XML and JSON regex stripping to prevent destroying valid outputs)
    
    # 3. Remove common technical headers/prefixes
    prefixes = [
        "Thought:", "Action:", "Action Input:", "Observation:", 
        "Final Answer:", "Tool Call:", "Calling tool:", "Output:"
    ]
    for p in prefixes:
        text = text.replace(p, "")
        
    # 4. Clean up formatting and whitespace
    text = text.strip()
    return text

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3665/3665922.png", width=80)
    st.title("SETTINGS")
    st.divider()
    st.divider()
    custom_info("💡 PRO-TIP: Try MIXTRAL if Llama is rate-limited. It usually has a separate data bucket.")
    st.divider()
    # High-compatibility options
    model_choice = st.selectbox("MODEL", [
        "llama-3.3-70b-versatile", 
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant"
    ], index=2)
    temp = st.slider("TEMP", 0.0, 1.0, 0.0)
    
st.title("🔭 PORTFOLIO ANALYST")
st.markdown("### COORDINATED MULTI-AGENT RESEARCH")

# User Input
question = st.text_area("INVESTMENT THESIS REQUEST", value=(
    "Analyze the impact of a potential 25bps interest rate cut on tech stocks, "
    "specifically focusing on NVIDIA (NVDA). What is a realistic price target?"
), height=100)

col1, col2 = st.columns([1.2, 0.8])

with col1:
    if st.button("RUN COMPREHENSIVE ANALYSIS"):
        import shutil
            
        if os.path.exists("./outputs"):
            for f in os.listdir("./outputs"):
                file_path = os.path.join("./outputs", f)
                try: 
                    if os.path.isfile(file_path): os.unlink(file_path)
                except: pass

            with st.status("investigating...", expanded=True) as status:
                st.write(f"Waking up {model_choice} Brain...")
                bundle = build_investment_agents(model=model_choice)
                
                async def run_analysis():
                    async with AsyncExitStack() as stack:
                        try:
                            # Handle MCP connections
                            for agent in [bundle.fundamental, bundle.quant]:
                                for server in getattr(agent, "mcp_servers", []):
                                    await server.connect()
                                    await stack.enter_async_context(server)

                            global_tracer.start_trace()
                            
                            @global_tracer.trace(span_type="agent", name="Agent: Head Portfolio Manager")
                            async def execute_agent():
                                return await Runner.run(bundle.head_pm, question, max_turns=40)

                            result = None
                            try:
                                result = await execute_agent()
                                return result
                            finally:
                                out_msg = result.final_output if hasattr(result, 'final_output') else (str(result) if result else "Execution Failed")
                                global_tracer.export_trace(
                                    out_msg,
                                    query=question,
                                    session_id="streamlit-session"
                                )
                        except Exception as e:
                            err_msg = str(e)
                            st.error(f"Execution Error: {err_msg}")
                            return None

                response = asyncio.run(run_analysis())
                
                if response:
                    status.update(label="Analysis Complete", state="complete", expanded=False)
                else:
                    status.update(label="Analysis Failed", state="error", expanded=True)

        if response:
            processed_output = clean_output(response.final_output)
            custom_success("INVESTMENT MEMO READY")
            
            # --- EXECUTIVE SUMMARY (TL;DR) ---
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.subheader("💡 EXECUTIVE SUMMARY (TL;DR)")
            # Extract first logical block or first 400 chars for a punchy summary
            summary_parts = processed_output.split('\n\n')
            tldr = summary_parts[0] if len(summary_parts[0]) > 50 else processed_output[:400]
            st.markdown(tldr)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- FULL DETAILED ANALYSIS ---
            with st.expander("📄 VIEW FULL DETAILED ANALYSIS", expanded=True):
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.markdown(processed_output)
                st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("ASSETS")
    asset_path = "./outputs"
    if os.path.exists(asset_path):
        files = [f for f in os.listdir(asset_path) if f.endswith(('.csv', '.png', '.json', '.md'))]
        if files:
            for f in sorted(files):
                with open(os.path.join(asset_path, f), "rb") as file:
                    st.download_button(label=f"DOWNLOAD {f}", data=file, file_name=f, key=f"dl_{f}")
        else:
            st.caption("NO FILES YET")

st.divider()
st.caption(f"INSTANCE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
