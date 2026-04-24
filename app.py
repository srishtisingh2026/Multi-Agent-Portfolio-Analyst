import streamlit as st
import asyncio
import os
import json
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the core logic from main and agents
from investment_agents.config import build_investment_agents
from agents import Runner
from utils import output_file

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

# Sidebar
with st.sidebar:
    st.title("📈 Settings")
    st.info("Using Groq API with Llama-3.1-8b-instant")
    st.divider()
    model_choice = st.selectbox("Select Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"])
    temp = st.slider("Temperature", 0.0, 1.0, 0.0)
    
st.title("📈 Multi-Agent Portfolio Analyst")
st.subheader("Simultaneously coordinate fundamental, macro, and quantitative research.")

# User Input
default_q = (
    "Analyze the impact of a potential 25bps interest rate cut on tech stocks, "
    "specifically focusing on NVIDIA (NVDA). What is a realistic price target?"
)
question = st.text_area("Investment Question", value=default_q, height=100)

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Run Comprehensive Analysis"):
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
            st.error("API Keys missing in .env! Please configure your keys first.")
        else:
            with st.status("🤖 Agents are collaborating...", expanded=True) as status:
                st.write("Initializing Agent Bundle (PM, Macro, Fundamental, Quant)...")
                bundle = build_investment_agents()
                
                # We'll use a placeholder for the output to show progress
                log_container = st.container()
                
                async def run_analysis():
                    try:
                        st.write("🛰️ Connecting to Yahoo Finance MCP Servers...")
                        # Handle MCP connections simplified for web
                        st.write("🧠 Portfolio Manager is delegating tasks...")
                        
                        response = await Runner.run(bundle.head_pm, question, max_turns=40)
                        
                        st.write("✅ Analysis complete. Synthesizing report...")
                        return response
                    except Exception as e:
                        st.error(f"Execution Error: {e}")
                        return None

                response = asyncio.run(run_analysis())
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            if response:
                st.success("Investment Memo Generated Successfully!")
                
                # Check for output file
                report_path = None
                if hasattr(response, 'final_output'):
                    output = response.final_output
                    if isinstance(output, str):
                        try:
                            data = json.loads(output)
                            if isinstance(data, dict) and 'file' in data:
                                report_path = output_file(data['file'])
                        except: pass

                if report_path and os.path.exists(report_path):
                    with open(report_path, "r") as f:
                        report_content = f.read()
                    
                    st.divider()
                    st.subheader("📄 Investment Memo")
                    st.markdown(f'<div class="report-card">', unsafe_allow_html=True)
                    st.markdown(report_content)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        label="📥 Download Report (.md)",
                        data=report_content,
                        file_name="investment_report.md",
                        mime="text/markdown"
                    )
                else:
                    st.warning("Report file generated, but could not be located in outputs/.")
                    st.write(response.final_output)

with col2:
    st.markdown("""
    ### 🛡️ Analyst Spans
    *This section shows the coordination between your agents.*
    """)
    
    with st.expander("Macro Analyst", expanded=True):
        st.write("Analyze Global Trends, Inflation, and Rates.")
        st.caption("Active Tools: FRED API, Web Search")
        
    with st.expander("Fundamental Analyst", expanded=True):
        st.write("Analyze Company Financials, Revenue, and Targets.")
        st.caption("Active Tools: Yahoo Finance MCP")
        
    with st.expander("Quantitative Analyst", expanded=True):
        st.write("Perform Statistical Regressions and Simulations.")
        st.caption("Active Tools: Local Code Interpreter")

st.divider()
st.caption(f"System Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
