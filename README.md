# 📈 Multi-Agent Portfolio Analyst

A sophisticated multi-agent collaborative system designed to perform deep investment research and generate comprehensive portfolio analysis. This project implements a **Hub-and-Spoke** architecture where a lead **Portfolio Manager (PM)** orchestrates a team of specialized AI agents.

---

## 🏗️ Architecture

The system consists of several autonomous agents collaborating in real-time:

- **👑 Head Portfolio Manager (Orchestrator)**: Delegates research tasks, synthesizes findings, and makes final investment recommendations.
- **📊 Fundamental Analyst**: Deep-dives into company financials, revenue growth, and valuation using Yahoo Finance data.
- **🌍 Macro Strategist**: Analyzes global economic trends, interest rates (FRED data), and geopolitical impacts.
- **🔢 Quantitative Analyst**: Performs statistical modeling, historical regressions, and price target projections.
- **✍️ Editor Agent**: Finalizes and formats professional investment memos.

---

## 🚀 Features

- **Multi-Agent Orchestration**: Real-time parallel execution of specialist workflows.
- **Groq Integration**: Optimized for high-speed inference using Llama 3 via Groq (supports high TPM limits).
- **Live Data Integration**: 
    - **Yahoo Finance**: Real-time stock quotes, financials, and news.
    - **FRED (St. Louis Fed)**: Real-time economic data integration.
- **Web Interface**: A premium **Streamlit Dashboard** for interactive analysis.
- **Professional Output**: Generates polished Markdown investment reports.

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/srishtisingh2026/Multi-Agent-Portfolio-Analyst.git
cd Multi-Agent-Portfolio-Analyst
```

### 2. Environment Configuration
Create a `.env` file in the root directory and add your API keys:
```env
# GROQ Configuration (Primary)
GROQ_API_KEY=your_groq_key_here

# Economic Data provider
FRED_API_KEY=your_fred_key_here
```

### 3. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Option 1: Web Dashboard (Recommended)
Launch the interactive Streamlit interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option 2: Terminal Mode
Run the standalone analysis script:
```bash
python main.py
```

---

## 📂 Project Structure
- `main.py`: CLI Entry point.
- `app.py`: Streamlit Web UI.
- `investment_agents/`: Core logic for PM, Fundamental, Macro, and Quant agents.
- `mcp/`: Model Context Protocol (MCP) servers for data tools.
- `outputs/`: Generated research reports and data files.
- `prompts/`: Managed system instructions for all specialized agents.
