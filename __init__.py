"""AI分类算法"黑箱"可视化——多模型决策边界对比探究智能体（LLM增强版）"""

import sys
import os
from pathlib import Path
import streamlit.web.cli as stcli

def main():
    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), "--server.port=8503"]
    stcli.main()
