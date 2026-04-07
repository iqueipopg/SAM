"""
SAM — Systematic Agent for Meta-analysis
Streamlit chat-based UI for the A1→A7 pipeline.
Run: streamlit run app.py
"""

import streamlit as st
import json, os, sys, gc, time, glob, base64
import torch
import yaml
from datetime import datetime
from typing import Optional

# ── Logo helpers ─────────────────────────────────────────────────────────────
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def _load_svg(filename: str) -> str:
    """Load an SVG file and return its raw content."""
    path = os.path.join(_ASSETS_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def _svg_to_base64(filename: str) -> str:
    """Load SVG and encode as base64 data URI for img tags."""
    svg = _load_svg(filename)
    if svg:
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    return ""

def _themed_logo(basename: str) -> str:
    """Return base64 data URI for a logo, picking light/dark variant."""
    dark = st.session_state.get("dark_mode", False)
    suffix = "dark" if dark else "light"
    return _svg_to_base64(f"{basename}_{suffix}.svg")

def _assistant_avatar() -> str:
    """Return the themed SAM icon as avatar for chat messages."""
    uri = _themed_logo("logo_icon")
    return uri if uri else "assets/favicon.png"

# Load .env before any agent imports (so env vars are available)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent1_protocol import Agent1Protocol
from agents.agent2_search import Agent2Search
from agents.agent3_screening import Agent3Screening
from agents.agent4_fulltext import Agent4FullText
from agents.agent5_extraction import Agent5Extraction
from agents.agent6_quality import Agent6Quality
from agents.agent7_synthesis import Agent7Synthesis
from agents.agent8_manuscript import Agent8Manuscript
from agents.prisma_flow import generate_prisma_flow
from agents.rob_plots import generate_rob_traffic_light, generate_rob_summary
from agents.exporters import export_csv, export_excel, export_revman, export_summary_table
from agents.logger import setup_logger, get_log_file_path
from agents.review_report import ReviewReportGenerator

# ── Constants ────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = {
    "CBT vs medication for depression in adults":
        "Is cognitive behavioral therapy more effective than medication for treating depression in adults?",
    "Mindfulness for anxiety in adolescents":
        "Is mindfulness meditation effective for reducing anxiety in adolescents?",
    "Exercise and cognition in dementia":
        "Does exercise improve cognitive function in elderly patients with dementia?",
    "Smoking and MI risk in diabetes":
        "Does smoking increase myocardial infarction risk in diabetic patients?",
    "Barriers to evidence-based nursing":
        "What barriers do nurses perceive in implementing evidence-based practice?",
}

CSS_THEME = """
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Hide menu & footer but KEEP sidebar toggle */
    #MainMenu, footer {visibility: hidden;}
    header {visibility: visible !important;}
    .block-container {padding-top: 1.2rem;}

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    section[data-testid="stSidebar"] hr {
        border-color: #E2E8F0;
        opacity: 0.5;
        margin: 0.5rem 0;
    }

    /* Sidebar logo container */
    .sam-sidebar-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.8rem 0.5rem 0.4rem 0.5rem;
    }
    .sam-sidebar-logo img {
        width: 100%;
        max-width: 220px;
        height: auto;
    }

    /* ── New Analysis primary CTA ── */
    .sam-new-btn > button {
        width: 100%;
        padding: 0.6rem 1rem !important;
        background: linear-gradient(135deg, #0D9488 0%, #06B6D4 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 0.6rem !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(13,148,136,0.25);
        transition: all 0.2s ease !important;
    }
    .sam-new-btn > button:hover {
        background: linear-gradient(135deg, #0F766E 0%, #0891B2 100%) !important;
        box-shadow: 0 4px 12px rgba(13,148,136,0.35) !important;
        transform: translateY(-1px) !important;
        color: #FFFFFF !important;
    }
    .sam-new-btn > button:active {
        transform: translateY(0) !important;
    }

    /* ── Sidebar section headers ── */
    .sam-section-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94A3B8;
        margin: 0.6rem 0 0.35rem 0.1rem;
    }

    /* ── History cards in sidebar ── */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        width: 100%;
        padding: 0.45rem 0.65rem;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        background: #FFFFFF;
        color: #475569;
        font-size: 0.74rem;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #0D9488;
        color: #0D9488;
        background: #F0FDFA;
        transform: translateX(2px);
        box-shadow: 0 2px 6px rgba(13,148,136,0.1);
    }

    /* ── Compare button special style ── */
    .sam-compare-btn > button {
        background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
    }
    .sam-compare-btn > button:hover {
        background: linear-gradient(135deg, #6D28D9 0%, #7C3AED 100%) !important;
        box-shadow: 0 4px 12px rgba(124,58,237,0.35) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* ── Resume pipeline cards ── */
    .sam-resume-btn > button {
        border-left: 3px solid #F59E0B !important;
        background: #FFFBEB !important;
        color: #92400E !important;
        font-size: 0.72rem !important;
    }
    .sam-resume-btn > button:hover {
        background: #FEF3C7 !important;
        border-left-color: #D97706 !important;
        color: #78350F !important;
        transform: translateX(2px) !important;
    }

    /* ── Pipeline Settings panel ── */
    .sam-settings-group {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 0.6rem 0.7rem;
        margin-bottom: 0.5rem;
    }
    .sam-settings-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        color: #0D9488;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    /* Cleaner slider & select styling in sidebar */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.76rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
        font-size: 0.72rem;
    }

    /* ── GPU chip ── */
    .sam-gpu-chip {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        background: #F0FDFA;
        border: 1px solid #CCFBF1;
        border-radius: 2rem;
        padding: 0.3rem 0.7rem;
        margin: 0.3rem auto;
        width: fit-content;
    }
    .sam-gpu-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10B981;
        animation: sam-pulse 2s infinite;
    }
    @keyframes sam-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .sam-gpu-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.65rem;
        color: #0D9488;
        font-weight: 500;
    }

    /* Sidebar footer */
    .sam-footer {
        font-family: 'Inter', sans-serif;
        color: #94A3B8;
        font-size: 0.58rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        text-align: center;
        padding: 0.4rem 0;
    }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] {
        color: #0D9488;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
    }

    /* ── Chat input ── */
    .stChatInput textarea {
        border-color: #E2E8F0 !important;
        border-radius: 0.75rem !important;
    }
    .stChatInput textarea:focus {
        border-color: #0D9488 !important;
        box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.15) !important;
    }

    /* ── Welcome hero ── */
    .sam-hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .sam-hero img {
        max-width: 340px;
        width: 80%;
        height: auto;
        margin-bottom: 0.3rem;
    }
    .sam-hero-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* ── Status containers (pipeline progress) ── */
    [data-testid="stStatusWidget"] {
        border-radius: 0.5rem;
    }

    /* ── Gradient accent bar ── */
    .sam-gradient-bar {
        height: 3px;
        background: linear-gradient(90deg, #0D9488, #06B6D4, #7C3AED);
        border-radius: 2px;
        margin: 0.5rem 0 1rem 0;
    }

    /* ── Agent step badges ── */
    .agent-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .agent-badge-teal { background: #CCFBF1; color: #0D9488; }
    .agent-badge-cyan { background: #CFFAFE; color: #0891B2; }
    .agent-badge-violet { background: #EDE9FE; color: #7C3AED; }

    /* ── Expanders ── */
    details[data-testid="stExpander"] summary {
        font-weight: 600;
    }
</style>
"""

CSS_DARK_OVERRIDES = """
<style>
    /* ── Dark mode overrides ── */

    /* Main background & text */
    .stApp, .main .block-container {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
    }
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp [data-testid="stMarkdownContainer"] h1,
    .stApp [data-testid="stMarkdownContainer"] h2,
    .stApp [data-testid="stMarkdownContainer"] h3,
    .stApp [data-testid="stMarkdownContainer"] h4,
    .stApp [data-testid="stMarkdownContainer"] h5 {
        color: #E2E8F0 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #334155 !important;
        opacity: 0.6;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: #CBD5E1 !important;
    }

    /* Section labels in dark */
    .sam-section-label {
        color: #64748B !important;
    }

    /* History cards dark — override all Streamlit button variants */
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"],
    section[data-testid="stSidebar"] .stButton > button[kind="secondaryFormSubmit"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"],
    section[data-testid="stSidebar"] .stButton button {
        background: #1E293B !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #CBD5E1 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover {
        border-color: #2DD4BF !important;
        color: #2DD4BF !important;
        background: #0F1D2E !important;
        background-color: #0F1D2E !important;
        box-shadow: 0 2px 8px rgba(45,212,191,0.15) !important;
    }

    /* New Analysis button dark — brighter gradient */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child .stButton:first-of-type > button {
        background: linear-gradient(135deg, #14B8A6 0%, #22D3EE 100%) !important;
        color: #0F172A !important;
        box-shadow: 0 2px 10px rgba(45,212,191,0.35) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child .stButton:first-of-type > button:hover {
        background: linear-gradient(135deg, #2DD4BF 0%, #38BDF8 100%) !important;
        box-shadow: 0 4px 16px rgba(45,212,191,0.45) !important;
        color: #0F172A !important;
    }

    /* GPU chip dark */
    .sam-gpu-chip {
        background: #1E293B !important;
        border-color: #334155 !important;
    }
    .sam-gpu-text {
        color: #2DD4BF !important;
    }

    /* Gradient bar — brighter for dark bg */
    .sam-gradient-bar {
        background: linear-gradient(90deg, #2DD4BF, #38BDF8, #A78BFA) !important;
        opacity: 0.7;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2DD4BF !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    /* Chat input dark */
    .stChatInput textarea {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
    }
    .stChatInput textarea:focus {
        border-color: #2DD4BF !important;
        box-shadow: 0 0 0 2px rgba(45,212,191,0.2) !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }

    /* Expanders */
    details[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    details[data-testid="stExpander"] summary {
        color: #CBD5E1 !important;
    }
    details[data-testid="stExpander"] summary:hover {
        color: #2DD4BF !important;
    }

    /* Status widget */
    [data-testid="stStatusWidget"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }

    /* Agent badges dark */
    .agent-badge-teal { background: rgba(45,212,191,0.15) !important; color: #2DD4BF !important; }
    .agent-badge-cyan { background: rgba(56,189,248,0.15) !important; color: #38BDF8 !important; }
    .agent-badge-violet { background: rgba(167,139,250,0.15) !important; color: #A78BFA !important; }

    /* Welcome subtitle */
    .sam-hero-subtitle {
        color: #94A3B8 !important;
    }

    /* Sidebar sliders/selects dark */
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #94A3B8 !important;
    }

    /* Selectbox/dropdown dark */
    [data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    [data-baseweb="select"] span {
        color: #CBD5E1 !important;
    }

    /* Toggle dark */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #CBD5E1;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2DD4BF !important;
    }

    /* Dividers */
    hr {
        border-color: #1E293B !important;
    }

    /* Resume buttons dark */
    .sam-resume-btn > button,
    [data-testid="stSidebar"] button[key*="resume_"] {
        border-left: 3px solid #F59E0B !important;
        background: #1C1917 !important;
        color: #FCD34D !important;
    }

    /* Settings panel dark */
    .sam-settings-group {
        background: #1E293B !important;
        border-color: #334155 !important;
    }
    .sam-settings-title {
        color: #2DD4BF !important;
    }

    /* Images (plots) — add subtle border */
    .stImage img {
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-color: #334155 !important;
    }

    /* Pills / chip selectors — catch all nesting contexts */
    [data-testid="stChatMessage"] button,
    [data-testid="stPills"] button,
    .stChatMessage button,
    .stPills button,
    [data-testid="stChatMessage"] [data-testid="stPills"] button,
    [data-testid="stChatMessage"] .stElementContainer button {
        background: #1E293B !important;
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #CBD5E1 !important;
    }
    [data-testid="stChatMessage"] button:hover,
    [data-testid="stPills"] button:hover,
    .stPills button:hover {
        border-color: #2DD4BF !important;
        color: #2DD4BF !important;
        background: rgba(45,212,191,0.10) !important;
        background-color: rgba(45,212,191,0.10) !important;
    }
    [data-testid="stPills"] button[aria-checked="true"],
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stChatMessage"] button[aria-checked="true"] {
        background: rgba(45,212,191,0.18) !important;
        background-color: rgba(45,212,191,0.18) !important;
        border-color: #2DD4BF !important;
        color: #2DD4BF !important;
    }

    /* ── Force dark on ALL white backgrounds ── */
    .stApp {
        background-color: #0F172A !important;
    }
    .main, .main > div, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: #0F172A !important;
    }

    /* Header bar */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        background-color: #0F172A !important;
    }

    /* Chat input container & bottom bar */
    .stChatInput,
    .stChatInput > div,
    [data-testid="stChatInput"],
    [data-testid="stChatInputContainer"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    .stChatInput textarea,
    [data-testid="stChatInput"] textarea {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
        caret-color: #2DD4BF !important;
    }
    .stChatInput textarea::placeholder {
        color: #64748B !important;
    }
    .stChatInput button,
    [data-testid="stChatInput"] button {
        background-color: #334155 !important;
        color: #CBD5E1 !important;
    }
    .stChatInput button:hover {
        background-color: #2DD4BF !important;
        color: #0F172A !important;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"],
    .stChatMessage {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        border: 1px solid #334155 !important;
        border-radius: 0.75rem !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong {
        color: #E2E8F0 !important;
    }
    [data-testid="stChatMessage"] strong {
        color: #F1F5F9 !important;
    }

    /* Sidebar toggle widget */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #CBD5E1 !important;
    }

    /* Sidebar expanders */
    section[data-testid="stSidebar"] details[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] summary span {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] details[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        background-color: #1E293B !important;
    }

    /* Selectbox — the visible selected value box */
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-baseweb="select"] [data-baseweb="input"],
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #CBD5E1 !important;
    }
    [data-baseweb="select"] span,
    [data-baseweb="select"] [data-baseweb="tag"] span,
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
        color: #CBD5E1 !important;
    }
    /* Selectbox dropdown arrow */
    [data-baseweb="select"] svg {
        fill: #94A3B8 !important;
    }

    /* Selectbox dropdown menu */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [data-baseweb="menu"] > div,
    ul[role="listbox"],
    ul[role="listbox"] > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
    }
    ul[role="listbox"] li,
    [data-baseweb="menu"] li {
        color: #CBD5E1 !important;
        background-color: #1E293B !important;
    }
    ul[role="listbox"] li:hover,
    [data-baseweb="menu"] li:hover,
    ul[role="listbox"] li[aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #334155 !important;
        color: #2DD4BF !important;
    }

    /* Slider — thumb value, min/max labels, track */
    [data-testid="stThumbValue"],
    section[data-testid="stSidebar"] [data-testid="stThumbValue"] {
        color: #E2E8F0 !important;
    }
    .stSlider > div > div > div,
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #94A3B8 !important;
    }
    .stSlider label p,
    section[data-testid="stSidebar"] .stSlider label p {
        color: #CBD5E1 !important;
    }
    /* Slider min/max text */
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"] {
        color: #64748B !important;
    }

    /* All labels inside sidebar expanders */
    section[data-testid="stSidebar"] .stSelectbox label p,
    section[data-testid="stSidebar"] .stSlider label p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #CBD5E1 !important;
    }

    /* Small colored A3/A4 labels */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span[style] {
        opacity: 1 !important;
    }

    /* Toggle switch styling */
    [data-baseweb="toggle"],
    section[data-testid="stSidebar"] [data-baseweb="toggle"] {
        background-color: #334155 !important;
    }

    /* Checkbox labels */
    section[data-testid="stSidebar"] .stCheckbox label span {
        color: #CBD5E1 !important;
    }

    /* Tooltip / help text */
    [data-testid="stTooltipContent"] {
        background-color: #1E293B !important;
        color: #CBD5E1 !important;
        border-color: #334155 !important;
    }

    /* Any remaining white container backgrounds */
    [data-testid="stForm"],
    [data-testid="stHorizontalBlock"],
    .element-container {
        background-color: transparent !important;
    }
</style>
"""


# ── Utilities ────────────────────────────────────────────────────────────────

def free_vram(agent=None):
    """Release GPU memory from an agent or just clear cache."""
    if agent is not None:
        if hasattr(agent, "model") and agent.model is not None:
            del agent.model
            agent.model = None
        if hasattr(agent, "tokenizer") and agent.tokenizer is not None:
            del agent.tokenizer
            agent.tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_r_objects(obj):
    """Recursively remove non-serializable R objects (_fit) from A7 result."""
    if isinstance(obj, dict):
        return {k: _clean_r_objects(v) for k, v in obj.items() if k != "_fit"}
    elif isinstance(obj, list):
        return [_clean_r_objects(item) for item in obj]
    return obj


def save_checkpoint(run_dir: str, agent_name: str, data: dict):
    """Save agent output as checkpoint. Data must be JSON-serializable."""
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, f"checkpoint_{agent_name}.json")
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_checkpoint(run_dir: str, agent_name: str) -> Optional[dict]:
    """Load checkpoint if it exists. Returns None if not found."""
    checkpoint_path = os.path.join(run_dir, f"checkpoint_{agent_name}.json")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def list_checkpoint_agents(run_dir: str) -> list:
    """List agents that have checkpoints in a run directory."""
    if not os.path.isdir(run_dir):
        return []
    agents = []
    for f in sorted(os.listdir(run_dir)):
        if f.startswith("checkpoint_") and f.endswith(".json"):
            agent_name = f.replace("checkpoint_", "").replace(".json", "")
            agents.append(agent_name)
    return agents


def get_resumable_runs() -> list:
    """Find run directories that have checkpoints but no final pipeline.json."""
    runs = []
    pattern = os.path.join("outputs", "run_*")
    for run_dir in sorted(glob.glob(pattern), reverse=True):
        checkpoints = list_checkpoint_agents(run_dir)
        if checkpoints:
            # Extract timestamp
            ts_part = os.path.basename(run_dir).replace("run_", "")
            try:
                dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%b %d, %H:%M")
            except ValueError:
                date_str = ts_part
            # Check if query is available from A1 checkpoint
            a1_data = load_checkpoint(run_dir, "a1_protocol")
            query_str = a1_data.get("query", "Unknown") if a1_data else "Unknown"
            has_pipeline = os.path.exists(os.path.join(run_dir, "pipeline.json"))
            runs.append({
                "path": run_dir,
                "date": date_str,
                "query": query_str,
                "checkpoints": checkpoints,
                "complete": has_pipeline,
            })
    return runs


def finalize_checkpoints(run_dir: str):
    """Remove checkpoint files after successful pipeline completion."""
    if not os.path.isdir(run_dir):
        return
    for f in os.listdir(run_dir):
        if f.startswith("checkpoint_") and f.endswith(".json"):
            os.remove(os.path.join(run_dir, f))


def _log_pipeline_stage(run_dir: str, stage: str, message: str):
    """Append a timestamped message to the pipeline log."""
    import datetime
    log_path = os.path.join(run_dir, "pipeline.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {stage}: {message}\n")


def _load_config_language():
    """Load manuscript language from config.yaml (default 'en')."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("manuscript", {}).get("language", "en")
    except Exception:
        return "en"


def init_session_state():
    """Initialize all session state keys with defaults."""
    defaults = {
        "messages": [],
        "show_welcome": True,
        "pending_query": None,
        "max_results": 30,
        "a3_ensemble": 3,
        "a3_vote_threshold": 0.60,
        "a4_ensemble": 1,
        "a4_vote_threshold": 0.60,
        "dedup_threshold": 0.90,
        "resume_run_dir": None,
        # Pipeline history
        "compare_mode": False,
        "compare_selections": [],
        # Language
        "manuscript_language": _load_config_language(),
        # Stop pipeline
        "pipeline_stop_requested": False,
        "pipeline_running": False,
        # Theme
        "dark_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── History ──────────────────────────────────────────────────────────────────

def scan_pipeline_outputs():
    """Scan outputs/ for pipeline JSON files, return list of rich summaries sorted newest-first.

    Each entry contains: path, date, date_sort (datetime), query, papers_found,
    papers_included, has_synthesis (bool), has_manuscript (bool), grade_levels.
    """
    new_pattern = os.path.join("outputs", "run_*", "pipeline.json")
    old_pattern = os.path.join("outputs", "pipeline_*.json")
    files = sorted(glob.glob(new_pattern) + glob.glob(old_pattern), reverse=True)
    history = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "/run_" in fp:
                ts_part = os.path.basename(os.path.dirname(fp)).replace("run_", "")
            else:
                ts_part = os.path.basename(fp).replace("pipeline_", "").replace(".json", "")
            try:
                dt = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%b %d, %H:%M")
            except ValueError:
                dt = datetime.now()
                date_str = ts_part

            prisma = data.get("prisma", {})
            protocol = data.get("protocol", {})
            query = data.get("query", protocol.get("research_question", "Unknown"))

            history.append({
                "path": fp,
                "query": query,
                "date": date_str,
                "date_sort": dt,
                # Legacy-compatible fields
                "included": prisma.get("included_final", 0),
                "identified": prisma.get("identified", 0),
                # Rich fields for pipeline history
                "papers_found": prisma.get("identified", prisma.get("screened", 0)),
                "papers_included": prisma.get("included_final", 0),
                "has_synthesis": bool(data.get("synthesis") and data["synthesis"].get("analyses")),
                "has_manuscript": bool(data.get("manuscript") and data["manuscript"].get("manuscript")),
                "grade_levels": _extract_grade_levels(data.get("synthesis")),
            })
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    history.sort(key=lambda x: x.get("date_sort", datetime.min), reverse=True)
    return history


def _extract_grade_levels(synthesis):
    """Extract GRADE final levels from synthesis data for comparison."""
    if not synthesis or not synthesis.get("analyses"):
        return {}
    grades = {}
    for measure, analysis in synthesis["analyses"].items():
        grade = analysis.get("grade", {})
        if grade.get("final_level"):
            grades[measure] = grade["final_level"]
    return grades


# Backward-compatible alias
scan_history = scan_pipeline_outputs


def load_historical_run(filepath):
    """Load a saved pipeline JSON and populate session_state.messages for replay."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    st.session_state.messages = [
        {"role": "user", "type": "query", "content": data.get("query", "")},
        {"role": "assistant", "type": "result", "content": "Pipeline completed.",
         "data": data},
    ]
    st.session_state.show_welcome = False
    st.session_state.pending_query = None


def load_pipeline_data(filepath):
    """Load pipeline JSON and return the data dict without modifying session state."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def render_comparison(data_a, data_b, label_a, label_b):
    """Render side-by-side comparison of two pipeline runs."""
    col1, col2 = st.columns(2)

    for col, data, label in [(col1, data_a, label_a), (col2, data_b, label_b)]:
        with col:
            st.markdown(f"**{label}**")
            prisma = data.get("prisma", {})
            st.metric("Papers Found", prisma.get("identified", 0))
            st.metric("Papers Included", prisma.get("included_final", 0))
            st.metric("Duplicates Removed", prisma.get("duplicates_removed", 0))

            # Synthesis results
            synthesis = data.get("synthesis")
            if synthesis and synthesis.get("analyses"):
                st.markdown("**Meta-Analysis:**")
                for measure, analysis in synthesis["analyses"].items():
                    model = analysis.get("model", {})
                    het = analysis.get("heterogeneity", {})
                    grade = analysis.get("grade", {})
                    st.markdown(
                        f"- **{measure}**: pooled = {model.get('pooled_estimate', '?')} "
                        f"[{model.get('ci_lower', '?')}, {model.get('ci_upper', '?')}], "
                        f"I\u00b2 = {het.get('I2', '?')}%"
                    )
                    st.markdown(f"  GRADE: **{grade.get('final_level', '?')}**")
            else:
                st.caption("No meta-analysis performed")

            # Quality summary
            quality = data.get("quality_assessment")
            if quality and quality.get("stats"):
                qs = quality["stats"]
                st.markdown(
                    f"**Quality:** RoB2: {qs.get('rob2_assessed', 0)} | "
                    f"ROBINS-I: {qs.get('robins_i_assessed', 0)} | "
                    f"JBI: {qs.get('jbi_assessed', 0)}"
                )

            # Manuscript
            manuscript = data.get("manuscript")
            if manuscript and manuscript.get("manuscript"):
                st.markdown("**Manuscript:** Generated")
            else:
                st.caption("No manuscript generated")


# ── CSS injection ────────────────────────────────────────────────────────────

def inject_css():
    st.markdown(CSS_THEME, unsafe_allow_html=True)
    if st.session_state.get("dark_mode", False):
        st.markdown(CSS_DARK_OVERRIDES, unsafe_allow_html=True)


# ── Welcome screen ───────────────────────────────────────────────────────────

def render_welcome():
    """Render the welcome hero with logo and suggestion chips."""
    # Hero section with full logo
    _hero_logo = _themed_logo("logo_horizontal")
    if _hero_logo:
        st.markdown(
            f'<div class="sam-hero">'
            f'<img src="{_hero_logo}" alt="SAM — Systematic Agent for Meta-analysis">'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="sam-gradient-bar"></div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=_assistant_avatar()):
        st.markdown(
            "Describe your **clinical research question** and I'll run a full "
            "systematic review pipeline: protocol generation, multi-database search "
            "(PubMed, Europe PMC, Semantic Scholar, Google Scholar), screening, "
            "full-text retrieval, data extraction, quality assessment (RoB 2.0 / ROBINS-I), "
            "meta-analysis (R/metafor), and manuscript generation."
        )
        st.markdown("**Try one of these:**")
        selected = st.pills(
            "example_queries",
            options=list(EXAMPLE_QUERIES.keys()),
            label_visibility="collapsed",
        )
        if selected:
            st.session_state.pending_query = EXAMPLE_QUERIES[selected]
            st.session_state.show_welcome = False
            st.rerun()


# ── Result renderers ─────────────────────────────────────────────────────────

def render_prisma_flow(prisma):
    """Render PRISMA flow metrics."""
    st.markdown("##### PRISMA Flow")
    # Database breakdown row
    per_db = prisma.get("identified_per_db", {})
    if per_db:
        _has_gs = per_db.get("google_scholar", 0) > 0
        db_cols = st.columns(6 if _has_gs else 5)
        db_cols[0].metric("PubMed", per_db.get("pubmed", 0))
        db_cols[1].metric("Europe PMC", per_db.get("europepmc", 0))
        db_cols[2].metric("Semantic Scholar", per_db.get("semantic_scholar", 0))
        _col_idx = 3
        if _has_gs:
            db_cols[_col_idx].metric("Google Scholar", per_db.get("google_scholar", 0))
            _col_idx += 1
        db_cols[_col_idx].metric("Duplicates Removed", prisma.get("duplicates_removed", 0))
        db_cols[_col_idx + 1].metric("After Dedup", prisma.get("after_dedup", prisma.get("identified", 0)))
    # Main PRISMA flow row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Screened", prisma.get("screened", prisma.get("identified", 0)))
    c2.metric("Excl. T/A", prisma.get("excluded_title_abstract", 0))
    c3.metric("Uncertain", prisma.get("uncertain_title_abstract", 0))
    c4.metric("Excl. Full-text", prisma.get("excluded_fulltext", 0))
    c5.metric("Included", prisma.get("included_final", 0))

    # Show PRISMA flow diagram if available
    prisma_img = prisma.get("prisma_flow_diagram")
    if prisma_img and os.path.exists(prisma_img):
        with st.expander("PRISMA 2020 Flow Diagram", expanded=False):
            st.image(prisma_img, caption="PRISMA 2020 Flow Diagram")


def render_protocol(protocol):
    """Render the protocol in an expander."""
    with st.expander("Protocol (A1)", expanded=False):
        fw = protocol.get("framework", "?")
        st.markdown(f"**Framework:** {fw}")
        st.markdown(f"**Population:** {protocol.get('population', '?')}")
        if fw == "PECO":
            st.markdown(f"**Exposure:** {protocol.get('exposure', protocol.get('intervention', '?'))}")
        elif fw == "SPIDER":
            st.markdown(f"**Phenomenon:** {protocol.get('phenomenon_of_interest', protocol.get('intervention', '?'))}")
        else:
            st.markdown(f"**Intervention:** {protocol.get('intervention', '?')}")
        st.markdown(f"**Comparator:** {protocol.get('comparator', '?')}")
        st.markdown(f"**Outcome:** {protocol.get('outcome', '?')}")
        inc = protocol.get("inclusion_criteria", [])
        exc = protocol.get("exclusion_criteria", [])
        if inc:
            st.markdown("**Inclusion:** " + " | ".join(inc))
        if exc:
            st.markdown("**Exclusion:** " + " | ".join(exc))

        # Protocol document (PRISMA-P)
        proto_doc = protocol.get("protocol_document", "")
        if proto_doc:
            with st.expander("Formal Protocol Document (PRISMA-P)", expanded=False):
                st.code(proto_doc, language=None)

        # PROSPERO registration draft
        prospero_draft = protocol.get("prospero_draft", "")
        if prospero_draft:
            with st.expander("PROSPERO Registration Draft", expanded=False):
                st.code(prospero_draft, language=None)
                st.download_button(
                    "Download PROSPERO Draft (.txt)",
                    data=prospero_draft,
                    file_name="prospero_draft.txt",
                    mime="text/plain",
                )


def render_extracted_paper(paper):
    """Render a single extracted paper in an expander."""
    q_icon = {"complete": "\u2705", "partial": "\u26a0\ufe0f", "failed": "\u274c"}.get(
        paper.get("extraction_quality", ""), "\u2753"
    )
    conf = paper.get("extraction_confidence", "")
    conf_icon = {"high": "\u2705", "medium": "\u26a0\ufe0f", "low": "\u274c"}.get(conf, "")
    conf_suffix = f" | {conf_icon} {conf}" if conf else ""
    label = f'{q_icon} [{paper["pmid"]}] {paper["title"][:90]}{conf_suffix}'
    with st.expander(label, expanded=False):
        _status_parts = [
            f'**Quality:** {paper.get("extraction_quality", "?")}',
            f'**Source:** {paper.get("text_source", "?")}',
            f'**Sections:** {", ".join(paper.get("sections_used", []))}',
        ]
        if conf:
            _status_parts.append(f'**Confidence:** {conf}')
        st.markdown(" | ".join(_status_parts))
        if paper.get("extraction_notes"):
            st.info(paper["extraction_notes"])
        # Concordance details (dual extraction)
        conc = paper.get("concordance")
        if conc:
            _conc_parts = [
                f'Matched: {conc.get("outcomes_matched", 0)}',
                f'Discrepant: {conc.get("outcomes_discrepant", 0)}',
                f'Pass1 only: {conc.get("outcomes_pass1_only", 0)}',
                f'Pass2 only: {conc.get("outcomes_pass2_only", 0)}',
                f'Sample match: {"Yes" if conc.get("sample_size_match") else "No"}',
            ]
            st.caption("Concordance: " + " | ".join(_conc_parts))
            if conc.get("flags"):
                with st.expander("Concordance flags", expanded=False):
                    for flag in conc["flags"]:
                        st.warning(flag)
        if paper.get("study_characteristics"):
            st.markdown("**Study Characteristics**")
            items = [f"- **{k}:** {v}" for k, v in paper["study_characteristics"].items() if v]
            st.markdown("\n".join(items))
        if paper.get("intervention_exposure"):
            st.markdown("**Intervention / Exposure**")
            items = [f"- **{k}:** {v}" for k, v in paper["intervention_exposure"].items() if v]
            st.markdown("\n".join(items))
        if paper.get("outcomes"):
            st.markdown(f'**Outcomes ({len(paper["outcomes"])})**')
            for i, o in enumerate(paper["outcomes"], 1):
                cols = [f"**{k}:** {v}" for k, v in o.items() if v]
                st.markdown(f"{i}. " + " | ".join(cols))


def render_quality_section(a6_data):
    """Render A6 quality assessment results."""
    if not a6_data or not a6_data.get("papers"):
        return

    papers = a6_data["papers"]
    stats = a6_data.get("stats", {})

    st.markdown(f'##### Quality Assessment ({len(papers)} papers)')
    st.caption(
        f'RoB 2: {stats.get("rob2_assessed", 0)} | '
        f'ROBINS-I: {stats.get("robins_i_assessed", 0)} | '
        f'JBI: {stats.get("jbi_assessed", 0)} | '
        f'Failed: {stats.get("failed", 0)}'
    )

    for p in papers:
        qa = p.get("quality_assessment", {})
        tool = qa.get("tool", "?")
        overall = qa.get("overall", "?")
        weight = p.get("quality_weight", "?")

        # Icon based on overall
        icon_map = {
            "Low": "\u2705", "Some concerns": "\u26a0\ufe0f", "High": "\u274c",
            "Moderate": "\u26a0\ufe0f", "Serious": "\u274c", "Critical": "\u274c",
            "No information": "\u2753",
            "Include": "\u2705", "Include with caution": "\u26a0\ufe0f", "Exclude": "\u274c",
        }
        icon = icon_map.get(overall, "\u2753")

        label = f'{icon} [{p.get("pmid", "?")}] {p.get("title", "")[:80]} — {tool}: {overall} (w={weight})'
        with st.expander(label, expanded=False):
            st.markdown(f'**Tool:** {tool} | **Design:** {p.get("study_design_detected", "?")} | **Weight:** {weight}')
            st.markdown(f'**Overall:** {overall} — {qa.get("overall_justification", "")}')

            if tool == "RoB2":
                domains = qa.get("domains", {})
                for dname, dinfo in domains.items():
                    j = dinfo.get("judgment", "?")
                    st.markdown(f'- **{dname}:** {j} — {dinfo.get("justification", "")}')
            elif tool == "ROBINS-I":
                domains = qa.get("domains", {})
                for dname, dinfo in domains.items():
                    j = dinfo.get("judgment", "?")
                    st.markdown(f'- **{dname}:** {j} — {dinfo.get("justification", "")}')
            elif tool == "JBI":
                items = qa.get("items", {})
                for iname, iinfo in items.items():
                    answer = iinfo.get("answer", "?")
                    answer_icon = {"Yes": "\u2705", "No": "\u274c", "Unclear": "\u2753", "Not applicable": "\u2796"}.get(answer, "")
                    st.markdown(f'- **{iname}:** {answer_icon} {answer} — {iinfo.get("justification", "")}')
                st.markdown(f'**Yes count:** {qa.get("yes_count", "?")}/10')

    # Show RoB plots if available
    rob_tl = a6_data.get("rob_traffic_light_plot")
    rob_sum = a6_data.get("rob_summary_plot")
    if rob_tl and os.path.exists(rob_tl):
        with st.expander("Risk of Bias Traffic Light Plot", expanded=False):
            st.image(rob_tl, caption="Risk of Bias Traffic Light")
    if rob_sum and os.path.exists(rob_sum):
        with st.expander("Risk of Bias Summary Plot", expanded=False):
            st.image(rob_sum, caption="Risk of Bias Summary")


def render_synthesis_section(a7_data):
    """Render A7 meta-analysis results."""
    if not a7_data:
        return

    analyses = a7_data.get("analyses", {})
    stats = a7_data.get("stats", {})

    if not analyses:
        st.info(f"No meta-analyses could be performed (usable data: {stats.get('papers_with_usable_data', 0)} papers).")
        return

    st.markdown(f'##### Meta-Analysis Results')
    st.caption(
        f'Papers with data: {stats.get("papers_with_usable_data", 0)} | '
        f'Analyses: {stats.get("analyses_performed", 0)}'
    )

    for measure, analysis in analyses.items():
        model = analysis.get("model", {})
        het = analysis.get("heterogeneity", {})
        grade = analysis.get("grade", {})

        st.markdown(f'**{measure}** ({model.get("k_studies", "?")} studies)')

        # Key metrics in columns — pooled estimate with prediction interval
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Pooled", f'{model.get("pooled_estimate", "?")}')
        mc2.metric("95% CI", f'[{model.get("ci_lower", "?")}, {model.get("ci_upper", "?")}]')
        mc3.metric("I\u00b2", f'{het.get("I2", "?")}%')
        mc4.metric("GRADE", grade.get("final_level", "?"))

        # Prediction interval display
        pi_lb = model.get("prediction_interval_lower")
        pi_ub = model.get("prediction_interval_upper")
        pi_text = f' | Prediction interval: [{pi_lb}, {pi_ub}]' if pi_lb is not None else ''
        total_p = analysis.get("total_participants", {})
        total_n = total_p.get("total_n")
        n_text = f' | Total N: {total_n}' if total_n else ''
        st.caption(f'p = {model.get("p_value", "?")} | {het.get("interpretation", "")}{pi_text}{n_text}')

        # Forest plot
        plots = analysis.get("plots", {})
        forest_path = plots.get("forest")
        if forest_path and os.path.exists(forest_path):
            st.image(forest_path, caption=f"Forest Plot ({measure})")

        # Summary of Findings table
        sof = analysis.get("summary_of_findings", {})
        if sof:
            with st.expander(f"Summary of Findings ({measure})", expanded=False):
                sof_data = {
                    "Outcome": [sof.get("outcome", "")],
                    "Studies (N)": [sof.get("n_studies", "")],
                    "Participants": [str(sof.get("n_participants", ""))],
                    "Effect": [sof.get("relative_effect", "")],
                    "Certainty": [sof.get("certainty", "")],
                }
                st.table(sof_data)
                aae = sof.get("anticipated_absolute_effects", {})
                if aae:
                    st.markdown(f'**Intervention effect:** {aae.get("intervention_effect", "")}')
                what = sof.get("what_happens", "")
                if what:
                    st.info(what)

        # Funnel plot in expander
        funnel_path = plots.get("funnel")
        if funnel_path and os.path.exists(funnel_path):
            with st.expander(f"Funnel Plot ({measure})", expanded=False):
                st.image(funnel_path)

        # GRADE details in expander (improved per-domain)
        if grade.get("domains"):
            with st.expander(f"GRADE Details ({measure})", expanded=False):
                st.markdown(f'**Starting level:** {grade.get("starting_level", "?")} | '
                           f'**Final level:** {grade.get("final_level", "?")} | '
                           f'**Total downgrades:** {grade.get("total_downgrades", 0)}')
                for dname, dinfo in grade["domains"].items():
                    dg = dinfo.get("downgrades", dinfo.get("downgrade", 0))
                    level = dinfo.get("level", "")
                    level_str = f' [{level}]' if level else ''
                    arrow = f' (**-{dg}**)' if dg > 0 else ''
                    st.markdown(f'- **{dname.replace("_", " ").title()}**{level_str}: {dinfo.get("reason", "")}{arrow}')

        # Subgroup analyses in expander
        subgroups = analysis.get("subgroup_analyses", {})
        has_subgroup_data = any(
            isinstance(v, dict) and not v.get("skipped")
            for v in subgroups.values()
        ) if isinstance(subgroups, dict) else False
        if has_subgroup_data:
            with st.expander(f"Subgroup Analyses ({measure})", expanded=False):
                for sg_name, sg_data in subgroups.items():
                    if not isinstance(sg_data, dict) or sg_data.get("skipped"):
                        continue
                    st.markdown(f'**{sg_name.replace("_", " ").title()}**')
                    # Render each subgroup's results
                    test_diff = sg_data.get("test_for_difference", {})
                    for sub_key, sub_val in sg_data.items():
                        if sub_key == "test_for_difference":
                            continue
                        if isinstance(sub_val, dict) and "pooled" in sub_val:
                            ci = sub_val.get("ci", [])
                            ci_str = f'[{ci[0]}, {ci[1]}]' if len(ci) == 2 else '?'
                            st.markdown(
                                f'- *{sub_key.replace("_", " ").title()}*: '
                                f'k={sub_val.get("k", "?")}, '
                                f'pooled={sub_val.get("pooled", "?")} {ci_str}, '
                                f'I\u00b2={sub_val.get("I2", "?")}%'
                            )
                    if test_diff and not test_diff.get("error"):
                        q_val = test_diff.get("Q_between", "?")
                        p_val = test_diff.get("p_value", "?")
                        st.markdown(f'- *Test for subgroup difference:* Q={q_val}, p={p_val}')
                    st.markdown("---")

        # Meta-regression in expander
        mr = analysis.get("meta_regression", {})
        has_mr_data = (isinstance(mr, dict) and not mr.get("skipped") and
                       any(isinstance(v, dict) and not v.get("skipped") and not v.get("error")
                           for v in mr.values()))
        if has_mr_data:
            with st.expander(f"Meta-Regression ({measure})", expanded=False):
                for mr_name, mr_data in mr.items():
                    if not isinstance(mr_data, dict) or mr_data.get("skipped") or mr_data.get("error"):
                        continue
                    sig_icon = "\u2705" if mr_data.get("significant") else "\u274c"
                    st.markdown(
                        f'**{mr_name.replace("_", " ").title()}:** '
                        f'coef = {mr_data.get("coefficient", "?")}, '
                        f'SE = {mr_data.get("se", "?")}, '
                        f'p = {mr_data.get("p_value", "?")} {sig_icon}'
                    )
                    st.caption(mr_data.get("interpretation", ""))

        # Sensitivity in expander
        sens = analysis.get("sensitivity", {})
        if isinstance(sens, dict) and sens.get("leave_one_out"):
            with st.expander(f"Sensitivity Analysis ({measure})", expanded=False):
                l1o = sens["leave_one_out"]
                st.markdown("**Leave-one-out:**")
                for item in l1o:
                    st.markdown(
                        f'- Remove {item.get("pmid_removed", "?")}: '
                        f'est = {item.get("estimate", "?")} '
                        f'[{item.get("ci_lower", "?")}, {item.get("ci_upper", "?")}]'
                    )
                excl = sens.get("exclude_high_risk", {})
                if excl and not excl.get("error"):
                    st.markdown(f'**Excluding high-risk studies:** '
                               f'k = {excl.get("k_remaining", "?")}, '
                               f'est = {excl.get("pooled_estimate", "?")} '
                               f'[{excl.get("ci_lower", "?")}, {excl.get("ci_upper", "?")}]')

        # Publication bias in expander
        pub = analysis.get("publication_bias", {})
        if isinstance(pub, dict) and (pub.get("egger_test") or pub.get("trim_and_fill")):
            with st.expander(f"Publication Bias ({measure})", expanded=False):
                egger = pub.get("egger_test", {})
                if egger.get("skipped"):
                    st.warning(egger.get("reason", "Egger's test skipped"))
                elif egger and not egger.get("error"):
                    sig = "Yes" if egger.get("significant") else "No"
                    st.markdown(f'**Egger\'s test:** z = {egger.get("z_value", "?")}, p = {egger.get("p_value", "?")} (significant: {sig})')
                tf = pub.get("trim_and_fill", {})
                if tf and not tf.get("error"):
                    st.markdown(
                        f'**Trim-and-fill:** {tf.get("estimated_missing", 0)} missing studies, '
                        f'adjusted est = {tf.get("adjusted_estimate", "?")} '
                        f'[{tf.get("adjusted_ci_lower", "?")}, {tf.get("adjusted_ci_upper", "?")}]'
                    )


def render_manuscript_section(a8_data):
    """Render A8 manuscript generation results."""
    if not a8_data or not a8_data.get("manuscript"):
        return

    manuscript = a8_data["manuscript"]
    references = a8_data.get("references", [])
    stats = a8_data.get("stats", {})

    st.markdown(f'##### Manuscript Draft')
    st.caption(
        f'Papers cited: {stats.get("total_papers_cited", 0)} | '
        f'RAG chunks: {stats.get("total_chunks", 0)} | '
        f'Time: {stats.get("elapsed_s", 0)}s'
    )

    section_labels = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "methods": "Methods",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
    }

    for sec_key in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        label = section_labels.get(sec_key, sec_key.title())
        text = manuscript.get(sec_key, "")
        word_count = len(text.split()) if text else 0
        expanded = sec_key == "abstract"
        with st.expander(f"{label} ({word_count} words)", expanded=expanded):
            st.markdown(text if text else "_Not generated_")

    # References
    if references:
        with st.expander(f"References ({len(references)})", expanded=False):
            for i, ref in enumerate(references, 1):
                st.markdown(
                    f'{i}. {ref.get("citation", "?")}. '
                    f'*{ref.get("title", "?")}*. '
                    f'{ref.get("journal", "?")}. '
                    f'PMID: {ref.get("pmid", "?")}'
                )

    # Download buttons — PDF (primary) + TXT (secondary)
    txt_parts = []
    for sec_key in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        label = section_labels.get(sec_key, sec_key.title())
        txt_parts.append(f"\n{'='*60}\n{label.upper()}\n{'='*60}\n\n{manuscript.get(sec_key, '')}\n")
    txt_parts.append(f"\n{'='*60}\nREFERENCES\n{'='*60}\n")
    for i, ref in enumerate(references, 1):
        txt_parts.append(
            f"{i}. {ref.get('citation', '?')}. {ref.get('title', '?')}. "
            f"{ref.get('journal', '?')}. PMID: {ref.get('pmid', '?')}\n"
        )
    manuscript_txt = "\n".join(txt_parts)

    dl_col1, dl_col2 = st.columns(2)
    pdf_path = a8_data.get("pdf_path")
    try:
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pf:
                pdf_bytes = pf.read()
            dl_col1.download_button(
                "\U0001f4d1 Download Manuscript (.pdf)",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
        else:
            dl_col1.caption("PDF not available")
    except Exception:
        dl_col1.caption("PDF not available (read error)")

    dl_col2.download_button(
        "\U0001f4c4 Download Manuscript (.txt)",
        data=manuscript_txt,
        file_name=f'manuscript_{time.strftime("%Y%m%d_%H%M%S")}.txt',
        mime="text/plain",
    )


def render_review_report(review_data):
    """Render the optional human review report."""
    if not review_data:
        return

    summary = review_data.get("summary", {})
    confidence = summary.get("pipeline_confidence", "unknown")

    # Color map for confidence badge
    conf_colors = {"high": "#16A34A", "moderate": "#D97706", "low": "#DC2626"}
    conf_color = conf_colors.get(confidence, "#6B7280")

    st.markdown("##### Human Review Report (Optional)")
    st.caption(
        "This report flags borderline or uncertain decisions across the pipeline "
        "for optional quality assurance. It does not affect the automated results."
    )

    # Summary metrics
    cols = st.columns(5)
    cols[0].metric("Total Flags", summary.get("total_flags", 0))
    cols[1].metric("High Priority", summary.get("high_priority", 0))
    cols[2].metric("Medium Priority", summary.get("medium_priority", 0))
    cols[3].metric("Low Priority", summary.get("low_priority", 0))
    cols[4].metric("Est. Review Time", f'{summary.get("estimated_review_minutes", 0)} min')

    # Confidence badge
    st.markdown(
        f'**Pipeline confidence:** '
        f'<span style="background:{conf_color};color:white;padding:2px 10px;'
        f'border-radius:4px;font-weight:600">{confidence.upper()}</span>',
        unsafe_allow_html=True,
    )

    # Helper to render a priority badge
    def _priority_badge(priority):
        colors = {"high": "#DC2626", "medium": "#D97706", "low": "#9CA3AF"}
        c = colors.get(priority, "#9CA3AF")
        return f'<span style="background:{c};color:white;padding:1px 6px;border-radius:3px;font-size:0.75em">{priority.upper()}</span>'

    # Screening borderline
    scr = review_data.get("screening_borderline", [])
    if scr:
        with st.expander(f"Screening Borderline (A3) \u2014 {len(scr)} flags", expanded=False):
            for item in sorted(scr, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)):
                badge = _priority_badge(item.get("priority", "low"))
                st.markdown(
                    f'{badge} **[{item.get("pmid", "?")}]** {item.get("title", "?")[:90]}',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f'Decision: {item.get("decision", "?")} | '
                    f'Confidence: {item.get("screening_confidence", "?")} | '
                    f'Include fraction: {item.get("include_fraction", "?")}'
                )
                for flag in item.get("flags", []):
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;- {flag}', unsafe_allow_html=True)

    # Full-text borderline
    ft = review_data.get("fulltext_borderline", [])
    if ft:
        with st.expander(f"Full-text Borderline (A4) \u2014 {len(ft)} flags", expanded=False):
            for item in sorted(ft, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)):
                badge = _priority_badge(item.get("priority", "low"))
                st.markdown(
                    f'{badge} **[{item.get("pmid", "?")}]** {item.get("title", "?")[:90]}',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f'Source: {item.get("text_source", "?")} | '
                    f'Decision: {item.get("decision", "?")} | '
                    f'Confidence: {item.get("screening_confidence", "?")}'
                )
                for flag in item.get("flags", []):
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;- {flag}', unsafe_allow_html=True)

    # Extraction concerns
    ext = review_data.get("extraction_concerns", [])
    if ext:
        with st.expander(f"Extraction Concerns (A5) \u2014 {len(ext)} flags", expanded=False):
            for item in sorted(ext, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)):
                badge = _priority_badge(item.get("priority", "low"))
                st.markdown(
                    f'{badge} **[{item.get("pmid", "?")}]** {item.get("title", "?")[:90]}',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f'Quality: {item.get("extraction_quality", "?")} | '
                    f'Confidence: {item.get("extraction_confidence", "?")} | '
                    f'Numeric complete: {item.get("n_numeric_complete", 0)}/{item.get("n_outcomes", 0)}'
                )
                for flag in item.get("flags", []):
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;- {flag}', unsafe_allow_html=True)

    # Quality concerns
    qa = review_data.get("quality_concerns", [])
    if qa:
        with st.expander(f"Quality Concerns (A6) \u2014 {len(qa)} flags", expanded=False):
            for item in sorted(qa, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)):
                badge = _priority_badge(item.get("priority", "low"))
                st.markdown(
                    f'{badge} **[{item.get("pmid", "?")}]** {item.get("title", "?")[:90]}',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f'Design: {item.get("study_design", "?")} | '
                    f'Tool: {item.get("tool", "?")} | '
                    f'Overall: {item.get("overall", "?")} | '
                    f'Weight: {item.get("quality_weight", "?")}'
                )
                for flag in item.get("flags", []):
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;- {flag}', unsafe_allow_html=True)
                dd = item.get("domain_details")
                if dd:
                    st.caption("Domains: " + ", ".join(f'{k}={v}' for k, v in dd.items()))

    # Synthesis concerns
    syn = review_data.get("synthesis_concerns", [])
    if syn:
        with st.expander(f"Synthesis Concerns (A7) \u2014 {len(syn)} flags", expanded=False):
            for item in sorted(syn, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("priority", "low"), 3)):
                badge = _priority_badge(item.get("priority", "low"))
                st.markdown(
                    f'{badge} **{item.get("measure", "?")}** analysis '
                    f'(k={item.get("k_studies", "?")})',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f'Pooled: {item.get("pooled_estimate", "?")} '
                    f'[{item.get("ci_lower", "?")}, {item.get("ci_upper", "?")}] | '
                    f'I\u00b2: {item.get("I2", "?")}% | '
                    f'GRADE: {item.get("grade_level", "?")}'
                )
                for flag in item.get("flags", []):
                    st.markdown(f'&nbsp;&nbsp;&nbsp;&nbsp;- {flag}', unsafe_allow_html=True)

    # Recommendations
    recs = review_data.get("recommendations", [])
    if recs:
        with st.expander("Recommendations", expanded=True):
            for i, rec in enumerate(recs, 1):
                st.markdown(f'{i}. {rec}')

    # Download button for full JSON report
    st.download_button(
        "\U0001f50d Download Review Report (JSON)",
        data=json.dumps(review_data, indent=2, ensure_ascii=False),
        file_name=f'review_report_{time.strftime("%Y%m%d_%H%M%S")}.json',
        mime="application/json",
    )


def _render_voting_sensitivity(results: dict, agent_label: str, ensemble_size: int,
                               voting_threshold: float, categories: list):
    """Render voting sensitivity analysis and Fleiss' kappa in an expander.

    Works for both A3 (include/exclude/uncertain lists) and A4 (papers list).
    """
    # Collect all papers depending on agent structure
    if "papers" in results:
        # A4 style: flat list in 'papers'
        all_papers = results["papers"]
    else:
        # A3 style: separate lists
        all_papers = (
            results.get("include", [])
            + results.get("exclude", [])
            + results.get("uncertain", [])
        )

    if not all_papers or ensemble_size < 2:
        return  # No sensitivity analysis possible with single voter

    # --- Compute sensitivity inline (no agent instance needed) ---
    thresholds = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    threshold_data = {}
    for t in thresholds:
        incl = excl = unc = 0
        for p in all_papers:
            # Determine which vote list to use
            votes = p.get("fulltext_votes", p.get("votes", []))
            if not votes:
                continue
            decisions = []
            for v in votes:
                decisions.append(v.get("decision", v) if isinstance(v, dict) else v)
            n = len(decisions)
            if n == 0:
                continue
            inc_frac = sum(1 for d in decisions if d == "INCLUDE") / n
            exc_frac = sum(1 for d in decisions if d == "EXCLUDE") / n
            if inc_frac >= t:
                incl += 1
            elif exc_frac >= t:
                excl += 1
            else:
                unc += 1
        threshold_data[t] = {"Included": incl, "Excluded": excl}
        if "UNCERTAIN" in categories:
            threshold_data[t]["Uncertain"] = unc

    # --- Fleiss' kappa ---
    n_raters = ensemble_size
    kappa = 0.0
    if n_raters >= 2:
        cat_count = len(categories)
        vote_matrix = []
        for p in all_papers:
            votes = p.get("fulltext_votes", p.get("votes", []))
            if not votes:
                continue
            counts = [0] * cat_count
            for v in votes:
                d = v.get("decision", v) if isinstance(v, dict) else v
                for ci, cat in enumerate(categories):
                    if d == cat:
                        counts[ci] += 1
                        break
                else:
                    # Unknown category -> last bucket
                    counts[-1] += 1
            vote_matrix.append(counts)

        if vote_matrix:
            p_values = []
            for counts in vote_matrix:
                n = sum(counts)
                if n < 2:
                    continue
                sum_sq = sum(c * c for c in counts)
                p_i = (sum_sq - n) / (n * (n - 1))
                p_values.append(p_i)

            if p_values:
                P_bar = sum(p_values) / len(p_values)
                total_votes = sum(sum(c) for c in vote_matrix)
                P_bar_e = 0.0
                if total_votes > 0:
                    for j in range(cat_count):
                        p_j = sum(c[j] for c in vote_matrix) / total_votes
                        P_bar_e += p_j * p_j
                if P_bar_e < 1.0:
                    kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)
                else:
                    kappa = 1.0

    # Interpret kappa
    if kappa < 0:
        kappa_label = "Poor"
    elif kappa < 0.20:
        kappa_label = "Slight"
    elif kappa < 0.40:
        kappa_label = "Fair"
    elif kappa < 0.60:
        kappa_label = "Moderate"
    elif kappa < 0.80:
        kappa_label = "Substantial"
    else:
        kappa_label = "Almost perfect"

    # --- Borderline papers ---
    borderline = [p for p in all_papers if p.get("borderline", False)]

    # --- Render ---
    with st.expander(f"{agent_label}: Ensemble Calibration & Sensitivity", expanded=False):
        # Kappa
        st.markdown(f"**Fleiss' kappa:** {kappa:.3f} ({kappa_label})")

        # Threshold sensitivity table
        st.markdown("**Threshold sensitivity:**")
        import pandas as pd
        rows = []
        for t in thresholds:
            row = {"Threshold": f"{t:.0%}"}
            row.update(threshold_data[t])
            if t == voting_threshold:
                row["Threshold"] += " (current)"
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Borderline papers
        if borderline:
            st.markdown(f"**Borderline papers (confidence < 0.3) -- {len(borderline)} for human review:**")
            for p in borderline:
                conf = p.get("screening_confidence", 0)
                st.markdown(f"- **[{p.get('pmid', '?')}]** {p.get('title', '?')[:80]}... (conf: {conf:.2f})")
        else:
            st.markdown("No borderline papers detected.")


def render_results(data):
    """Render full pipeline results inside a chat message."""
    prisma = data.get("prisma", {})
    protocol = data.get("protocol", {})
    a4 = data.get("fulltext_retrieval")
    a5 = data.get("data_extraction")
    a6 = data.get("quality_assessment")
    a7 = data.get("synthesis")
    a8 = data.get("manuscript")
    screening = data.get("screening", {})

    # PRISMA metrics
    render_prisma_flow(prisma)

    st.divider()

    # Protocol
    render_protocol(protocol)

    # Included papers (A4)
    if a4 and a4.get("include"):
        st.markdown(f'##### Included Papers ({len(a4["include"])})')
        for p in a4["include"]:
            with st.expander(f'\u2705 [{p["pmid"]}] {p["title"][:100]}', expanded=False):
                st.markdown(f'**Journal:** {p.get("journal", "?")} ({p.get("year", "?")})')
                st.markdown(f'**Authors:** {", ".join(p.get("authors", []))}')
                st.markdown(f'**DOI:** {p.get("doi", "\u2014")}')
                st.markdown(f'**Source:** {p.get("text_source", "?")} ({p.get("full_text_length", 0)} chars)')
                st.markdown(f'**Reasoning:** {p.get("fulltext_reasoning", "\u2014")}')

    # Extracted data (A5)
    if a5 and a5.get("papers"):
        st.markdown(f'##### Extracted Data ({len(a5["papers"])} papers)')
        for ep in a5["papers"]:
            render_extracted_paper(ep)

    # Quality assessment (A6)
    render_quality_section(a6)

    # Meta-analysis (A7)
    render_synthesis_section(a7)

    # Manuscript (A8)
    render_manuscript_section(a8)

    # Human Review Report (optional)
    review_report = data.get("review_report")
    if review_report:
        st.divider()
        render_review_report(review_report)

    # Excluded full-text (A4) — collapsed
    if a4 and a4.get("exclude"):
        with st.expander(f'Excluded at Full-text ({len(a4["exclude"])})', expanded=False):
            for p in a4["exclude"]:
                st.markdown(
                    f'- **[{p["pmid"]}]** {p["title"][:80]}... \u2014 {p.get("exclusion_reason", "?")}'
                )

    # Excluded T/A (A3) — collapsed
    if screening.get("exclude"):
        with st.expander(f'Excluded at Title/Abstract ({len(screening["exclude"])})', expanded=False):
            for p in screening["exclude"]:
                st.markdown(f'- **[{p["pmid"]}]** {p["title"][:80]}...')

    # Uncertain (A3) — collapsed
    if screening.get("uncertain"):
        with st.expander(f'Uncertain \u2014 manual review ({len(screening["uncertain"])})', expanded=False):
            for p in screening["uncertain"]:
                st.markdown(f'- **[{p["pmid"]}]** {p["title"][:80]}...')

    # Calibration: A3 sensitivity (in results view)
    cfg = data.get("config", {})
    a3_ens = cfg.get("a3_ensemble", screening.get("stats", {}).get("ensemble_size", 3))
    a3_thr = cfg.get("a3_vote_threshold", screening.get("stats", {}).get("voting_threshold", 0.60))
    if screening:
        _render_voting_sensitivity(
            screening, "A3", ensemble_size=a3_ens, voting_threshold=a3_thr,
            categories=["INCLUDE", "EXCLUDE", "UNCERTAIN"],
        )

    # Calibration: A4 sensitivity (in results view)
    a4_ens = cfg.get("a4_ensemble", 1)
    a4_thr = cfg.get("a4_vote_threshold", a3_thr)
    if a4 and a4.get("papers"):
        _render_voting_sensitivity(
            a4, "A4", ensemble_size=a4_ens, voting_threshold=a4_thr,
            categories=["INCLUDE", "EXCLUDE"],
        )

    # Download — JSON + additional export formats
    st.markdown("##### Export Data")
    dl_cols = st.columns(4)

    dl_cols[0].download_button(
        "\U0001f4e5 Download JSON",
        data=json.dumps(data, indent=2, ensure_ascii=False),
        file_name=f'meta_analysis_{time.strftime("%Y%m%d_%H%M%S")}.json',
        mime="application/json",
    )

    # CSV export
    try:
        import tempfile
        _csv_path = os.path.join(tempfile.gettempdir(), "sam_export.csv")
        export_csv(data, _csv_path)
        with open(_csv_path, "r", encoding="utf-8") as _cf:
            _csv_data = _cf.read()
        dl_cols[1].download_button(
            "\U0001f4cb Download CSV",
            data=_csv_data,
            file_name=f'meta_analysis_{time.strftime("%Y%m%d_%H%M%S")}.csv',
            mime="text/csv",
        )
        os.remove(_csv_path)
    except Exception:
        dl_cols[1].caption("CSV export unavailable")

    # Excel export
    try:
        import tempfile
        _xlsx_path = os.path.join(tempfile.gettempdir(), "sam_export.xlsx")
        export_excel(data, _xlsx_path)
        with open(_xlsx_path, "rb") as _xf:
            _xlsx_data = _xf.read()
        dl_cols[2].download_button(
            "\U0001f4ca Download Excel",
            data=_xlsx_data,
            file_name=f'meta_analysis_{time.strftime("%Y%m%d_%H%M%S")}.xlsx',
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        os.remove(_xlsx_path)
    except Exception:
        dl_cols[2].caption("Excel export unavailable")

    # RevMan XML export
    try:
        import tempfile
        _xml_path = os.path.join(tempfile.gettempdir(), "sam_export_revman.xml")
        export_revman(data, _xml_path)
        with open(_xml_path, "r", encoding="utf-8") as _rf:
            _xml_data = _rf.read()
        dl_cols[3].download_button(
            "\U0001f4c4 Download RevMan XML",
            data=_xml_data,
            file_name=f'revman_{time.strftime("%Y%m%d_%H%M%S")}.xml',
            mime="application/xml",
        )
        os.remove(_xml_path)
    except Exception:
        dl_cols[3].caption("RevMan export unavailable")


# ── Pipeline execution ───────────────────────────────────────────────────────

def _check_stop(last_agent_label=""):
    """Check if user requested a stop. Returns True if stop was requested."""
    if st.session_state.get("pipeline_stop_requested", False):
        st.warning(
            f"Pipeline stopped by user after **{last_agent_label}**. "
            f"Checkpoints saved — you can resume later from the sidebar."
        )
        return True
    return False


def run_pipeline(query, resume_dir=None):
    """Run the full A1→A8 pipeline with st.status progress and checkpointing.

    If resume_dir is provided, load checkpoints from that directory and skip
    already-completed agents. Otherwise create a new run directory.

    Returns output dict (partial if stopped early).
    """
    st.session_state.pipeline_running = True
    st.session_state.pipeline_stop_requested = False

    # ── Force-clear any leftover VRAM before starting ──────────
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    max_results = st.session_state.max_results
    dedup_threshold = st.session_state.dedup_threshold
    a3_ensemble = st.session_state.a3_ensemble
    a3_vote_thr = st.session_state.a3_vote_threshold
    a4_ensemble = st.session_state.a4_ensemble
    a4_vote_thr = st.session_state.get("a4_vote_threshold", a3_vote_thr)

    # Determine run directory
    if resume_dir and os.path.isdir(resume_dir):
        run_dir = resume_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("outputs", f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # Initialize structured logging for this run
    setup_logger("SAM", log_dir=run_dir)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    protocol = None
    pubmed_query = None
    papers = None
    a2_stats = None
    a3_results = None
    a4_result = None
    a5_result = None

    # ── A1: Protocol ─────────────────────────────────────────
    cached_a1 = load_checkpoint(run_dir, "a1_protocol")
    if cached_a1:
        protocol = cached_a1.get("protocol")
        pubmed_query = cached_a1.get("pubmed_query")
        with st.status("A1: Protocol (cached)", expanded=False) as s1:
            st.write(f"**Framework:** {protocol.get('framework', '?')}")
            st.write(f"**PubMed query:** `{pubmed_query}`")
            s1.update(label="A1: Protocol \u2714 (cached)", state="complete", expanded=False)
    else:
        with st.status("A1: Generating protocol...", expanded=True) as s1:
            try:
                # Force VRAM cleanup before loading model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                a1 = Agent1Protocol()
                protocol = a1.generate_protocol(query)
                if protocol.get("_fallback"):
                    st.warning("Protocol generation failed — using generic fallback. Results may be inaccurate.")
                pubmed_query = a1.generate_pubmed_query(query, protocol)
                fw = protocol.get("framework", "?")
                st.write(f"**Framework:** {fw}")
                st.write(f"**Population:** {protocol.get('population', '?')}")
                if fw == "PECO":
                    st.write(f"**Exposure:** {protocol.get('exposure', protocol.get('intervention', '?'))}")
                else:
                    st.write(f"**Intervention:** {protocol.get('intervention', '?')}")
                st.write(f"**PubMed query:** `{pubmed_query}`")
                # Generate PROSPERO draft
                try:
                    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
                    with open(_cfg_path) as _cf:
                        _cfg = yaml.safe_load(_cf) or {}
                    prospero_draft = a1.generate_prospero_draft(protocol, config=_cfg)
                    protocol["prospero_draft"] = prospero_draft
                    # Save PROSPERO draft to file
                    _prospero_path = os.path.join(run_dir, "prospero_draft.txt")
                    with open(_prospero_path, "w", encoding="utf-8") as _pf:
                        _pf.write(prospero_draft)
                except Exception:
                    pass  # Non-critical
                free_vram(a1)
                save_checkpoint(run_dir, "a1_protocol", {"query": query, "protocol": protocol, "pubmed_query": pubmed_query})
                s1.update(label="A1: Protocol \u2714", state="complete", expanded=False)
            except Exception as e:
                import traceback
                s1.update(label=f"A1: Failed \u2014 {e}", state="error", expanded=True)
                st.error(f"Agent 1 failed: {e}")
                st.code(traceback.format_exc(), language="text")
                free_vram(a1 if 'a1' in dir() else None)
                return None

    if _check_stop("A1: Protocol"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A2: Search ───────────────────────────────────────────
    cached_a2 = load_checkpoint(run_dir, "a2_search")
    if cached_a2:
        papers = cached_a2.get("papers")
        a2_stats = cached_a2.get("stats")
        with st.status("A2: Search (cached)", expanded=False) as s2:
            st.write(f"After dedup: **{a2_stats.get('total_after_dedup', len(papers) if papers else 0)}** papers")
            s2.update(label=f"A2: Search \u2714 (cached, {a2_stats.get('total_after_dedup', len(papers) if papers else 0)} papers)", state="complete", expanded=False)
    else:
        with st.status("A2: Searching databases...", expanded=True) as s2:
            try:
                a2 = Agent2Search()
                a2_result = a2.search(protocol, pubmed_query, max_results=max_results)
                papers = a2_result["papers"]
                a2_stats = a2_result["stats"]
                _db_parts = []
                for _dbk, _dbl in [("pubmed","PubMed"),("europepmc","Europe PMC"),("semantic_scholar","S2"),("google_scholar","GScholar")]:
                    _dbv = a2_stats.get(_dbk, 0)
                    if _dbv:
                        _db_parts.append(f"{_dbl}: **{_dbv}**")
                st.write(" | ".join(_db_parts) + f" | After dedup: **{a2_stats['total_after_dedup']}**")
                free_vram(a2)
                save_checkpoint(run_dir, "a2_search", {"papers": papers, "stats": a2_stats})
                _n_dbs = sum(1 for k in ('pubmed','europepmc','semantic_scholar','google_scholar') if a2_stats.get(k, 0))
                s2.update(
                    label=f"A2: Search \u2714 ({a2_stats['total_after_dedup']} papers from {_n_dbs} DBs)",
                    state="complete", expanded=False,
                )
            except Exception as e:
                free_vram(a2)
                s2.update(label=f"A2: Failed \u2014 {e}", state="error", expanded=True)
                st.error(f"Agent 2 failed: {e}")
                return None

    if not papers:
        st.warning("No papers found. Try a different query or increase max results.")
        save_checkpoint(run_dir, "a1_protocol", {"query": query, "protocol": protocol, "pubmed_query": pubmed_query})
        return None

    _log_pipeline_stage(run_dir, "A2", f"Search returned {len(papers)} papers")

    if _check_stop("A2: Search"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A3: Title/Abstract Screening ─────────────────────────
    cached_a3 = load_checkpoint(run_dir, "a3_screening")
    if cached_a3:
        a3_results = cached_a3
        stats3 = a3_results.get("stats", {})
        with st.status("A3: Screening (cached)", expanded=False) as s3:
            st.write(f"INCLUDE: **{stats3.get('included', 0)}** | EXCLUDE: **{stats3.get('excluded', 0)}** | UNCERTAIN: **{stats3.get('uncertain', 0)}**")
            s3.update(label=f"A3: Screening \u2714 (cached, {stats3.get('included', 0)} included)", state="complete", expanded=False)
    else:
        with st.status(f"A3: Screening {len(papers)} papers...", expanded=True) as s3:
            try:
                a3 = Agent3Screening(ensemble_size=a3_ensemble, voting_threshold=a3_vote_thr)
                a3_results = a3.screen_batch(papers, protocol)
                stats3 = a3_results["stats"]
                st.write(f"INCLUDE: **{stats3['included']}** | EXCLUDE: **{stats3['excluded']}** | UNCERTAIN: **{stats3['uncertain']}**")
                st.write(f"Time: {stats3['elapsed_s']}s ({stats3['s_per_paper']}s/paper)")
                free_vram(a3)
                save_checkpoint(run_dir, "a3_screening", a3_results)
                s3.update(
                    label=f"A3: Screening \u2714 ({stats3['included']} included)",
                    state="complete", expanded=False,
                )
            except Exception as e:
                s3.update(label=f"A3: Failed \u2014 {e}", state="error", expanded=True)
                st.error(f"Agent 3 failed: {e}")
                return None

    included_papers = a3_results.get("include", [])
    uncertain_papers = a3_results.get("uncertain", [])
    # Pass UNCERTAIN papers to A4 for full-text screening (they need more info to decide)
    papers_for_a4 = included_papers + uncertain_papers

    stats3 = a3_results.get("stats", {})
    _log_pipeline_stage(run_dir, "A3", f"Screening: {stats3.get('included', 0)} INCLUDE, {stats3.get('excluded', 0)} EXCLUDE, {stats3.get('uncertain', 0)} UNCERTAIN → {len(papers_for_a4)} papers to A4")

    if _check_stop("A3: Screening"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A3 Calibration & Sensitivity ──────────────────────────
    _render_voting_sensitivity(
        a3_results, "A3", ensemble_size=a3_ensemble, voting_threshold=a3_vote_thr,
        categories=["INCLUDE", "EXCLUDE", "UNCERTAIN"],
    )

    # ── A4: Full-text ────────────────────────────────────────
    if not papers_for_a4:
        st.info("No papers passed A3 screening. Skipping A4 & A5.")
    else:
        if uncertain_papers:
            st.info(f"Passing {len(included_papers)} INCLUDE + {len(uncertain_papers)} UNCERTAIN papers to A4 full-text screening.")
        cached_a4 = load_checkpoint(run_dir, "a4_fulltext")
        if cached_a4:
            a4_result = cached_a4
            stats4 = a4_result.get("stats", {})
            with st.status("A4: Full-text (cached)", expanded=False) as s4:
                st.write(f"INCLUDE: **{stats4.get('fulltext_included', 0)}** | EXCLUDE: **{stats4.get('fulltext_excluded', 0)}**")
                s4.update(label=f"A4: Full-text \u2714 (cached, {stats4.get('fulltext_included', 0)} included)", state="complete", expanded=False)
        else:
            with st.status(f"A4: Full-text screening ({len(papers_for_a4)} papers)...", expanded=True) as s4:
                try:
                    a4 = Agent4FullText(ensemble_size=a4_ensemble, voting_threshold=a4_vote_thr)
                    a4_result = a4.run(papers_for_a4, papers, protocol)
                    stats4 = a4_result["stats"]
                    st.write(
                        f"Retrieval: PMC={stats4['pmc_fulltext']}, "
                        f"Unpaywall={stats4['unpaywall_pdf']}, "
                        f"EuropePMC={stats4.get('europepmc_fulltext', 0)}, "
                        f"S2={stats4.get('semanticscholar_pdf', 0)}, "
                        f"OpenAlex={stats4.get('openalex_pdf', 0)}, "
                        f"CrossRef={stats4.get('crossref_pdf', 0)}, "
                        f"DOAJ={stats4.get('doaj_pdf', 0)}, "
                        f"Lens={stats4.get('lens_fulltext', 0)}, "
                        f"CORE={stats4.get('core_fulltext', 0)}, "
                        f"GScholar={stats4.get('google_scholar_pdf', 0)}, "
                        f"SciHub={stats4.get('scihub_pdf', 0)}, "
                        f"Abstract={stats4['abstract_only']}"
                    )
                    st.write(f"INCLUDE: **{stats4['fulltext_included']}** | EXCLUDE: **{stats4['fulltext_excluded']}**")
                    free_vram(a4)
                    save_checkpoint(run_dir, "a4_fulltext", a4_result)
                    s4.update(
                        label=f"A4: Full-text \u2714 ({stats4['fulltext_included']} included)",
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    s4.update(label=f"A4: Failed \u2014 {e}", state="error", expanded=True)
                    st.error(f"Agent 4 failed: {e}")

    if a4_result:
        stats4 = a4_result.get("stats", {})
        _log_pipeline_stage(run_dir, "A4", f"Full-text: {stats4.get('fulltext_included', 0)} INCLUDE, {stats4.get('fulltext_excluded', 0)} EXCLUDE")

    if _check_stop("A4: Full-text"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A4 Calibration & Sensitivity ──────────────────────────
    if a4_result and a4_result.get("papers"):
        _render_voting_sensitivity(
            a4_result, "A4", ensemble_size=a4_ensemble, voting_threshold=a4_vote_thr,
            categories=["INCLUDE", "EXCLUDE"],
        )

    # ── A5: Extraction ───────────────────────────────────────
    if a4_result and a4_result.get("include"):
        cached_a5 = load_checkpoint(run_dir, "a5_extraction")
        if cached_a5:
            a5_result = cached_a5
            stats5 = a5_result.get("stats", {})
            with st.status("A5: Extraction (cached)", expanded=False) as s5:
                st.write(f"Complete: **{stats5.get('complete_extraction', 0)}** | Partial: **{stats5.get('partial_extraction', 0)}**")
                s5.update(label=f"A5: Extraction \u2714 (cached)", state="complete", expanded=False)
        else:
            with st.status(f"A5: Extracting data ({len(a4_result['include'])} papers)...", expanded=True) as s5:
                try:
                    a5 = Agent5Extraction()
                    a5_result = a5.extract_all(a4_result["include"], protocol)
                    stats5 = a5_result["stats"]
                    _conf_line = ""
                    if stats5.get("dual_extraction"):
                        _conf_line = (
                            f" | Confidence: "
                            f"H **{stats5.get('high_confidence', 0)}** / "
                            f"M **{stats5.get('medium_confidence', 0)}** / "
                            f"L **{stats5.get('low_confidence', 0)}**"
                        )
                    st.write(
                        f"Complete: **{stats5['complete_extraction']}** | "
                        f"Partial: **{stats5['partial_extraction']}** | "
                        f"Failed: **{stats5['failed_extraction']}**"
                        f"{_conf_line}"
                    )
                    free_vram(a5)
                    save_checkpoint(run_dir, "a5_extraction", a5_result)
                    _log_pipeline_stage(run_dir, "A5", f"Extraction: {stats5.get('complete_extraction', 0)} complete, {stats5.get('numeric_complete', 0)} with numeric outcomes")
                    # Sanity check: warn if no numeric outcomes extracted
                    n_numeric = stats5.get("numeric_complete", 0)
                    if n_numeric == 0:
                        st.warning(
                            f"⚠️ A5 extracted 0 numeric outcomes from {stats5['total']} papers. "
                            "Meta-analysis (A7) will not be possible. Common causes: "
                            "study protocols without results, abstract-only papers, or papers "
                            "not reporting quantitative effect sizes."
                        )
                    _s5_label = f"A5: Extraction \u2714 ({stats5['complete_extraction']} complete)"
                    if stats5.get("dual_extraction"):
                        _s5_label += (
                            f" | {stats5.get('high_confidence', 0)}H/"
                            f"{stats5.get('medium_confidence', 0)}M/"
                            f"{stats5.get('low_confidence', 0)}L"
                        )
                    s5.update(
                        label=_s5_label,
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    s5.update(label=f"A5: Failed \u2014 {e}", state="error", expanded=True)
                    st.error(f"Agent 5 failed: {e}")
    else:
        if included_papers:
            st.info("No papers passed A4. Skipping A5.")

    if _check_stop("A5: Extraction"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A6: Quality / Risk of Bias ──────────────────────────
    a6_result = None
    a6_papers = None

    if a5_result and a5_result.get("papers"):
        a6_papers = a5_result["papers"]
    elif a4_result and a4_result.get("include"):
        a6_papers = a4_result["include"]

    if a6_papers:
        cached_a6 = load_checkpoint(run_dir, "a6_quality")
        if cached_a6:
            a6_result = cached_a6
            stats6 = a6_result.get("stats", {})
            with st.status("A6: Quality (cached)", expanded=False) as s6:
                st.write(f"RoB 2: **{stats6.get('rob2_assessed', 0)}** | ROBINS-I: **{stats6.get('robins_i_assessed', 0)}** | JBI: **{stats6.get('jbi_assessed', 0)}**")
                s6.update(label=f"A6: Quality \u2714 (cached)", state="complete", expanded=False)
        else:
            with st.status(f"A6: Quality assessment ({len(a6_papers)} papers)...", expanded=True) as s6:
                try:
                    a6 = Agent6Quality()
                    a6_result = a6.assess_all(a6_papers, protocol)
                    stats6 = a6_result["stats"]
                    st.write(
                        f"RoB 2: **{stats6['rob2_assessed']}** | "
                        f"ROBINS-I: **{stats6['robins_i_assessed']}** | "
                        f"JBI: **{stats6['jbi_assessed']}** | "
                        f"Failed: **{stats6['failed']}**"
                    )
                    weights = [p["quality_weight"] for p in a6_result["papers"]]
                    avg_w = sum(weights) / len(weights) if weights else 0
                    st.write(f"Avg quality weight: **{avg_w:.2f}**")
                    free_vram(a6)
                    save_checkpoint(run_dir, "a6_quality", a6_result)
                    s6.update(
                        label=f"A6: Quality \u2714 ({stats6['rob2_assessed']} RoB2, {stats6['robins_i_assessed']} ROBINS-I, {stats6['jbi_assessed']} JBI)",
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    s6.update(label=f"A6: Failed \u2014 {e}", state="error", expanded=True)
                    st.error(f"Agent 6 failed: {e}")

    if _check_stop("A6: Quality"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── Generate RoB plots after A6 ──────────────────────────
    rob_tl_path = None
    rob_sum_path = None
    if a6_result and a6_result.get("papers"):
        try:
            rob_tl_path = generate_rob_traffic_light(
                a6_result, os.path.join(plots_dir, "rob_traffic_light.png")
            )
            rob_sum_path = generate_rob_summary(
                a6_result, os.path.join(plots_dir, "rob_summary.png")
            )
            if rob_tl_path:
                a6_result["rob_traffic_light_plot"] = rob_tl_path
            if rob_sum_path:
                a6_result["rob_summary_plot"] = rob_sum_path
        except Exception as e:
            st.caption(f"RoB plot generation failed: {e}")

    # ── A7: Statistical Synthesis ───────────────────────────
    a7_result = None

    # A7 input: prefer A6 output; fallback to A5 with default weights
    a7_input = None
    if a6_result and a6_result.get("papers"):
        a7_input = a6_result
    elif a5_result and a5_result.get("papers"):
        a7_input = {
            "papers": [{**p, "quality_weight": 0.5} for p in a5_result["papers"]]
        }

    if a7_input and len(a7_input.get("papers", [])) >= 2:
        cached_a7 = load_checkpoint(run_dir, "a7_synthesis")
        if cached_a7:
            a7_result = cached_a7
            stats7 = a7_result.get("stats", {})
            with st.status("A7: Synthesis (cached)", expanded=False) as s7:
                st.write(f"Analyses: **{stats7.get('analyses_performed', 0)}**")
                s7.update(label=f"A7: Synthesis \u2714 (cached)", state="complete", expanded=False)
        else:
            with st.status("A7: Statistical meta-analysis...", expanded=True) as s7:
                try:
                    # Load config for A7 synthesis settings
                    _a7_config = {}
                    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
                    if os.path.exists(_config_path):
                        try:
                            with open(_config_path) as _cf:
                                _a7_config = yaml.safe_load(_cf) or {}
                        except Exception:
                            pass
                    a7 = Agent7Synthesis(config=_a7_config)
                    a7_result = a7.synthesize(a7_input, protocol, output_dir=run_dir)
                    stats7 = a7_result["stats"]
                    st.write(
                        f"Papers with usable data: **{stats7['papers_with_usable_data']}** | "
                        f"Analyses: **{stats7['analyses_performed']}**"
                    )
                    for measure, analysis in a7_result.get("analyses", {}).items():
                        model = analysis.get("model", {})
                        grade = analysis.get("grade", {})
                        st.write(
                            f"**{measure}:** pooled = {model.get('pooled_estimate', '?')} "
                            f"[{model.get('ci_lower', '?')}, {model.get('ci_upper', '?')}], "
                            f"I\u00b2 = {model.get('I2', '?')}%, "
                            f"GRADE: {grade.get('final_level', '?')}"
                        )
                    # Clean R objects before checkpointing
                    a7_clean_for_ckpt = _clean_r_objects(a7_result)
                    save_checkpoint(run_dir, "a7_synthesis", a7_clean_for_ckpt)
                    s7.update(
                        label=f"A7: Synthesis \u2714 ({stats7['analyses_performed']} analyses)",
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    s7.update(label=f"A7: Failed \u2014 {e}", state="error", expanded=True)
                    st.error(f"Agent 7 failed: {e}")

    if a7_result:
        n_analyses = a7_result.get("stats", {}).get("analyses_performed", 0) if isinstance(a7_result.get("stats"), dict) else 0
        _log_pipeline_stage(run_dir, "A7", f"Synthesis: {n_analyses} analyses performed")
        if isinstance(a7_result.get("stats"), dict):
            if a7_result["stats"].get("analyses_performed", 0) == 0:
                st.warning("⚠️ A7 performed 0 meta-analyses. No usable numeric data was available from A5 extraction.")

    if _check_stop("A7: Synthesis"):
        return {"_stopped": True, "_run_dir": run_dir, "query": query, "protocol": protocol}

    # ── A8: Manuscript Generation ──────────────────────────────
    a8_result = None

    a8_papers = None
    if a4_result and a4_result.get("include"):
        a8_papers = a4_result["include"]

    if a8_papers and len(a8_papers) >= 2:
        cached_a8 = load_checkpoint(run_dir, "a8_manuscript")
        if cached_a8:
            a8_result = cached_a8
            stats8 = a8_result.get("stats", {})
            with st.status("A8: Manuscript (cached)", expanded=False) as s8:
                st.write(f"Papers cited: **{stats8.get('total_papers_cited', 0)}**")
                s8.update(label=f"A8: Manuscript \u2714 (cached)", state="complete", expanded=False)
        else:
            with st.status("A8: Generating manuscript (RAG)...", expanded=True) as s8:
                try:
                    _ms_lang = st.session_state.get("manuscript_language", "en")
                    a8 = Agent8Manuscript(language=_ms_lang)
                    a7_for_a8 = _clean_r_objects(a7_result) if a7_result else None
                    _s3 = a3_results.get("stats", {}) if a3_results else {}
                    _s4 = a4_result.get("stats", {}) if a4_result else {}
                    _n_ext = 0
                    if a5_result and a5_result.get("stats"):
                        _n_ext = a5_result["stats"].get("complete_extraction", 0) + a5_result["stats"].get("partial_extraction", 0)
                    _n_qa = 0
                    if a6_result and a6_result.get("stats"):
                        _n_qa = a6_result["stats"].get("rob2_assessed", 0) + a6_result["stats"].get("robins_i_assessed", 0) + a6_result["stats"].get("jbi_assessed", 0)
                    prisma_for_a8 = {
                        "identified": _s3.get("total", 0),
                        "excluded_title_abstract": _s3.get("excluded", 0),
                        "uncertain_title_abstract": _s3.get("uncertain", 0),
                        "assessed_fulltext": _s3.get("included", 0),
                        "excluded_fulltext": _s4.get("fulltext_excluded", 0),
                        "included_final": _s4.get("fulltext_included", len(a8_papers)),
                        "extracted": _n_ext,
                        "quality_assessed": _n_qa,
                    }
                    protocol_for_a8 = dict(protocol)
                    protocol_for_a8["pubmed_query"] = pubmed_query
                    a8_result = a8.generate_manuscript(
                        protocol=protocol_for_a8,
                        included_papers=a8_papers,
                        extraction=a5_result,
                        quality=a6_result,
                        synthesis=a7_for_a8,
                        output_dir=run_dir,
                        prisma=prisma_for_a8,
                    )
                    stats8 = a8_result["stats"]
                    st.write(
                        f"Sections: **6** | "
                        f"Papers cited: **{stats8['total_papers_cited']}** | "
                        f"RAG chunks: **{stats8['total_chunks']}**"
                    )
                    st.write(f"Time: {stats8['elapsed_s']}s")
                    free_vram(a8)
                    save_checkpoint(run_dir, "a8_manuscript", a8_result)
                    s8.update(
                        label=f"A8: Manuscript \u2714 ({stats8['total_papers_cited']} papers cited)",
                        state="complete", expanded=False,
                    )
                except Exception as e:
                    s8.update(label=f"A8: Failed \u2014 {e}", state="error", expanded=True)
                    st.error(f"Agent 8 failed: {e}")

    # ── Build PRISMA summary ─────────────────────────────────
    stats3 = a3_results.get("stats", {}) if a3_results else {}
    n_extracted = 0
    if a5_result and a5_result.get("stats"):
        n_extracted = a5_result["stats"].get("complete_extraction", 0) + a5_result["stats"].get("partial_extraction", 0)

    n_quality_assessed = 0
    if a6_result and a6_result.get("stats"):
        n_quality_assessed = a6_result["stats"].get("rob2_assessed", 0) + a6_result["stats"].get("robins_i_assessed", 0) + a6_result["stats"].get("jbi_assessed", 0)

    _a2s = a2_stats or {}
    prisma = {
        "identified": _a2s.get("total_before_dedup", stats3.get("total", 0)),
        "identified_per_db": {
            "pubmed": _a2s.get("pubmed", 0),
            "europepmc": _a2s.get("europepmc", 0),
            "semantic_scholar": _a2s.get("semantic_scholar", 0),
        },
        "after_dedup": _a2s.get("total_after_dedup", stats3.get("total", 0)),
        "duplicates_removed": _a2s.get("duplicates_removed", 0),
        "screened": stats3.get("total", 0),
        "excluded_title_abstract": stats3.get("excluded", 0),
        "uncertain_title_abstract": stats3.get("uncertain", 0),
        "assessed_fulltext": stats3.get("included", 0) + stats3.get("uncertain", 0),
        "excluded_fulltext": a4_result["stats"]["fulltext_excluded"] if a4_result and a4_result.get("stats") else 0,
        "included_final": a4_result["stats"]["fulltext_included"] if a4_result and a4_result.get("stats") else 0,
        "extracted": n_extracted,
        "quality_assessed": n_quality_assessed,
    }

    # ── Generate PRISMA flow diagram ─────────────────────────
    prisma_flow_path = None
    try:
        n_included = prisma.get("included_final", 0)
        n_synthesis = 0
        if a7_result and a7_result.get("stats"):
            n_synthesis = a7_result["stats"].get("papers_with_usable_data", 0)

        prisma_diagram_data = {
            "identified": prisma.get("identified", 0),
            "identified_per_db": prisma.get("identified_per_db", {}),
            "duplicates_removed": prisma.get("duplicates_removed", 0),
            "screened": prisma.get("screened", 0),
            "excluded_screening": prisma.get("excluded_title_abstract", 0),
            "fulltext_assessed": prisma.get("assessed_fulltext", 0),
            "excluded_fulltext": prisma.get("excluded_fulltext", 0),
            "not_retrieved": 0,
            "included_review": n_included,
            "included_synthesis": n_synthesis,
            "quality_assessed": n_quality_assessed,
        }
        prisma_flow_path = generate_prisma_flow(
            prisma_diagram_data, os.path.join(plots_dir, "prisma_flow.png")
        )
        if prisma_flow_path:
            prisma["prisma_flow_diagram"] = prisma_flow_path
    except Exception as e:
        st.caption(f"PRISMA diagram generation failed: {e}")

    # Clean A7 result: remove non-serializable R objects
    a7_clean = None
    if a7_result:
        a7_clean = _clean_r_objects(a7_result)

    # ── Review Report (optional, non-blocking) ──────────────
    review_report_data = None
    try:
        _interim = {
            "screening": a3_results,
            "fulltext_retrieval": a4_result,
            "data_extraction": a5_result,
            "quality_assessment": a6_result,
            "synthesis": a7_clean,
        }
        rr_gen = ReviewReportGenerator()
        review_report_data = rr_gen.generate(_interim)

        # Save JSON report
        rr_json_path = os.path.join(run_dir, "review_report.json")
        with open(rr_json_path, "w", encoding="utf-8") as f:
            json.dump(review_report_data, f, indent=2, ensure_ascii=False)

        # Save plain text report
        rr_txt = rr_gen.generate_text_report(review_report_data)
        rr_txt_path = os.path.join(run_dir, "review_report.txt")
        with open(rr_txt_path, "w", encoding="utf-8") as f:
            f.write(rr_txt)

        n_flags = review_report_data.get("summary", {}).get("total_flags", 0)
        confidence = review_report_data.get("summary", {}).get("pipeline_confidence", "?")
        st.caption(
            f"Review report: {n_flags} flags, pipeline confidence: {confidence}"
        )
    except Exception as e:
        st.caption(f"Review report generation failed (non-blocking): {e}")

    # Collect system info + full config for reproducibility
    _sys_info = {"timestamp": datetime.now().isoformat()}
    try:
        _sys_info["python"] = sys.version.split()[0]
        _sys_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            _sys_info["gpu"] = torch.cuda.get_device_name(0)
            _sys_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
            _sys_info["cuda"] = torch.version.cuda
    except Exception:
        pass

    _full_cfg = {}
    try:
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(_cfg_path) as _cf:
            _full_cfg = yaml.safe_load(_cf) or {}
    except Exception:
        pass

    output = {
        "query": query,
        "protocol": protocol,
        "pubmed_query": pubmed_query,
        "config_snapshot": _full_cfg,
        "system_info": _sys_info,
        "prisma": prisma,
        "search_stats": _a2s,
        "screening": a3_results,
        "fulltext_retrieval": a4_result,
        "data_extraction": a5_result,
        "quality_assessment": a6_result,
        "synthesis": a7_clean,
        "manuscript": a8_result,
        "review_report": review_report_data,
    }

    # Save final pipeline JSON
    out_path = os.path.join(run_dir, "pipeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    # Save additional export formats alongside pipeline output
    try:
        export_csv(output, os.path.join(run_dir, "included_studies.csv"))
    except Exception:
        pass
    try:
        export_excel(output, os.path.join(run_dir, "meta_analysis.xlsx"))
    except Exception:
        pass
    try:
        export_revman(output, os.path.join(run_dir, "revman_export.xml"))
    except Exception:
        pass
    try:
        export_summary_table(output, os.path.join(run_dir, "summary_table.md"))
    except Exception:
        pass

    # Clean up checkpoint files after successful completion (if config allows)
    _pipeline_cfg = {}
    try:
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(_cfg_path) as _cf:
            _pipeline_cfg = (yaml.safe_load(_cf) or {}).get("pipeline", {})
    except Exception:
        pass
    if _pipeline_cfg.get("cleanup_checkpoints_on_success", False):
        finalize_checkpoints(run_dir)

    st.session_state.pipeline_running = False
    return output


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

_favicon_path = os.path.join(_ASSETS_DIR, "favicon.png")
st.set_page_config(
    page_title="SAM \u2014 Systematic Agent for Meta-analysis",
    page_icon=_favicon_path if os.path.exists(_favicon_path) else "\U0001f9ec",
    layout="wide",
)

init_session_state()
inject_css()

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding — SVG logo from logo kit
    _pill_logo = _themed_logo("logo_pill")
    if _pill_logo:
        st.markdown(
            f'<div class="sam-sidebar-logo">'
            f'<img src="{_pill_logo}" alt="SAM">'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### SAM")

    # ── Theme toggle ──
    _theme_cols = st.columns([1, 3, 1])
    with _theme_cols[1]:
        _dark_toggle = st.toggle(
            "Dark mode",
            value=st.session_state.dark_mode,
            key="dark_mode_toggle",
        )
    if _dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = _dark_toggle
        st.rerun()

    # ── New Analysis CTA ──
    _new_btn = st.container()
    with _new_btn:
        st.markdown('<style>.sam-new-btn-wrap .stButton{margin:0;}</style>', unsafe_allow_html=True)
        if st.button("New Analysis", use_container_width=True, key="btn_new_analysis"):
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.session_state.pending_query = None
            st.session_state.compare_mode = False
            st.session_state.compare_selections = []
            st.rerun()
    # Apply gradient CTA styling via wrapper class
    st.markdown(
        '<script>document.querySelectorAll("[data-testid=\\"stSidebar\\"] .stButton").forEach('
        'function(el){if(el.querySelector("button") && el.querySelector("button").textContent.trim()==="New Analysis")'
        '{el.classList.add("sam-new-btn")}})</script>',
        unsafe_allow_html=True,
    )
    # CSS-only fallback: target first button in sidebar
    st.markdown("""<style>
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child .stButton:first-of-type > button {
            background: linear-gradient(135deg, #0D9488 0%, #06B6D4 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 0.6rem !important;
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            padding: 0.6rem 1rem !important;
            box-shadow: 0 2px 8px rgba(13,148,136,0.25);
            transition: all 0.2s ease !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child .stButton:first-of-type > button:hover {
            background: linear-gradient(135deg, #0F766E 0%, #0891B2 100%) !important;
            box-shadow: 0 4px 12px rgba(13,148,136,0.35) !important;
            transform: translateY(-1px) !important;
            color: #FFFFFF !important;
        }
    </style>""", unsafe_allow_html=True)

    st.divider()

    # ── Pipeline History ──────────────────────────────────────────
    st.markdown('<p class="sam-section-label">Pipeline History</p>', unsafe_allow_html=True)
    history = scan_pipeline_outputs()

    if history:
        # Compare mode toggle
        st.session_state.compare_mode = st.toggle(
            "Compare mode", value=st.session_state.compare_mode,
            help="Select 2 pipelines to compare side-by-side",
        )

        for entry in history[:10]:
            trunc = entry["query"][:45] + ("..." if len(entry["query"]) > 45 else "")
            n_incl = entry.get("papers_included", entry.get("included", 0))
            synth_icon = " \u2713" if entry.get("has_synthesis") else ""
            ms_icon = " \U0001f4d1" if entry.get("has_manuscript") else ""
            label = f'{trunc}'
            sublabel = f'{entry["date"]} \u00b7 {n_incl} papers{synth_icon}{ms_icon}'

            if st.session_state.compare_mode:
                selected = entry["path"] in st.session_state.compare_selections
                if st.checkbox(label, value=selected, key=f'cmp_{entry["path"]}'):
                    if entry["path"] not in st.session_state.compare_selections:
                        st.session_state.compare_selections.append(entry["path"])
                        if len(st.session_state.compare_selections) > 2:
                            st.session_state.compare_selections = st.session_state.compare_selections[-2:]
                else:
                    if entry["path"] in st.session_state.compare_selections:
                        st.session_state.compare_selections.remove(entry["path"])
            else:
                if st.button(label, key=f'hist_{entry["path"]}', help=sublabel):
                    load_historical_run(entry["path"])
                    st.rerun()

        if st.session_state.compare_mode and len(st.session_state.compare_selections) == 2:
            _cmp_container = st.container()
            with _cmp_container:
                if st.button("Compare Selected", use_container_width=True, key="btn_compare"):
                    st.session_state.show_welcome = False
                    st.rerun()
            st.markdown("""<style>
                [data-testid="stSidebar"] button[kind="secondary"]:has(+ div),
                button[data-testid="baseButton-secondary"][key*="btn_compare"] {
                    background: linear-gradient(135deg, #7C3AED 0%, #8B5CF6 100%) !important;
                    color: #FFFFFF !important; border: none !important;
                    font-weight: 600 !important; border-radius: 0.6rem !important;
                    box-shadow: 0 2px 8px rgba(124,58,237,0.25) !important;
                }
            </style>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#94A3B8;font-size:0.76rem;text-align:center;padding:0.5rem 0;">'
            'No previous runs</p>',
            unsafe_allow_html=True,
        )

    # ── Resume from checkpoint ──────────────────────────────────
    resumable = get_resumable_runs()
    incomplete_runs = [r for r in resumable if not r["complete"]]
    if incomplete_runs:
        st.divider()
        st.markdown('<p class="sam-section-label">Resume Pipeline</p>', unsafe_allow_html=True)
        for run_info in incomplete_runs[:5]:
            trunc_q = run_info["query"][:40] + ("..." if len(run_info["query"]) > 40 else "")
            ckpts = ", ".join(c.upper().replace("_", "") for c in run_info["checkpoints"])
            btn_label = f'{trunc_q} \u00b7 {run_info["date"]}'
            _resume_container = st.container()
            with _resume_container:
                if st.button(btn_label, key=f'resume_{run_info["path"]}', help=f"Checkpoints: {ckpts}"):
                    a1_ckpt = load_checkpoint(run_info["path"], "a1_protocol")
                    if a1_ckpt:
                        st.session_state.pending_query = a1_ckpt.get("query", "")
                        st.session_state.resume_run_dir = run_info["path"]
                        st.session_state.show_welcome = False
                        st.rerun()
        # Style resume buttons with amber accent
        st.markdown("""<style>
            [data-testid="stSidebar"] button[key*="resume_"] {
                border-left: 3px solid #F59E0B !important;
                background: #FFFBEB !important;
                color: #92400E !important;
            }
        </style>""", unsafe_allow_html=True)

    st.divider()

    # ── Pipeline Settings ─────────────────────────────────────────
    st.markdown('<p class="sam-section-label">Pipeline Settings</p>', unsafe_allow_html=True)

    with st.expander("Output", expanded=False, icon="\U0001f4dd"):
        _lang_options = ["en", "es", "pt", "fr", "de"]
        _lang_labels = {
            "en": "English", "es": "Espanol", "pt": "Portugues",
            "fr": "Francais", "de": "Deutsch",
        }
        _current_lang = st.session_state.manuscript_language
        _lang_idx = _lang_options.index(_current_lang) if _current_lang in _lang_options else 0
        st.session_state.manuscript_language = st.selectbox(
            "Manuscript language",
            _lang_options,
            index=_lang_idx,
            format_func=lambda x: _lang_labels.get(x, x),
            help="Language for the generated manuscript (A8)",
        )

    with st.expander("Search & Dedup", expanded=False, icon="\U0001f50d"):
        st.session_state.max_results = st.slider(
            "Max papers", 10, 200,
            st.session_state.max_results, step=10,
        )
        st.session_state.dedup_threshold = st.slider(
            "Dedup similarity", 0.80, 0.99,
            st.session_state.dedup_threshold, step=0.01,
        )

    with st.expander("Screening (A3/A4)", expanded=False, icon="\u2696\ufe0f"):
        col_a3, col_a4 = st.columns(2)
        with col_a3:
            st.markdown('<span style="font-size:0.7rem;font-weight:600;color:#0D9488;">A3</span>', unsafe_allow_html=True)
            _a3_ens_options = [1, 3, 5]
            _a3_ens_idx = _a3_ens_options.index(st.session_state.a3_ensemble) if st.session_state.a3_ensemble in _a3_ens_options else 1
            st.session_state.a3_ensemble = st.selectbox(
                "Ensemble", _a3_ens_options,
                index=_a3_ens_idx,
                key="a3_ens_sel",
            )
            st.session_state.a3_vote_threshold = st.slider(
                "Threshold", 0.50, 0.90,
                st.session_state.a3_vote_threshold, step=0.05,
                key="a3_vote_sl",
            )
        with col_a4:
            st.markdown('<span style="font-size:0.7rem;font-weight:600;color:#06B6D4;">A4</span>', unsafe_allow_html=True)
            _a4_ens_options = [1, 3]
            _a4_ens_idx = _a4_ens_options.index(st.session_state.a4_ensemble) if st.session_state.a4_ensemble in _a4_ens_options else 0
            st.session_state.a4_ensemble = st.selectbox(
                "Ensemble", _a4_ens_options,
                index=_a4_ens_idx,
                key="a4_ens_sel",
            )
            st.session_state.a4_vote_threshold = st.slider(
                "Threshold", 0.50, 0.90,
                st.session_state.a4_vote_threshold, step=0.05,
                key="a4_vote_sl",
            )

    st.divider()

    # ── GPU info ──────────────────────────────────────────────────
    try:
        if torch.cuda.is_available():
            _gpu_name = torch.cuda.get_device_name(0)
            _vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            _vram_used = torch.cuda.memory_allocated() / 1e9
            st.markdown(
                f'<div class="sam-gpu-chip">'
                f'<div class="sam-gpu-dot"></div>'
                f'<span class="sam-gpu-text">{_gpu_name} \u00b7 {_vram_used:.1f}/{_vram_total:.0f} GB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sam-gpu-chip">'
                '<div class="sam-gpu-dot" style="background:#F59E0B;"></div>'
                '<span class="sam-gpu-text">CPU mode</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<div class="sam-gpu-chip">'
            '<div class="sam-gpu-dot" style="background:#F59E0B;"></div>'
            '<span class="sam-gpu-text">GPU info unavailable</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Footer with icon
    _icon_logo = _themed_logo("logo_icon")
    if _icon_logo:
        st.markdown(
            f'<div style="text-align:center;padding:0.3rem 0;">'
            f'<img src="{_icon_logo}" width="24" style="opacity:0.4;">'
            f'</div>',
            unsafe_allow_html=True,
        )
    _sam_version = "0.2.0"
    try:
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(_cfg_path) as _vf:
            _sam_version = (yaml.safe_load(_vf) or {}).get("project", {}).get("version", "0.2.0")
    except Exception:
        pass
    st.markdown(
        f'<p class="sam-footer">SAM v{_sam_version}</p>',
        unsafe_allow_html=True,
    )

# ── Main area: message replay ────────────────────────────────────────────────

# Comparison mode rendering
if (st.session_state.compare_mode
        and len(st.session_state.get("compare_selections", [])) == 2):
    st.markdown("### Pipeline Comparison")
    sel = st.session_state.compare_selections
    try:
        data_a = load_pipeline_data(sel[0])
        data_b = load_pipeline_data(sel[1])
        label_a = data_a.get("query", "Run A")[:60]
        label_b = data_b.get("query", "Run B")[:60]
        render_comparison(data_a, data_b, label_a, label_b)
    except Exception as _cmp_err:
        st.error(f"Could not load pipelines for comparison: {_cmp_err}")

# Show welcome on empty state
if st.session_state.show_welcome and not st.session_state.messages:
    render_welcome()

# Render historical messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="\U0001f464"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant", avatar=_assistant_avatar()):
            if msg["type"] == "result" and msg.get("data"):
                render_results(msg["data"])
            else:
                st.markdown(msg.get("content", ""))

# ── Chat input ───────────────────────────────────────────────────────────────

# Handle pending query from pill click
if st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Describe your research question...")

if user_input:
    # Hide welcome
    st.session_state.show_welcome = False

    # Add user message
    st.session_state.messages.append({
        "role": "user", "type": "query", "content": user_input,
    })
    with st.chat_message("user", avatar="\U0001f464"):
        st.markdown(user_input)

    # Check for resume
    _resume_dir = st.session_state.get("resume_run_dir")
    st.session_state.resume_run_dir = None  # Clear after use

    # Run pipeline inside assistant message
    with st.chat_message("assistant", avatar=_assistant_avatar()):
        # Stop button — visible during pipeline execution
        stop_col1, stop_col2 = st.columns([4, 1])
        with stop_col2:
            if st.button("\u23F9 Stop Pipeline", type="secondary", use_container_width=True):
                st.session_state.pipeline_stop_requested = True

        output = run_pipeline(user_input, resume_dir=_resume_dir)

        st.session_state.pipeline_running = False

        if output:
            if output.get("_stopped"):
                st.info(
                    "Pipeline stopped. Checkpoints saved — "
                    "use **Resume Pipeline** in the sidebar to continue later."
                )
                st.session_state.messages.append({
                    "role": "assistant", "type": "result",
                    "content": "Pipeline stopped (partial results saved).",
                    "data": output,
                })
            else:
                st.divider()
                render_results(output)
                st.session_state.messages.append({
                    "role": "assistant", "type": "result",
                    "content": "Pipeline completed.",
                    "data": output,
                })
        else:
            fail_msg = "Pipeline did not complete. Check the errors above."
            st.warning(fail_msg)
            st.session_state.messages.append({
                "role": "assistant", "type": "error",
                "content": fail_msg,
            })
