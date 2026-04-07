"""
main.py — Tactical Decision Engine (Human-in-the-Loop)
Stack  : LangGraph · LangChain-Ollama (llama3.2:3b) · Streamlit
Target : Ryzen 5 / 16 GB RAM  (CPU inference via Ollama)
"""

# ── Standard library ──────────────────────────────────────────────────────────
import uuid
from typing import Annotated, Literal

# ── LangGraph / LangChain ────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

# ── Streamlit ────────────────────────────────────────────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tactical Decision Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING  — dark military / tactical aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

:root {
    --bg-primary:   #0a0c0e;
    --bg-secondary: #111418;
    --bg-card:      #161b22;
    --border:       #21262d;
    --accent-green: #39d353;
    --accent-red:   #f85149;
    --accent-amber: #e3b341;
    --accent-blue:  #58a6ff;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --font-mono:    'Share Tech Mono', monospace;
    --font-ui:      'Barlow Condensed', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-ui) !important;
}

/* ── Header ── */
.tde-header {
    border-bottom: 1px solid var(--border);
    padding: 1.2rem 0 1rem 0;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: baseline;
    gap: 1rem;
}
.tde-header h1 {
    font-family: var(--font-ui);
    font-weight: 700;
    font-size: 1.9rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-primary);
    margin: 0;
}
.tde-header .sub {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent-green);
    letter-spacing: 0.2em;
    margin: 0;
}

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
}
.status-idle    { background: #21262d; color: var(--text-muted); }
.status-running { background: #1c2a1c; color: var(--accent-green);
                  box-shadow: 0 0 8px #39d35340; }
.status-waiting { background: #2d1f06; color: var(--accent-amber);
                  box-shadow: 0 0 8px #e3b34140; animation: pulse 1.4s infinite; }
.status-done    { background: #0d1117; color: var(--accent-blue); }
.status-threat  { background: #2d0c0c; color: var(--accent-red);
                  box-shadow: 0 0 8px #f8514940; }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.55; }
}

/* ── Cards ── */
.tde-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.tde-card-title {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* ── Vision input ── */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px #58a6ff25 !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] button {
    font-family: var(--font-ui) !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 0.45rem 1.6rem !important;
    transition: all 0.15s ease !important;
}

/* Primary / Analyze */
div[data-testid="stButton"]:nth-of-type(1) button {
    background: #1a2d1a !important;
    border: 1px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
}
div[data-testid="stButton"]:nth-of-type(1) button:hover {
    background: #39d35320 !important;
    box-shadow: 0 0 10px #39d35340 !important;
}

/* Approve */
.approve-btn button {
    background: #162716 !important;
    border: 1px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
    font-size: 1rem !important;
}
.approve-btn button:hover { background: #39d35325 !important; }

/* Deny */
.deny-btn button {
    background: #270e0e !important;
    border: 1px solid var(--accent-red) !important;
    color: var(--accent-red) !important;
    font-size: 1rem !important;
}
.deny-btn button:hover { background: #f8514925 !important; }

/* Reset */
.reset-btn button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
}

/* ── Report block ── */
.tde-report {
    background: #0d1117;
    border-left: 3px solid var(--accent-blue);
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.4rem;
    font-family: var(--font-ui);
    font-size: 1rem;
    line-height: 1.75;
    color: #c9d1d9;
    white-space: pre-wrap;
}

/* ── Action executed block ── */
.action-executed {
    background: #0d1f0d;
    border: 1px solid var(--accent-green);
    border-radius: 6px;
    padding: 1rem 1.4rem;
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--accent-green);
    letter-spacing: 0.08em;
}

/* ── Analysis result ── */
.analysis-result {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Threat level bar ── */
.threat-no  { color: var(--accent-green); font-family: var(--font-mono); font-size: 0.82rem; }
.threat-yes { color: var(--accent-red);   font-family: var(--font-mono); font-size: 0.82rem;
              animation: pulse 1s infinite; }

/* Divider */
.tde-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LLM  (Ollama — CPU-friendly, 3B parameters)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Ollama model…")
def get_llm():
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0.2,          # low temp → consistent threat decisions
        num_predict=1024,         # token budget for tactical report
    )


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────
class TacticalState(TypedDict):
    messages:      Annotated[list, add_messages]
    vision_string: str
    threat_found:  bool
    analysis:      str          # analyzer's raw reasoning
    final_report:  str          # long tactical description (deny path)
    action_taken:  bool         # True when attack_tool() fires


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS / ACTIONS
# ─────────────────────────────────────────────────────────────────────────────
def attack_tool() -> str:
    """Dummy action executed when operator approves a threat."""
    print("ACTION EXECUTED")            # terminal confirmation
    return "ACTION EXECUTED"


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────────────────────────
ANALYZE_PROMPT = """You are a tactical AI threat-assessment system.

Analyze the following visual scene description from drone footage.
Determine if it contains HIGH-RISK elements such as:
- Weapons, firearms, explosives
- Active conflict or violence
- Humans in danger or under threat
- Critical infrastructure under attack
- Hazardous materials or events

Vision string:
\"\"\"{vision}\"\"\"

Respond in EXACTLY this format (nothing else):
THREAT: YES or NO
REASON: <one concise sentence explaining your decision>"""

DENY_REPORT_PROMPT = """You are a senior tactical intelligence analyst.
The following drone observation was flagged as a potential threat but the operator DENIED the action.

Vision string:
\"\"\"{vision}\"\"\"

Write a thorough 3-paragraph tactical assessment:
Paragraph 1 — Detailed situational analysis of what was observed.
Paragraph 2 — Risk evaluation: what could go wrong, what threats remain.
Paragraph 3 — Recommended non-kinetic courses of action for the operator.

Write formally, precisely, and with strategic depth."""


def node_analyzer(state: TacticalState) -> dict:
    """Node 1: LLM decides if the vision string contains a threat."""
    llm    = get_llm()
    prompt = ANALYZE_PROMPT.format(vision=state["vision_string"])
    response: AIMessage = llm.invoke([HumanMessage(content=prompt)])

    raw = response.content.strip()

    # Parse structured response
    threat_found = "THREAT: YES" in raw.upper()
    reason_line  = next(
        (ln for ln in raw.splitlines() if ln.upper().startswith("REASON:")),
        "REASON: (see raw output)"
    )
    reason = reason_line.split(":", 1)[-1].strip()

    return {
        "messages":     [response],
        "threat_found": threat_found,
        "analysis":     reason,
    }


def node_hitl_gate(state: TacticalState) -> dict:
    """
    Interrupt node — pauses the graph and surfaces the decision to the human.
    LangGraph's interrupt() raises an Interrupt exception that the MemorySaver
    checkpointer captures, letting us resume later with a Command(resume=...).
    """
    decision = interrupt({
        "question": "Threat detected. Do you APPROVE or DENY the action?",
        "vision":   state["vision_string"],
        "analysis": state["analysis"],
    })
    # 'decision' is whatever value was passed to Command(resume=...)
    return {"messages": [HumanMessage(content=f"Operator decision: {decision}")]}


def node_action(state: TacticalState) -> dict:
    """Approved path: execute the attack tool."""
    result = attack_tool()
    return {
        "action_taken": True,
        "messages": [AIMessage(content=result)],
    }


def node_deny_report(state: TacticalState) -> dict:
    """Denied path: LLM generates a long tactical description."""
    llm    = get_llm()
    prompt = DENY_REPORT_PROMPT.format(vision=state["vision_string"])
    response: AIMessage = llm.invoke([HumanMessage(content=prompt)])
    return {
        "final_report": response.content.strip(),
        "messages":     [response],
    }


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL ROUTING
# ─────────────────────────────────────────────────────────────────────────────
def route_after_analysis(state: TacticalState) -> Literal["hitl_gate", "__end__"]:
    """Route to human checkpoint only when a threat is found."""
    return "hitl_gate" if state["threat_found"] else END


def route_after_hitl(state: TacticalState) -> Literal["action", "deny_report"]:
    """
    Inspect the most recent HumanMessage to determine operator decision.
    Command(resume="APPROVE") → action node
    Command(resume="DENY")    → deny report node
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content.upper()
            if "APPROVE" in text:
                return "action"
            if "DENY" in text:
                return "deny_report"
    return "deny_report"     # safe default


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building LangGraph…")
def build_graph():
    memory = MemorySaver()           # in-memory checkpointer for pause/resume

    builder = StateGraph(TacticalState)

    # Register nodes
    builder.add_node("analyzer",    node_analyzer)
    builder.add_node("hitl_gate",   node_hitl_gate)
    builder.add_node("action",      node_action)
    builder.add_node("deny_report", node_deny_report)

    # Entry point
    builder.set_entry_point("analyzer")

    # Conditional edge: analyzer → (hitl_gate | END)
    builder.add_conditional_edges("analyzer", route_after_analysis)

    # Conditional edge: hitl_gate → (action | deny_report)
    builder.add_conditional_edges("hitl_gate", route_after_hitl)

    # Terminal edges
    builder.add_edge("action",      END)
    builder.add_edge("deny_report", END)

    return builder.compile(checkpointer=memory)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "thread_id":    str(uuid.uuid4()),
        "app_status":   "idle",       # idle | running | waiting | done
        "vision_input": "",
        "analysis":     "",
        "threat_found": False,
        "final_report": "",
        "action_taken": False,
        "error":        "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    keys_to_clear = [
        "thread_id", "app_status", "vision_input",
        "analysis", "threat_found", "final_report", "action_taken", "error"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    init_session()


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH EXECUTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def start_analysis(graph, vision_string: str):
    """Invoke the graph from the beginning."""
    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    initial_state: TacticalState = {
        "messages":      [],
        "vision_string": vision_string,
        "threat_found":  False,
        "analysis":      "",
        "final_report":  "",
        "action_taken":  False,
    }

    try:
        result = graph.invoke(initial_state, config=thread_config)

        # Check if graph is paused at the interrupt
        graph_state = graph.get_state(thread_config)

        if graph_state.next:                      # graph is paused → HITL waiting
            st.session_state.app_status   = "waiting"
            st.session_state.threat_found = True
            st.session_state.analysis     = result.get("analysis", "")
        else:                                      # no threat → finished cleanly
            st.session_state.app_status   = "done"
            st.session_state.threat_found = False
            st.session_state.analysis     = result.get("analysis", "")

    except Exception as exc:
        st.session_state.error      = str(exc)
        st.session_state.app_status = "idle"


def resume_graph(graph, decision: str):
    """Resume from the interrupt with the operator's APPROVE or DENY."""
    from langgraph.types import Command

    thread_config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        result = graph.invoke(
            Command(resume=decision),
            config=thread_config,
        )
        st.session_state.app_status   = "done"
        st.session_state.final_report = result.get("final_report", "")
        st.session_state.action_taken = result.get("action_taken", False)

    except Exception as exc:
        st.session_state.error      = str(exc)
        st.session_state.app_status = "done"


# ─────────────────────────────────────────────────────────────────────────────
# UI RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class="tde-header">
        <h1>⬡ Tactical Decision Engine</h1>
        <p class="sub">HITL · LangGraph · Ollama llama3.2:3b</p>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge():
    status = st.session_state.app_status
    label_map = {
        "idle":    ("STANDBY",         "status-idle"),
        "running": ("ANALYZING…",      "status-running"),
        "waiting": ("⚠ AWAITING INPUT","status-waiting"),
        "done":    ("COMPLETE",         "status-done"),
    }
    label, cls = label_map.get(status, ("UNKNOWN", "status-idle"))
    st.markdown(
        f'<span class="status-badge {cls}">{label}</span>',
        unsafe_allow_html=True
    )


def render_input_panel():
    st.markdown('<div class="tde-card-title">VISION STRING INPUT</div>', unsafe_allow_html=True)

    vision = st.text_area(
        label="Vision String",
        label_visibility="collapsed",
        placeholder=(
            "Paste drone vision output here…\n"
            "e.g. 'Two individuals are seen carrying automatic rifles near a fuel depot.'"
        ),
        height=110,
        key="vision_text_area",
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        analyze_clicked = st.button("⬡  ANALYZE THREAT", use_container_width=True)
    with col2:
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        reset_clicked = st.button("↺  RESET", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    return vision.strip(), analyze_clicked, reset_clicked


def render_analysis_card():
    if not st.session_state.analysis:
        return

    threat = st.session_state.threat_found
    threat_html = (
        '<span class="threat-yes">■ THREAT DETECTED</span>'
        if threat else
        '<span class="threat-no">■ NO THREAT — CLEAR</span>'
    )

    st.markdown(f"""
    <div class="tde-card">
        <div class="tde-card-title">ANALYZER ASSESSMENT</div>
        {threat_html}
        <div style="margin-top:0.6rem;" class="analysis-result">
            {st.session_state.analysis}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_hitl_panel(graph):
    if st.session_state.app_status != "waiting":
        return

    st.markdown('<hr class="tde-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div class="tde-card" style="border-color:#e3b341;">
        <div class="tde-card-title" style="color:#e3b341;">⚠ HUMAN AUTHORIZATION REQUIRED</div>
        <p style="font-size:0.95rem; color:#c9d1d9; margin:0.3rem 0 1rem 0;">
            A potential threat has been flagged. Review the analysis above and select your response.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="approve-btn">', unsafe_allow_html=True)
        approved = st.button("✔  APPROVE ACTION", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="deny-btn">', unsafe_allow_html=True)
        denied = st.button("✖  DENY  ACTION", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if approved:
        with st.spinner("Executing action…"):
            resume_graph(graph, "APPROVE")
        st.rerun()

    if denied:
        with st.spinner("Generating tactical report…"):
            resume_graph(graph, "DENY")
        st.rerun()


def render_output_panel():
    status = st.session_state.app_status
    if status != "done":
        return

    st.markdown('<hr class="tde-divider">', unsafe_allow_html=True)

    # ── Path A: action executed ───────────────────────────────────────────
    if st.session_state.action_taken:
        st.markdown("""
        <div class="action-executed">
            ✔ ACTION EXECUTED<br>
            <span style="color:#8b949e;font-size:0.75rem;">
            Engagement authorized and confirmed at operator level.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Path B: tactical report ───────────────────────────────────────────
    elif st.session_state.final_report:
        st.markdown("""
        <div class="tde-card-title" style="margin-bottom:0.5rem;">
            TACTICAL ASSESSMENT REPORT
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            f'<div class="tde-report">{st.session_state.final_report}</div>',
            unsafe_allow_html=True
        )

    # ── Path C: no threat → no action needed ──────────────────────────────
    elif not st.session_state.threat_found:
        st.markdown("""
        <div class="tde-card">
            <span style="color:#39d353;font-family:'Share Tech Mono',monospace;font-size:0.85rem;">
            ■ SITUATION CLEAR — No threat elements detected. No action required.
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_error():
    if st.session_state.error:
        st.error(f"**Graph error:** {st.session_state.error}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    init_session()
    graph = build_graph()

    render_header()

    left, right = st.columns([1, 1], gap="large")

    with left:
        render_status_badge()
        st.markdown("<br>", unsafe_allow_html=True)
        vision, analyze_clicked, reset_clicked = render_input_panel()

        if reset_clicked:
            reset_session()
            st.rerun()

        if analyze_clicked:
            if not vision:
                st.warning("Paste a vision string before analyzing.")
            elif st.session_state.app_status in ("running", "waiting"):
                st.warning("Analysis already in progress.")
            else:
                st.session_state.vision_input = vision
                st.session_state.app_status   = "running"
                # Reset per-run fields
                st.session_state.analysis     = ""
                st.session_state.threat_found = False
                st.session_state.final_report = ""
                st.session_state.action_taken = False
                st.session_state.error        = ""
                st.session_state.thread_id    = str(uuid.uuid4())
                with st.spinner("Running threat analysis…"):
                    start_analysis(graph, vision)
                st.rerun()

    with right:
        render_analysis_card()
        render_hitl_panel(graph)
        render_output_panel()
        render_error()


if __name__ == "__main__":
    main()