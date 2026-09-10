"""
app.py — LearnMate Streamlit UI

Layout:
  Left sidebar  → student management (new / returning student)
  Main panel    → two-tab layout:
                    Tab 1: Chat interface (assessment, Q&A, feedback)
                    Tab 2: Roadmap viewer (week-by-week plan + progress tracker)
"""

import io
import time

import streamlit as st
from fpdf import FPDF

import memory
import agent
import rag_utils

# ---------------------------------------------------------------------------
# Mock-mode indicator (read from agent so the flag lives in one place)
# ---------------------------------------------------------------------------
MOCK_MODE: bool = agent.USE_MOCK

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="LearnMate — AI Learning Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* Chat bubbles */
    .chat-user {
        background: #1d4ed8;
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 10px 16px;
        margin: 6px 0 6px 20%;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 0.95rem;
    }
    .chat-assistant {
        background: #f1f5f9;
        color: #1e293b;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 16px;
        margin: 6px 20% 6px 0;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
        font-size: 0.95rem;
    }
    .chat-wrapper { overflow: hidden; width: 100%; margin-bottom: 4px; }

    /* Agent step pipeline */
    .agent-step {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-left: 4px solid #0ea5e9;
        border-radius: 6px;
        padding: 8px 14px;
        margin: 4px 0;
        font-size: 0.9rem;
        color: #0c4a6e;
    }
    .agent-step.done {
        background: #f0fdf4;
        border-color: #bbf7d0;
        border-left-color: #22c55e;
        color: #14532d;
    }

    /* Week card */
    .week-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .week-card.done { border-left-color: #22c55e; background: #f0fdf4; }
    .week-card h4 { margin: 0 0 8px 0; color: #1e40af; }
    .week-card.done h4 { color: #15803d; }
    .topic-chip {
        display: inline-block;
        background: #dbeafe;
        color: #1e40af;
        border-radius: 12px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.82rem;
    }
    .topic-chip.done {
        background: #dcfce7;
        color: #166534;
        text-decoration: line-through;
    }

    /* Sidebar refinements */
    .sidebar-section { font-size: 0.8rem; color: #64748b; margin-top: -8px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* Scrollable chat area */
    .chat-scroll { max-height: 520px; overflow-y: auto; padding: 8px 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

DEFAULTS = {
    "student_id": None,
    "student_name": None,
    "domain": None,
    "phase": "onboarding",   # onboarding | assessing | learning
    "chat_history": [],      # list of {"role": "user"|"assistant", "content": str}
    "assessment_history": [], # internal assessment conversation
    "roadmap": None,
    "completed_topics": [],
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Ensure RAG index is built
rag_utils.build_index()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

AVAILABLE_DOMAINS = rag_utils.get_available_domains()


def _push_chat(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def _render_chat(history: list[dict]) -> None:
    """Render the chat history as styled bubbles."""
    for msg in history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-wrapper"><div class="chat-user">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-wrapper"><div class="chat-assistant">{msg["content"]}</div></div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Agentic step sequencer (feature 1)
# ---------------------------------------------------------------------------

_AGENT_STEPS = [
    (
        "🔍 Assessment Agent",
        "Analysing your skill-level responses and determining your learning profile…",
        0.6,
    ),
    (
        "📚 Retrieval Agent",
        "Querying the knowledge base for the most relevant {domain} topics at {level} level…",
        0.8,
    ),
    (
        "🗺️ Planning Agent",
        "Building your personalised 8-week roadmap and sequencing topics for optimal learning…",
        0.7,
    ),
    (
        "✅ Roadmap ready",
        "All agents completed. Your personalised roadmap is ready!",
        0.0,
    ),
]


def _run_agentic_steps(student_id: str) -> dict:
    """
    Display a sequential agentic-pipeline UI (named step labels + spinners),
    then call agent.generate_roadmap().  Returns the generated roadmap dict.

    The step labels are shown one at a time with a brief pause so the viewer
    can clearly see the pipeline progressing.
    """
    domain = st.session_state.domain
    state  = memory.get_student_state(student_id)
    level  = state["profile"]["skill_level"] if state else "beginner"

    step_placeholder = st.empty()
    roadmap = None

    for i, (label, description, pause) in enumerate(_AGENT_STEPS):
        desc_rendered = description.format(domain=domain, level=level)
        is_last = i == len(_AGENT_STEPS) - 1

        if is_last:
            # Final "done" state — no spinner needed
            step_placeholder.markdown(
                f'<div class="agent-step done">✅ <strong>{label}</strong> — {desc_rendered}</div>',
                unsafe_allow_html=True,
            )
            break

        # Show the active step with a Streamlit spinner overlaid
        with step_placeholder.container():
            st.markdown(
                f'<div class="agent-step">⚙️ <strong>{label}</strong> — {desc_rendered}</div>',
                unsafe_allow_html=True,
            )

        if MOCK_MODE:
            # In mock mode the actual call is instant; add a small cosmetic pause
            time.sleep(pause * 0.8)
        else:
            time.sleep(0.1)

        # On the Planning Agent step (index 2) trigger the actual generation
        if i == 1:
            with step_placeholder.container():
                st.markdown(
                    f'<div class="agent-step">⚙️ <strong>{_AGENT_STEPS[2][0]}</strong> — '
                    f'{_AGENT_STEPS[2][1].format(domain=domain, level=level)}</div>',
                    unsafe_allow_html=True,
                )
            with st.spinner("Planning Agent working…"):
                roadmap = agent.generate_roadmap(student_id)
            if MOCK_MODE:
                time.sleep(_AGENT_STEPS[2][2] * 0.8)

    # Clear intermediate placeholder — the chat messages will carry the result
    step_placeholder.empty()
    return roadmap


# ---------------------------------------------------------------------------
# PDF generation (feature 4)
# ---------------------------------------------------------------------------

# Map of Unicode characters that appear naturally in roadmap content to their
# plain-ASCII equivalents.  Applied to every string before it enters the PDF
# so Helvetica's latin-1 encoding never sees an out-of-range codepoint.
_PDF_CHAR_MAP = str.maketrans({
    "\u2014": "-",   # em dash  —  → -
    "\u2013": "-",   # en dash  –  → -
    "\u2018": "'",   # left single quote  '
    "\u2019": "'",   # right single quote / apostrophe  '
    "\u201c": '"',   # left double quote  "
    "\u201d": '"',   # right double quote  "
    "\u2022": "*",   # bullet  •  → *
    "\u2713": "[x]", # check mark  ✓  → [x]
    "\u2714": "[x]", # heavy check mark  ✔
    "\u2026": "...", # ellipsis  …
    "\u00e9": "e",   # é  (appears in e.g. "résumé")
    "\u00e0": "a",   # à
    "\u00fc": "u",   # ü
    "\u00f6": "o",   # ö
    "\u00e4": "a",   # ä
    "\u00e8": "e",   # è
    "\u00ea": "e",   # ê
    "\u00e2": "a",   # â
    "\u00ee": "i",   # î
    "\u00f4": "o",   # ô
    "\u00fb": "u",   # û
    "\u2665": "<3",  # heart ♥
    "\u00b7": ".",   # middle dot ·
    "\u00b0": " deg",# degree sign °
    "\u00d7": "x",   # multiplication sign ×
    "\u00f7": "/",   # division sign ÷
})


def _pdf_safe(text: str) -> str:
    """
    Convert a Unicode string to a Helvetica/latin-1 safe string by:
      1. Substituting known special characters with ASCII equivalents.
      2. Dropping any remaining non-latin-1 codepoints (emoji, CJK, etc.)
         rather than letting fpdf2 raise FPDFUnicodeEncodingException.
    """
    translated = str(text).translate(_PDF_CHAR_MAP)
    return translated.encode("latin-1", errors="ignore").decode("latin-1")


def _generate_pdf(roadmap: dict, completed_topics: list[str]) -> bytes:
    """
    Render the roadmap as a clean PDF and return the raw bytes.
    Uses fpdf2 (pure-Python, no system dependencies).
    All text is passed through _pdf_safe() before being added so that
    Helvetica's latin-1 encoding never encounters an out-of-range character.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title block ──────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    title = _pdf_safe(roadmap.get("roadmap_title", "Learning Roadmap"))
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    name   = _pdf_safe(roadmap.get("student_name", ""))
    domain = _pdf_safe(roadmap.get("domain", ""))
    level  = _pdf_safe(roadmap.get("level", "").capitalize())
    if name:
        pdf.cell(0, 7, f"Student: {name}   |   Domain: {domain}   |   Level: {level}",
                 new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # ── Progress summary ─────────────────────────────────────────────────────
    all_topics = [t for w in roadmap.get("weeks", []) for t in w.get("topics", [])]
    n_done  = sum(1 for t in all_topics if t in completed_topics)
    n_total = len(all_topics)
    pct     = int(100 * n_done / n_total) if n_total else 0
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, f"Progress: {n_done}/{n_total} topics completed ({pct}%)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Week cards ───────────────────────────────────────────────────────────
    for week_data in roadmap.get("weeks", []):
        week_num = week_data.get("week", "?")
        theme    = _pdf_safe(week_data.get("theme", f"Week {week_num}"))
        topics   = week_data.get("topics", [])
        task     = _pdf_safe(week_data.get("hands_on_task", ""))
        hours    = week_data.get("estimated_hours", "")

        week_done = all(t in completed_topics for t in topics) and len(topics) > 0
        done_tag  = " [done]" if week_done else ""

        # Week header
        pdf.set_fill_color(219, 234, 254)   # blue-100
        if week_done:
            pdf.set_fill_color(220, 252, 231)  # green-100
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, f"Week {week_num} - {theme}{done_tag}",
                 new_x="LMARGIN", new_y="NEXT", fill=True)

        # Topics
        pdf.set_font("Helvetica", "", 9)
        for t in topics:
            tick    = "[x] " if t in completed_topics else " -  "
            safe_t  = _pdf_safe(t)
            pdf.cell(6)
            pdf.cell(0, 5, f"{tick}{safe_t}", new_x="LMARGIN", new_y="NEXT")

        # Task
        if task:
            pdf.set_font("Helvetica", "I", 9)
            pdf.cell(6)
            pdf.cell(0, 5, f"Task: {task}   (~{hours}h)",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # ── Final project ─────────────────────────────────────────────────────────
    final = _pdf_safe(roadmap.get("final_project", ""))
    if final:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Final Project", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, final)

    # ── Resources note ────────────────────────────────────────────────────────
    note = _pdf_safe(roadmap.get("resources_note", ""))
    if note:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Resources: {note}")

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Roadmap renderer (features 2, 3, 4 integrated here)
# ---------------------------------------------------------------------------

def _render_roadmap(roadmap: dict, completed_topics: list[str]) -> None:
    """Render the week-by-week roadmap with completion indicators."""
    if not roadmap:
        st.info("No roadmap generated yet. Complete the assessment in the Chat tab.")
        return

    st.markdown(f"## 📚 {roadmap.get('roadmap_title', 'Your Learning Roadmap')}")

    # ── Top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Domain", roadmap.get("domain", "—"))
    col2.metric("Level", roadmap.get("level", "—").capitalize())
    col3.metric("Progress", f"{len(completed_topics)} topics done")

    # ── Progress bar (feature 3) ─────────────────────────────────────────────
    all_topics = [t for w in roadmap.get("weeks", []) for t in w.get("topics", [])]
    n_total    = len(all_topics)
    n_done     = sum(1 for t in all_topics if t in completed_topics)
    pct        = n_done / n_total if n_total else 0.0

    st.markdown("**Overall Progress**")
    st.progress(pct, text=f"{n_done} / {n_total} topics completed ({int(pct * 100)}%)")
    st.divider()

    # ── Week cards ───────────────────────────────────────────────────────────
    weeks = roadmap.get("weeks", [])
    for week_data in weeks:
        week_num = week_data.get("week", "?")
        theme    = week_data.get("theme", f"Week {week_num}")
        topics   = week_data.get("topics", [])
        task     = week_data.get("hands_on_task", "")
        hours    = week_data.get("estimated_hours", "")

        week_done  = all(t in completed_topics for t in topics) and len(topics) > 0
        card_class = "week-card done" if week_done else "week-card"

        topic_chips = "".join(
            f'<span class="topic-chip {"done" if t in completed_topics else ""}">{t}</span>'
            for t in topics
        )

        st.markdown(
            f"""
            <div class="{card_class}">
                <h4>Week {week_num} — {theme} {"✅" if week_done else ""}</h4>
                <div style="margin-bottom:8px;">{topic_chips}</div>
                <div style="font-size:0.88rem;color:#475569;">
                    🛠️ <strong>Task:</strong> {task}
                    &nbsp;&nbsp; ⏱️ <em>~{hours}h</em>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    final_project = roadmap.get("final_project", "")
    if final_project:
        st.markdown(f"### 🏆 Final Project\n{final_project}")

    resources_note = roadmap.get("resources_note", "")
    if resources_note:
        st.caption(f"📌 {resources_note}")

    # ── Why this roadmap? expander (feature 2) ────────────────────────────────
    if st.session_state.get("student_id"):
        with st.expander("🧠 Why this roadmap?"):
            reasons = agent.build_roadmap_reasoning(st.session_state.student_id)
            for reason in reasons:
                st.markdown(f"- {reason}")
            revision_note = roadmap.get("_revision_note")
            if revision_note:
                st.info(f"💬 Latest revision: {revision_note}")

    # ── Completion tracking ───────────────────────────────────────────────────
    remaining = [t for t in all_topics if t not in completed_topics]
    if remaining:
        st.divider()
        st.markdown("#### ✅ Mark Topics as Complete")
        selected = st.multiselect(
            "Select topics you've finished:",
            options=remaining,
            key="topic_completions",
        )
        if st.button("Save Progress", type="primary"):
            for topic in selected:
                wk = next(
                    (w["week"] for w in weeks if topic in w.get("topics", [])), 0
                )
                memory.mark_topic_complete(st.session_state.student_id, topic, wk)
            st.session_state.completed_topics = memory.get_completed_topic_names(
                st.session_state.student_id
            )
            st.success(f"Marked {len(selected)} topic(s) complete! 🎉")
            st.rerun()

    # ── Download PDF button (feature 4) ──────────────────────────────────────
    st.divider()
    pdf_bytes = _generate_pdf(roadmap, completed_topics)
    safe_title = roadmap.get("roadmap_title", "roadmap").replace(" ", "_").replace("—", "-")
    st.download_button(
        label="📄 Download Roadmap as PDF",
        data=pdf_bytes,
        file_name=f"{safe_title}.pdf",
        mime="application/pdf",
    )


# ---------------------------------------------------------------------------
# Sidebar — student management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
        width=80,
    )
    st.title("🎓 LearnMate")

    st.caption("Powered by IBM watsonx.ai · Granite")
    st.divider()

    mode = st.radio("👤 Session", ["New Student", "Returning Student"], horizontal=True)

    if mode == "New Student":
        st.subheader("Start Learning")
        new_name = st.text_input("Your name", placeholder="e.g. Alex")
        new_domain = st.selectbox("Domain", AVAILABLE_DOMAINS)
        if st.button("🚀 Start Learning Journey", type="primary"):
            if not new_name.strip():
                st.error("Please enter your name.")
            else:
                # Reset session
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.domain = new_domain
                st.session_state.student_name = new_name.strip()
                st.session_state.phase = "assessing"
                st.session_state.chat_history = []
                st.session_state.assessment_history = []

                # Greeting
                greeting = (
                    f"👋 Hi **{new_name.strip()}**! Welcome to LearnMate.\n\n"
                    f"I'm your AI learning coach, here to build you a personalised "
                    f"**{new_domain}** roadmap.\n\n"
                    f"First, let me assess your current level with a few quick questions."
                )
                _push_chat("assistant", greeting)

                # First assessment question
                result = agent.run_assessment_turn(
                    new_domain, st.session_state.assessment_history
                )
                _push_chat("assistant", result["message"])
                st.rerun()

    else:  # Returning Student
        st.subheader("Continue Learning")
        students = memory.list_students()
        if not students:
            st.info("No students found. Create a new student first.")
        else:
            options = {f"{s['name']} ({s['domain']})": s["student_id"] for s in students}
            chosen_label = st.selectbox("Select student", list(options.keys()))
            if st.button("📂 Load Profile", type="primary"):
                sid = options[chosen_label]
                state = memory.get_student_state(sid)
                if state:
                    for k, v in DEFAULTS.items():
                        st.session_state[k] = v
                    st.session_state.student_id = sid
                    st.session_state.student_name = state["profile"]["name"]
                    st.session_state.domain = state["profile"]["domain"]
                    st.session_state.roadmap = state["roadmap"]
                    st.session_state.completed_topics = state["completed_topics"]
                    st.session_state.phase = "learning"

                    welcome_back = (
                        f"👋 Welcome back, **{state['profile']['name']}**!\n\n"
                        f"You're continuing your **{state['profile']['domain']}** "
                        f"journey at **{state['profile']['skill_level']}** level.\n\n"
                        f"You can:\n"
                        f"- Ask me questions about your topics\n"
                        f"- Give feedback on your roadmap (e.g. *'this is too easy'*)\n"
                        f"- Check your roadmap in the **Roadmap** tab"
                    )
                    _push_chat("assistant", welcome_back)
                    st.rerun()

    # Current session info
    if st.session_state.student_id:
        st.divider()
        st.markdown("**Current Session**")
        st.markdown(f"👤 {st.session_state.student_name}")
        st.markdown(f"📘 {st.session_state.domain}")
        completed = st.session_state.get("completed_topics", [])
        st.markdown(f"✅ {len(completed)} topics done")
        if st.button("🔄 New Session"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

st.title("🎓 LearnMate — Your AI Learning Coach")

if st.session_state.phase == "onboarding":
    # Welcome screen
    st.markdown(
        """
        ### Welcome to LearnMate!
        An intelligent, adaptive learning coach powered by **IBM watsonx.ai** and **Granite**.

        **How it works:**
        1. 👤 Enter your name and choose a learning domain in the sidebar
        2. 🧠 The AI assesses your current skill level through a short conversation
        3. 📋 Receive a personalised week-by-week learning roadmap grounded in real course content
        4. 💬 Chat with your coach, give feedback, and watch your roadmap adapt in real time
        5. ✅ Track your progress as you complete topics

        **Available Domains**
        """
    )
    cols = st.columns(len(AVAILABLE_DOMAINS))
    for i, domain in enumerate(AVAILABLE_DOMAINS):
        desc = rag_utils.get_domain_description(domain)
        with cols[i]:
            st.markdown(
                f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                            padding:14px;text-align:center;min-height:140px;">
                    <strong>{domain}</strong><br/>
                    <span style="font-size:0.82rem;color:#64748b;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.info("👈 Use the sidebar to start your learning journey!")

else:
    tab_chat, tab_roadmap = st.tabs(["💬 Chat", "📋 Roadmap"])

    # -----------------------------------------------------------------------
    # TAB 1: Chat
    # -----------------------------------------------------------------------
    with tab_chat:
        chat_container = st.container()
        with chat_container:
            _render_chat(st.session_state.chat_history)

        st.markdown("---")

        # Feedback shortcut buttons (only in learning phase)
        if st.session_state.phase == "learning" and st.session_state.roadmap:
            st.markdown("**Quick Feedback:**")
            fb_cols = st.columns(4)
            feedback_options = [
                ("😅 Too Easy", "The content feels too easy, I want to go faster"),
                ("😰 Too Hard", "The content is too difficult, please slow down"),
                ("⏩ Skip Ahead", "I want to skip ahead and learn more advanced topics"),
                ("🛠️ More Practice", "I need more hands-on tasks and projects"),
            ]
            for i, (label, text) in enumerate(feedback_options):
                if fb_cols[i].button(label, key=f"fb_{i}"):
                    _push_chat("user", text)
                    with st.spinner("Revising your roadmap..."):
                        revised = agent.revise_roadmap(st.session_state.student_id, text)
                    st.session_state.roadmap = revised
                    response = (
                        f"✅ I've revised your roadmap based on your feedback: *\"{text}\"*\n\n"
                        f"Switch to the **Roadmap** tab to see the updated plan!"
                    )
                    _push_chat("assistant", response)
                    st.rerun()

        # Input box
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Message",
                placeholder="Type a message, question, or feedback...",
                label_visibility="collapsed",
            )
            send = st.form_submit_button("Send ➤", type="primary")

        if send and user_input.strip():
            user_msg = user_input.strip()
            _push_chat("user", user_msg)

            # ── Assessment phase ──────────────────────────────────────────
            if st.session_state.phase == "assessing":
                with st.spinner("Thinking..."):
                    result = agent.run_assessment_turn(
                        st.session_state.domain,
                        st.session_state.assessment_history,
                        student_answer=user_msg,
                    )

                _push_chat("assistant", result["message"])

                if result["complete"]:
                    skill_level = result["skill_level"]

                    # Create student in DB
                    sid = memory.create_student(
                        name=st.session_state.student_name,
                        domain=st.session_state.domain,
                        skill_level=skill_level,
                    )
                    st.session_state.student_id = sid

                    # ── Agentic step pipeline (feature 1) ─────────────────
                    _push_chat(
                        "assistant",
                        "🤖 **Assessment complete!** Spinning up the multi-agent pipeline to build your roadmap…",
                    )
                    st.rerun()  # flush the message before showing steps

                st.rerun()

            # ── Learning phase ────────────────────────────────────────────
            elif st.session_state.phase == "learning":
                # Detect feedback intent
                feedback_keywords = [
                    "too easy", "too hard", "too fast", "too slow",
                    "skip", "faster", "slower", "more practice",
                    "go back", "change roadmap", "revise", "update",
                    "boring", "difficult", "challenging", "struggle",
                ]
                is_feedback = any(kw in user_msg.lower() for kw in feedback_keywords)

                if is_feedback:
                    with st.spinner("Revising your roadmap..."):
                        revised = agent.revise_roadmap(
                            st.session_state.student_id, user_msg
                        )
                    st.session_state.roadmap = revised
                    memory.save_feedback(st.session_state.student_id, user_msg)
                    response = (
                        f"✅ Got your feedback! I've revised your roadmap accordingly.\n\n"
                        f"Switch to the **Roadmap** tab to see the updated plan. "
                        f"Keep going — you're doing great! 💪"
                    )
                else:
                    with st.spinner("Thinking..."):
                        response = agent.answer_question(
                            st.session_state.student_id, user_msg
                        )

                _push_chat("assistant", response)
                st.rerun()

        # ── Trigger agentic pipeline after assessment completes ───────────────
        # This runs on the *next* rerun after assessment is marked complete,
        # so the "pipeline spinning up" message is visible before the steps start.
        if (
            st.session_state.phase == "assessing"
            and st.session_state.student_id
            and st.session_state.chat_history
            and st.session_state.chat_history[-1]["content"].startswith("🤖 **Assessment complete!**")
        ):
            roadmap = _run_agentic_steps(st.session_state.student_id)
            st.session_state.roadmap = roadmap
            st.session_state.phase = "learning"

            roadmap_ready_msg = (
                f"🎉 Your personalised **{st.session_state.domain}** roadmap is ready!\n\n"
                f"You're starting at the **{memory.get_student(st.session_state.student_id)['skill_level']}** "
                f"level with an 8-week plan tailored specifically for you.\n\n"
                f"👉 Click the **Roadmap** tab above to see your full week-by-week plan!\n\n"
                f"From here you can:\n"
                f"- Ask me anything about your topics\n"
                f"- Give feedback to adapt the roadmap\n"
                f"- Track completed topics in the Roadmap tab"
            )
            _push_chat("assistant", roadmap_ready_msg)
            st.rerun()

    # -----------------------------------------------------------------------
    # TAB 2: Roadmap
    # -----------------------------------------------------------------------
    with tab_roadmap:
        if st.session_state.phase == "assessing":
            st.info(
                "🧠 Complete the skill assessment in the **Chat** tab first. "
                "Your personalised roadmap will appear here once it's generated."
            )
        else:
            # Refresh completed topics from DB
            if st.session_state.student_id:
                st.session_state.completed_topics = memory.get_completed_topic_names(
                    st.session_state.student_id
                )

            _render_roadmap(
                st.session_state.roadmap,
                st.session_state.completed_topics,
            )

            # Feedback section in roadmap tab
            if st.session_state.roadmap and st.session_state.student_id:
                st.divider()
                with st.expander("💬 Give Feedback on This Roadmap"):
                    with st.form("feedback_form"):
                        fb_text = st.text_area(
                            "Share your feedback",
                            placeholder=(
                                'e.g. "This week was too easy, please add harder content" '
                                'or "I want to focus more on hands-on projects"'
                            ),
                        )
                        submit_fb = st.form_submit_button("Submit Feedback", type="primary")
                    if submit_fb and fb_text.strip():
                        with st.spinner("Revising roadmap based on your feedback..."):
                            revised = agent.revise_roadmap(
                                st.session_state.student_id, fb_text.strip()
                            )
                        st.session_state.roadmap = revised
                        _push_chat("user", fb_text.strip())
                        _push_chat(
                            "assistant",
                            "✅ Roadmap revised based on your feedback. "
                            "Check the updated plan above!",
                        )
                        st.success("Roadmap updated! ✅")
                        st.rerun()

                # Feedback history
                if st.session_state.student_id:
                    history = memory.get_feedback_history(st.session_state.student_id)
                    if history:
                        with st.expander("📜 Feedback History"):
                            for entry in history:
                                st.markdown(
                                    f"- **{entry['timestamp']}** — {entry['feedback']}"
                                )
