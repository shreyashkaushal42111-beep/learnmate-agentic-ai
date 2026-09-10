"""
agent.py — Core agentic logic for LearnMate.

Orchestrates:
  1. Onboarding conversation (collects domain + name)
  2. Skill assessment (2-3 adaptive questions → determines beginner/intermediate/advanced)
  3. RAG-grounded roadmap generation via IBM watsonx.ai Granite
  4. Feedback-driven roadmap revision (core agentic loop)
  5. Progress tracking via memory.py

Mock mode
---------
Set the environment variable USE_MOCK=true (case-insensitive) to run the
full agentic flow with pre-written realistic responses instead of calling
the real watsonx.ai API.  The real API code is preserved intact below and
is used automatically when USE_MOCK is false or unset.
"""

import os
import json
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Load .env file FIRST so USE_MOCK (and credentials) are available via
# os.getenv() before anything else runs.  python-dotenv is a no-op when the
# file doesn't exist, so this is safe in all environments.
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)   # don't clobber real env vars already set
except ImportError:
    pass  # dotenv not installed — rely on the shell environment

# ---------------------------------------------------------------------------
# Mock-mode flag — evaluated AFTER load_dotenv() so .env values are visible
# ---------------------------------------------------------------------------

USE_MOCK: bool = os.getenv("USE_MOCK", "false").strip().lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# watsonx.ai imports — skipped in mock mode to avoid missing-credentials
# errors at startup when the API is not yet configured.
# ---------------------------------------------------------------------------

if not USE_MOCK:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

import memory
import rag_utils
import mock_responses

# ---------------------------------------------------------------------------
# watsonx.ai client configuration  (real API — unchanged)
# ---------------------------------------------------------------------------

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm/granite-4-h-small")

# Chat parameters passed per-call (max_tokens is the correct key for the
# chat endpoint; GenParams keys only apply to the text-generation endpoint).
_CHAT_PARAMS = {
    "max_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.95,
}


def _get_model():  # -> ModelInference (only valid when USE_MOCK=false)
    if USE_MOCK:
        raise RuntimeError(
            "_get_model() must not be called when USE_MOCK=true. "
            "All mock-mode code paths should have returned before reaching this function."
        )
    credentials = Credentials(
        api_key=WATSONX_API_KEY,
        url=WATSONX_URL,
    )
    return ModelInference(
        model_id=GRANITE_MODEL_ID,
        credentials=credentials,
        project_id=WATSONX_PROJECT_ID,
    )


def _chat(system_prompt: str, user_message: str) -> str:
    """Send a single-turn message to Granite and return the response text."""
    if USE_MOCK:
        raise RuntimeError(
            "_chat() must not be called when USE_MOCK=true. "
            "All mock-mode code paths should have returned before reaching this function."
        )
    model = _get_model()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    response = model.chat(messages=messages, params=_CHAT_PARAMS)
    return response["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Assessment questions bank (used when LLM is not available)
# ---------------------------------------------------------------------------

ASSESSMENT_QUESTIONS: dict[str, list[dict]] = {
    "Frontend Development": [
        {
            "question": "Have you built a complete webpage using HTML and CSS before?",
            "beginner_signals": ["no", "never", "just started", "learning"],
            "advanced_signals": ["yes, many", "regularly", "professionally"],
        },
        {
            "question": "Are you comfortable writing JavaScript functions, loops, and working with the DOM?",
            "beginner_signals": ["no", "not really", "a little"],
            "advanced_signals": ["yes", "very comfortable", "daily"],
        },
        {
            "question": "Have you used any JavaScript framework (React, Vue, Angular) in a real project?",
            "beginner_signals": ["no", "never", "heard of it"],
            "advanced_signals": ["yes", "react", "vue", "angular", "next"],
        },
    ],
    "Cybersecurity": [
        {
            "question": "Are you familiar with basic networking concepts like IP addresses, DNS, and TCP/IP?",
            "beginner_signals": ["no", "not really", "vaguely"],
            "advanced_signals": ["yes", "very familiar", "studied networking"],
        },
        {
            "question": "Have you ever used tools like Nmap, Wireshark, or Metasploit?",
            "beginner_signals": ["no", "never", "don't know them"],
            "advanced_signals": ["yes", "nmap", "wireshark", "metasploit", "burp"],
        },
        {
            "question": "Have you completed any CTF challenges or formal security training/certification?",
            "beginner_signals": ["no", "none"],
            "advanced_signals": ["yes", "ctf", "ceh", "oscp", "cissp", "security+"],
        },
    ],
    "UI/UX Design": [
        {
            "question": "Have you designed user interfaces using any design tool (Figma, Sketch, Adobe XD)?",
            "beginner_signals": ["no", "never", "not yet"],
            "advanced_signals": ["yes", "figma", "sketch", "xd", "adobe"],
        },
        {
            "question": "Are you familiar with concepts like user personas, wireframes, or design systems?",
            "beginner_signals": ["no", "not sure", "heard of it"],
            "advanced_signals": ["yes", "regularly use", "design system"],
        },
        {
            "question": "Have you conducted user research or usability testing in a project?",
            "beginner_signals": ["no", "never"],
            "advanced_signals": ["yes", "user testing", "usability", "research"],
        },
    ],
    "Data Science": [
        {
            "question": "Are you comfortable with Python programming and using libraries like NumPy or Pandas?",
            "beginner_signals": ["no", "learning python", "basic python"],
            "advanced_signals": ["yes", "comfortable", "numpy", "pandas", "daily"],
        },
        {
            "question": "Have you built and evaluated a machine learning model before?",
            "beginner_signals": ["no", "never", "not yet"],
            "advanced_signals": ["yes", "sklearn", "classification", "regression"],
        },
        {
            "question": "Have you worked with deep learning frameworks (TensorFlow, PyTorch) or deployed ML models?",
            "beginner_signals": ["no", "never"],
            "advanced_signals": ["yes", "tensorflow", "pytorch", "deployment", "mlops"],
        },
    ],
    "Cloud Computing": [
        {
            "question": "Have you used any cloud platform (AWS, Azure, GCP) to deploy or manage resources?",
            "beginner_signals": ["no", "never", "just exploring"],
            "advanced_signals": ["yes", "aws", "azure", "gcp", "ec2", "s3"],
        },
        {
            "question": "Are you comfortable with Docker containers and writing Dockerfiles?",
            "beginner_signals": ["no", "not yet", "heard of docker"],
            "advanced_signals": ["yes", "docker", "containers", "compose"],
        },
        {
            "question": "Have you worked with Kubernetes, Terraform, or CI/CD pipelines on cloud?",
            "beginner_signals": ["no", "never"],
            "advanced_signals": ["yes", "kubernetes", "k8s", "terraform", "cicd"],
        },
    ],
}

# Default questions for domains not in the bank
_DEFAULT_QUESTIONS = [
    "How would you describe your current experience level with this subject? (none / some / substantial)",
    "Have you completed any formal courses or projects in this area before?",
    "Are you comfortable reading technical documentation and learning independently?",
]


# ---------------------------------------------------------------------------
# Skill level determination
# ---------------------------------------------------------------------------

def _score_answers(domain: str, answers: list[str]) -> str:
    """
    Simple heuristic: tally beginner vs advanced signals across answers.
    Returns 'beginner', 'intermediate', or 'advanced'.
    """
    questions = ASSESSMENT_QUESTIONS.get(domain, [])
    if not questions:
        # Fallback: look for level keywords in all answers combined
        combined = " ".join(answers).lower()
        if any(w in combined for w in ["advanced", "expert", "senior", "professional"]):
            return "advanced"
        if any(w in combined for w in ["intermediate", "some experience", "familiar"]):
            return "intermediate"
        return "beginner"

    beginner_score = 0
    advanced_score = 0

    for i, q_data in enumerate(questions):
        if i >= len(answers):
            break
        ans = answers[i].lower()
        if any(sig in ans for sig in q_data["beginner_signals"]):
            beginner_score += 1
        if any(sig in ans for sig in q_data["advanced_signals"]):
            advanced_score += 1

    if advanced_score >= 2:
        return "advanced"
    elif advanced_score == 1 or beginner_score <= 1:
        return "intermediate"
    return "beginner"


# ---------------------------------------------------------------------------
# Roadmap generation
# ---------------------------------------------------------------------------

ROADMAP_SYSTEM_PROMPT = """You are LearnMate, an expert learning coach AI. Your job is to create
personalized, structured week-by-week learning roadmaps based on student profiles and course content.

Guidelines:
- Create a clear, motivating 8-week roadmap
- Each week has a theme, 3-5 specific topics, and one hands-on task
- Adapt complexity to the student's skill level
- Reference the provided course topics naturally — do not invent topics not present in the context
- Format the output as valid JSON matching the schema provided
- Be encouraging and specific, not generic
"""

ROADMAP_SCHEMA = """
{
  "roadmap_title": "string",
  "student_name": "string",
  "domain": "string",
  "level": "string",
  "total_weeks": 8,
  "weeks": [
    {
      "week": 1,
      "theme": "string",
      "topics": ["topic1", "topic2", "topic3"],
      "hands_on_task": "string",
      "estimated_hours": 5
    }
  ],
  "final_project": "string",
  "resources_note": "string"
}
"""


def generate_roadmap(student_id: str) -> dict:
    """
    Generate a personalised 8-week roadmap for the student.

    In mock mode: returns a pre-written realistic roadmap (or a RAG-grounded
    dynamically built one) without calling the Granite API.
    In real mode: uses RAG context from ChromaDB + Granite LLM for reasoning.
    Persists the result via memory.py.
    """
    # ── Mock path ────────────────────────────────────────────────────────────
    if USE_MOCK:
        return mock_responses.mock_generate_roadmap(student_id, _build_fallback_roadmap)

    # ── Real API path (unchanged) ────────────────────────────────────────────
    state = memory.get_student_state(student_id)
    if state is None:
        raise ValueError(f"Student {student_id} not found")

    profile = state["profile"]
    domain = profile["domain"]
    level = profile["skill_level"]
    name = profile["name"]

    # RAG retrieval
    rag_context = rag_utils.retrieve_topics_for_roadmap(domain, level, num_weeks=8)
    topics_text = "\n".join(f"- {t}" for t in rag_context["topics"])
    projects_text = "\n".join(f"- {p}" for p in rag_context["project_suggestions"])

    user_message = f"""
Create a personalised 8-week learning roadmap for the following student.

STUDENT PROFILE:
- Name: {name}
- Domain: {domain}
- Skill Level: {level}

AVAILABLE COURSE TOPICS (use these as the basis for the roadmap):
{topics_text}

SUGGESTED PROJECTS:
{projects_text}

OUTPUT FORMAT (return ONLY valid JSON, no markdown fences, no extra text):
{ROADMAP_SCHEMA}
"""

    raw = _chat(ROADMAP_SYSTEM_PROMPT, user_message)

    # Strip any accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        roadmap = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fallback: build a basic roadmap from RAG topics without LLM
        roadmap = _build_fallback_roadmap(name, domain, level, rag_context)

    memory.save_roadmap(student_id, roadmap)
    return roadmap


def _build_fallback_roadmap(
    name: str, domain: str, level: str, rag_context: dict
) -> dict:
    """Construct a simple roadmap from RAG topics when LLM JSON parsing fails."""
    weekly_chunks = rag_context["weekly_chunks"]
    projects = rag_context["project_suggestions"]
    weeks = []
    for i, chunk in enumerate(weekly_chunks[:8], start=1):
        weeks.append(
            {
                "week": i,
                "theme": f"Week {i}: {chunk[0].split(':')[0] if chunk else 'Core Concepts'}",
                "topics": chunk[:5],
                "hands_on_task": projects[i % len(projects)] if projects else f"Practice Week {i} topics",
                "estimated_hours": 5,
            }
        )
    return {
        "roadmap_title": f"{domain} Learning Path — {level.capitalize()}",
        "student_name": name,
        "domain": domain,
        "level": level,
        "total_weeks": len(weeks),
        "weeks": weeks,
        "final_project": projects[0] if projects else "Build a capstone project",
        "resources_note": "Use free resources: MDN, freeCodeCamp, YouTube, official docs.",
    }


# ---------------------------------------------------------------------------
# Roadmap reasoning  (explains *why* the roadmap was shaped this way)
# ---------------------------------------------------------------------------

_LEVEL_REASONING: dict[str, str] = {
    "beginner": (
        "Your answers indicated you are new to this domain, so the roadmap starts from "
        "first principles and builds foundational skills progressively before introducing "
        "any intermediate tooling or frameworks."
    ),
    "intermediate": (
        "Your answers showed you already have working knowledge of the basics, so the "
        "roadmap skips introductory material and focuses on consolidating core skills, "
        "adding modern tooling, and tackling real-world projects."
    ),
    "advanced": (
        "Your answers demonstrated strong existing expertise, so the roadmap prioritises "
        "depth over breadth — advanced patterns, performance, architecture, and "
        "industry-grade practices are front-loaded."
    ),
}

_DOMAIN_EMPHASIS: dict[str, str] = {
    "Frontend Development": "JavaScript fluency and component-based architecture",
    "Cybersecurity": "hands-on lab work, CTF practice, and tool proficiency",
    "UI/UX Design": "a design-thinking workflow from research through high-fidelity prototyping",
    "Data Science": "a data-to-model pipeline covering cleaning, EDA, and ML evaluation",
    "Cloud Computing": "infrastructure-as-code, containerisation, and CI/CD automation",
}


def build_roadmap_reasoning(student_id: str) -> list[str]:
    """
    Return a list of bullet-point strings explaining why the roadmap was
    structured the way it was for this student.  Used by the UI's
    "Why this roadmap?" expander.
    """
    state = memory.get_student_state(student_id)
    if state is None:
        return ["Reasoning unavailable — student record not found."]

    profile = state["profile"]
    roadmap = state["roadmap"]
    domain  = profile["domain"]
    level   = profile["skill_level"]
    name    = profile["name"]

    reasons: list[str] = []

    # 1 — skill-level rationale
    reasons.append(_LEVEL_REASONING.get(level, f"Tailored for a {level}-level learner."))

    # 2 — domain emphasis
    emphasis = _DOMAIN_EMPHASIS.get(domain, "domain-specific best practices")
    reasons.append(
        f"For **{domain}**, the curriculum places special emphasis on {emphasis}."
    )

    # 3 — RAG-grounded topic count
    try:
        rag_ctx = rag_utils.retrieve_topics_for_roadmap(domain, level, num_weeks=8)
        n_topics = len(rag_ctx["topics"])
        reasons.append(
            f"**{n_topics} topics** were retrieved from the course knowledge base and "
            f"distributed across 8 weeks so each session stays under 6–8 hours."
        )
    except Exception:
        pass

    # 4 — project grounding
    if roadmap:
        final = roadmap.get("final_project", "")
        if final:
            reasons.append(
                f"The roadmap culminates in a hands-on capstone: *\"{final}\"* — "
                f"chosen because it exercises the full range of {domain} skills at "
                f"the {level} level."
            )

    # 5 — revision history note
    feedback_log = state.get("feedback_history", [])
    if feedback_log:
        reasons.append(
            f"This roadmap has been **revised {len(feedback_log)} time(s)** based on "
            f"your feedback — the content you see reflects those adjustments."
        )

    return reasons


# ---------------------------------------------------------------------------
# Roadmap revision (core agentic behaviour)
# ---------------------------------------------------------------------------

REVISION_SYSTEM_PROMPT = """You are LearnMate, an adaptive learning coach AI. A student has shared
feedback about their current learning roadmap. Your task is to revise the roadmap intelligently.

Rules:
- If feedback indicates the content is too easy → accelerate the pace, remove basic topics, add advanced ones
- If feedback indicates content is too hard → slow down, add foundational material, split complex topics
- If student wants to skip ahead → move them forward in the roadmap
- If student wants more practice → add extra hands-on tasks
- Preserve the overall JSON structure exactly
- Return ONLY valid JSON, no markdown, no extra text
"""


def revise_roadmap(student_id: str, feedback: str) -> dict:
    """
    Revise the existing roadmap based on student feedback.
    This is the core agentic behaviour — the agent dynamically adapts the plan.

    In mock mode: applies keyword-driven revision rules without calling Granite.
    In real mode: sends the full roadmap + feedback to Granite for LLM revision.
    """
    # ── Mock path ────────────────────────────────────────────────────────────
    if USE_MOCK:
        return mock_responses.mock_revise_roadmap(student_id, feedback, _build_fallback_roadmap)

    # ── Real API path (unchanged) ────────────────────────────────────────────
    state = memory.get_student_state(student_id)
    if state is None:
        raise ValueError(f"Student {student_id} not found")

    profile = state["profile"]
    current_roadmap = state["roadmap"]
    completed = state["completed_topics"]
    past_feedback = state["feedback_history"]

    if current_roadmap is None:
        raise ValueError("No roadmap found. Generate a roadmap first.")

    # Save this feedback to history
    memory.save_feedback(student_id, feedback)

    domain = profile["domain"]
    level = profile["skill_level"]

    # Get additional RAG topics in case we need to enrich the roadmap
    rag_context = rag_utils.retrieve_topics_for_roadmap(domain, level, num_weeks=8)
    all_topics = rag_context["topics"]

    user_message = f"""
STUDENT PROFILE:
- Name: {profile['name']}
- Domain: {domain}
- Current Skill Level: {level}

CURRENT ROADMAP:
{json.dumps(current_roadmap, indent=2)}

COMPLETED TOPICS SO FAR:
{json.dumps(completed)}

PAST FEEDBACK:
{json.dumps(past_feedback)}

NEW FEEDBACK FROM STUDENT:
"{feedback}"

AVAILABLE TOPICS FROM KNOWLEDGE BASE (for enrichment if needed):
{json.dumps(all_topics)}

Please revise the roadmap based on the new feedback. Return the complete updated roadmap as JSON.
"""

    raw = _chat(REVISION_SYSTEM_PROMPT, user_message)

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw.strip(), flags=re.MULTILINE)

    try:
        revised_roadmap = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Return the original if revision parsing fails
        revised_roadmap = current_roadmap
        revised_roadmap["_revision_note"] = (
            "Auto-revision failed to parse; roadmap unchanged. Try rephrasing your feedback."
        )

    memory.save_roadmap(student_id, revised_roadmap)
    return revised_roadmap


# ---------------------------------------------------------------------------
# Conversational assessment
# ---------------------------------------------------------------------------

ASSESSMENT_SYSTEM_PROMPT = """You are LearnMate, a friendly AI learning coach. You are assessing
a student's current skill level in their chosen domain by asking them 3 focused questions.

Rules:
- Ask one question at a time, in a friendly and encouraging tone
- Questions should reveal whether the student is a beginner, intermediate, or advanced learner
- Do not ask more than 3 questions total
- After the 3rd answer, output a JSON summary like:
  {"assessment_complete": true, "skill_level": "beginner|intermediate|advanced", "reasoning": "..."}
- Before the 3rd answer is received, just ask the next question naturally — no JSON
"""


def run_assessment_turn(
    domain: str,
    conversation_history: list[dict],
    student_answer: Optional[str] = None,
) -> dict:
    """
    Conduct one turn of the skill assessment conversation.

    Parameters
    ----------
    domain              : learning domain
    conversation_history: list of {"role": "assistant"|"user", "content": str}
    student_answer      : the student's latest message (None for the opening)

    Returns
    -------
    {
        "message": str,           # agent's next message to display
        "complete": bool,         # True when assessment is done
        "skill_level": str|None,  # set when complete
    }

    In mock mode: runs a scripted 3-question exchange with heuristic scoring.
    In real mode: calls the Granite LLM for adaptive question generation.
    """
    # ── Mock path ────────────────────────────────────────────────────────────
    if USE_MOCK:
        return mock_responses.mock_assessment_turn(
            domain,
            conversation_history,
            student_answer,
            ASSESSMENT_QUESTIONS,
            _score_answers,
        )

    # ── Real API path (unchanged) ────────────────────────────────────────────
    if student_answer is not None:
        conversation_history.append({"role": "user", "content": student_answer})

    user_turn_count = sum(1 for m in conversation_history if m["role"] == "user")

    if user_turn_count == 0:
        # Opening: ask first question
        questions = ASSESSMENT_QUESTIONS.get(domain, [])
        first_q = questions[0]["question"] if questions else _DEFAULT_QUESTIONS[0]
        msg = (
            f"Great choice! Let me ask you a few quick questions to tailor your {domain} "
            f"learning path.\n\n**Question 1:** {first_q}"
        )
        conversation_history.append({"role": "assistant", "content": msg})
        return {"message": msg, "complete": False, "skill_level": None}

    # Build messages for LLM with domain context
    system = ASSESSMENT_SYSTEM_PROMPT + f"\n\nThe student has chosen to learn: {domain}."
    messages = [{"role": "system", "content": system}] + conversation_history

    model = _get_model()
    response = model.chat(messages=messages)
    assistant_reply = response["choices"][0]["message"]["content"].strip()

    conversation_history.append({"role": "assistant", "content": assistant_reply})

    # Check for completion JSON in the reply
    json_match = re.search(r"\{[^{}]*\"assessment_complete\"[^{}]*\}", assistant_reply, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if result.get("assessment_complete"):
                skill_level = result.get("skill_level", "beginner")
                # Also run heuristic cross-check
                answers = [m["content"] for m in conversation_history if m["role"] == "user"]
                heuristic_level = _score_answers(domain, answers)
                # Prefer LLM's assessment; use heuristic as fallback
                final_level = skill_level if skill_level in ("beginner", "intermediate", "advanced") else heuristic_level

                clean_message = assistant_reply[: json_match.start()].strip()
                if not clean_message:
                    clean_message = (
                        f"Thanks for your answers! Based on our conversation, "
                        f"I've assessed your level as **{final_level}**. "
                        f"Let me generate your personalised roadmap now! 🚀"
                    )
                return {
                    "message": clean_message,
                    "complete": True,
                    "skill_level": final_level,
                }
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: if we've had 3 user turns and no JSON, determine level heuristically
    if user_turn_count >= 3:
        answers = [m["content"] for m in conversation_history if m["role"] == "user"]
        skill_level = _score_answers(domain, answers)
        return {
            "message": (
                f"Thanks! Based on your responses, I've placed you at the "
                f"**{skill_level}** level. Generating your roadmap now... 🚀"
            ),
            "complete": True,
            "skill_level": skill_level,
        }

    return {"message": assistant_reply, "complete": False, "skill_level": None}


# ---------------------------------------------------------------------------
# Chat Q&A (general questions about the roadmap / topics)
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = """You are LearnMate, a helpful and encouraging AI learning coach.
You help students understand their learning roadmap, answer questions about topics,
suggest resources, and motivate them to keep going.
Keep answers concise and practical. Use bullet points for lists.
"""


def answer_question(student_id: str, question: str) -> str:
    """
    Answer a general question from the student in the context of their roadmap.

    In mock mode: matches keywords and returns a pre-written helpful answer.
    In real mode: calls Granite with full student context.
    """
    # ── Mock path ────────────────────────────────────────────────────────────
    if USE_MOCK:
        return mock_responses.mock_answer_question(student_id, question)

    # ── Real API path (unchanged) ────────────────────────────────────────────
    state = memory.get_student_state(student_id)
    context_parts = []

    if state:
        profile = state["profile"]
        roadmap = state["roadmap"]
        completed = state["completed_topics"]

        context_parts.append(
            f"Student: {profile['name']}, "
            f"learning {profile['domain']} at {profile['skill_level']} level."
        )
        if completed:
            context_parts.append(f"Completed topics: {', '.join(completed[:5])}.")
        if roadmap:
            context_parts.append(
                f"Current roadmap title: {roadmap.get('roadmap_title', 'N/A')}."
            )

    context = " ".join(context_parts)
    system = CHAT_SYSTEM_PROMPT + (f"\n\nStudent context: {context}" if context else "")

    return _chat(system, question)
