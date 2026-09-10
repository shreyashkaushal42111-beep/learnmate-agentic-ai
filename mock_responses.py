"""
mock_responses.py — Pre-written realistic mock responses for LearnMate.

Used when USE_MOCK=true to demonstrate the full agentic flow without
calling the real watsonx.ai / Granite API.

All public functions mirror the signatures of the live agent functions they
replace so that agent.py can swap them in with a simple branch.
"""

import json
import re
from typing import Optional

import memory
import rag_utils

# ---------------------------------------------------------------------------
# Assessment mock — 3-question scripted conversation
# ---------------------------------------------------------------------------

# The mock assessment drives a fixed 3-question flow keyed by the number of
# user turns seen so far.  After the 3rd answer it returns the completion
# signal with a synthesised skill level derived from the keyword heuristic
# already present in agent.py (_score_answers).

_ASSESSMENT_QUESTION_TEMPLATES = {
    1: (
        "Great choice! Let me ask you a few quick questions to tailor your "
        "{domain} learning path.\n\n"
        "**Question 1:** {q1}"
    ),
    2: "Got it! **Question 2:** {q2}",
    3: "Almost done! **Question 3:** {q3}",
}

# Friendly default questions for domains with no bespoke bank entry
_GENERIC_QUESTIONS = [
    "How would you describe your current experience with {domain}? (none / a little / quite a bit)",
    "Have you completed any formal course or hands-on project in {domain} before?",
    "Are you comfortable reading technical documentation and learning independently in this area?",
]


def mock_assessment_turn(
    domain: str,
    conversation_history: list[dict],
    student_answer: Optional[str],
    assessment_questions_bank: dict,
    score_answers_fn,
) -> dict:
    """
    Simulate one turn of the skill-assessment conversation.

    Parameters match agent.run_assessment_turn so the call-site is unchanged.
    """
    if student_answer is not None:
        conversation_history.append({"role": "user", "content": student_answer})

    user_turn_count = sum(1 for m in conversation_history if m["role"] == "user")

    # Pull domain-specific questions or fall back to generic ones
    qs = assessment_questions_bank.get(domain, [])
    q_texts = [q["question"] for q in qs] if qs else [
        q.format(domain=domain) for q in _GENERIC_QUESTIONS
    ]
    # Ensure we always have 3 questions
    while len(q_texts) < 3:
        q_texts.append(f"How confident are you with advanced {domain} concepts?")

    if user_turn_count == 0:
        msg = (
            f"Great choice! Let me ask you a few quick questions to tailor your "
            f"**{domain}** learning path.\n\n"
            f"**Question 1:** {q_texts[0]}"
        )
        conversation_history.append({"role": "assistant", "content": msg})
        return {"message": msg, "complete": False, "skill_level": None}

    if user_turn_count == 1:
        msg = f"Thanks for sharing! **Question 2:** {q_texts[1]}"
        conversation_history.append({"role": "assistant", "content": msg})
        return {"message": msg, "complete": False, "skill_level": None}

    if user_turn_count == 2:
        msg = f"Almost there! **Question 3:** {q_texts[2]}"
        conversation_history.append({"role": "assistant", "content": msg})
        return {"message": msg, "complete": False, "skill_level": None}

    # user_turn_count >= 3 → assessment complete
    answers = [m["content"] for m in conversation_history if m["role"] == "user"]
    skill_level = score_answers_fn(domain, answers)

    level_descriptions = {
        "beginner": "you're at the **beginner** level — a great place to start building solid fundamentals",
        "intermediate": "you're at the **intermediate** level — you have a useful base to build on",
        "advanced": "you're at the **advanced** level — we'll focus on depth and cutting-edge topics",
    }
    description = level_descriptions.get(skill_level, f"you're at the **{skill_level}** level")

    msg = (
        f"Thanks for your answers! Based on our conversation, {description}.\n\n"
        f"Let me generate your personalised roadmap now! 🚀"
    )
    conversation_history.append({"role": "assistant", "content": msg})
    return {"message": msg, "complete": True, "skill_level": skill_level}


# ---------------------------------------------------------------------------
# Roadmap generation mock
# ---------------------------------------------------------------------------

# One realistic 8-week roadmap per domain × level combination.
# Only a subset are pre-written verbatim; the rest are generated dynamically
# from the RAG context so the mock still exercises the full RAG retrieval path.

_MOCK_ROADMAPS: dict[str, dict[str, dict]] = {
    "Frontend Development": {
        "beginner": {
            "roadmap_title": "Frontend Development Learning Path — Beginner",
            "domain": "Frontend Development",
            "level": "beginner",
            "total_weeks": 8,
            "weeks": [
                {
                    "week": 1,
                    "theme": "HTML5 Foundations",
                    "topics": [
                        "HTML5 Fundamentals: tags, semantic structure, forms, tables",
                        "Debugging with Browser DevTools",
                    ],
                    "hands_on_task": "Build a fully semantic 3-page static website about a topic you enjoy.",
                    "estimated_hours": 5,
                },
                {
                    "week": 2,
                    "theme": "Styling with CSS3",
                    "topics": [
                        "CSS3 Basics: selectors, box model, colors, fonts, backgrounds",
                        "Responsive Design: flexbox, CSS Grid, media queries",
                    ],
                    "hands_on_task": "Style your Week 1 website with a consistent colour palette and mobile-first layout.",
                    "estimated_hours": 6,
                },
                {
                    "week": 3,
                    "theme": "JavaScript Basics",
                    "topics": [
                        "JavaScript Basics: variables, data types, conditionals, loops",
                    ],
                    "hands_on_task": "Write a quiz game in vanilla JS that keeps score and shows a result at the end.",
                    "estimated_hours": 6,
                },
                {
                    "week": 4,
                    "theme": "DOM & Interactivity",
                    "topics": [
                        "DOM Manipulation: querySelector, events, innerHTML",
                    ],
                    "hands_on_task": "Add a live search/filter feature to a list of items on a webpage.",
                    "estimated_hours": 5,
                },
                {
                    "week": 5,
                    "theme": "Version Control",
                    "topics": [
                        "Introduction to Git and GitHub: init, commit, push, pull requests",
                    ],
                    "hands_on_task": "Host your portfolio website on GitHub Pages via a pull-request workflow.",
                    "estimated_hours": 4,
                },
                {
                    "week": 6,
                    "theme": "Responsive Projects",
                    "topics": [
                        "Building static HTML/CSS pages: portfolio, landing pages",
                    ],
                    "hands_on_task": "Build a responsive product landing page with a sticky nav, hero section, and footer.",
                    "estimated_hours": 6,
                },
                {
                    "week": 7,
                    "theme": "Review & Polish",
                    "topics": [
                        "HTML5 Fundamentals: tags, semantic structure, forms, tables",
                        "CSS3 Basics: selectors, box model, colors, fonts, backgrounds",
                    ],
                    "hands_on_task": "Audit your existing pages for accessibility and semantic correctness, then fix issues.",
                    "estimated_hours": 4,
                },
                {
                    "week": 8,
                    "theme": "Capstone",
                    "topics": [
                        "Responsive Design: flexbox, CSS Grid, media queries",
                        "DOM Manipulation: querySelector, events, innerHTML",
                    ],
                    "hands_on_task": "Build and deploy a personal portfolio website showcasing your Week 1–7 projects.",
                    "estimated_hours": 8,
                },
            ],
            "final_project": "Personal portfolio website hosted on GitHub Pages with at least 3 projects documented.",
            "resources_note": "Free resources: MDN Web Docs, freeCodeCamp, The Odin Project, CSS-Tricks.",
        },
        "intermediate": {
            "roadmap_title": "Frontend Development Learning Path — Intermediate",
            "domain": "Frontend Development",
            "level": "intermediate",
            "total_weeks": 8,
            "weeks": [
                {
                    "week": 1,
                    "theme": "Modern JavaScript (ES6+)",
                    "topics": [
                        "ES6+ JavaScript: arrow functions, destructuring, spread, promises, async/await",
                        "Fetch API and REST: GET, POST, handling JSON responses",
                    ],
                    "hands_on_task": "Fetch data from a public REST API (e.g. Open Trivia DB) and render it dynamically.",
                    "estimated_hours": 6,
                },
                {
                    "week": 2,
                    "theme": "React Fundamentals",
                    "topics": [
                        "React.js Fundamentals: JSX, components, props, state, hooks (useState, useEffect)",
                    ],
                    "hands_on_task": "Convert your vanilla JS quiz from before into a React app with functional components.",
                    "estimated_hours": 7,
                },
                {
                    "week": 3,
                    "theme": "Routing & State",
                    "topics": [
                        "React Router: SPA navigation, dynamic routes, nested routes",
                        "State Management with Context API and Redux Toolkit",
                    ],
                    "hands_on_task": "Add multi-page navigation to your React app using React Router v6.",
                    "estimated_hours": 6,
                },
                {
                    "week": 4,
                    "theme": "Styling at Scale",
                    "topics": [
                        "CSS Preprocessors: SASS/SCSS variables, nesting, mixins",
                        "Component Libraries: Material UI, Tailwind CSS integration",
                    ],
                    "hands_on_task": "Refactor your app's styles to use Tailwind CSS utility classes.",
                    "estimated_hours": 5,
                },
                {
                    "week": 5,
                    "theme": "TypeScript & Tooling",
                    "topics": [
                        "TypeScript basics for React projects",
                        "Webpack and Vite: bundling, hot reload, build optimization",
                    ],
                    "hands_on_task": "Migrate your React app from JavaScript to TypeScript, adding type annotations.",
                    "estimated_hours": 7,
                },
                {
                    "week": 6,
                    "theme": "Testing",
                    "topics": [
                        "Testing with Jest and React Testing Library",
                    ],
                    "hands_on_task": "Write unit and integration tests for at least 5 components in your project.",
                    "estimated_hours": 6,
                },
                {
                    "week": 7,
                    "theme": "Full-Stack Integration",
                    "topics": [
                        "Fetch API and REST: GET, POST, handling JSON responses",
                        "State Management with Context API and Redux Toolkit",
                    ],
                    "hands_on_task": "Build a weather dashboard that fetches live data, handles loading and error states.",
                    "estimated_hours": 7,
                },
                {
                    "week": 8,
                    "theme": "Capstone",
                    "topics": [
                        "React.js Fundamentals: JSX, components, props, state, hooks (useState, useEffect)",
                        "TypeScript basics for React projects",
                    ],
                    "hands_on_task": "Build and deploy a complete e-commerce product listing with cart functionality.",
                    "estimated_hours": 9,
                },
            ],
            "final_project": "E-commerce product listing app built with React, TypeScript, and Tailwind CSS, deployed on Vercel.",
            "resources_note": "Free resources: React docs (react.dev), TypeScript Handbook, Tailwind docs, Scrimba React course.",
        },
    },
    "Data Science": {
        "beginner": {
            "roadmap_title": "Data Science Learning Path — Beginner",
            "domain": "Data Science",
            "level": "beginner",
            "total_weeks": 8,
            "weeks": [
                {
                    "week": 1,
                    "theme": "Python for Data",
                    "topics": [
                        "Python for Data Science: NumPy, Pandas basics, data types",
                        "Jupyter Notebooks: environment setup, markdown, cells",
                    ],
                    "hands_on_task": "Load the Titanic dataset in Pandas, inspect it, and compute basic statistics.",
                    "estimated_hours": 6,
                },
                {
                    "week": 2,
                    "theme": "Data Cleaning",
                    "topics": ["Data Cleaning: handling nulls, duplicates, type casting"],
                    "hands_on_task": "Clean a messy CSV dataset: fill/drop nulls, fix data types, remove duplicates.",
                    "estimated_hours": 5,
                },
                {
                    "week": 3,
                    "theme": "Exploratory Analysis",
                    "topics": [
                        "Exploratory Data Analysis (EDA): descriptive statistics, distributions",
                        "Data Visualization: Matplotlib, Seaborn, basic chart types",
                    ],
                    "hands_on_task": "Create an EDA notebook with 5 visualisations that tell a story about a dataset.",
                    "estimated_hours": 6,
                },
                {
                    "week": 4,
                    "theme": "Statistics Foundations",
                    "topics": [
                        "Introduction to Statistics: mean, median, variance, normal distribution"
                    ],
                    "hands_on_task": "Compute and visualise the distribution of a real dataset column; interpret the results.",
                    "estimated_hours": 5,
                },
                {
                    "week": 5,
                    "theme": "SQL for Analysis",
                    "topics": ["SQL for Data Analysis: SELECT, GROUP BY, JOINs, subqueries"],
                    "hands_on_task": "Answer 10 business questions from a SQLite sales database using SQL queries.",
                    "estimated_hours": 6,
                },
                {
                    "week": 6,
                    "theme": "Intro to ML",
                    "topics": [
                        "Introduction to Machine Learning concepts: supervised vs unsupervised"
                    ],
                    "hands_on_task": "Train your first scikit-learn classifier (Decision Tree) on the Iris dataset.",
                    "estimated_hours": 6,
                },
                {
                    "week": 7,
                    "theme": "Review & Portfolio",
                    "topics": [
                        "Data Visualization: Matplotlib, Seaborn, basic chart types",
                        "Exploratory Data Analysis (EDA): descriptive statistics, distributions",
                    ],
                    "hands_on_task": "Polish your EDA notebook and publish it to Kaggle or GitHub.",
                    "estimated_hours": 4,
                },
                {
                    "week": 8,
                    "theme": "Capstone",
                    "topics": [
                        "Python for Data Science: NumPy, Pandas basics, data types",
                        "Data Cleaning: handling nulls, duplicates, type casting",
                    ],
                    "hands_on_task": "Complete an end-to-end EDA on the Titanic or Iris dataset and present findings.",
                    "estimated_hours": 8,
                },
            ],
            "final_project": "EDA on Titanic dataset with a Jupyter notebook published on GitHub, including narrative and visualisations.",
            "resources_note": "Free resources: Kaggle Learn, Google Colab, Python Data Science Handbook (free online), StatQuest YouTube.",
        }
    },
}


def mock_generate_roadmap(student_id: str, build_fallback_fn) -> dict:
    """
    Return a pre-written roadmap if one exists for (domain, level),
    otherwise build one dynamically from RAG topics (exercising the full
    retrieval pipeline without calling the LLM).
    """
    state = memory.get_student_state(student_id)
    if state is None:
        raise ValueError(f"Student {student_id} not found")

    profile = state["profile"]
    domain = profile["domain"]
    level = profile["skill_level"]
    name = profile["name"]

    # Try pre-written roadmap first
    pre = _MOCK_ROADMAPS.get(domain, {}).get(level)
    if pre:
        roadmap = dict(pre)  # shallow copy
        roadmap["student_name"] = name
        # Ensure level key is present (some templates omit it)
        roadmap.setdefault("level", level)
    else:
        # Fall back to RAG-grounded dynamic construction (no LLM needed)
        rag_context = rag_utils.retrieve_topics_for_roadmap(domain, level, num_weeks=8)
        roadmap = build_fallback_fn(name, domain, level, rag_context)

    memory.save_roadmap(student_id, roadmap)
    return roadmap


# ---------------------------------------------------------------------------
# Roadmap revision mock
# ---------------------------------------------------------------------------

# Revision rules applied to the stored roadmap based on feedback keywords.

def mock_revise_roadmap(student_id: str, feedback: str, build_fallback_fn) -> dict:
    """
    Simulate an intelligent roadmap revision based on feedback keywords.

    Modifies the current roadmap in memory and returns the updated version.
    """
    state = memory.get_student_state(student_id)
    if state is None:
        raise ValueError(f"Student {student_id} not found")

    current_roadmap = state["roadmap"]
    if current_roadmap is None:
        raise ValueError("No roadmap found. Generate a roadmap first.")

    memory.save_feedback(student_id, feedback)

    fb = feedback.lower()
    import copy
    revised = copy.deepcopy(current_roadmap)

    if any(kw in fb for kw in ["too easy", "faster", "skip", "bore", "basic"]):
        # Accelerate: reduce hours, append "(Advanced)" hint to themes
        revised["_revision_note"] = (
            "Roadmap accelerated — advanced topics surfaced and pace increased based on your feedback."
        )
        for w in revised.get("weeks", []):
            if not w["theme"].endswith("⚡"):
                w["theme"] = w["theme"] + " ⚡"
            w["estimated_hours"] = max(4, w.get("estimated_hours", 5) - 1)
            # Append an advanced topic hint if topics list is short
            if len(w["topics"]) < 4:
                w["topics"].append(f"Advanced extension: deeper dive into {w['topics'][0].split(':')[0]}")

    elif any(kw in fb for kw in ["too hard", "slow", "difficult", "struggle", "confus"]):
        # Slow down: add foundational notes
        revised["_revision_note"] = (
            "Roadmap adjusted — additional foundational material added and pace reduced based on your feedback."
        )
        for w in revised.get("weeks", []):
            if not w["theme"].startswith("🔄 "):
                w["theme"] = "🔄 " + w["theme"]
            w["estimated_hours"] = w.get("estimated_hours", 5) + 1
            w["topics"].insert(0, f"Review & reinforce: core concepts from previous week")

    elif any(kw in fb for kw in ["practice", "project", "hands-on", "hands on", "build"]):
        # More practical tasks
        revised["_revision_note"] = (
            "Extra hands-on tasks added to every week based on your feedback."
        )
        for w in revised.get("weeks", []):
            w["hands_on_task"] = (
                w.get("hands_on_task", "Practice exercise") +
                " — **Bonus:** Build a mini-project applying all topics from this week."
            )

    else:
        # Generic positive acknowledgement — minor tweak to show something changed
        revised["_revision_note"] = (
            f"Roadmap reviewed based on your feedback: \"{feedback[:80]}\". "
            "Minor adjustments applied to better match your learning goals."
        )
        if revised.get("weeks"):
            revised["weeks"][-1]["hands_on_task"] = (
                revised["weeks"][-1].get("hands_on_task", "") +
                " (updated based on your latest feedback)"
            )

    memory.save_roadmap(student_id, revised)
    return revised


# ---------------------------------------------------------------------------
# General Q&A mock
# ---------------------------------------------------------------------------

_MOCK_QA_RESPONSES = [
    (
        ["resource", "learn", "where", "recommend", "book", "course"],
        (
            "Great question! Here are some top free resources:\n\n"
            "- **MDN Web Docs** — the definitive reference for web technologies\n"
            "- **freeCodeCamp** — structured, project-based curriculum\n"
            "- **The Odin Project** — full-stack open-source curriculum\n"
            "- **YouTube** — channels like Traversy Media, Fireship, and Codevolution\n"
            "- **Official docs** — always the most accurate and up-to-date source\n\n"
            "Start with one and go deep rather than jumping between many. 📚"
        ),
    ),
    (
        ["how long", "time", "finish", "complete", "week"],
        (
            "The roadmap is designed for **~5–8 hours per week** over **8 weeks**.\n\n"
            "That said, everyone learns differently:\n"
            "- At **5 hrs/week** → ~8 weeks to completion\n"
            "- At **10 hrs/week** → ~4 weeks\n"
            "- At **2 hrs/week** → ~18 weeks\n\n"
            "Consistency beats speed — even 30 minutes a day compounds quickly. 💪"
        ),
    ),
    (
        ["project", "build", "practice", "exercise"],
        (
            "Hands-on projects are the fastest way to cement knowledge! Here are a few ideas:\n\n"
            "- Rebuild a site/app you use daily (weather app, to-do list, portfolio)\n"
            "- Contribute to an open-source project on GitHub\n"
            "- Complete a challenge on Frontend Mentor, Kaggle, or HackTheBox\n"
            "- Build the project suggested in your weekly hands-on task\n\n"
            "The goal is to struggle a little — that's where the real learning happens. 🛠️"
        ),
    ),
    (
        ["stuck", "help", "don't understand", "confused", "error"],
        (
            "Totally normal to feel stuck! Here's a productive debugging loop:\n\n"
            "1. **Read the error message carefully** — it often tells you exactly what's wrong\n"
            "2. **Search the error** on Stack Overflow or GitHub Issues\n"
            "3. **Rubber duck it** — explain your code out loud step by step\n"
            "4. **Take a short break** — fresh eyes spot bugs faster\n"
            "5. **Ask a community** — Discord servers, Reddit r/learnprogramming, or open a GitHub issue\n\n"
            "You've got this! Every bug you squash makes you a better developer. 🐛➡️✅"
        ),
    ),
]

_MOCK_QA_DEFAULT = (
    "That's a great question! Based on your current roadmap, I'd suggest focusing on the topics "
    "scheduled for this week before branching out — building in layers keeps things from getting "
    "overwhelming.\n\n"
    "If you're looking for a deeper explanation of a specific topic, feel free to ask me directly "
    "and I'll break it down for you. Keep up the great work! 🎓"
)


def mock_answer_question(student_id: str, question: str) -> str:
    """Return a contextual pre-written answer matched by keywords."""
    q_lower = question.lower()
    for keywords, response in _MOCK_QA_RESPONSES:
        if any(kw in q_lower for kw in keywords):
            return response
    return _MOCK_QA_DEFAULT
