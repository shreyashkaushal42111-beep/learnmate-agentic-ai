# LearnMate 🎓 — Agentic AI for Personalised Learning Pathways

LearnMate is an **agentic AI system** that creates personalised, adaptive week-by-week learning roadmaps for students. It is powered by **IBM watsonx.ai** (Granite model) for reasoning, **ChromaDB** for vector-based retrieval (RAG), and **Streamlit** for the chat-first UI.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                    │
│   ┌─────────────────┐        ┌──────────────────────────┐   │
│   │  Chat Interface │        │  Roadmap Display Panel    │   │
│   │  (assessment,   │        │  (week cards, progress,   │   │
│   │  Q&A, feedback) │        │   topic completion)       │   │
│   └────────┬────────┘        └──────────────┬───────────┘   │
└────────────│──────────────────────────────────│──────────────┘
             │                                  │
             ▼                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Core (agent.py)                     │
│  • Onboarding & assessment conversation (Granite LLM)       │
│  • Roadmap generation (RAG + Granite LLM)                   │
│  • Feedback-driven roadmap revision (agentic loop)          │
│  • General Q&A                                              │
└──────────┬──────────────────────────┬────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐       ┌──────────────────────────────────┐
│   memory.py      │       │        rag_utils.py              │
│  SQLite DB       │       │  ChromaDB + SentenceTransformers  │
│  • students      │       │  • Embeds data/courses.json       │
│  • roadmaps      │       │  • Retrieves relevant topics      │
│  • completed     │       │    for domain + level             │
│  • feedback      │       └──────────────────────────────────┘
└──────────────────┘
           │
           ▼
┌──────────────────────┐
│  IBM watsonx.ai      │
│  Granite Model       │
│  (ibm-granite-3-3-   │
│   8b-instruct)       │
└──────────────────────┘
```

---

## Project Structure

```
learnmate/
├── app.py                # Streamlit UI (chat + roadmap panels)
├── agent.py              # Core agentic logic (assessment, generation, revision)
├── rag_utils.py          # ChromaDB vector retrieval utilities
├── memory.py             # SQLite-backed student state tracking
├── data/
│   └── courses.json      # Knowledge base: topics per domain and level
├── chroma_db/            # Auto-created: persistent ChromaDB vector store
├── learnmate.db          # Auto-created: SQLite student progress database
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Features

| Feature | Description |
|---|---|
| **Skill Assessment** | 2-3 adaptive questions to determine beginner / intermediate / advanced level |
| **RAG-Grounded Roadmaps** | Topics retrieved from ChromaDB (vector search) ensure roadmaps are grounded in real content |
| **Personalised 8-Week Plan** | Granite LLM generates a week-by-week roadmap tailored to the student |
| **Adaptive Feedback Loop** | Student feedback (e.g. "too easy") triggers roadmap revision — the core agentic behaviour |
| **Progress Tracking** | Mark topics complete; SQLite persists progress across sessions |
| **Returning Students** | Load any previously created student profile from the sidebar |
| **5 Domains** | Frontend Development, Cybersecurity, UI/UX Design, Data Science, Cloud Computing |

---

## Prerequisites

- Python **3.10+**
- An **IBM Cloud account** (free Lite tier works)
- Access to **IBM watsonx.ai** (free trial available)

---

## 1 — IBM Cloud & watsonx.ai Setup

### 1.1 Create an IBM Cloud Account
1. Go to [cloud.ibm.com](https://cloud.ibm.com) and sign up for a **free Lite account**.
2. No credit card required for the Lite tier.

### 1.2 Provision a watsonx.ai Instance
1. In the IBM Cloud console, search for **watsonx.ai** in the catalog.
2. Select the **Lite** plan (free, includes 50k token/month).
3. Click **Create**.

### 1.3 Create a Project in watsonx.ai
1. Go to [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com).
2. Click **New project → Create an empty project**.
3. Name it (e.g. `LearnMate`).
4. Note the **Project ID** from the project settings page.

### 1.4 Generate an API Key
1. Go to **Manage → Access (IAM) → API keys** in IBM Cloud.
2. Click **Create an IBM Cloud API key**.
3. Copy and save the key securely — it is shown only once.

### 1.5 Find the watsonx.ai URL
The default URL depends on your IBM Cloud region:

| Region | URL |
|---|---|
| US South (Dallas) | `https://us-south.ml.cloud.ibm.com` |
| EU (Frankfurt) | `https://eu-de.ml.cloud.ibm.com` |
| Tokyo | `https://jp-tok.ml.cloud.ibm.com` |
| London | `https://eu-gb.ml.cloud.ibm.com` |

---

## 2 — Local Setup

### 2.1 Clone / Download the project
```bash
git clone https://github.com/your-org/learnmate.git
cd learnmate
```

### 2.2 Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2.3 Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB).

### 2.4 Configure environment variables

Create a `.env` file in the project root (or export the variables directly):

```env
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_watsonx_project_id_here
GRANITE_MODEL_ID=ibm/granite-3-3-8b-instruct
```

Then load it before running:

**Windows (PowerShell):**
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.+)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
    }
}
```

**macOS / Linux:**
```bash
export $(grep -v '^#' .env | xargs)
```

Or install `python-dotenv` and add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py`.

### 2.5 Run the application
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## 3 — Available Granite Models

The default model is `ibm/granite-3-3-8b-instruct`. You can switch to any model available in your watsonx.ai instance:

| Model ID | Description |
|---|---|
| `ibm/granite-3-3-8b-instruct` | Balanced — recommended (default) |
| `ibm/granite-3-2-8b-instruct` | Slightly faster, similar quality |
| `ibm/granite-3-8b-instruct` | Earlier version |
| `ibm/granite-13b-instruct-v2` | Larger, higher quality, slower |

Set `GRANITE_MODEL_ID` in your `.env` to switch models.

---

## 4 — Usage Guide

### Starting as a New Student
1. Open the app at `http://localhost:8501`
2. In the **sidebar**, choose *New Student*
3. Enter your name and select a learning domain
4. Click **Start Learning Journey**
5. Answer the 2-3 assessment questions in the **Chat** tab
6. Your personalised roadmap will be generated and displayed in the **Roadmap** tab

### Giving Feedback (Agentic Adaptation)
In the Chat tab, you can type (or click) feedback like:
- *"This is too easy, I want to go faster"*
- *"The content is too difficult, please slow down"*
- *"I want more hands-on projects"*
- *"Skip ahead to advanced topics"*

The agent will revise your roadmap in real time.

### Returning Students
Select *Returning Student* in the sidebar, pick your name, and click **Load Profile** to resume where you left off.

### Tracking Progress
In the **Roadmap** tab, use the multiselect at the bottom to mark topics as complete. Progress is saved to the SQLite database and persists across sessions.

---

## 5 — Extending the Knowledge Base

Edit `data/courses.json` to add new domains or topics. After editing, force a ChromaDB rebuild:

```python
# In a Python REPL or a small script:
import rag_utils
rag_utils.build_index(force_rebuild=True)
```

---

## 6 — Troubleshooting

| Problem | Solution |
|---|---|
| `AuthenticationError` | Check `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` are set correctly |
| `ModelNotFound` | Verify the model ID exists in your watsonx.ai region; try `ibm/granite-3-3-8b-instruct` |
| Roadmap shows raw JSON | LLM response parsing failed; the fallback roadmap is used — try refreshing |
| ChromaDB error on startup | Delete the `chroma_db/` folder and restart; it will be rebuilt automatically |
| Slow first startup | The sentence-transformer model (~90 MB) is being downloaded — wait for it to complete |
| SQLite locked | Close any other process using `learnmate.db` and restart the app |

---

## 7 — Tech Stack

| Component | Technology |
|---|---|
| LLM | IBM watsonx.ai — Granite (ibm/granite-3-3-8b-instruct) |
| Vector DB | ChromaDB (local, persistent) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| State / Memory | SQLite via Python `sqlite3` |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## 8 — watsonx.ai Integration Status

This application is fully built to integrate with **IBM watsonx.ai Granite models** for all agentic reasoning — skill assessment, roadmap generation, and feedback-based revision. The complete integration is implemented in [`agent.py`](agent.py) using the `ibm-watsonx-ai` SDK and is activated by the credentials in `.env`.

Due to a temporary IBM Cloud account provisioning issue during development, this demo currently runs with **`USE_MOCK=true`**, which substitutes realistic pre-written responses for live Granite API calls. Every agentic flow (assessment → RAG retrieval → roadmap generation → feedback revision) runs end-to-end; only the LLM call itself is simulated.

**To switch to the live watsonx.ai API**, set the following in your `.env` file and restart the app:

```env
USE_MOCK=false
WATSONX_API_KEY=your_ibm_cloud_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_watsonx_project_id_here
```

No other code changes are required — the real API path in `agent.py` is fully intact and production-ready.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
