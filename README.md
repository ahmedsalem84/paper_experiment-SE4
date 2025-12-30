# Semanti-MOP: Bridging the Semantic Gap in Automated Architecture Refactoring via Instability-Aware Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-orange.svg)](https://ollama.ai/)
[![Pymoo](https://img.shields.io/badge/Optimization-Pymoo-red.svg)](https://pymoo.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b.svg)](https://streamlit.io/)

## 🧬 Overview

**Semanti-MOP** (Semantic Multi-Objective Optimization) is a hybrid framework designed to automate the decomposition of monolithic software into microservices. It addresses the fundamental **Semantic–Structural Conflict**—where purely mathematical optimization (coupling/cohesion) often leads to service boundaries that lack domain coherence and violate architectural intent.

The core innovation is an **Instability-Triggered Semantic Repair Operator**. By monitoring the assignments of classes across a Many-Objective Evolutionary population, the framework identifies "Borderline Classes" (high entropy). It then selectively invokes a Large Language Model (LLM) to act as a semantic arbiter, resolving ambiguities using Retrieval-Augmented Generation (RAG) and Domain-Driven Design (DDD) principles.

---

## 🚀 Key Features

*   **7 State-of-the-Art Optimizers:** Support for NSGA-II, NSGA-III, MOEA/D, AGE-MOEA-II, R-NSGA-II, U-NSGA-III, and Random Search.
*   **Targeted LLM Steering:** Uses Pareto-front instability to invoke LLMs (Llama 2, Llama 3, Mistral) only where structural metrics are ambiguous.
*   **RAG-Enhanced Reasoning:** Contextualizes LLM prompts based on specific architectural scenarios (Semantic Ambiguity vs. Structural Conflict).
*   **Memetic Repair Operator:** Forcibly steers 50% of the population toward semantically valid solutions to snap the search trajectory out of logical local optima.
*   **Real-Time Proof Metrics:** Automatically calculates MoJoFM (Accuracy), Hypervolume (Search Quality), and A12 Effect Size.

---

## 🛠️ The 5-Phase Methodology

1.  **Phase 1: Semantic-Structural Graph Construction**  
    Parses source code into a weighted graph embedding static dependencies, semantic embeddings, and hierarchy weights.
2.  **Phase 2: Many-Objective Evolutionary Optimization**  
    Runs metaheuristics to minimize inter-service coupling and maximize intra-service cohesion.
3.  **Phase 3: LLM-Guided Semantic Steering (Core Novelty)**  
    Detects borderline classes and uses RAG-augmented LLM reasoning to resolve architectural deadlocks.
4.  **Phase 4: Convergence and Validation**  
    Monitors Pareto-optimal trade-offs and applies memetic repair to stabilize solution sets.
5.  **Phase 5: Output and Interpretation**  
    Generates explainable AI logs, architecture visualizations, and statistical comparisons.

---

## 📦 Installation

### 1. System Requirements
- **Python:** 3.10 or higher.
- **LLM Engine:** [Ollama](https://ollama.ai/) installed and running.
- **Hardware:** GPU recommended for local LLM inference.

### 2. Setup Repository
```bash
git clone https://github.com/ahmedsalem84/SE4.git
cd SE4

3. Install Dependencies

pip install streamlit pymoo networkx pandas plotly scikit-learn requests javalang matplotlib scipy gitpython numba

4. Setup Local AI Models

Ensure Ollama is running and pull the required models:
ollama pull llama2
ollama pull llama3

5. Download Benchmark Datasets
Run the automated downloader to fetch the three industrial datasets from GitHub:

python download_datasets.py

📊 Running the Experimental Suite

Launch the interactive dashboard to replicate the study:streamlit run main.py
Usage Steps:

    Connection Test: Click 🔌 Test AI Connection in the sidebar to ensure Ollama is responding.

    Dataset Selection: Choose between Spring-PetClinic, Cargo, or Shopizer.

    Scenario Selection: Choose Semantic Ambiguity (naming overlaps) or Structural Conflict (infrastructure pull).

    Algorithm Selection: Select any combination of the 7 optimizers.

    Execution: Click 🚀 EXECUTE EXPERIMENT.

📈 Understanding the Outputs

    Final Metrics: Side-by-side boxplots comparing structural coupling vs. human-aligned accuracy (MoJoFM).

    Hypervolume/Cost: Visualizes the quality of the non-dominated front and proves the 90% reduction in LLM API calls.

    Real-Time Proof: Bar charts showing the "MoJoFM Delta"—the instantaneous accuracy gain achieved after each LLM intervention.

    Explainable AI Log: A complete RAG-driven ledger of every intervention, class name, and the domain-aware reasoning used by the AI.

📁 Project Structure
/SE4
├── main.py                     # Streamlit Dashboard & Main Loop
├── download_datasets.py        # Dataset Fetcher
├── logic/
│   ├── algorithms.py           # MaOP & Memetic Repair Logic (7 Optimizers)
│   ├── llm_client.py           # Ollama API & RAG Context Engine
│   ├── data_loader.py          # AST Parser & Dependency Extraction
│   └── metrics.py              # MoJoFM, Hypervolume, & A12 Stats
└── datasets/                   # (Created after download) Spring, Cargo, Shopizer

📖 Citation

If you use this framework in your research, please cite our work:

    Salem, A. (2025). Semanti-MOP: Bridging the Semantic Gap in Automated Architecture Refactoring via Instability-Aware Large Language Models. Information and Software Technology (Preprint).

🔗 Subject Systems

    Spring PetClinic

    Cargo Tracker

    Shopizer

Author: Ahmed Salem
Contact: a.salem@aast.edu
ORCID: [0000-0002-0456-2276](https://orcid.org/0000-0002-0456-2276)
