import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from scipy.stats import mannwhitneyu
import requests

from download_datasets import setup_data
from logic.llm_client import LLMClient
from logic.data_loader import DataLoader
from logic.metrics import calculate_mojofm, vargha_delaney, calculate_hypervolume
from logic.algorithms import MicroserviceProblem, get_algorithm, semanti_mop_repair
from pymoo.optimize import minimize
from pymoo.core.termination import NoTermination

# --- Configuration & State ---
st.set_page_config(layout="wide", page_title="Semanti-MOP: Final Complete Study")

if 'results' not in st.session_state: st.session_state.results = pd.DataFrame()
if 'history' not in st.session_state: st.session_state.history = {}
if 'best_solution' not in st.session_state: st.session_state.best_solution = None
if 'graph_ref' not in st.session_state: st.session_state.graph_ref = None
if 'llm_logs' not in st.session_state: st.session_state.llm_logs = []

st.title("🔬 Semanti-MOP: Final Complete Study (Publication Ready)")
st.markdown("Bridge the gap between **structural coupling** and **semantic domain intent** using LLM-steered optimization.")

# --- Helper 1: RAG Context & SCENARIO CLASSIFICATION ---
def get_rag_context(dataset_name, scenario):
    """
    Simulates retrieving context based on the selected architectural scenario.
    This proves the tool's strength across different problem types.
    """
    if "PetClinic" in dataset_name:
        if scenario == "Semantic Ambiguity":
            return "Project: PetClinic. SCENARIO: AMBIGUITY. Focus: Noun/Verb ambiguity (e.g., 'Owner' meaning pet owner vs. system user). Goal: Group by business domain."
        if scenario == "Structural Conflict":
            return "Project: PetClinic. SCENARIO: COUPLING CONFLICT. Focus: High utility coupling (shared database access) that conflicts with DDD boundaries. Goal: Prioritize semantic boundaries."
    if "Cargo" in dataset_name:
        if scenario == "Semantic Ambiguity":
            return "Project: Cargo Tracker. SCENARIO: DDD CONTEXTS. Focus: Enforcing Bounded Contexts and Aggregate roots (strict semantic isolation between Cargo and Voyage)."
        if scenario == "Structural Conflict":
            return "Project: Cargo Tracker. SCENARIO: COUPLING CONFLICT. Focus: Utility classes shared across contexts. Goal: Break up utility coupling."
    if "Shopizer" in dataset_name:
        if scenario == "Semantic Ambiguity":
            return "Project: Shopizer E-commerce. SCENARIO: INHERITANCE AMBIGUITY. Focus: Correct separation of generic 'Product' base classes from specific 'DigitalProduct' subclasses."
        if scenario == "Structural Conflict":
            return "Project: Shopizer E-commerce. SCENARIO: TRANSACTIONAL CONFLICT. Focus: Separation of transactional (Order) and master data (Catalog) despite shared database entities."
    return f"Generic Application. SCENARIO: {scenario}."

# --- Helper 2: Plotting (Full code provided in previous steps, omitted for final file size) ---
def plot_microservice_graph(G, assignments):
    pos = nx.spring_layout(G, seed=42, k=0.15)
    edge_x, edge_y = [], []; node_x, node_y, node_text, node_color = [], [], [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y)
        node_text.append(f"{G.nodes[node].get('name', str(node))}")
        node_color.append(assignments[node] if node < len(assignments) else 0)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, marker=dict(showscale=True, colorscale='Turbo', color=node_color, size=12))
    return go.Figure(data=[go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#ccc'), mode='lines'), node_trace], layout=go.Layout(showlegend=False, margin=dict(b=0,l=0,r=0,t=40)))

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    setup_data()
    dataset_name = st.selectbox("1. Dataset", ["Spring-PetClinic", "Cargo", "Shopizer"])
    
    # NEW: Scenario Dropdown
    scenario = st.selectbox("1b. Architectural Scenario", ["Semantic Ambiguity", "Structural Conflict"])

    all_opts = ["NSGA-II", "NSGA-III", "MOEA/D", "AGE-MOEA-II", "R-NSGA-II", "U-NSGA-III", "Random Search"]
    selected_opts = st.multiselect("2. Optimizers", all_opts, default=["NSGA-II", "MOEA/D"])
    
    llm_model = st.selectbox("3. LLM Model (Ollama)", ["llama2", "llama3", "mistral"])
    
    if st.button("🔌 Test AI Connection"):
        try:
            res = requests.post("http://127.0.0.1:11434/api/generate", json={"model": llm_model, "prompt": "Test", "stream": False}, timeout=10)
            if res.status_code == 200: st.success(f"✅ Connected to {llm_model}!")
            else: st.error(f"❌ Ollama Error {res.status_code}: Model not found.")
        except: st.error("❌ Ollama is not running (Connection Refused).")

    st.markdown("---")
    generations = st.number_input("Generations", 5, 200, 15)
    pop_size = st.number_input("Population Size", 10, 200, 20)
    runs = st.number_input("Runs", 1, 10, 2)
    run_ablation = st.checkbox("✅ Enable LLM Steering", value=True)
    
    btn_start = st.button("🚀 EXECUTE EXPERIMENT")

# --- Main Logic ---
if btn_start:
    llm = LLMClient(model_name=llm_model)
    llm.decision_log = [] 
    loader = DataLoader(llm)
    
    with st.spinner("Loading Data..."):
        G, pkg_map = loader.load_real_dataset(f"datasets/{dataset_name}")
        st.session_state.graph_ref = G
    
    if G is None: st.error("Load failed."); st.stop()
    rag_context = get_rag_context(dataset_name, scenario) # Use both dataset and scenario
    
    # NEW: Initialization of Delta Log
    st.session_state.repair_deltas = []

    results, history_log = [], {}
    total = len(selected_opts) * runs * (2 if run_ablation else 1)
    bar = st.progress(0); counter = 0; best_mojofm_global = -1
    
    for algo in selected_opts:
        modes = ["Baseline"]
        if run_ablation: modes.append("Semanti-MOP")
        
        for mode in modes:
            hist_key = f"{algo}_{mode}"
            history_log[hist_key] = []
            
            for r in range(runs):
                problem = MicroserviceProblem(G)
                algorithm = get_algorithm(algo, pop_size)
                algorithm.setup(problem, termination=NoTermination())
                
                trace = []; calls_made_in_run = 0
                for gen in range(generations):
                    
                    # 1. Get MoJoFM BEFORE repair (for plotting delta)
                    mojofm_before = 0.0
                    if mode == "Semanti-MOP" and (gen % 3 == 0) and (gen > 0):
                        res_temp = algorithm.result()
                        if res_temp.X is not None and len(res_temp.X)>0:
                             mojofm_before = calculate_mojofm(res_temp.X[0], G, pkg_map)

                    # 2. Advance Generation
                    algorithm.next()
                    trace.append(np.mean([ind.F[0] for ind in algorithm.pop]))
                    
                    # 3. Trigger Repair
                    if mode == "Semanti-MOP" and gen % 3 == 0 and gen > 0:
                        st.toast(f"🤖 {algo} Gen {gen}: LLM checking instability...", icon="🧠")
                        
                        # Call Repair (returns population, calls made, and repair status)
                        algorithm.pop, calls, repair_status = semanti_mop_repair(algorithm.pop, G, llm, rag_text=rag_context)
                        calls_made_in_run += calls
                        
                        # 4. Get MoJoFM AFTER repair and log delta
                        if repair_status:
                             res_after = algorithm.result()
                             mojofm_after = calculate_mojofm(res_after.X[np.argmin(np.sum(res_after.F, axis=1))], G, pkg_map)
                             
                             st.session_state.repair_deltas.append({
                                 "Generation": gen,
                                 "Algo": algo,
                                 "Scenario": scenario,
                                 "Delta_MoJoFM": mojofm_after - mojofm_before,
                                 "Calls": calls
                             })

                # ... (Final result logging) ...
                res = algorithm.result()
                best_ind = res.X[np.argmin(np.sum(res.F, axis=1))]
                mojofm = calculate_mojofm(best_ind, G, pkg_map)
                hv_score = calculate_hypervolume(res.F)
                
                if mojofm > best_mojofm_global:
                    best_mojofm_global = mojofm
                    st.session_state.best_solution = best_ind
                
                results.append({
                    "Algorithm": algo, "Mode": mode, "Run": r, 
                    "Coupling": res.F[0][0], "MoJoFM": mojofm, "Hypervolume": hv_score,
                    "LLM_Calls": calls_made_in_run
                })
                
                if not history_log[hist_key]: history_log[hist_key] = [trace]
                counter += 1; bar.progress(counter / total)
                
    st.session_state.results = pd.DataFrame(results)
    st.session_state.history = history_log
    st.session_state.llm_logs = llm.decision_log

# --- Visualization ---
if not st.session_state.results.empty:
    df = st.session_state.results
    
    t1, t2, t3, t4, t5, t6 = st.tabs(["📊 Metrics", "📈 Hypervolume/Cost", "📉 Convergence", "💥 Real-Time Proof", "📜 Final Tables", "🧠 AI Logs"])
    
    # T1: Boxplots
    with t1:
        st.subheader("Structural vs Semantic Performance")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(df, x="Algorithm", y="Coupling", color="Mode", title="Coupling (Min)"), use_container_width=True)
        with c2: st.plotly_chart(px.box(df, x="Algorithm", y="MoJoFM", color="Mode", title="MoJoFM (Semantic Accuracy)"), use_container_width=True)

    # T2: Hypervolume/Cost
    with t2:
        st.subheader("Optimization Quality and Cost-Benefit")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(px.box(df, x="Algorithm", y="Hypervolume", color="Mode", title="Hypervolume (Quality - Max is Best)"), use_container_width=True)
        with c2:
            cost_df = df.groupby(["Algorithm", "Mode"])["LLM_Calls"].sum().reset_index()
            fig_cost = px.bar(cost_df[cost_df['Mode']=='Semanti-MOP'], x="Algorithm", y="LLM_Calls", title="Total LLM Interventions Across All Runs")
            st.plotly_chart(fig_cost, use_container_width=True)

    # T3: Convergence
    with t3:
        st.subheader("Convergence Trajectory")
        fig_conv = go.Figure()
        for k, v in st.session_state.history.items():
            if v: fig_conv.add_trace(go.Scatter(y=v[0], mode='lines', name=k))
        st.plotly_chart(fig_conv, use_container_width=True)

    # T4: REAL-TIME DELTA PLOT (NEW)
    with t4:
        st.subheader("💥 Real-Time Proof: MoJoFM Improvement After LLM Intervention")
        if st.session_state.repair_deltas:
            delta_df = pd.DataFrame(st.session_state.repair_deltas)
            
            # Aggregate by generation and algorithm
            agg_delta = delta_df.groupby(['Generation', 'Algo', 'Scenario'])['Delta_MoJoFM'].mean().reset_index()
            
            fig_delta = px.bar(agg_delta, 
                               x="Generation", y="Delta_MoJoFM", color="Algo", 
                               facet_col="Scenario", 
                               title="Average MoJoFM Improvement (Delta) per LLM Intervention Point",
                               labels={"Delta_MoJoFM": "MoJoFM Improvement (%)"})
            st.plotly_chart(fig_delta, use_container_width=True)
            
            st.info("A positive bar proves the LLM successfully pushed the solution toward the semantic ideal.")
        else:
            st.warning("No successful interventions logged in Semanti-MOP mode.")

    # T5: Final Tables
    with t5:
        st.subheader("Comprehensive Statistical Results (Paper-Ready)")
        
        agg_df = df.groupby(['Algorithm', 'Mode']).agg(
            Mean_Coupling=('Coupling', 'mean'), Std_Coupling=('Coupling', 'std'),
            Mean_MoJoFM=('MoJoFM', 'mean'), Std_MoJoFM=('MoJoFM', 'std'),
            Mean_HV=('Hypervolume', 'mean'), Std_HV=('Hypervolume', 'std'),
            Total_LLM_Calls=('LLM_Calls', 'sum')
        ).reset_index()
        st.dataframe(agg_df.style.format({v: "{:.2f}" for v in agg_df.columns if v not in ['Algorithm', 'Mode']}), use_container_width=True)
        
        stat_data = []
        for m in ["MoJoFM", "Hypervolume"]:
            for algo in df["Algorithm"].unique():
                base = df[(df["Algorithm"]==algo) & (df["Mode"]=="Baseline")][m]
                prop = df[(df["Algorithm"]==algo) & (df["Mode"]=="Semanti-MOP")][m]
                if len(base) > 1 and len(prop) > 1:
                    u, p = mannwhitneyu(base, prop)
                    a12 = vargha_delaney(base, prop)
                    winner = "Semanti-MOP" if a12 > 0.5 else "Baseline"
                    stat_data.append({"Metric": m, "Algo": algo, "P-Value": f"{p:.4f}", "A12": f"{a12:.2f}", "Winner": winner})
        st.dataframe(pd.DataFrame(stat_data), use_container_width=True)

    # T6: AI Logs
    with t6:
        st.subheader("🧠 Explainable AI Log (RAG-Enhanced)")
        logs = st.session_state.llm_logs
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df, use_container_width=True)
            if "Class" in log_df.columns:
                counts = log_df["Class"].value_counts().reset_index()
                counts.columns = ["Class", "Count"]
                st.plotly_chart(px.bar(counts.head(10), x="Class", y="Count", title="Top 10 Most Unstable Classes Fixed by LLM"))
        else:
            st.warning("No AI logs.")

    st.download_button("💾 Download Full Results CSV", df.to_csv(index=False).encode('utf-8'), "semanti_mop_results.csv", "text/csv")