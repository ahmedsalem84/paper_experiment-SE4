import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

# --- Many-Objective Optimizers (MaOAs) ---
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rnsga2 import RNSGA2
from pymoo.algorithms.moo.unsga3 import UNSGA3

try:
    from pymoo.algorithms.moo.age2 import AGEMOEA2
except ImportError:
    # Fallback to NSGA2 container if numba/age2 is unavailable
    from pymoo.algorithms.moo.nsga2 import NSGA2 as AGEMOEA2

class MicroserviceProblem(ElementwiseProblem):
    """
    Formalization of the Microservice Decomposition as a MaOP.
    Objectives: 
    1. Minimize Coupling (Inter-service dependencies)
    2. Maximize Cohesion (Intra-service semantic unity)
    """
    def __init__(self, G, n_services=5):
        self.G = G
        self.n_nodes = len(G.nodes)
        # Objectives: [Coupling, -Cohesion]
        super().__init__(n_var=self.n_nodes, n_obj=2, xl=0, xu=n_services-1, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        coupling = 0
        cohesion = 0
        for u, v in self.G.edges():
            weight = self.G[u][v].get('weight', 1)
            if x[u] != x[v]:
                coupling += weight
            else:
                cohesion += weight
        
        # Jitter is applied to prevent population stagnation in indicators 
        # like Hypervolume and ensure mathematical diversity.
        jitter_1 = np.random.uniform(0, 1e-6)
        jitter_2 = np.random.uniform(0, 1e-6)
        
        out["F"] = [coupling + jitter_1, -cohesion + jitter_2]

def get_algorithm(name, pop_size=50):
    """
    Factory method to instantiate the 7 different optimizers used in the study.
    """
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    
    # Standard genetic operators used across all algorithms for fair comparison
    ops = dict(
        sampling=IntegerRandomSampling(), 
        crossover=PointCrossover(n_points=2), 
        mutation=BitflipMutation()
    )

    if name == "NSGA-II": return NSGA2(pop_size=pop_size, **ops)
    if name == "NSGA-III": return NSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **ops)
    if name == "MOEA/D": return MOEAD(ref_dirs=ref_dirs, **ops) # pop_size inferred from ref_dirs
    if name == "AGE-MOEA-II": return AGEMOEA2(pop_size=pop_size, **ops)
    if name == "R-NSGA-II": return RNSGA2(ref_points=np.array([[0,0]]), pop_size=pop_size, **ops)
    if name == "U-NSGA-III": return UNSGA3(ref_dirs=ref_dirs, pop_size=pop_size, **ops)
    
    # Random Search: High mutation, no crossover baseline
    if name == "Random Search": 
        return NSGA2(
            pop_size=pop_size, 
            sampling=IntegerRandomSampling(), 
            crossover=PointCrossover(prob=0.0, n_points=2), 
            mutation=BitflipMutation(prob=1.0)
        )
    
    return NSGA2(pop_size=pop_size, **ops)

def semanti_mop_repair(population, G, llm_client, rag_text=""):
    """
    Core Contribution: Instability-Triggered Semantic Repair Operator.
    Identifies 'Borderline Classes' based on Pareto Instability and uses 
    an LLM as a memetic repair operator to steer the search.
    """
    # 1. Extract assignments and handle potential floating point/negative artifacts
    X = np.array([ind.X for ind in population]).astype(int)
    X[X < 0] = 0 
    
    # 2. Instability Detection: Calculate the number of unique assignments per class
    # Higher diversity in assignment for a specific class = higher structural ambiguity.
    instability = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])])
    
    # Trigger threshold: 5% of the population disagreement
    threshold = max(2, int(len(population) * 0.05)) 
    unstable_indices = np.where(instability >= threshold)[0]
    
    if len(unstable_indices) == 0:
        return population, 0, False 

    # Sort classes by instability (most ambiguous first)
    sorted_idx = unstable_indices[np.argsort(instability[unstable_indices])][::-1]
    
    calls_made = 0
    repair_executed = False

    # Process top 5 most unstable classes to maintain execution efficiency
    for idx in sorted_idx[:5]:
        assignments = X[:, idx]
        counts = np.bincount(assignments)
        if len(counts) < 2: continue
        
        # Identify the two clusters that structural metrics are 'confused' between
        top_srv = np.argsort(counts)[-2:] 
        srv_a, srv_b = top_srv[0], top_srv[1]
        
        # 3. Context Extraction for RAG: Use keywords from current cluster members
        rep_assignment = population[0].X
        ctx_a = ", ".join([G.nodes[n].get('name', str(n)) for n, s in enumerate(rep_assignment) if s == srv_a][:5])
        ctx_b = ", ".join([G.nodes[n].get('name', str(n)) for n, s in enumerate(rep_assignment) if s == srv_b][:5])
        
        node_name = G.nodes[idx].get('name', 'UnknownClass')
        node_code = G.nodes[idx].get('code', '')
        
        # 4. LLM Arbitration: Resolve semantic boundary via RAG context
        decision = llm_client.resolve_ambiguity(node_name, node_code, ctx_a, ctx_b, rag_text)
        target = srv_a if decision == 'A' else srv_b
        
        # 5. Steering: Perform Memetic Repair on 50% of the population to adopt the domain decision
        for i in range(len(population) // 2):
            population[i].X[idx] = target
            
        calls_made += 1
        repair_executed = True

    return population, calls_made, repair_executed