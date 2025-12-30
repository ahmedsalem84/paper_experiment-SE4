import numpy as np
from scipy.stats import rankdata

def calculate_mojofm(individual, G, node_to_package_map):
    """
    Calculates MoJoFM (Move-Join) distance between the generated decomposition
    and the original package structure (Ground Truth). 100% is a perfect match.
    """
    if not node_to_package_map:
        return 0

    # 1. Group nodes by Generated Service
    generated_clusters = {}
    for node_idx, service_id in enumerate(individual):
        generated_clusters.setdefault(service_id, set()).add(node_idx)

    # 2. Group nodes by Ground Truth (Package)
    true_clusters = {}
    for node_idx, package_name in node_to_package_map.items():
        true_clusters.setdefault(package_name, set()).add(node_idx)

    # 3. Compute accuracy (Jaccard-based MoJo approximation)
    matches = 0
    for g_set in generated_clusters.values():
        max_overlap = 0
        for t_set in true_clusters.values():
            overlap = len(g_set.intersection(t_set))
            if overlap > max_overlap:
                max_overlap = overlap
        matches += max_overlap
    
    n = len(node_to_package_map)
    mojofm = (matches / n) * 100
    return mojofm

def vargha_delaney(m, n):
    """
    Computes Vargha-Delaney A12 effect size.
    A12 > 0.5 means group 'n' is generally better than group 'm'.
    """
    m = list(m)
    n = list(n)
    if not m or not n:
        return 0.5
    
    r = rankdata(m + n)
    r1 = sum(r[:len(m)])
    m_len, n_len = len(m), len(n)
    return (r1 / m_len - (m_len + 1) / 2) / n_len

def calculate_hypervolume(F):
    """
    Calculates the Hypervolume of a Pareto Front F.
    FIXED: Handles negative objectives (Cohesion) correctly.
    """
    try:
        from pymoo.indicators.hv import HV
        
        if F is None or len(F) == 0:
            return 0.0
            
        # Ensure F is a 2D numpy array
        F_array = np.array(F)
        if F_array.ndim == 1:
            F_array = F_array.reshape(1, -1)

        # FIND THE REFERENCE POINT:
        # For minimization, the reference point must be numerically GREATER 
        # than the maximum value in the front for each objective.
        max_objs = np.max(F_array, axis=0)
        
        # We add 10% of the absolute value plus a constant offset of 1.0
        # This ensures that even if max is 0, the reference point is worse (1.0).
        # This works for positive coupling and negative cohesion.
        ref_point = max_objs + np.abs(max_objs) * 0.1 + 1.0
        
        # Initialize Pymoo Hypervolume indicator
        ind = HV(ref_point=ref_point)
        hv_value = ind(F_array)
        
        return float(hv_value)
        
    except Exception as e:
        # In a real run, you can log this, but for the dashboard we return 0
        return 0.0