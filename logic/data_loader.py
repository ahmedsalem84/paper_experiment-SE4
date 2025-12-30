import os
import networkx as nx

class DataLoader:
    def __init__(self, llm_client):
        self.llm = llm_client

    def load_real_dataset(self, folder_path):
        G = nx.DiGraph()
        class_registry = {} 
        file_contents = {}
        package_map = {} # Map node_id -> package_name (Ground Truth)

        idx = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".java") and "Test" not in file:
                    class_name = file.replace(".java", "")
                    path = os.path.join(root, file)
                    
                    # Extract Package Name as Ground Truth
                    package = "default"
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            for line in content.splitlines():
                                if line.strip().startswith("package "):
                                    package = line.strip().replace("package ", "").replace(";", "")
                                    break
                            
                            class_registry[class_name] = idx
                            file_contents[idx] = content
                            package_map[idx] = package
                            G.add_node(idx, name=class_name, package=package)
                            idx += 1
                    except:
                        continue
        
        if idx == 0: return None, None

        # Build Edges
        for i in G.nodes:
            content = file_contents[i]
            for name, target_id in class_registry.items():
                if i == target_id: continue
                if name in content:
                    G.add_edge(i, target_id, weight=1)

        return G, package_map