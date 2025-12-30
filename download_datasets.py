import os
import sys

try:
    from git import Repo
except ImportError:
    print("FATAL ERROR: 'gitpython' library is missing. Run: pip install gitpython")
    sys.exit(1)

# Dataset URLs
DATASETS = {
    "Spring-PetClinic": "https://github.com/spring-projects/spring-petclinic.git",
    "Cargo": "https://github.com/citerus/dddsample-core.git",
    "Shopizer": "https://github.com/shopizer-ecommerce/shopizer.git"
}

def setup_data():
    """
    Clones the required datasets from GitHub into the 'datasets' folder.
    """
    base_dir = "datasets"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for name, url in DATASETS.items():
        target_path = os.path.join(base_dir, name)
        
        # Check if folder exists and has content (to avoid re-cloning)
        if os.path.exists(target_path) and os.listdir(target_path):
            print(f"⚡ {name} is already downloaded.")
            continue
            
        print(f"📥 Cloning {name} from GitHub...")
        try:
            Repo.clone_from(url, target_path, depth=1)
            print(f"✅ {name} cloned successfully.")
        except Exception as e:
            print(f"❌ Failed to clone {name}. Error: {e}")

if __name__ == "__main__":
    setup_data()