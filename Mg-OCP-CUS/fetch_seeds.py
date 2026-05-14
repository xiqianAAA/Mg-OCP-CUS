import os
import glob
import json
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from monty.serialization import dumpfn
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

API_KEY = ""
SAVE_DIR = "./seed_structures"

OQMD_DIR = "./oqmd_cifs"
ICSD_DIR = "./icsd_cifs"

ALLOY_ELEMENTS = ["Al", "Zn", "Y", "Gd", "Nd"]
MAX_E_HULL = 0.05 

TARGET_MP = 85
TARGET_OQMD = 32
TARGET_ICSD = 15

def fetch_real_mp_seeds(api_key):

    seeds = []
    metadata_log = []
    
    with MPRester(api_key) as mpr:

        docs_mg = mpr.materials.summary.search(
            elements=["Mg"], 
            num_elements=(1, 1),
            energy_above_hull=(0.0, MAX_E_HULL),
            fields=["material_id", "structure", "formula_pretty", "energy_above_hull"]
        )
        
        for doc in docs_mg:
            seeds.append(doc.structure)
            metadata_log.append({
                "id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "e_hull": doc.energy_above_hull,
                "source": "MP"
            })
        
        for alloy in ALLOY_ELEMENTS:
            docs_alloy = mpr.materials.summary.search(
                elements=["Mg", alloy],
                num_elements=(2, 2),
                energy_above_hull=(0.0, MAX_E_HULL),
                fields=["material_id", "structure", "formula_pretty", "energy_above_hull"]
            )
            
            for doc in docs_alloy:
                seeds.append(doc.structure)
                metadata_log.append({
                    "id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "e_hull": doc.energy_above_hull,
                    "source": "MP"
                })

    seeds = seeds[:TARGET_MP]
    metadata_log = metadata_log[:TARGET_MP]
    
    return seeds, metadata_log

def load_local_cifs(directory, source_name, target_count):

    structures = []
    metadata_log = []
    
    cif_files = glob.glob(os.path.join(directory, "*.cif"))
    
    if len(cif_files) == 0:
        return [], []
        
    for cif in tqdm(cif_files[:target_count], desc=f"解析 {source_name}"):
        try:
            struct = Structure.from_file(cif)
            filename = os.path.basename(cif)
            struct.properties = {"source": source_name, "filename": filename}
            
            structures.append(struct)
            metadata_log.append({
                "id": filename,
                "formula": struct.composition.reduced_formula,
                "e_hull": "N/A (Local)",
                "source": source_name
            })
        except Exception as e:
            print(f"[❌] failed to load {cif}: {e}")
            
    return structures, metadata_log

if __name__ == "__main__":

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OQMD_DIR, exist_ok=True)
    os.makedirs(ICSD_DIR, exist_ok=True)
    
    print("="*50)
    print("🚀 Starting multi-source seed structure pool (MP + OQMD + ICSD) integration engine")
    print("="*50)
    
    mp_seeds, mp_metadata = fetch_real_mp_seeds(API_KEY)
    
    oqmd_seeds, oqmd_metadata = load_local_cifs(OQMD_DIR, "OQMD", TARGET_OQMD)
    icsd_seeds, icsd_metadata = load_local_cifs(ICSD_DIR, "ICSD", TARGET_ICSD)
    
    all_seeds = mp_seeds + oqmd_seeds + icsd_seeds
    all_metadata = mp_metadata + oqmd_metadata + icsd_metadata
    
    print("\n" + "="*50)
    print(f"🎉 Integration complete! The seed structure pool contains a total of {len(all_seeds)} configurations.")
    print(f"  - MP (Ground State): {len(mp_seeds)}")
    print(f"  - OQMD (Metastable): {len(oqmd_seeds)}")
    print(f"  - ICSD (Experimental): {len(icsd_seeds)}")
    
    if len(all_seeds) < (TARGET_MP + TARGET_OQMD + TARGET_ICSD):
        print(f"💡 Note: The current count ({len(all_seeds)}) is below the target of 132 specified in the paper. Please ensure the {OQMD_DIR} and {ICSD_DIR} directories contain enough .cif files.")
    
    save_path = os.path.join(SAVE_DIR, "all_mixed_seeds.json")
    dumpfn(all_seeds, save_path)
    
    log_path = os.path.join(SAVE_DIR, "seed_metadata_log.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n[*] 100% complete seed pool successfully saved to {save_path}")