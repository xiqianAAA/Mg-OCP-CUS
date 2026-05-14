import os
import warnings
from monty.serialization import loadfn, dumpfn
from pymatgen.core.surface import generate_all_slabs
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_DIR = "./seed_structures"
INPUT_FILE = os.path.join(DATA_DIR, "mp_seed_structures.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "mg_cleaved_slabs.json")

MAX_INDEX = 2 
MIN_SLAB_SIZE = 15.0
MIN_VACUUM_SIZE = 20.0
MAX_ATOMS_LIMIT = 150

def cleave_and_mask_real_slabs(bulk_seeds):
    all_valid_slabs = []
    
    print(f"[*] 准备对 {len(bulk_seeds)} 个块体种子进行高通量晶面切分...")
    
    for i, bulk in enumerate(tqdm(bulk_seeds, desc="切分进度")):
        try:
            sga = SpacegroupAnalyzer(bulk)
            std_bulk = sga.get_conventional_standard_structure()
            
            slabs = generate_all_slabs(
                std_bulk, 
                max_index=MAX_INDEX, 
                min_slab_size=MIN_SLAB_SIZE, 
                min_vacuum_size=MIN_VACUUM_SIZE,
                center_slab=True
            )
            
            for slab in slabs:

                slab.make_supercell([2, 2, 1])

                if len(slab) > MAX_ATOMS_LIMIT:
                    continue
                
                z_coords = [site.coords[2] for site in slab]
                z_min = min(z_coords)
                
                mask = []
                for site in slab:
                    if site.coords[2] > z_min + 6.0:
                        mask.append(1.0)
                    else:
                        mask.append(0.0)
                        
                slab.add_site_property("active_mask", mask)
                all_valid_slabs.append(slab)
                
        except Exception as e:
            continue

    print(f"\n{'-'*40}")
    print(f"🎉 切分完毕！从 {len(bulk_seeds)} 个块体中，成功切分出 {len(all_valid_slabs)} 个带有掩码的表面平板结构！")
    return all_valid_slabs

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到种子文件: {INPUT_FILE}，请确认第一步已成功运行。")
        exit(1)
        
    print("[*] 正在加载块体种子数据...")
    bulk_seeds = loadfn(INPUT_FILE)
    
    cleaved_slabs = cleave_and_mask_real_slabs(bulk_seeds)
    
    print(f"[*] 正在将表面数据序列化保存至: {OUTPUT_FILE}")
    dumpfn(cleaved_slabs, OUTPUT_FILE)
    print("✅ 第二阶段：平板切分与掩码冻结全流程执行成功！")