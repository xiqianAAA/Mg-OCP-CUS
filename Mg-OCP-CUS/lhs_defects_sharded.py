import os
import random
import warnings
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from monty.serialization import loadfn, dumpfn

warnings.filterwarnings("ignore")

DATA_DIR = "./seed_structures"
INPUT_FILE = os.path.join(DATA_DIR, "mg_cleaved_slabs.json")

OUTPUT_SUBDIR = os.path.join(DATA_DIR, "lhs_shards") 
os.makedirs(OUTPUT_SUBDIR, exist_ok=True)

TARGET_NUM_SAMPLES = 125000
SHARD_SIZE = 1000
ALLOY_ELEMENTS = ["Al", "Zn", "Y", "Gd", "Nd"]

MAX_VACANCY_RATIO = 0.15  
MAX_SUBST_RATIO = 0.10    
MAX_ADSORB_RATIO = 0.05   

def apply_defects_to_slab(args):
    slab_idx, original_slab, p = args
    slab = original_slab.copy()
    
    mask = slab.site_properties.get("active_mask", [])
    active_indices = [i for i, m in enumerate(mask) if m > 0.5]
    
    if not active_indices:
        return None 

    num_active = len(active_indices)
    vac_num = int(num_active * p[0] * MAX_VACANCY_RATIO)
    sub_num = int(num_active * p[1] * MAX_SUBST_RATIO)
    ada_num = int(num_active * p[2] * MAX_ADSORB_RATIO)
    
    dopant_idx = int(p[3] * len(ALLOY_ELEMENTS))
    dopant_idx = min(dopant_idx, len(ALLOY_ELEMENTS) - 1)
    dopant = ALLOY_ELEMENTS[dopant_idx]

    random.shuffle(active_indices)

    sub_targets = active_indices[:sub_num]
    for idx in sub_targets:
        slab.replace(idx, dopant)

    z_coords = [site.coords[2] for site in slab]
    top_z = max(z_coords)
    for _ in range(ada_num):
        base_site = slab[random.choice(active_indices)]
        ada_pos = base_site.coords + np.array([0, 0, 2.5])
        slab.append(dopant, ada_pos, coords_are_cartesian=True)
        mask.append(1.0)

    vac_targets = active_indices[sub_num : sub_num+vac_num]
    if vac_targets:
        slab.remove_sites(vac_targets)
        mask = [m for i, m in enumerate(mask) if i not in vac_targets]

    slab.add_site_property("active_mask", mask)
    return slab

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：找不到输入文件 {INPUT_FILE}")
        exit(1)

    print("[*] 正在加载纯净平板数据...")
    base_slabs = loadfn(INPUT_FILE)
    
    print(f"[*] 初始化 LHS 引擎，生成 {TARGET_NUM_SAMPLES} 个高维参数点...")
    sampler = qmc.LatinHypercube(d=4, seed=42)
    lhs_points = sampler.random(n=TARGET_NUM_SAMPLES)
    
    tasks = []
    for i in range(TARGET_NUM_SAMPLES):
        slab_choice = random.choice(base_slabs)
        tasks.append((i, slab_choice, lhs_points[i]))
        
    print(f"[*] 启动多核并行演化 (使用 {cpu_count()} 个逻辑核心)...")
    
    current_shard = []
    shard_count = 0
    processed_count = 0

    with Pool(processes=cpu_count()) as pool:

        for result in tqdm(pool.imap(apply_defects_to_slab, tasks), total=TARGET_NUM_SAMPLES, desc="总进度"):
            if result is not None:
                current_shard.append(result)
                processed_count += 1
            
            if len(current_shard) >= SHARD_SIZE:

                shard_name = f"mg_lhs_part_{str(shard_count).zfill(2)}.json"
                shard_file = os.path.join(OUTPUT_SUBDIR, shard_name)
                
                dumpfn(current_shard, shard_file)
                shard_count += 1
                current_shard = []

        if current_shard:
            shard_name = f"mg_lhs_part_{str(shard_count).zfill(2)}.json"
            shard_file = os.path.join(OUTPUT_SUBDIR, shard_name)
            dumpfn(current_shard, shard_file)
            shard_count += 1

    print(f"\n{'-'*40}")
    print(f"[*] 成功生成结构: {processed_count} 个")
    print(f"[*] 产出分片文件: {shard_count} 个 (每个约包含 {SHARD_SIZE} 个结构)")
    print(f"[*] 存储路径: {os.path.abspath(OUTPUT_SUBDIR)}")