import os
import glob
import warnings
import numpy as np
from monty.serialization import loadfn
from tqdm import tqdm
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

torch.set_num_threads(1)
warnings.filterwarnings("ignore")

DATA_DIR = "./seed_structures"
INPUT_DIR = os.path.join(DATA_DIR, "relaxed_shards") 
OUTPUT_DIR = "./dataset_pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CN_MAX = 12.0 
CUTOFF_RADIUS = 4.0

def extract_cus_features(slab, global_index):
    num_atoms = len(slab)
    
    frac_coords = slab.frac_coords
    cart_coords = slab.cart_coords
    lat_mat = slab.lattice.matrix
    atom_types = np.array([site.specie.Z for site in slab], dtype=np.int64)
    
    mask_list = slab.site_properties.get("active_mask", [1.0] * num_atoms)
    mask = np.array(mask_list, dtype=np.float32)

    diff = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]
    
    diff -= np.round(diff)
    
    cart_diff = diff @ lat_mat
    
    dist_matrix = np.linalg.norm(cart_diff, axis=-1)
    
    src_indices, dst_indices = np.where((dist_matrix <= CUTOFF_RADIUS) & (dist_matrix > 0.1))
    
    CN_array = np.zeros(num_atoms, dtype=np.float32)
    for i in range(num_atoms):
        CN_array[i] = np.sum(src_indices == i)
    
    c_GCN = np.zeros(num_atoms, dtype=np.float32)
    for i in range(num_atoms):
        neighbors_of_i = dst_indices[src_indices == i]
        sum_cn_j = np.sum(CN_array[neighbors_of_i])
        c_GCN[i] = sum_cn_j / CN_MAX

    c_chem = np.zeros((num_atoms, 2), dtype=np.float32)
    for i, site in enumerate(slab):
        el = site.specie
        c_chem[i, 0] = float(el.atomic_mass)
        try:
            elec_neg = float(el.X)
        except (ValueError, TypeError, AttributeError):
            elec_neg = 1.0
        c_chem[i, 1] = elec_neg

    z_coords = cart_coords[:, 2]
    z_max = np.max(z_coords) if np.max(z_coords) > 0.1 else 1.0
    phi_v = (z_coords / z_max).reshape(-1, 1)

    gamma_val = slab.properties.get("formation_energy_jm2", 0.0)
    energy = float(gamma_val)

    data_dict = {
        'frac_coords': torch.tensor(frac_coords, dtype=torch.float32),
        'cart_coords': torch.tensor(cart_coords, dtype=torch.float32),
        'atom_types': torch.tensor(atom_types, dtype=torch.long),
        'mask': torch.tensor(mask, dtype=torch.float32),
        'c_GCN': torch.tensor(c_GCN, dtype=torch.float32),
        'c_chem': torch.tensor(c_chem, dtype=torch.float32),
        'phi_v': torch.tensor(phi_v, dtype=torch.float32),
        'energy': torch.tensor(energy, dtype=torch.float32),
        'edge_index_voro': torch.tensor(np.vstack((src_indices, dst_indices)), dtype=torch.long)
    }
    
    save_path = os.path.join(OUTPUT_DIR, f"mg_cus_{global_index}.pt")
    torch.save(data_dict, save_path)
    return save_path

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录: {INPUT_DIR}")
        exit(1)
        
    shard_files = sorted(glob.glob(os.path.join(INPUT_DIR, "mg_relaxed_part_*.json")))
    if not shard_files:
        print(f"❌ 在 {INPUT_DIR} 中没有找到分片文件！")
        exit(1)
        
    print(f"[*] 发现 {len(shard_files)} 个弛豫分片文件。")
    print("[*] 核心系统已接管：采用纯 Numpy 矩阵空间投影算法 (物理防崩版)。")
    
    global_counter = 0  
    
    for shard_path in shard_files:
        shard_filename = os.path.basename(shard_path)
        relaxed_slabs = loadfn(shard_path)
        
        for slab in tqdm(relaxed_slabs, desc=f"提取 {shard_filename}"):
            extract_cus_features(slab, global_counter)
            global_counter += 1
            
    print(f"\n{'-'*40}")
    print(f"[*] 共生成了 {global_counter} 个 .pt 文件。")