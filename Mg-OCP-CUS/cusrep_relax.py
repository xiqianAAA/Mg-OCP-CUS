import os
import glob
import warnings
import numpy as np
import torch
from monty.serialization import loadfn, dumpfn
from tqdm import tqdm
from pymatgen.core.structure import Structure
import json
from cusrep_model import CUSRepModel
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")

DATA_DIR = "./seed_structures"
INPUT_DIR = os.path.join(DATA_DIR, "lhs_shards")
OUTPUT_DIR = os.path.join(DATA_DIR, "relaxed_shards")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MU_MG = -1.5
MU_DOPANTS = {
    "Al": -3.5,
    "Zn": -1.2,
    "Y": -6.5,
    "Gd": -5.0,
    "Nd": -4.8
}

FMAX = 0.05
MAX_STEPS = 200

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] 当前使用设备: {device}")

class CUSRepASECalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        cell = torch.tensor(atoms.get_cell()[:], dtype=torch.float32, device=self.device)

        positions.requires_grad_(True)
        
        energy_pred = self.model(positions, atomic_numbers, cell)

        forces_pred = -torch.autograd.grad(
            outputs=energy_pred,
            inputs=positions,
            grad_outputs=torch.ones_like(energy_pred),
            create_graph=False,
            retain_graph=False
        )[0]

        self.results['energy'] = energy_pred.detach().cpu().numpy().item()
        self.results['forces'] = forces_pred.detach().cpu().numpy()


class CUSRepOptimizer:

    def __init__(self, device):
        self.device = device
        
        self.model = CUSRepModel().to(self.device)
        self.model.load_state_dict(torch.load("best_weights.pth"))
        self.model.eval() 

    def relax(self, structure: Structure, fmax: float, steps: int):

        atoms = AseAtomsAdaptor.get_atoms(structure)

        calc = CUSRepASECalculator(model=self.model, device=self.device)
        atoms.calc = calc

        opt = LBFGS(atoms, logfile=None)
        
        opt.run(fmax=fmax, steps=steps)

        final_energy = atoms.get_potential_energy()
        final_structure = AseAtomsAdaptor.get_structure(atoms)

        return {
            'final_structure': final_structure,
            'final_energy': final_energy
        }

optimizer = CUSRepOptimizer(device=device)

def calculate_surface_energy(slab, relaxed_energy):

    a_vec = slab.lattice.matrix[0]
    b_vec = slab.lattice.matrix[1]
    area = np.linalg.norm(np.cross(a_vec, b_vec))
    
    comp = slab.composition
    n_mg = comp.get("Mg", 0)
    
    mu_sum = n_mg * MU_MG
    for el in MU_DOPANTS.keys():
        n_el = comp.get(el, 0)
        mu_sum += n_el * MU_DOPANTS[el]
        
    ev_per_ang2 = (relaxed_energy - mu_sum) / (2 * area)
    gamma_jm2 = ev_per_ang2 * 16.02176634
    
    return gamma_jm2

def relax_slabs(unrelaxed_slabs, shard_name):
    relaxed_results = []
    failed_count = 0
    metadata_log = []
    
    for i, slab in enumerate(tqdm(unrelaxed_slabs, desc=f"CUS-Rep 弛豫 {shard_name}")):
        try:
            opt_result = optimizer.relax(
                slab, 
                fmax=FMAX, 
                steps=MAX_STEPS
            )
            
            final_structure = opt_result['final_structure']
            final_energy = opt_result['final_energy'] 
            
            original_mask = slab.site_properties.get("active_mask", [])
            final_structure.add_site_property("active_mask", original_mask)
            
            gamma = calculate_surface_energy(final_structure, final_energy)
            final_structure.properties = {"formation_energy_jm2": gamma}
            
            relaxed_results.append(final_structure)
            
            metadata_log.append({
                "index": i,
                "formula": final_structure.formula,
                "energy_ev": float(final_energy),
                "gamma_jm2": float(gamma)
            })
            
        except Exception as e:
            failed_count += 1
            print(e)
            pass
            
    return relaxed_results, metadata_log, failed_count

if __name__ == "__main__":
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录: {INPUT_DIR}，请先执行前置步骤。")
        exit(1)
        
    shard_files = sorted(glob.glob(os.path.join(INPUT_DIR, "mg_lhs_part_*.json")))
    
    if not shard_files:
        print(f"❌ 在 {INPUT_DIR} 中没有找到任何分片文件！")
        exit(1)
        
    print(f"[*] 发现 {len(shard_files)} 个待弛豫的分片文件。")
    print(f"[*] 全量生产模式已开启。\n")

    total_success = 0
    total_failed = 0
    total_processed = 0

    for shard_path in shard_files:
        shard_filename = os.path.basename(shard_path)
        
        out_filename = shard_filename.replace("lhs", "relaxed")
        log_filename = shard_filename.replace("mg_lhs_", "log_").replace(".json", "_relax.json")
        
        output_file = os.path.join(OUTPUT_DIR, out_filename)
        log_file = os.path.join(OUTPUT_DIR, log_filename)
        
        if os.path.exists(output_file):
            print(f"⏭️ {out_filename} 已存在，跳过弛豫。")
            continue
            
        unrelaxed_slabs = loadfn(shard_path)
        total_processed += len(unrelaxed_slabs)
        
        relaxed_slabs, log_data, failed_count = relax_slabs(unrelaxed_slabs, shard_filename)
        
        dumpfn(relaxed_slabs, output_file)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
            
        total_success += len(relaxed_slabs)
        total_failed += failed_count

    print(f"\n{'-'*40}")
    print(f"✅ 第四阶段全量弛豫 (CUS-Rep 驱动) 执行完毕！")
    print(f"[*] 本次运行共处理结构: {total_processed} 个。")
    print(f"[*] 成功收敛: {total_success} 个，失败/未收敛: {total_failed} 个。")
    print(f"[*] 结果已保存至目录: {OUTPUT_DIR}")