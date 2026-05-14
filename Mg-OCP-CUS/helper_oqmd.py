import os
import requests
from pymatgen.core.structure import Structure
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings("ignore")

SAVE_DIR = "./oqmd_cifs"
TARGET_OQMD = 53

os.makedirs(SAVE_DIR, exist_ok=True)

ALLOY_ELEMENTS = ["Al", "Zn", "Y", "Gd", "Nd"]

MIN_STABILITY = 0.001
MAX_STABILITY = 0.10

def fetch_oqmd_metastable_structures():
    print("="*50)
    print(f"🚀 启动 OQMD 亚稳态固溶体自动化爬虫引擎")
    print("="*50)
    print(f"[*] 目标：寻找 {TARGET_OQMD} 个含 {ALLOY_ELEMENTS} 且处于亚稳态的镁合金结构...")
    
    structures_found = []
    
    for alloy in ALLOY_ELEMENTS:
        if len(structures_found) >= TARGET_OQMD:
            break
            
        print(f"\n[*] 正在搜索 Mg-{alloy} 体系...")
        api_url = f"http://oqmd.org/oqmdapi/formationenergy?composition=Mg,{alloy}&limit=100"
        
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", [])
                
                for item in results:
                    stability = item.get("stability", None)
                    
                    if stability is not None and MIN_STABILITY <= stability <= MAX_STABILITY:
                        sites = item.get("sites", [])
                        unit_cell = item.get("unit_cell", [])
                        
                        if not sites or not unit_cell:
                            continue
                        
                        try:
                            if isinstance(unit_cell[0], (int, float)):
                                lattice = [
                                    [unit_cell[0], unit_cell[1], unit_cell[2]],
                                    [unit_cell[3], unit_cell[4], unit_cell[5]],
                                    [unit_cell[6], unit_cell[7], unit_cell[8]]
                                ]
                            else:
                                lattice = [
                                    [unit_cell[0][0], unit_cell[0][1], unit_cell[0][2]],
                                    [unit_cell[1][0], unit_cell[1][1], unit_cell[1][2]],
                                    [unit_cell[2][0], unit_cell[2][1], unit_cell[2][2]]
                                ]
                        except Exception:
                            continue
                        
                        species = []
                        frac_coords = []
                        valid_site = True
                        
                        try:
                            for site in sites:
                                if isinstance(site, dict):
                                    species.append(site.get("label", site.get("species", "X")))
                                    frac_coords.append([float(site["x"]), float(site["y"]), float(site["z"])])
                                elif isinstance(site, str):
                                    parts = site.split("@")
                                    if len(parts) == 2:
                                        element = re.sub(r'[^A-Za-z]', '', parts[0])
                                        species.append(element)
                                        coord_str = parts[1].strip().split()
                                        frac_coords.append([float(c) for c in coord_str])
                                    else:
                                        valid_site = False
                        except Exception:
                            valid_site = False
                            
                        if not valid_site or len(species) == 0:
                            continue
                            
                        try:
                            struct = Structure(lattice, species, frac_coords)
                            struct.properties = {
                                "source": "OQMD", 
                                "formula": struct.composition.reduced_formula,
                                "stability": stability,
                                "entry_id": item.get("entry_id")
                            }
                            
                            structures_found.append(struct)
                            print(f"  [+] 找到候选: {struct.composition.reduced_formula} (Stability: {stability:.4f} eV/atom, ID: {item.get('entry_id')})")
                            
                            if len(structures_found) >= TARGET_OQMD:
                                break
                        except Exception:
                            continue
                            
            else:
                print(f"  [❌] 请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"  [❌] 网络请求异常: {e}")

    return structures_found

if __name__ == "__main__":
    downloaded_structs = fetch_oqmd_metastable_structures()
    
    print("\n" + "="*50)
    print(f"🎉 搜索完毕！共找到 {len(downloaded_structs)} 个符合物理要求且解析成功的亚稳态结构。")
    
    if downloaded_structs:
        print(f"[*] 正在将它们转换为 .cif 文件并保存到 {SAVE_DIR} ...")
        
        for i, struct in enumerate(tqdm(downloaded_structs, desc="导出 CIF")):
            formula = struct.properties["formula"]
            entry_id = struct.properties.get("entry_id", f"unk_{i}")
            
            filename = f"oqmd_{i:02d}_{formula}_{entry_id}.cif"
            save_path = os.path.join(SAVE_DIR, filename)
            
            cif_string = struct.to(fmt="cif")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(cif_string)
    else:
        print("\n[⚠️] 未能找到足够的结构，请检查网络是否通畅。")