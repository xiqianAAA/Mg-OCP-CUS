import os
from monty.serialization import loadfn
from pymatgen.io.cif import CifWriter

input_file = "./seed_structures/mg_relaxed_slabs.json"
relaxed_slabs = loadfn(input_file)

os.makedirs("./vesta_view", exist_ok=True)

for i, slab in enumerate(relaxed_slabs):
    filename = f"./vesta_view/relaxed_slab_{i}.cif"

    cif_string = str(CifWriter(slab))
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cif_string)
        
    print(f"✅ Successfully exported: {filename}")

print("🎉 Complete!")