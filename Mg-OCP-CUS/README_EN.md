- # Magnesium Alloy CUS Defect Surface Data Processing Guide

  This project is dedicated to building a high-fidelity dataset (Mg-OCP-CUS) containing **15,421** magnesium alloy surface configurations with Coordinatively Unsaturated Sites (CUS) defects from scratch.

  ## 📂 Directory Structure & Status

  ```text
  .
  ├── dataset_pt/           # [Destination] Stores the final generated PyTorch .pt tensor files (To be generated)
  ├── seed_structures/      # [Data Hub] Stores intermediate crystal structure files across the pipeline
  │   ├── lhs_shards/       # [To be generated] Reserved for the 125 LHS defect shard files
  │   └── relaxed_shards/   # [To be generated] Reserved for relaxed final structures and execution logs
  ├── vesta_view/           # [Tool] Directory for storing .cif files for 3D visualization in VESTA
  ├── oqmd_cifs/            # [Raw Data] Stores metastable solid solution .cif files fetched from OQMD (53 files)
  ├── icsd_cifs/            # [Raw Data] Stores experimental high-precision .cif files from ICSD/COD (22 files)
  ├── fetch_seeds.py        # Step 1: Multi-source seed pool integration (MP + OQMD + ICSD)
  ├── cleave_slabs.py       # Step 2: Pure surface cleavage and vacuum layer construction
  ├── lhs_defects_sharded.py # Step 3: LHS defect evolution and parallel sharding
  ├── cusrep_relax.py       # Step 4: CUS-Rep physical relaxation and energy labeling
  ├── feature_extraction.py # Step 5: Matrix-driven physical feature extraction and .pt tensorization
  └── helper_oqmd.py        # [Helper] Automated OQMD metastable structure scraper
  ```

  ------

  ## 🚀 Full-Pipeline Execution Manual (Step-by-Step)

  Please execute the scripts in the terminal strictly in the following order. **Prerequisite**: Ensure that your `oqmd_cifs/` (can be auto-fetched via `helper_oqmd.py`) and `icsd_cifs/` (manually downloaded) folders are populated with the required `.cif` crystal files.

  ### 🟢 Phase 1: Basic Data Generation (Steps 1-3)

  **Step 1: Seed Structure Integration**

  - **Command**: `python fetch_seeds.py`
  - **Goal**: Extract and merge ground-state data from the Materials Project with local metastable and experimental data.
  - **Output**: `seed_structures/all_mixed_seeds.json` and corresponding log files.

  **Step 2: Pure Surface Cleavage**

  - **Command**: `python cleave_slabs.py`
  - **Goal**: Cleave seed structures along different close-packed planes and add vacuum layers to generate ideal, pure slabs.
  - **Output**: `seed_structures/mg_cleaved_slabs.json`.

  **Step 3: LHS Defect Sampling & Parallel Evolution**

  - **Command**: `python lhs_defects_sharded.py`
  - **Goal**: Introduce complex defects (vacancies, substitutions, adsorptions) and generate the initial target-scale configurations in parallel.
  - **Output**: **125 shard files** (totaling ~125,000 structures) generated in the `seed_structures/lhs_shards/` directory.

  ------

  ### 🟡 Phase 2: Neural Force Field Relaxation (Step 4)

  **Step 4: Execute Relaxation via CUS-Rep Black-Box Optimizer** Before running the full scale, ensure there are no truncation limits in the code (i.e., **DO NOT** limit `unrelaxed_slabs = unrelaxed_slabs[:2]`).

  - **Command**: `python cusrep_relax.py`
  - **Goal**: Iteratively load the 125 shards, utilize the CUS-Rep surrogate model combined with the ASE optimizer to find local energy minima, and calculate the generalized surface defect formation energy.
  - **Time Warning**: Relaxing 125,000 structures is computationally intensive and is estimated to take **6-7 days** (depending on GPU capabilities). It is highly recommended to run this in the background on a server using `nohup` or `tmux`.
  - **Output**: 125 relaxed shard files and JSON logs generated in the `seed_structures/relaxed_shards/` directory.

  ------

  ### 🔵 Phase 3: Tensorization & Dataset Closure (Step 5)

  **Step 5: Physical Feature Extraction** Once Step 4 is completely finished (or if you wish to extract features from already completed shards):

  - **Command**: `python feature_extraction.py`
  - **Goal**: Based on a pure Numpy Minimum Image Convention (MIC) matrix algorithm, extract Voronoi topology graphs, Generalized Coordination Numbers (GCN), and coordinate fields, converting them into PyTorch-readable `.pt` tensor dictionaries.
  - **Output**: A complete tensor dataset ranging from `mg_cus_0.pt` to `mg_cus_124999.pt` generated in the `dataset_pt/` directory.

  ------

  ## ⚠️ Core Precautions & Troubleshooting

  1. **Checkpoint Resume**: `cusrep_relax.py` features a built-in resume capability. If Step 4 stops midway due to power loss or VRAM overflow, simply re-run the script. It will automatically skip any successfully generated `mg_relaxed_part_XX.json` files.
  2. **Lock Data Consistency**: Once Step 3 is completed and LHS shards are generated, **ABSOLUTELY DO NOT** modify the parameters in Step 3 to regenerate them. Doing so will cause severe misalignment between the initial structures and the subsequent feature extraction/energy labels.
  3. **High Storage Requirements**: Due to the inclusion of high-dimensional graph structures and masks, the 125,000 `.pt` tensor files are expected to consume **40-60 GB** of disk space. Please ensure sufficient storage capacity is available beforehand.
  4. **Ultimate Data Funneling**:  After CUS-Rep convergence checks, the removal of anomalous configurations (e.g., structural disintegration due to extreme atomic overlap during relaxation), and final physical consistency cleaning, we will precisely filter down to **15,421** high-quality CUS defect surface structures required for the final Mg-OCP-CUS dataset.

  We provide a representative benchmark subset (e.g., N configurations) in this repository to ensure the full reproducibility of the CUS-Diff pipeline and for immediate community evaluation. Due to file size constraints and ongoing subsequent research, the complete generation of the 15,421 configurations can be exactly reproduced using the provided lhs_defects_sharded.py and cusrep_relax.py scripts. 
