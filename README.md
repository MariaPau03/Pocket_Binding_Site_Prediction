# Pocket Binding Prediction

A machine learning pipeline for predicting protein-ligand binding sites from PDB structures, inspired by the P2Rank approach. The pipeline extracts geometric, physicochemical and evolutionary features from protein surfaces and prepares them for training a Random Forest classifier.

---

## Project structure

```
pocket_binding_prediction/
├── data/
│   ├── *.pdb                   # input protein structures
│   ├── fastas/                 # generated FASTA sequences
│   ├── pssms/                  # generated conservation scores
│   ├── pdb_parser.py           # PDB loading and data model
│   └── prepro.py               # PDB → FASTA conversion
├── geometry/
│   ├── neighbors.py            # KDTree-based neighbor search
│   ├── sas.py                  # SAS point generation
│   └── features.py             # feature extraction (39 features)
├── model/
│   └── labels.py               # SAS point labelling (binding / non-binding)
├── output/
│   └── pocket_writer.py        # residue CSV + visualization PDB writer
├── evolution.py                # mock PSSM generator
├── predict.py                  # quick prediction script (score CSV only)
└── main.py                     # full pipeline entry point (train + predict)
```

---

## Installation

```bash
pip install biopython freesasa scipy numpy scikit-learn
```

---

## Usage

**Convert PDB → FASTA:**
```bash
python prepro.py
```
Make sure there is (at least) a PDB file inside the `data/` folder.

**Generate PSSM conservation scores from FASTA:**
```bash
python evolution.py
```
Produces a CSV file of mock conservation scores in `data/pssms/`.

---

### `predict.py` — quick prediction (score summary only)

Loads a trained model and outputs a pocket-level score CSV to `csv/`.

```bash
# Single file
python predict.py data/1GUA.pdb
python predict.py data/subset_chen11/a.001.001.001_1s69a.pdb

# Whole directory
python predict.py data/
```

Output: `csv/<protein_name>_results.csv` with pocket center coordinates and scores.

---

### `main.py` — full pipeline (train + predict + full outputs)

**Training:**
```bash
python main.py train data/chen11/
python main.py train data/chen11/ --pssm_dir data/pssms/
python main.py train data/chen11/ --model_out my_model.pkl
```

**Prediction — single file:**
```bash
python main.py predict data/1GUA.pdb
python main.py predict data/1GUA.pdb --model rf_model.pkl
python main.py predict data/1GUA.pdb --model rf_model.pkl --threshold 0.5
python main.py predict data/1GUA.pdb --model rf_model.pkl --pssm_dir data/pssms/ --output_dir results/
```

**Prediction — whole directory:**
```bash
python main.py predict data/chen11/
python main.py predict data/chen11/ --model rf_model.pkl --threshold 0.4 --output_dir results/
```

**Built-in help:**
```bash
python main.py --help
python main.py train --help
python main.py predict --help
```

| Flag | Mode | Default | Purpose |
|---|---|---|---|
| `--pssm_dir` | both | `None` | Folder with `.pssm` conservation files |
| `--model_out` | train | `rf_model.pkl` | Where to save the trained model |
| `--model` | predict | `rf_model.pkl` | Which model to load |
| `--threshold` | predict | `0.3` | Min probability to count a point as binding |
| `--output_dir` | predict | `output/` | Where to write the output files |

---

### Difference between `predict.py` and `main.py predict`

Use `predict.py` for a quick score summary. Use `main.py` for the full outputs needed for visualization and residue analysis.


---

## Pipeline 

The pipeline runs the following sequential steps for each PDB file:

```
[1] Load protein         pdb_parser.py     → Protein object (atoms, residues, ligands)
[2] Build KDTree         neighbors.py      → Efficient spatial search index
[3] Generate SAS points  sas.py            → Surface point cloud
[4] Generate PSSM        evolution.py      → Conservation scores per residue
[5] Extract features     features.py       → Feature matrix (N_points × 39)
[6] Label points         labels.py         → Binary labels (binding / non-binding)
[7] Train / load model   main.py           → RandomForest classifier
[8] Cluster pockets      main.py           → DBSCAN on high-probability SAS points
[9] Write outputs        pocket_writer.py  → Residue CSV + visualization PDB
```

[1] `python prepro.py` --> Convert PDBs to FASTA
[2] `python evolution.py` --> Obtain from FASTA the PSSMs
[3] `python main.py train data/chen11/ --model_out my_model.pkl > train.log` --> Model training
[4] `python main.py predict data/subset_holo4k/ --model my_model.pkl --threshold 0.4 --output_dir output/ > predict.log` --> Prediction
[5] `pymol results/pockets/target_pockets.pdb` --> Visualizatoin



### Output files per protein (e.g. `1GUA`)

| File | Script | Description |
|---|---|---|
| `data/fastas/1GUA.fasta` | `prepro.py` | Protein sequence in FASTA format |
| `data/pssms/1GUA.pssm` | `evolution.py` | Per-residue conservation scores |
| `csv/1GUA_results.csv` | `predict.py` | Pocket centers, sizes and scores |
| `output/csv/1GUA_residues.csv` | `main.py` | Amino acids involved in each pocket (with coordinates) |
| `output/pockets/1GUA_pockets.pdb` | `main.py` | Visualization file for PyMOL / ChimeraX |

The visualization PDB contains the full protein as `ATOM` records and each predicted pocket as `HETATM` records on a separate chain (B, C, D…), allowing independent colouring in PyMOL or ChimeraX:

```bash
pymol output/pockets/1GUA_pockets.pdb
chimerax output/pockets/1GUA_pockets.pdb
```

---

## Modules

### `data/pdb_parser.py`

Parses a `.pdb` file into three Python objects:

- `Atom` — stores 3D coordinates, element, name and residue info
- `Residue` — groups atoms and computes the geometric center
- `Protein` — top-level container that separates protein atoms from ligand atoms (water molecules are excluded)

```python
protein = Protein("data/1GUA.pdb")
protein.load()

protein.atoms         # list of Atom objects
protein.residues      # list of Residue objects
protein.ligand_atoms  # list of Atom objects (ligands only)
```

### `data/pdb_to_fasta.py`

Converts a PDB file to FASTA format using `Bio.PPBuilder`, which physically traces connected amino acids in the structure. Handles multi-chain proteins.

```python
pdb_to_fasta("data/1GUA.pdb", "data/fastas/")
# → data/fastas/1GUA.fasta
```

### `geometry/neighbors.py`

A thin wrapper around `scipy.spatial.KDTree`. Built once from all atom coordinates and used throughout the pipeline for efficient radius searches.

```python
neighbor_search = NeighborSearch(coords)
indices = neighbor_search.query(point, radius=10.0)
```

### `geometry/sas.py`

Generates the Solvent Accessible Surface (SAS) point cloud:

1. Uses `freesasa` (Shrake-Rupley algorithm) to identify exposed atoms (SASA > threshold)
2. Places random points around each surface atom at a fixed distance
3. Filters out any point that falls inside the protein (closer than 1.2 Å to any atom)

```python
generator  = SASPointGenerator(protein, neighbor_search)
sas_points = generator.generate_SAS()   # numpy array (N, 3)
```

### `geometry/features.py`

Extracts a 39-dimensional feature vector for each SAS point. Features are computed from the local atomic environment within a 10 Å radius.

| Group | Features | Count |
|---|---|---|
| Geometry | Neighbor count, mean/std/min distance, local density, depth, normalized depth | 7 |
| Physicochemical | Hydrophobicity (mean/std), charge (total/pos/neg), H-bond donors/acceptors, amino acid composition (fractions) | 27 |
| Advanced geometry | Surface curvature (PCA), pocket depth (ray casting), hydrophobic patch score, charge dipole magnitude | 4 |
| Evolutionary | PSSM conservation score of nearest residue | 1 |
| **Total** | | **39** |

```python
extractor = FeatureExtractor(protein, neighbor_search, radius=10.0)
extractor.load_pssm("data/pssms/1GUA.pssm")

feature_vector = extractor.extract(sas_point)       # shape: (39,)
feature_matrix = extractor.extract_all(sas_points)  # shape: (N, 39)
```

**Feature descriptions:**

- `neighbor_count` — number of protein atoms within 10 Å; pockets have higher counts than flat surfaces
- `mean_distance / std_distance` — low std indicates a tight enclosed pocket
- `density` — atoms per Å³; deep pockets are denser
- `depth / depth_norm` — distance to protein centroid normalized by protein radius
- `mean_hydrophobicity` — Kyte-Doolittle scale average over neighboring residues
- `total_charge / n_positive / n_negative` — net charge and charge counts of unique neighboring residues at pH 7
- `n_hbond_donors / n_hbond_acceptors` — atom-level counts of hydrogen bond participants
- `aa_fractions` — fraction of each of the 20 canonical amino acids among neighboring residues
- `curvature` — PCA eigenvalue ratio; high value = concave/curved (pocket-like)
- `pocket_depth` — fraction of outward rays blocked by protein atoms (0 = exposed, 1 = fully enclosed)
- `hydrophobic_patch_score` — inverse mean pairwise distance of hydrophobic atoms; high = clustered patch
- `charge_dipole` — magnitude of the spatial charge separation vector
- `conservation_score` — PSSM score of the nearest residue (0 = variable, 1 = conserved)

### `evolution.py`

Generates a mock PSSM CSV with random conservation scores. This is a placeholder for real PSI-BLAST output, designed so that the rest of the pipeline can be tested without configuring a BLAST server.

The output format is:
```
Residue_Index,Residue,Conservation_Score
1,M,0.8543
2,K,0.2341
...
```

To replace with real PSSM data, run PSI-BLAST and parse the output into the same CSV format — no other changes are needed.

---

## Feature lookup tables

All lookup tables are defined at module level in `features.py`:

- `HYDROPHOBICITY` — Kyte-Doolittle scale for all 20 amino acids
- `CHARGE` — formal charge at pH 7 (ARG/LYS +1, HIS +0.1, ASP/GLU -1)
- `HBOND_DONORS` — atom names that donate hydrogen bonds
- `HBOND_ACCEPTORS` — atom names that accept hydrogen bonds
- `AMINO_ACIDS / AA_INDEX` — ordered list and index map for one-hot encoding
- `HYDROPHOBIC_RESIDUES` — set of 9 hydrophobic residue names used for patch scoring

---

## RESULTS

The following image shows the trainig results with the chen11 dataset, lasting aproximately one hour. 

![alt text](training_results.png)

This protein-ligand binding site prediction model shows high reliability but follows a conservative prediction strategy. With a precision of 0.96, the model is extremely accurate when it identifies a binding site, meaning nearly all predicted pockets are likely to be biologically relevant. However, the lower recall of 0.46 indicates that it currently captures less than half of the total true binding surface, likely due to the significant class imbalance where binding points represent only 4.5% of the 2.8 million samples. This "safe" behavior results in a respectable F1-score of 0.62, suggesting that while the model misses some peripheral binding areas, it effectively identifies the core "hotspots" of the pockets.

---

## References

- Xia Y., Pan X., Shen H-B. (2024). *A comprehensive survey on protein-ligand binding site prediction*. Current Opinion in Structural Biology, 86, 102793.
- Krivák R., Hoksza D. (2018). *P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure*. J Cheminformatics, 10, 1–12.
- Jumper J. et al. (2021). *Highly accurate protein structure prediction with AlphaFold*. Nature, 596, 583–589.