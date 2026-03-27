"""
main.py — Entry point for the binding-site prediction pipeline.

Pipeline:
  1. Parse PDB → Protein object
  2. Build NeighborSearch (KD-tree)
  3. Generate SAS points (FreeSASA + local sampling)
  4. Extract features for every SAS point
  5. Label points (requires ligand in PDB; skipped in predict-only mode)
  6. Train / load a RandomForest model
  7. Cluster high-probability SAS points → pockets
  8. Write outputs:
       • CSV listing residues per pocket   (output/<name>_residues.csv)
       • Visualization PDB for PyMOL/Chimera (output/<name>_pockets.pdb)
"""

import os
import argparse
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.cluster import DBSCAN

# ── Project modules ────────────────────────────────────────────────────────────
from data.pdb_parser import Protein
from geometry.neighbors import NeighborSearch
from geometry.sas import SASPointGenerator
from geometry.features import FeatureExtractor
from model.labels import LabelGenerator
from output.pocket_writer import PocketWriter


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROTEIN PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_protein(pdb_path, pssm_path=None):
    """
    Parse a PDB file and compute per-point features + labels.

    Returns
    -------
    X          : np.ndarray (N, F)  — feature matrix
    y          : np.ndarray (N,)    — binary labels (1 = binding, 0 = not)
                 All zeros when no ligand is present in the structure.
    sas_points : np.ndarray (N, 3) — 3-D coordinates of every SAS point
    protein    : Protein            — parsed protein object (needed by writers)
    ns         : NeighborSearch     — KD-tree built on protein atoms
    """
    print(f"\n[1] Parsing: {pdb_path}")
    protein = Protein(pdb_path)
    print(f"    {len(protein.atoms)} protein atoms | "
          f"{len(protein.ligand_atoms)} ligand atoms | "
          f"{len(protein.residues)} residues")

    print("[2] Building neighbour search …")
    atom_coords = protein.get_atom_coordinates()
    ns = NeighborSearch(atom_coords)

    print("[3] Generating SAS points …")
    sas_gen    = SASPointGenerator(protein, ns)
    sas_points = sas_gen.generate_SAS()
    print(f"    {len(sas_points)} SAS points generated")

    print("[4] Extracting features …")
    extractor = FeatureExtractor(protein, ns)
    if pssm_path and os.path.exists(pssm_path):
        extractor.load_pssm(pssm_path)
    X = extractor.extract_all(sas_points)

    print("[5] Labelling points …")
    labeller = LabelGenerator(protein, ns)
    y = labeller.label_all(sas_points)
    n_binding = int(y.sum())
    print(f"    Binding: {n_binding} / {len(y)}  "
          f"({'no ligand found' if n_binding == 0 else 'OK'})")

    return X, y, sas_points, protein, ns


# ══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTERING  →  POCKET OBJECTS
# ══════════════════════════════════════════════════════════════════════════════

def cluster_points(sas_points, probabilities, threshold=0.3,
                   eps=4.0, min_samples=5):
    """
    Cluster SAS points with probability >= threshold using DBSCAN.

    Returns
    -------
    pockets : list of dicts, each with keys:
        'center'     np.ndarray (3,)
        'size'       int
        'score'      float   (mean probability of cluster members)
        'points'     np.ndarray (K, 3)  — SAS points in this cluster
    """
    mask = probabilities >= threshold
    candidate_points = sas_points[mask]
    candidate_probs  = probabilities[mask]

    if len(candidate_points) == 0:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(candidate_points)

    pockets = []
    for label in set(labels):
        if label == -1:          # noise
            continue
        idx    = labels == label
        pts    = candidate_points[idx]
        probs  = candidate_probs[idx]
        pockets.append({
            "center": pts.mean(axis=0),
            "size":   int(idx.sum()),
            "score":  float(probs.mean()),
            "points": pts,
        })

    # Sort by score descending
    pockets.sort(key=lambda p: p["score"], reverse=True)
    return pockets


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUTS  (residue list + visualization PDB)
# ══════════════════════════════════════════════════════════════════════════════

def write_outputs(pockets, protein, ns, protein_name, output_dir="output"):
    """
    Write the two requested output files for a set of predicted pockets.

    Files created
    -------------
    output/csv/<protein_name>_residues.csv     — amino acids per pocket
    output/pockets/<protein_name>_pockets.pdb  — visualisation file (PyMOL / Chimera)
    """
    if not pockets:
        print("\n[!] No pockets to write.")
        return

    csv_dir = os.path.join(output_dir, "csv")
    pdb_dir = os.path.join(output_dir, "pockets")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(pdb_dir, exist_ok=True)

    writer = PocketWriter(protein, ns)

    # ── 3a. CSV: list of residues per pocket ──────────────────────────────────
    csv_path = os.path.join(csv_dir, f"{protein_name}_residues.csv")
    pocket_centers = [p["center"] for p in pockets]
    print(f"\n[6a] Writing residue list → {csv_path}")
    writer.write_residues_csv(pocket_centers, csv_path)

    # ── 3b. PDB: visualisation for PyMOL / Chimera ───────────────────────────
    pdb_path = os.path.join(pdb_dir, f"{protein_name}_pockets.pdb")
    sas_pts_per_pocket = [p["points"] for p in pockets]
    print(f"[6b] Writing visualisation PDB → {pdb_path}")
    writer.write_visualization_pdb(pocket_centers, sas_pts_per_pocket, pdb_path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING MODE
# ══════════════════════════════════════════════════════════════════════════════

def train(pdb_files, pssm_dir=None, model_out="my_model.pkl"):
    """
    Train a RandomForest on a list of PDB files that contain ligands.
    Saves the trained model to *model_out*.
    """
    all_X, all_y = [], []

    for pdb in pdb_files:
        name      = os.path.splitext(os.path.basename(pdb))[0]
        pssm_path = os.path.join(pssm_dir, f"{name}.pssm") if pssm_dir else None
        X, y, _, _, _ = process_protein(pdb, pssm_path)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        print("No data collected — aborting training.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\nTotal samples: {len(y)}  |  binding: {int(y.sum())}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    print("\n── Evaluation on training set ──")
    print(classification_report(y_te, clf.predict(X_te)))

    joblib.dump(clf, model_out)
    print(f"Model saved → {model_out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION MODE
# ══════════════════════════════════════════════════════════════════════════════

def predict(pdb_path, model_path, pssm_dir=None,
            threshold=0.3, output_dir="output"):
    """
    Run the full prediction pipeline on a single PDB file and write outputs.
    """
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        return

    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")

    name      = os.path.splitext(os.path.basename(pdb_path))[0]
    pssm_path = os.path.join(pssm_dir, f"{name}.pssm") if pssm_dir else None

    X, _, sas_points, protein, ns = process_protein(pdb_path, pssm_path)
    if X is None:
        return

    print("[5] Predicting binding probabilities …")
    probs   = model.predict_proba(X)[:, 1]
    pockets = cluster_points(sas_points, probs, threshold=threshold)

    print(f"\n    Pockets found: {len(pockets)}")
    for i, p in enumerate(pockets[:5], 1):
        print(f"    Pocket {i}: score={p['score']:.3f}  size={p['size']}")

    write_outputs(pockets, protein, ns, name, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Binding-pocket predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Train a new model")
    tr.add_argument("pdb_dir",  help="Directory of training PDB files")
    tr.add_argument("--pssm_dir", default=None, help="Directory of .pssm files")
    tr.add_argument("--model_out", default="my_model.pkl",
                    help="Output path for the saved model")

    # ── predict ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("predict", help="Predict pockets for a PDB file")
    pr.add_argument("pdb_path",  help="Path to a .pdb file or directory of .pdb files")
    pr.add_argument("--model",   default="my_model.pkl", help="Trained model (.pkl)")
    pr.add_argument("--pssm_dir", default=None, help="Directory of .pssm files")
    pr.add_argument("--threshold", type=float, default=0.3,
                    help="Minimum predicted probability to consider a point")
    pr.add_argument("--output_dir", default="output",
                    help="Directory for output files")

    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "train":
        pdb_files = [
            os.path.join(args.pdb_dir, f)
            for f in os.listdir(args.pdb_dir)
            if f.endswith(".pdb")
        ]
        if not pdb_files:
            print(f"No .pdb files found in {args.pdb_dir}")
            return
        train(pdb_files, pssm_dir=args.pssm_dir, model_out=args.model_out)

    elif args.mode == "predict":
        # Accept a single file or a whole directory
        targets = []
        if os.path.isdir(args.pdb_path):
            targets = [
                os.path.join(args.pdb_path, f)
                for f in os.listdir(args.pdb_path)
                if f.endswith(".pdb")
            ]
        elif os.path.isfile(args.pdb_path):
            targets = [args.pdb_path]
        else:
            print(f"Path not found: {args.pdb_path}")
            return

        for pdb in sorted(targets):
            predict(
                pdb_path    = pdb,
                model_path  = args.model,
                pssm_dir    = args.pssm_dir,
                threshold   = args.threshold,
                output_dir  = args.output_dir,
            )
        print("\nDone.")


if __name__ == "__main__":
    main()
