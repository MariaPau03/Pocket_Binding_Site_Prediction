import joblib
import os
import argparse
import numpy as np
import pandas as pd
from pocket_binding_prediction.main import process_protein, cluster_points 

def run_prediction(pdb_path, model_path):
    output_dir = "csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Carpeta creada: {output_dir}")

    protein_name = os.path.splitext(os.path.basename(pdb_path))[0]
    output_csv = os.path.join(output_dir, f"{protein_name}_results.csv")

    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"--- Modelo cargado: {model_path} ---")
    X, y_true, sas_points = process_protein(pdb_path)

    if X is None:
        return
    probs = model.predict_proba(X)[:, 1]
    
    pockets = cluster_points(sas_points, probs, threshold=0.3)

    print(f"\nResultados para: {protein_name}")
    print(f"Bolsillos encontrados: {len(pockets)}")

    results = []
    for i, p in enumerate(pockets):
        results.append({
            "protein": protein_name,
            "pocket_id": i + 1,
            "center_x": p['center'][0],
            "center_y": p['center'][1],
            "center_z": p['center'][2],
            "size": p['size'],
            "score": p['score']
        })
        if i < 3: 
            print(f" Pocket {i+1}: Score={p['score']:.2f}, Size={p['size']}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n[OK] Resultados guardados en: {output_csv}")
    else:
        print("\n[!] No se detectaron bolsillos significativos con el umbral actual.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path", help="Path to a PDB file or directory containing PDB files")
    parser.add_argument("--model", default="rf_model.pkl", help="Path to the model .pkl file")
    args = parser.parse_args()

    if not args.pdb_path:
        print("Error: Provide a PDB file or directory path")
        exit(1)

    if os.path.isdir(args.pdb_path):
        # Directory mode: process all .pdb files in the directory
        pdb_files = [
            os.path.join(args.pdb_path, f)
            for f in os.listdir(args.pdb_path)
            if f.endswith('.pdb')
        ]
        if not pdb_files:
            print(f"No PDB files found in directory '{args.pdb_path}'")
            exit(1)

        print(f"Found {len(pdb_files)} PDB file(s): {[os.path.basename(f) for f in pdb_files]}")

        for pdb_file in sorted(pdb_files):
            run_prediction(pdb_file, args.model)

        print("\nAll predictions completed.")

    elif os.path.isfile(args.pdb_path):
        # Single file mode
        if not args.pdb_path.endswith('.pdb'):
            print("Error: File must have .pdb extension")
            exit(1)
        run_prediction(args.pdb_path, args.model)

    else:
        print(f"Error: Path '{args.pdb_path}' does not exist.")
        exit(1)