# output/pocket_writer.py
import numpy as np
import csv
import os


class PocketWriter:
    def __init__(self, protein, neighbor_search, radius=5.0):
        """
        protein         : Protein object from pdb_parser.py
        neighbor_search : NeighborSearch built from protein atom coords
        radius          : distance threshold to assign residues to a pocket (Å)
        """
        self.protein         = protein
        self.neighbor_search = neighbor_search
        self.radius          = radius

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def get_pocket_residues(self, pocket_center):
        """
        Given a 3D pocket center, return all unique residues within self.radius.
        Returns a list of dicts with residue info.
        """
        neighbor_idx = self.neighbor_search.query(pocket_center, self.radius)

        seen = set()
        residues = []

        for i in neighbor_idx:
            atom = self.protein.atoms[i]
            key  = (atom.chain_id, atom.residue_id)

            if key not in seen:
                seen.add(key)
                residues.append({
                    "chain":        atom.chain_id,
                    "residue_id":   atom.residue_id,
                    "residue_name": atom.residue_name,
                })

        # Sort by chain then residue number for readability
        residues.sort(key=lambda r: (r["chain"], r["residue_id"]))
        return residues

    def write_residues_csv(self, pockets, output_path):
        """
        Write a CSV listing all residues involved in each predicted pocket.

        pockets : list of 3D numpy arrays (pocket centers)

        Output columns: pocket_id, chain, residue_id, residue_name,
                        center_x, center_y, center_z
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build a lookup: (chain_id, residue_id) -> Residue object for fast access
        residue_lookup = {
            (res.chain_id, res.residue_id): res
            for res in self.protein.residues
        }

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "pocket_id", "chain", "residue_id", "residue_name",
                    "center_x", "center_y", "center_z"
                ]
            )
            writer.writeheader()

            for pocket_id, center in enumerate(pockets, start=1):
                residues = self.get_pocket_residues(center)

                if len(residues) == 0:
                    print(f"  Warning: pocket {pocket_id} has no nearby residues")
                    continue

                for res in residues:
                    key = (res["chain"], res["residue_id"])
                    res_obj = residue_lookup.get(key)
                    if res_obj is not None:
                        cx, cy, cz = res_obj.get_center()
                    else:
                        cx, cy, cz = float("nan"), float("nan"), float("nan")

                    writer.writerow({
                        "pocket_id":    pocket_id,
                        "chain":        res["chain"],
                        "residue_id":   res["residue_id"],
                        "residue_name": res["residue_name"],
                        "center_x":     round(float(cx), 3),
                        "center_y":     round(float(cy), 3),
                        "center_z":     round(float(cz), 3),
                    })

                print(f"  Pocket {pocket_id}: {len(residues)} residues "
                      f"({residues[0]['residue_name']}{residues[0]['residue_id']} "
                      f"... {residues[-1]['residue_name']}{residues[-1]['residue_id']})")

        print(f"\nResidue list saved to {output_path}")

    def write_visualization_pdb(self, pockets, sas_points_per_pocket, output_path):
        """
        Write a PDB file with two types of records:
          - ATOM   : the original protein atoms (so the protein is visible)
          - HETATM : the predicted pocket SAS points (one chain per pocket)

        This lets PyMOL/ChimeraX show the protein + coloured pockets together.

        pockets              : list of 3D pocket centers (numpy arrays)
        sas_points_per_pocket: list of numpy arrays, one per pocket
                               (the SAS points belonging to each pocket cluster)
        output_path          : path to write the .pdb file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Pocket chains: use letters B, C, D... (A is reserved for the protein)
        pocket_chains = "BCDEFGHIJKLMNOPQRSTUVWXYZ"

        with open(output_path, "w") as f:

            # ── 1. Write protein atoms as ATOM records ────────────────────
            for i, atom in enumerate(self.protein.atoms, start=1):
                x, y, z = atom.coord
                f.write(
                    f"ATOM  {i:5d} {atom.name:<4s} {atom.residue_name:>3s} "
                    f"{atom.chain_id}{atom.residue_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                    f"{atom.element:>2s}\n"
                )

            f.write("TER\n")

            # ── 2. Write pocket SAS points as HETATM records ──────────────
            hetatm_idx = 1

            for pocket_id, (center, sas_pts) in enumerate(
                zip(pockets, sas_points_per_pocket), start=1
            ):
                chain = pocket_chains[pocket_id - 1] if pocket_id <= 25 else "Z"

                # Write the pocket center as a special marker (element Xe)
                x, y, z = center
                f.write(
                    f"HETATM{hetatm_idx:5d}  Xe  PCT "
                    f"{chain}{pocket_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          Xe\n"
                )
                hetatm_idx += 1

                # Write each SAS point belonging to this pocket (element C)
                for pt in sas_pts:
                    x, y, z = pt
                    f.write(
                        f"HETATM{hetatm_idx:5d}  C   PKT "
                        f"{chain}{pocket_id:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                    )
                    hetatm_idx += 1

            f.write("END\n")

        print(f"Visualization PDB saved to {output_path}")