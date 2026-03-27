import numpy as np


class LabelGenerator:
    def __init__(self, protein, neighbor_search):
        """
        Initialize the label generator.
        Extracts and stores the 3D coordinates of all ligand atoms.
        """
        self.protein = protein
        self.neighbor_search = neighbor_search

        # Extract coordinates of ligand atoms (shape: N_ligand_atoms × 3)
        self.ligand_coords = np.array([a.coord for a in protein.ligand_atoms])

    def label_point(self, point, threshold=4.0):
        """
        Assign a binary label to a single SAS point.

        - point : 3D coordinates of the SAS point
        - threshold : If the point is closer than this distance to any ligand atom, it is considered part of a binding site.

        A SAS point is labeled as "binding (1)" if it lies close enough to the ligand.
        """

        # If there are no ligands in the structure
        if len(self.ligand_coords) == 0:
            return 0        # No ligand → no binding site → all points labeled as non-binding (0)

        # Compute distances from the point to all ligand atoms
        dists = np.linalg.norm(self.ligand_coords - point, axis=1)      # array of distances (N_ligand_atoms,)

        # Take the minimum distance
        min_dist = np.min(dists)    # closest ligand atom distance

        # Assign label based on distance threshold -> 1 (binding), 0 (non-binding)
        return 1 if min_dist < threshold else 0

    def label_all(self, points):
        """
        Label all SAS points.
        
        Applies label_point() to each SAS point to generate the full label vector.
        Return: Array of labels (N_points) which will be used as the target variable (y) for machine learning.
        """
        return np.array([self.label_point(p) for p in points])