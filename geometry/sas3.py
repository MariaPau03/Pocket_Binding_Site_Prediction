import numpy as np
import freesasa


class SASPointGenerator:
    def __init__(self, protein, neighbor_search):
        self.protein = protein
        self.neighbor_search = neighbor_search

    def generate_SAS(self, sasa_threshold=1.0, n_points=20, distance=1.5):
        """
        Generate SAS points using FreeSASA + local sampling.
        Steps:
        1 - Identify surface atoms using FreeSASA
        2 - For each surface atom, generate points around it
        """
        # 1 - Identify surface atoms using FreeSASA
        surface_atoms = self._get_surface_atoms(sasa_threshold)

        sas_points = []

        for atom in surface_atoms:
            # 2 - Generate points around each surface atom
            points = self._generate_points_around_atom(atom, n_points, distance)
            sas_points.extend(points)

        return np.array(sas_points)
    
    def _get_surface_atoms(self, sasa_threshold):
        """
        Identify surface atoms using FreeSASA which computes the solvent accessible surface area (SASA) for each atom. 
        Atoms with SASA > threshold are considered exposed.

        Returns:
            List of atoms considered to be on the surface
        """
        # Load the structure into FreeSASA
        structure = freesasa.Structure(self.protein.pdb_file)
        # Calculate the accessible surface area for each atom
        result = freesasa.calc(structure)
        
        surface_atoms = []
        # Iterate over the protein atoms and check their SASA against the threshold
        for i, atom in enumerate(self.protein.atoms):
            # Get the SASA for each atom
            sasa = result.atomArea(i)
            # If the SASA is greater than the threshold (exposed), consider it a surface atom
            if sasa > sasa_threshold:
                surface_atoms.append(atom)

        return surface_atoms

    def _generate_points_around_atom(self, atom, n_points, distance):
        '''
        Generate points in a sphere around an atom.

        Each point is placed at a fixed distance from the atom in a random direction.

        Returns:
            List of generated 3D points
        '''
        points = []

        for _ in range(n_points):
            # Generate a random direction
            direction = np.random.normal(size=3)    # random 3D vector
            direction /= np.linalg.norm(direction)  # normalize to unit length

            # Place the point at a fixed distance from the atom in the random direction
            point = atom.coord + direction * distance

            # Keep only points that are not inside the protein (i.e., not too close to any atom)
            if self._is_accessible(point):
                points.append(point)

        return points
    
    def _is_accessible(self, point, min_distance=1.2):
        """
        Check if a point is accessible (not inside the protein).
        A point is considered accessible if there are no atoms closer than a given distance.
        """
        # Search for neighboring atoms within the minimum distance
        neighbors = self.neighbor_search.query(point, radius=min_distance)
        return len(neighbors) == 0
    
    