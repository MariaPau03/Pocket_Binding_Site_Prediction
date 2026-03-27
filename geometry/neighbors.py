from scipy.spatial import KDTree


class NeighborSearch:
    def __init__(self, atom_coords):
        """
        atom_coords: numpy array (N_atoms, 3)
            containing the 3D coordinates of all atoms in the protein
        """
        # Build a KDTree, a data structure that allows efficient neighbor searches in 3D space 
        self.tree = KDTree(atom_coords)

    def query(self, point, radius):
        """
        Return indices of atoms within a given radius
        """
        # query_ball_point of KDTree returns a list of indices of atoms that are within the specified radius from the point
        return self.tree.query_ball_point(point, radius)