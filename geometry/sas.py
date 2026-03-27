import numpy as np
import freesasa

class SASPointGenerator:
    def __init__(self, protein, neighbor_search):
        self.protein = protein
        self.neighbor_search = neighbor_search

    def generate_SAS(self, sasa_threshold=1.0, n_points=20, distance=1.5):
        """
        Genera puntos SAS usando FreeSASA y un muestreo local.
        1 - Identifica átomos en la superficie.
        2 - Genera una nube de puntos alrededor de cada átomo superficial.
        """
        # 1 - Identificar átomos de superficie de forma robusta
        surface_atoms = self._get_surface_atoms(sasa_threshold)

        sas_points = []
        for atom in surface_atoms:
            # 2 - Generar puntos alrededor de cada átomo
            points = self._generate_points_around_atom(atom, n_points, distance)
            sas_points.extend(points)

        return np.array(sas_points)
    
    def _get_surface_atoms(self, sasa_threshold):
        """
        Identifica átomos en la superficie pasando coordenadas y radios 
        directamente a FreeSASA para evitar errores de índices.
        """
        # Preparamos las listas de coordenadas y radios de NUESTROS átomos
        coords = []
        radii = []
        
        for atom in self.protein.atoms:
            # coords debe ser una lista plana [x1, y1, z1, x2, y2, z2...]
            coords.extend(atom.coord)
            # Asignamos radios de Van der Waals básicos según el elemento
            # (1.7Å es un promedio seguro para C, N, O)
            radii.append(1.7)

        # Usamos calcCoord en lugar de calc(structure)
        # Esto hace que FreeSASA use exactamente nuestra lista de átomos
        result = freesasa.calcCoord(coords, radii)
        
        surface_atoms = []
        # Ahora el índice 'i' coincide perfectamente con self.protein.atoms
        for i in range(len(self.protein.atoms)):
            sasa = result.atomArea(i)
            if sasa > sasa_threshold:
                surface_atoms.append(self.protein.atoms[i])

        return surface_atoms

    def _generate_points_around_atom(self, atom, n_points, distance):
        """
        Genera puntos en una esfera alrededor de un átomo.
        """
        points = []
        for _ in range(n_points):
            # Dirección aleatoria 3D
            direction = np.random.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm == 0: continue
            direction /= norm

            # Colocar punto a una distancia fija
            point = atom.coord + direction * distance

            # Solo nos quedamos con puntos que no choquen con el interior de la proteína
            if self._is_accessible(point):
                points.append(point)
        return points
    
    def _is_accessible(self, point, min_distance=1.2):
        """
        Verifica si un punto es accesible (no está dentro de la proteína).
        """
        neighbors = self.neighbor_search.query(point, radius=min_distance)
        return len(neighbors) == 0