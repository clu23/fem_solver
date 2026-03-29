"""Assemblage des matrices globales K, M et du vecteur de force F."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from femsolver.core.mesh import (
    BoundaryConditions,
    BodyForce,
    DistributedLineLoad,
    ElementData,
    Mesh,
    PressureLoad,
)


# ---------------------------------------------------------------------------
# Helpers pour les charges distribuées et surfaciques
# ---------------------------------------------------------------------------


def _find_element(mesh: Mesh, node_ids: tuple[int, ...]) -> ElementData | None:
    """Trouve l'élément du maillage dont les nœuds contiennent ``node_ids``.

    Parameters
    ----------
    mesh : Mesh
        Maillage dans lequel chercher.
    node_ids : tuple[int, ...]
        Ensemble de nœuds à trouver (sous-ensemble ou égalité exacte).

    Returns
    -------
    ElementData or None
        Premier élément trouvé, ou ``None`` si aucun ne correspond.
    """
    face_set = set(node_ids)
    for elem_data in mesh.elements:
        if face_set.issubset(set(elem_data.node_ids)):
            return elem_data
    return None


def _pressure_nodal_force_2d(
    mesh: Mesh,
    load: PressureLoad,
) -> tuple[list[int], np.ndarray]:
    """Forces nodales sur une arête 2D soumise à une pression.

    Convention CCW : arête A→B avec le domaine à gauche.
    Normale sortante : n̂ = (dy, -dx) / L.

    La normale est vérifiée en cherchant l'élément adjacent et en s'assurant
    que le vecteur centroïde→face pointe dans le sens de n̂.

    Parameters
    ----------
    mesh : Mesh
        Maillage (pour récupérer les coordonnées et l'élément adjacent).
    load : PressureLoad
        Pression surfacique (node_ids de taille 2).

    Returns
    -------
    dofs : list[int]
        DDL globaux des 2 nœuds (taille 4 pour n_dim=2).
    f_edge : np.ndarray, shape (4,)
        Forces nodales [Fx_A, Fy_A, Fx_B, Fy_B].

    Raises
    ------
    ValueError
        Si la normale calculée pointe vers l'intérieur du domaine
        (ordre des nœuds non CCW).
    """
    id_a, id_b = load.node_ids
    A = mesh.nodes[id_a]   # shape (2,)
    B = mesh.nodes[id_b]   # shape (2,)

    dx = B[0] - A[0]
    dy = B[1] - A[1]
    L = float(np.hypot(dx, dy))
    if L < 1e-14:
        raise ValueError(
            f"PressureLoad : longueur nulle pour l'arête ({id_a}, {id_b})"
        )

    # Normale sortante (domaine à gauche de A→B)
    n_hat = np.array([dy / L, -dx / L])

    # Vérification CCW via le centroïde de l'élément adjacent
    elem_data = _find_element(mesh, load.node_ids)
    if elem_data is not None:
        elem_nodes = mesh.nodes[list(elem_data.node_ids)]
        centroid = elem_nodes.mean(axis=0)
        edge_center = 0.5 * (A + B)
        # Vecteur du centroïde vers le centre de l'arête (doit être dans le sens n̂)
        outward = edge_center - centroid
        if float(np.dot(outward, n_hat)) < 0:
            raise ValueError(
                f"PressureLoad : la normale de l'arête ({id_a}→{id_b}) pointe "
                f"vers l'intérieur. Inverser l'ordre des nœuds ({id_b}→{id_a})."
            )

    # Force par nœud = -p · n̂ · L/2  (compression positive → force vers intérieur)
    dpn = mesh.dpn
    f_node = -load.magnitude * n_hat * (L / 2.0)
    dofs = mesh.global_dofs((id_a, id_b))
    f_edge = np.zeros(2 * dpn)
    # Pour les éléments 2D (dpn=2) : les 2 premiers DDL de chaque nœud sont ux, uy
    f_edge[0:2] = f_node
    f_edge[dpn : dpn + 2] = f_node
    return dofs, f_edge


def _pressure_nodal_force_3d_tri(
    mesh: Mesh,
    load: PressureLoad,
) -> tuple[list[int], np.ndarray]:
    """Forces nodales sur une face triangulaire 3D soumise à une pression.

    Convention CCW : nœuds A→B→C avec normale sortante n̂ = (B-A)×(C-A).

    Parameters
    ----------
    mesh : Mesh
        Maillage.
    load : PressureLoad
        Pression (node_ids de taille 3).

    Returns
    -------
    dofs : list[int]
        DDL globaux (taille 9 pour n_dim=3, dpn=3).
    f_face : np.ndarray, shape (9,)
        Forces nodales (3 composantes par nœud).

    Raises
    ------
    ValueError
        Si la normale pointe vers l'intérieur.
    """
    id_a, id_b, id_c = load.node_ids
    A = mesh.nodes[id_a]
    B = mesh.nodes[id_b]
    C = mesh.nodes[id_c]

    AB = B - A
    AC = C - A
    cross = np.cross(AB, AC)
    area = float(np.linalg.norm(cross)) / 2.0
    if area < 1e-28:
        raise ValueError(
            f"PressureLoad : aire nulle pour la face ({id_a}, {id_b}, {id_c})"
        )
    n_hat = cross / (2.0 * area)   # normale unitaire

    # Vérification CCW
    elem_data = _find_element(mesh, load.node_ids)
    if elem_data is not None:
        elem_nodes = mesh.nodes[list(elem_data.node_ids)]
        centroid = elem_nodes.mean(axis=0)
        face_center = (A + B + C) / 3.0
        if float(np.dot(face_center - centroid, n_hat)) < 0:
            raise ValueError(
                f"PressureLoad : la normale de la face {load.node_ids} pointe "
                f"vers l'intérieur. Inverser l'ordre des nœuds."
            )

    # Chaque nœud reçoit 1/3 de la force totale = -p · n̂ · area
    f_node = -load.magnitude * n_hat * area / 3.0
    dofs = mesh.global_dofs(load.node_ids)
    dpn = mesh.dpn
    f_face = np.zeros(3 * dpn)
    for k in range(3):
        f_face[k * dpn : k * dpn + 3] = f_node
    return dofs, f_face


def _pressure_nodal_force_3d_quad(
    mesh: Mesh,
    load: PressureLoad,
) -> tuple[list[int], np.ndarray]:
    """Forces nodales sur une face quadrangulaire 3D soumise à une pression.

    Convention CCW : nœuds dans l'ordre CCW vu de l'extérieur.
    Intégration 2×2 Gauss sur la face bilinéaire.

    Parameters
    ----------
    mesh : Mesh
        Maillage.
    load : PressureLoad
        Pression (node_ids de taille 4).

    Returns
    -------
    dofs : list[int]
        DDL globaux (taille 12 pour dpn=3).
    f_face : np.ndarray, shape (12,)
        Forces nodales.

    Raises
    ------
    ValueError
        Si la normale pointe vers l'intérieur.
    """
    ids = load.node_ids   # 4 nœuds
    face_nodes = mesh.nodes[list(ids)]   # (4, 3)

    _GP = 1.0 / np.sqrt(3.0)
    gauss_2x2 = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]

    # Vérification CCW via l'élément adjacent (au premier point de Gauss)
    n_hat_check = None
    elem_data = _find_element(mesh, ids)

    dpn = mesh.dpn
    f_face = np.zeros(4 * dpn)

    for xi, eta in gauss_2x2:
        # Fonctions de forme bilinéaires Q4
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ])
        dN_dxi = 0.25 * np.array([
            [-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)],
            [-(1 - xi),  -(1 + xi),  (1 + xi),   (1 - xi)],
        ])
        # Vecteurs tangents sur la face
        t1 = dN_dxi[0] @ face_nodes   # ∂x/∂ξ  shape (3,)
        t2 = dN_dxi[1] @ face_nodes   # ∂x/∂η  shape (3,)
        cross = np.cross(t1, t2)
        dA = float(np.linalg.norm(cross))
        if dA < 1e-28:
            continue
        n_hat = cross / dA

        # Vérification CCW au premier point de Gauss
        if n_hat_check is None and elem_data is not None:
            n_hat_check = n_hat
            elem_nodes = mesh.nodes[list(elem_data.node_ids)]
            centroid = elem_nodes.mean(axis=0)
            face_center = face_nodes.mean(axis=0)
            if float(np.dot(face_center - centroid, n_hat)) < 0:
                raise ValueError(
                    f"PressureLoad : la normale de la face {ids} pointe vers "
                    f"l'intérieur. Inverser l'ordre des nœuds."
                )

        # Force par point de Gauss : f += -p · n̂ · dA (poids = 1 en 2×2 Gauss)
        f_gp = -load.magnitude * n_hat * dA   # shape (3,)
        for k in range(4):
            f_face[k * dpn : k * dpn + 3] += N[k] * f_gp

    return mesh.global_dofs(ids), f_face


class Assembler:
    """Assemble les matrices globales K, M et le vecteur F à partir d'un maillage.

    L'assemblage utilise le format COO (triplets) puis convertit en CSR pour la
    résolution. Les matrices élémentaires (K_e, M_e) sont calculées à la volée.

    Parameters
    ----------
    mesh : Mesh
        Maillage contenant nœuds, éléments et matériaux.

    Examples
    --------
    >>> assembler = Assembler(mesh)
    >>> K = assembler.assemble_stiffness()
    >>> M = assembler.assemble_mass()
    >>> F = assembler.assemble_forces(bc)
    """

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def assemble_stiffness(self) -> csr_matrix:
        """Assemble la matrice de rigidité globale K (format CSR).

        Returns
        -------
        K : csr_matrix, shape (n_dof, n_dof)
            Matrice de rigidité globale creuse symétrique.

        Notes
        -----
        Le pattern COO → CSR permet l'addition automatique des contributions
        en DDL partagés entre éléments adjacents.
        """
        mesh = self.mesh
        n_dof = mesh.n_dof
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for elem_data in mesh.elements:
            elem = elem_data.get_element()
            node_coords = mesh.node_coords(elem_data.node_ids)
            K_e = elem.stiffness_matrix(elem_data.material, node_coords, elem_data.properties)
            dofs = mesh.global_dofs(elem_data.node_ids)
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    rows.append(di)
                    cols.append(dj)
                    vals.append(K_e[i, j])

        return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    def assemble_mass(self) -> csr_matrix:
        """Assemble la matrice de masse globale M consistante (format CSR).

        Returns
        -------
        M : csr_matrix, shape (n_dof, n_dof)
            Matrice de masse globale creuse symétrique.
        """
        mesh = self.mesh
        n_dof = mesh.n_dof
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for elem_data in mesh.elements:
            elem = elem_data.get_element()
            node_coords = mesh.node_coords(elem_data.node_ids)
            M_e = elem.mass_matrix(elem_data.material, node_coords, elem_data.properties)
            dofs = mesh.global_dofs(elem_data.node_ids)
            for i, di in enumerate(dofs):
                for j, dj in enumerate(dofs):
                    rows.append(di)
                    cols.append(dj)
                    vals.append(M_e[i, j])

        return coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

    def assemble_forces(self, bc: BoundaryConditions) -> np.ndarray:
        """Assemble le vecteur de forces nodales F à partir des conditions aux limites.

        Contributions assemblées :

        1. **Neumann ponctuel** : forces nodales directes (``bc.neumann``).
        2. **Charges linéiques** : Bar2D (axiale) et Beam2D (Hermite) via
           ``bc.distributed`` → ``element.distributed_load_vector()``.
        3. **Force de volume** : gravité ou accélération uniforme via
           ``bc.body_force`` → ``element.body_force_vector()``.
        4. **Pression surfacique** : arêtes 2D ou faces 3D via ``bc.pressure``
           → forces nodales équivalentes (convention CCW stricte).

        Parameters
        ----------
        bc : BoundaryConditions
            Conditions aux limites complètes.

        Returns
        -------
        F : np.ndarray, shape (n_dof,)
            Vecteur de forces nodales [N].

        Raises
        ------
        ValueError
            Si une pression surfacique est définie avec une orientation non CCW
            (normale pointant vers l'intérieur).
        KeyError
            Si un DistributedLineLoad référence des nœuds qui ne correspondent
            à aucun élément du maillage.
        """
        mesh = self.mesh
        F = np.zeros(mesh.n_dof)

        # 1. Forces ponctuelles (Neumann)
        for node_id, dof_forces in bc.neumann.items():
            for dof, force in dof_forces.items():
                global_dof = mesh.dpn * node_id + dof
                F[global_dof] += force

        # 2. Charges linéiques distribuées (Bar2D, Beam2D)
        for dist_load in bc.distributed:
            elem_data = _find_element(mesh, dist_load.node_ids)
            if elem_data is None:
                raise KeyError(
                    f"DistributedLineLoad : aucun élément trouvé pour les nœuds "
                    f"{dist_load.node_ids}."
                )
            elem = elem_data.get_element()
            node_coords = mesh.node_coords(elem_data.node_ids)
            f_e = elem.distributed_load_vector(
                elem_data.material,
                node_coords,
                elem_data.properties,
                dist_load.qx,
                dist_load.qy,
            )
            dofs = mesh.global_dofs(elem_data.node_ids)
            for i, di in enumerate(dofs):
                F[di] += f_e[i]

        # 3. Force de volume (gravité, accélération uniforme)
        if bc.body_force is not None:
            b = np.array(bc.body_force.acceleration, dtype=float)
            for elem_data in mesh.elements:
                elem = elem_data.get_element()
                node_coords = mesh.node_coords(elem_data.node_ids)
                f_e = elem.body_force_vector(
                    elem_data.material,
                    node_coords,
                    elem_data.properties,
                    b,
                )
                dofs = mesh.global_dofs(elem_data.node_ids)
                for i, di in enumerate(dofs):
                    F[di] += f_e[i]

        # 4. Pressions surfaciques
        for pressure_load in bc.pressure:
            n_face = len(pressure_load.node_ids)
            if n_face == 2:
                dofs, f_face = _pressure_nodal_force_2d(mesh, pressure_load)
            elif n_face == 3:
                dofs, f_face = _pressure_nodal_force_3d_tri(mesh, pressure_load)
            elif n_face == 4:
                dofs, f_face = _pressure_nodal_force_3d_quad(mesh, pressure_load)
            else:
                raise ValueError(
                    f"PressureLoad : nombre de nœuds non supporté ({n_face}). "
                    f"Attendu : 2 (arête 2D), 3 (face tri 3D) ou 4 (face quad 3D)."
                )
            for i, di in enumerate(dofs):
                F[di] += f_face[i]

        return F
