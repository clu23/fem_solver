"""Assemblage des matrices globales K, M et du vecteur de force F."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from femsolver.core.mesh import BoundaryConditions, Mesh


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
        """Assemble le vecteur de forces nodales F à partir des conditions de Neumann.

        Parameters
        ----------
        bc : BoundaryConditions
            Conditions aux limites (seule la partie Neumann est utilisée ici).

        Returns
        -------
        F : np.ndarray, shape (n_dof,)
            Vecteur de forces nodales [N].

        Notes
        -----
        Seules les forces ponctuelles nodales (Neumann) sont assemblées ici.
        Les forces distribuées surfaciques/volumiques seront ajoutées en Phase 2.
        """
        F = np.zeros(self.mesh.n_dof)
        for node_id, dof_forces in bc.neumann.items():
            for dof, force in dof_forces.items():
                global_dof = self.mesh.n_dim * node_id + dof
                F[global_dof] += force
        return F
