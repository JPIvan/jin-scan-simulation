from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, Union, Optional

from jin_scan_simulation.raycasting import RaycastMixin


BoundsTuple = namedtuple("bounds", ["min", "max"])
HullTuple = namedtuple("coordinates", ["x", "y"])


class GridPolygon(RaycastMixin):
    """
    Attributes
    ----------
    self.rng
    self.vertices
    self.hull
    """
    default_start_vertices = [(0, 0), (3, 0), (3, 3), (0, 3)]

    def __init__(self, V=None, *, seed: Optional[int] = None) -> None:
        """This class is used for creating and handling grid-aligned polygons.

        Parameters
        ----------
        V : iterable, optional
            array-like which will create a (N, 2) array, by default None
        seed : Optional[int], optional
            seed for attached random number generator, by default None
        """
        if V is None:
            V = self.default_start_vertices
        self.rng = np.random.default_rng(seed=seed)
        self.vertices = self._init_vertices(V)
        self.hull = HullTuple(
            x=BoundsTuple(
                self.vertices[:, 0].min(), self.vertices[:, 0].max()
            ),
            y=BoundsTuple(
                self.vertices[:, 1].min(), self.vertices[:, 1].max()
            )
        )

    def _init_vertices(self, V):
        """Vertices in GridPolygons are integers, derived classes
        should override this
        """
        return np.array(V, dtype=int)

    def edf_at_point(self, x: np.ndarray) -> np.ndarray:
        """Get the edf at a shape (2, ) or (n_queries, 2) query x

        Parameters
        ----------
        x : np.ndarray
            shape (2, ) or (n_queries, 2) set of points that the edf value
            will be calculated at

        Returns
        -------
        np.ndarray
            edf values at each query

        Raises
        ------
        ValueError
            If query dimension is not 1 or 2
        """
        # we reshape things like this so we can describe everything in terms
        # of vertices with shape (n_vertices, 2)
        # and queries with shape (n_queries, 1, 2)
        if x.ndim == 1 and x.shape == (2, ):
            x = x[np.newaxis, np.newaxis, :]
        elif x.ndim == 2 and x.shape[1] == 2:
            x = x[:, np.newaxis, :]
        else:
            raise ValueError(
                f"Bad shape, {x.shape}, should be (2, ) or (n, 2)."
            )
        e = self._get_evs()
        # take dot product between (unit) edge vector and x - v_i
        # get (n_queries, n_vertices) array of dot products
        alpha = np.einsum("ij,kij->ki", e, x - self.vertices) / (
            np.einsum("ij,ij->i", e, e))
        # clip alpha
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        # (n_queries, n_vertices, 1) array of dot products
        alpha = alpha[:, :, np.newaxis]
        # alpha * e is (n_queries, n_vertices, 2)
        # from x = v + alpha*e + d
        d = x - self.vertices - alpha*e
        # d is (n_queries, n_vertices, 2)
        # take norm of all vectors ->  (n_queries, n_vertices)
        # minimum over all vertices -> (n_queries, )
        return np.linalg.norm(d, axis=2).min(axis=1)

    def plot(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        V = np.concatenate((self.vertices, self.vertices[[0], :]), axis=0)
        ax.plot(V[:, 0], V[:, 1], color='white', alpha=0.5)
        ax.scatter(V[:, 0], V[:, 1], color='white', marker=',', s=5, zorder=5)
        ax.grid(linestyle='dotted')
        ax.set_aspect('equal')
        ax.set_xlim(self.hull.x.min - 1, self.hull.x.max + 1)
        ax.set_ylim(self.hull.y.min - 1, self.hull.y.max + 1)

        X, Y = np.meshgrid(
            np.linspace(self.hull.x.min - 1, self.hull.x.max + 1, 41),
            np.linspace(self.hull.y.min - 1, self.hull.y.max + 1, 41)
        )
        Z = self.edf_at_point(
            np.array([X.flatten(), Y.flatten()]).T
        )
        ax.contourf(X, Y, Z.reshape(X.shape), levels=80)
        return ax

    def sample_boundary(self, n_samples: int) -> np.ndarray:
        """Saple points along the boundary of the polygon.

        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        np.ndarray
            numpy array of shape (n_samples, 2) containing samples on boundary
        """
        idxs1 = self.rng.integers(self.n_vertices, size=n_samples)
        idxs2 = self._next_vertex(idxs1)
        a = self.rng.uniform(size=(n_samples, 1))
        # shape because we want to multiply (n_samples, 1) * (n_samples, 2)
        return a*self.vertices[idxs1, :] + (1-a)*self.vertices[idxs2, :]

    def sample_interior(
                self, n_samples: int, *, minimium_dist=1e-1
            ) -> np.ndarray:
        """Sample points uniformly in the interior of the polygon.

        Parameters
        ----------
        n_samples : int
            Number of samples
        minimium_dist : float
            Minimum distance to boundary
            default 1e-1

        Returns
        -------
        np.ndarray
            (n_samples, 2) array of sampled points
        """
        if n_samples is None:
            n_samples = 1
        # sample interior
        interior_samples = np.zeros(shape=(0, 2))
        for _ in range(100):  # exit if sampling fails
            sample = self.rng.uniform(size=(n_samples, 2))
            sample[:, 0] = (
                sample[:, 0]*self.hull.x.min
                + (1 - sample[:, 0])*self.hull.x.max
            )
            sample[:, 1] = (
                sample[:, 1]*self.hull.y.min
                + (1 - sample[:, 1])*self.hull.y.max
            )
            interior = self._is_on_interior(
                sample, tol=minimium_dist
            ).astype(bool)
            interior_samples = np.concatenate(
                [interior_samples, sample[interior, :]], axis=0
            )
            if interior_samples.shape[0] >= n_samples:
                break
        else:
            raise RuntimeError("Could not sample interior.")

        return interior_samples[:n_samples, :]

    def transform(
                    self,
                    angle: Optional[float] = None,
                    scalex: Optional[float] = None,
                    scaley: Optional[float] = None
            ):
        if angle is None:
            angle = self.rng.uniform() * 2 * np.pi
        if scalex is None:
            scalex = self.rng.uniform() * 1.5 + 0.5
        if scaley is None:
            scaley = self.rng.uniform() * 1.5 + 0.5
        scale_mat = np.diag([scalex, scaley])
        rotation_mat = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        new_v = np.matmul(self.vertices, scale_mat)
        new_v = np.matmul(new_v, rotation_mat.T)
        return RectilinearPolygon(V=new_v, seed=self.rng.integers(2**16))

    def _is_on_interior(
                self, p: np.ndarray, *, open=True, tol: float = 0.0
            ) -> np.ndarray:
        """Check if the points in p are on the interior of the polygon

        Parameters
        ----------
        p : np.ndarray
            Point(s) of interest. Shape (2, ) or (n_ponts, 2)
        open : bool, optional
            whether the interior is open or closed, by default True
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 0.0

        Returns
        -------
        np.ndarray
            boolean array of size (n_points, )
            True if point is on interior.
        """
        w = self._get_winding_number(p, tol=tol)
        on_interior = (w != 0)
        if open:
            on_boundary = np.array(0)  # boundary check irrelvant
        else:
            on_boundary = self._is_on_boundary(p, tol=tol)
        return on_interior | on_boundary

    def _get_winding_number(
                self, p: np.ndarray, *, tol: float = 0.0
            ) -> np.ndarray:
        """Compute the winding number of the points given in `p`.

        Parameters
        ----------
        p : np.ndarray
            Point(s) of interest. Shape (2, ) or (n_points, 2)
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 0.0

        Returns
        -------
        np.ndarray
            Winding number of each point, shape (n_points, )

        Raises
        ------
        ValueError
            If any point is on the boundary of the polygon.
            Winding number not defined in this case.
        """
        # 7.121 in Hughes et. al
        # Hughes uses v - p and we use p - v
        # also, Hughes ignores the sign of the angle, which does not work
        if any(self._is_on_boundary(p, open=False, tol=tol)):
            raise ValueError(
                "Winding number not defined for point on boundary."
            )
        # these have shape (n_test_points, n_vertices, 2)
        pv1, pv2 = self._get_pvs(p)
        # norms, (n_test_points, n_vertices)
        norm1, norm2 = np.linalg.norm(pv1, axis=2), np.linalg.norm(pv2, axis=2)
        # dot products, (n_test_points, n_vertices)
        dots = np.einsum("ijk,ijk->ij", pv1, pv2)
        # compute the angle from cos relation to dot product
        # compute sign of angle using determinant
        angles = np.arccos(dots / (norm1 * norm2)) * np.sign(
            np.linalg.det(
                np.concatenate(
                    [
                        pv1[:, :, np.newaxis, :],
                        pv2[:, :, np.newaxis, :]
                    ], axis=2
                )
            )
        )
        # sum over vertices
        angles = np.sum(angles, axis=1) / np.pi / 2
        # round to int
        return np.rint(
            angles,
            out=np.zeros_like(angles, dtype=int),
            casting="unsafe"
        )

    def _is_on_boundary(
                self, p: np.ndarray, *, open=True, tol: float = 0.0
            ) -> np.ndarray:
        """Checks if the point p is on the boundary of the polygon.

        Parameters
        ----------
        p : numpy array of shape (2, ) or (n_test_points, 2)
            test point
        open : bool, optional
            if True, then vertexes are not considered part of the boundary,
            by default True
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 0.0

        Returns
        -------
        np.ndarray
            shape (n_test_points, ) array of True/False for boundary check
        """
        ev, pv1, pv2 = self._get_evs(), *self._get_pvs(p)
        # check determinant of matrix using basis pv1, pv2
        is_collinear = self._check_vectors_collinear(pv1, pv2, tol=tol)
        dots1 = np.einsum("ij,kij->ki", ev, pv1)
        dots2 = np.einsum("ij,kij->ki", ev, pv2)
        is_in_bounds = ((dots2 < 0) & (0 < dots1))

        # if p is collinear and in bounds with any edge -> on the boundary
        on_open_bd = (is_collinear & is_in_bounds).any(axis=-1)
        if open:
            return on_open_bd
        else:
            return on_open_bd | self._is_vertex(p)

    def _check_vectors_collinear(
                self, v1: np.ndarray, v2: np.ndarray, *, tol: float = 0.0
            ) -> np.ndarray:
        """Check if vectors in the given arrays are collinear.
        The arrays are both assumed to have the same shape (..., 2).
        The last dimension is treated as the vectors of interest.

        Parameters
        ----------
        v1 : np.ndarray
            (..., 2) array
        v2 : np.ndarray
            (..., 2) array
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 0.0

        Returns
        -------
        np.ndarray
            Boolean array of shape (..., ) showing whether the given vectors
            are collinear.
        """
        # check determinant of matrix using basis v1, v2
        det = np.linalg.det(
                np.concatenate(
                    [
                        v1[..., np.newaxis, :],
                        v2[..., np.newaxis, :]
                    ], axis=-2
                )
            )
        # (n_test_points, n_vertices) array of where determinant is 0
        # note that n_vertices = n_edges
        # if det = 0, p is colliniear with an edge
        return self._collinearity_condition(det, tol=tol)

    def _collinearity_condition(self, det: np.array, *, tol: float = 0.0):
        """Collinearity condition for this type of polygon.

        For some polygons we can check strict equality.
        For others we have to deal with floating-point representations.

        Here we check equality.
        (Note that abs(det) <= tol will still work if tol is 0.)

        Parameters
        ----------
        det : np.array
            iterable of determinants
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 0.0
        """
        return (np.abs(det) <= tol)

    def _get_pvs(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get vectors from point p to all vertices.

        Parameters
        ----------
        p : numpy array of shape (2, ) or (n, 2)
            test point

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            pair of shape (n_test_points, n_vertices, 2) array of vectors
            from point to vertices
            First element: p - v_i
            Second element p - v_{i+1}
        """
        p = p.reshape(-1, 1, 2)  # (n_test_points, 1, 2)
        return p - self.vertices, p - self._vertices_offset_forward

    def _get_evs(self) -> np.ndarray:
        """Get `edge vectors`, i.e. vectors oriented from vertex v_i to v_{i+1}

        Returns
        -------
        np.ndarray
            shape (n_vertices, 2) array of edge vectors
        """
        return self._vertices_offset_forward - self.vertices

    def _is_vertex(self, p: np.ndarray) -> np.ndarray:
        """Checks if points are vertices of the gridpolygon

        Parameters
        ----------
        p : np.ndarray
            shape (2, ) or (n_test_points, 2) array of test points

        Returns
        -------
        np.ndarray
            bool array of shape (n_test_points) showing if p is a vertex
        """
        p = p.reshape(-1, 1, 2)  # (n_test_points, 1, 2)
        # .all() checks if both coordinates are the same
        # .any() checks if test point perfectly matched any vertex
        return (p == self.vertices).all(axis=-1).any(axis=-1)

    def _next_vertex(self, i: Union[int, np.ndarray]) \
            -> Union[int, np.ndarray]:
        """Convenience method for modular indexing.

        Parameters
        ----------
        i : Union[int, np.ndarray]
            reference index (or indices)

        Returns
        -------
        Union[int, np.ndarray]
            next index
        """
        return (i + 1) % self.n_vertices

    def _prev_vertex(self, i: Union[int, np.ndarray]) \
            -> Union[int, np.ndarray]:
        """Convenience method for modular indexing.

        Parameters
        ----------
        i : Union[int, np.ndarray]
            reference index (or indices)

        Returns
        -------
        Union[int, np.ndarray]
            next index
        """
        return (i - 1) % self.n_vertices

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def _vertices_offset_forward(self) -> np.ndarray:
        return self.vertices[self._next_vertex(np.arange(self.n_vertices))]


class RectilinearPolygon(GridPolygon):
    def __init__(self, V=None, *, seed: Optional[int] = None) -> None:
        super().__init__(V=V, seed=seed)

    def _init_vertices(self, V):
        """Vertices in RectilinearPolygon are can be floats.
        Vertices in base class are only integers, so this is overriden.
        """
        return np.array(V)

    def _collinearity_condition(self, det, *, tol: float = 1e-3):
        """Collinearity condition for this type of polygon.

        For some polygons we can check strict equality.
        For others we have to deal with floating-point representations.

        Here we have to deal with floating points.

        Parameters
        ----------
        det : np.array
            iterable of determinants
        tol : float, optional
            tolerance.
            A point closer than tol to a boundary will be
            considered on the boundary.
            by default 1e-3
        """
        if tol == 0:
            raise ValueError(
                "Tolerance for collinearity in rectilinear polygons "
                "cannot be 0!"
            )
        return np.isclose(det, 0, rtol=tol)
