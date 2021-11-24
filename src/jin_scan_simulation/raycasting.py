from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from typing import Union


class RaycastMixin(ABC):
    """This mixin can be used for any class that defines
    self.edf_at_point
    """
    @abstractmethod
    def edf_at_point(self, x: np.ndarray) -> np.ndarray:
        pass

    def raycast(
                self,
                source: np.ndarray,
                absolute_theta: Union[float, np.ndarray],
                n_rays: int = 127,
                total_angle: float = 2*np.pi * 0.75,
                tol: float = 1e-2,
                maxiter=128,
            ) -> np.ndarray:
        """Perform raycasting from the specified source point.

        Parameters
        ----------
        source : np.ndarray
            (2, ) or (n_sources, 2) array of source point coordinates
        absolute_theta : Union[float, np.ndarray]
            float, (1, ), (n_sources, ) array
            direction in global coordinate frame for the central ray
        n_rays : int, optional
            number of rays, by default 127
        total_angle : float, optional
            angular width of the cast rays
            the rays will be evenly spaced between
            absolute_theta -+ total_angle/2, by default 2*np.pi*0.75
        tol : float, optional
            tolerance of raycasting, rays will stop once they are within
            `tol` units of an obstuction, by default 1e-6
        maxiter : int, optional
            maximum number of iterations for raycasting, by default 128

        Returns
        -------
        np.ndarray
            (n_sources, n_rays, 2) endpoints of rays

        Raises
        ------
        RuntimeError
            if raycasting fails to converge within specified tolerance in the
            given number of iterations
            unconverged results are passed along with the exception if the
            caller wants them
        """
        self._check_raycast_inputs(source, absolute_theta)
        # angles of rays, shape (n_rays, )
        theta = np.linspace(-total_angle/2, total_angle/2, n_rays)
        theta = theta[np.newaxis, :]  # (1, n_rays)
        absolute_theta = np.array(absolute_theta)  # if theta is float
        absolute_theta = absolute_theta.reshape(-1, 1)
        theta = theta + absolute_theta  # (n_sources, n_rays)

        # unit vector in direction of rays, (n_sources, n_rays, 2)
        u = np.concatenate(
            (np.cos(theta)[:, :, np.newaxis], np.sin(theta)[:, :, np.newaxis]),
            axis=-1
        )

        source = source.reshape(-1, 1, 2)  # (n_sources, 1, 2)
        # positions of ray endpoints, all start at source
        # shape (n_sources, n_rays, 2)
        p = np.tile(source, (1, n_rays, 1))
        for _ in range(maxiter):  # limit number of steps
            edf = self.edf_at_point(p.reshape(-1, 2)).reshape(-1, n_rays)
            edf *= (edf > tol/100)  # clip very small values to zero
            p += u*edf[:, :, np.newaxis]
            if (edf < tol).all():
                break
        else:
            raise RuntimeError("Raycasting failed to converge.", p)
        return p

    def plot_rays(
                self,
                ax: plt.Axes,
                source: np.ndarray,
                endpoints: np.ndarray,
                absolute_theta: Union[float, np.ndarray],
            ) -> None:
        """Plot rays on given figure.

        Parameters
        ----------
        ax : plt.Axes
            axis to plot on
        source : np.ndarray
            (2, ) or (n_sources, 2) array of source point coordinates
        endpoints : np.array
            (n_sources, n_rays, 2) endpoints of rays
        absolute_theta : Union[float, np.ndarray]
            float, (1, ), (n_sources, ) array
            direction in global coordinate frame for the central ray
        """
        self._check_raycast_inputs(source, absolute_theta)
        absolute_theta = np.array(absolute_theta, ndmin=1)  # if theta is float
        source = source.reshape(-1, 2)
        for t, s in zip(absolute_theta, source):
            ax.scatter(
                s[0], s[1],
                marker=(3, 0, 180*t/np.pi - 90),
                linewidths=1, color='red', zorder=3
            )
        ax.scatter(
            endpoints[:, :, 0].flatten(), endpoints[:, :, 1].flatten(),
            marker='x', linewidths=1, color='white', s=8
        )
        n_rays = endpoints.shape[1]
        _source = np.repeat(source[:, np.newaxis, :], n_rays, axis=1)
        _source = _source.reshape(-1, 2)
        endpoints = endpoints.reshape(-1, 2)
        lines_x = np.vstack((_source[:, 0], endpoints[:, 0]))
        lines_y = np.vstack((_source[:, 1], endpoints[:, 1]))
        ax.plot(lines_x, lines_y, color='white', lw=0.5, alpha=0.5)
        return ax

    def _check_raycast_inputs(
                self,
                source: np.ndarray,
                absolute_theta: Union[float, np.ndarray]
            ):
        if isinstance(absolute_theta, float):
            if not(source.shape == (2, ) or source.shape == (1, 2)):
                raise ValueError(
                    f"source.shape '{source.shape}' incompatible with "
                    "absolute_theta of type float."
                )
        elif isinstance(absolute_theta, np.ndarray):
            if absolute_theta.ndim != 1:
                raise ValueError("absolute_theta must be 1D")
            if source.shape == (2, ):
                if absolute_theta.shape != (1, ):
                    raise ValueError(
                        f"source.shape '{source.shape}' and "
                        f"absolute_theta.shape '{absolute_theta.shape}' not "
                        "compatible."
                    )
            elif source.ndim == 2 and source.shape[1] == 2:
                if absolute_theta.shape[0] != source.shape[0]:
                    raise ValueError(
                        f"source.shape '{source.shape}' and "
                        f"absolute_theta.shape '{absolute_theta.shape}' not "
                        "compatible."
                    )
            else:
                raise ValueError(
                    f"source.shape '{source.shape}' invalid."
                )
