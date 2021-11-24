from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from jin_scan_simulation.datasets import GridPolygonDataset

data_folder = Path(__file__).resolve().parent.parent / "data" / "vertex_data"
gpd = GridPolygonDataset(seed=2)
# this seed is used to set the seed of all polygons in the dataset
gpd.load_polygons(name="run1k", _VERTEX_DATA_DIR=data_folder)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
gp1, gp2 = gpd[0], gpd[-1]
gp1.plot(ax1)
gp2.plot(ax2)

# plot rays for grid polygon
# single pose
source1 = gp1.sample_interior(n_samples=1, minimium_dist=0.01)
thetas1 = np.array([0])
# multiple (2) poses
source2 = gp2.sample_interior(n_samples=2, minimium_dist=0.01)
thetas2 = np.array([0, np.pi/4])
# NOTE: the minimum_dist keywords does not work as expected at the moment

# raycast for single pose
ray_endpoints1 = gp1.raycast(
    source=source1, absolute_theta=thetas1, n_rays=27, tol=1e-2,
    maxiter=128
)
# raycast for multiple poses
ray_endpoints2 = gp2.raycast(
    source=source2, absolute_theta=thetas2, n_rays=27, tol=1e-2,
    maxiter=128
)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
gp1.plot(ax1)
gp2.plot(ax2)
gp1.plot_rays(
    ax1, source1, ray_endpoints1, absolute_theta=thetas1
)
gp2.plot_rays(
    ax2, source2, ray_endpoints2, absolute_theta=thetas2
)

# transform grid polygons

rp1 = gp2.transform(angle=np.pi/8, scalex=1.5, scaley=0.6)
rp2 = gp2.transform()  # random transform

# plot rays for rectilinear polygon
# single pose
source1 = rp1.sample_interior(n_samples=1, minimium_dist=0.01)
thetas1 = np.array([0])
# multiple (2) poses
source2 = rp2.sample_interior(n_samples=2, minimium_dist=0.01)
thetas2 = np.array([0, np.pi/4])
# NOTE: the minimum_dist keywords does not work as expected at the moment

# raycast for single pose
ray_endpoints1 = rp1.raycast(
    source=source1, absolute_theta=thetas1, n_rays=27, tol=1e-2,
    maxiter=128
)
# raycast for multiple poses
ray_endpoints2 = rp2.raycast(
    source=source2, absolute_theta=thetas2, n_rays=27, tol=1e-2,
    maxiter=128
)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
rp1.plot(ax1)
rp2.plot(ax2)
rp1.plot_rays(
    ax1, source1, ray_endpoints1, absolute_theta=thetas1
)
rp2.plot_rays(
    ax2, source2, ray_endpoints2, absolute_theta=thetas2
)
