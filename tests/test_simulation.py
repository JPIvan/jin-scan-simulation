from pytest import mark, fixture
import itertools
import numpy as np

from jin_scan_simulation.polygons import GridPolygon, RectilinearPolygon


# Arrange
@fixture
def vertices_default():
    return np.array(
        [
            [-1, -1], [1, -1], [1, 1], [-5, 1], [-5, -3],
            [-2, -3], [-2, -2], [-1, -2],
        ]
    )


@fixture
def vertices_double_square():
    return np.array(
        [
            [-1, -1], [1, -1], [1, 1], [-1, 1],
            [-1, -1], [1, -1], [1, 1], [-1, 1],
        ]
    )


@fixture
def poly_default(vertices_default):
    return GridPolygon(V=vertices_default, seed=0)


@fixture
def poly_default_points_with_edf_value_1():
    return np.array([[0, 0], [-6, -1], [-1, -3], [-4, -1]])


@fixture
def poly_default_points_with_edf_value_05():
    return np.array(
        [
            [0, -0.5], [-5-np.sqrt(2)/4, -3-np.sqrt(2)/4],
            [0, -1.5]
        ]
    )


@fixture
def poly_double_square(vertices_double_square):
    return GridPolygon(V=vertices_double_square, seed=0)


@fixture
def poly_test_points_in_interior():
    return np.array(
        [
            [0, 0],
            [0, 0.5], [0, -0.5],
            [0.5, 0], [-0.5, 0],
            [0.5, 0.5], [-0.5, -0.5]
        ]
    )


@fixture
def default_tolerance():
    return 0.0


class TestGridPolygon:
    @mark.parametrize("dim_inputs", (1, 2))
    def test_get_pvs(self, dim_inputs, poly_default):
        """Check if we handle 1 and n inputs correctly."""
        if dim_inputs == 1:
            pv1, pv2 = poly_default._get_pvs(np.array([0, 0]))
            assert (pv1 == -poly_default.vertices).all()
            assert (pv2 == -poly_default._vertices_offset_forward).all()
        elif dim_inputs == 2:
            pv1, pv2 = poly_default._get_pvs(np.array([[0, 0]]*3))
            assert pv1.shape == (3, poly_default.n_vertices, 2)
            assert pv2.shape == (3, poly_default.n_vertices, 2)
            assert (pv1 == -poly_default.vertices).all()
            assert (pv2 == -poly_default._vertices_offset_forward).all()
        else:
            raise ValueError(f"Unexpected parameter: '{dim_inputs}'.")

    def test_sample_boundary(self, poly_default):
        """Check shape of output.
        Cannot check simply correctness of this method:
        depends on correctness of _is_on_boundary and vice versa.
        """
        assert poly_default.sample_boundary(n_samples=1).shape == (1, 2)
        assert poly_default.sample_boundary(n_samples=50).shape == (50, 2)

    @mark.parametrize("point_type,open", list(itertools.product(
        ("vertices", "boundary_points"), (True, False)
    )))
    def test_is_on_boundary(
                self, poly_default, default_tolerance, point_type, open
            ):
        if point_type == "vertices":
            testp = poly_default.vertices
        elif point_type == "boundary_points":
            testp = poly_default.sample_boundary(n_samples=50)
        else:
            raise ValueError(f"Unexpected parameter: '{point_type}'.")

        if open and point_type == "vertices":
            assert not any(
                poly_default._is_on_boundary(
                    testp, open=open, tol=default_tolerance
                )
            )
        else:
            assert all(
                poly_default._is_on_boundary(
                    testp, open=open, tol=default_tolerance
                )
            )

    @mark.parametrize("query_type,n_queries", list(itertools.product(
        ("vertices", "boundary_points", "general"),
        (1, "many")
    )))
    def test_edf_at_point(
                self,
                poly_default,
                poly_default_points_with_edf_value_1,
                query_type,
                n_queries
            ):
        # make query according to query_type
        if query_type == "vertices":
            query = poly_default.vertices
            expected_edf = 0
        elif query_type == "boundary_points":
            query = poly_default.sample_boundary(n_samples=50)
            expected_edf = 0
        elif query_type == "general":
            query = poly_default_points_with_edf_value_1
            expected_edf = 1
        else:
            raise ValueError(f"Unexpected parameter: '{query_type}'.")

        if n_queries == 1:
            query = query[0, :]  # shape (2, )
            assert query.shape == (2,)
            assert np.isclose(
                poly_default.edf_at_point(query),
                np.array([expected_edf])
            )
            query = query.reshape(1, 2)
            assert np.isclose(
                poly_default.edf_at_point(query),
                np.array([expected_edf])
            )
        elif n_queries == "many":
            assert query.ndim == 2
            assert np.isclose(
                poly_default.edf_at_point(query),
                np.array([expected_edf]*query.shape[0])
            ).all()
        else:
            raise ValueError(f"Unexpected parameter: '{n_queries}'.")

    @mark.parametrize("shape,interior,n_queries", list(itertools.product(
        ("default", "double_square"), (True, False), (1, "1_2d", "many")
    )))
    def test_get_winding_number(
                self,
                poly_default,
                poly_double_square,
                poly_test_points_in_interior,
                default_tolerance,
                shape,
                interior,
                n_queries
            ):
        if shape == "default":
            poly = poly_default
            ew = 1  # expected winding number
        elif shape == "double_square":
            poly = poly_double_square
            ew = 2  # expected winding number
        else:
            raise ValueError(f"Unexpected parameter: '{shape}'.")
        ew = ew * interior  # winding number should be 0 from exterior

        if n_queries == 1:
            p = poly_test_points_in_interior[0]
            expected_winding = np.array([ew])
        elif n_queries == "1_2d":
            p = poly_test_points_in_interior[0].reshape(1, 2)
            expected_winding = np.array([ew])
        elif n_queries == "many":
            p = poly_test_points_in_interior.reshape(-1, 2)
            expected_winding = np.array([ew]*p.shape[0]) * interior

        # move to exterior if test requires it
        p += 20 * (0 if interior else 1)

        assert (
            poly._get_winding_number(
                p, tol=default_tolerance
            ) == expected_winding
        ).all()

    @mark.parametrize("test_property,dim", list(itertools.product(
        ("shape", "correct_winding"), (1, "many")
    )))
    def test_sample_interior(
                self, poly_default, default_tolerance, test_property, dim
            ):
        n = 1 if dim == 1 else 50 if dim == "many" else None
        if test_property == "shape":
            print(n)
            assert (
                poly_default.sample_interior(
                    n_samples=n, minimium_dist=default_tolerance
                ).shape == (n, 2)
            )
        elif test_property == "correct_winding":
            p = poly_default.sample_interior(
                n_samples=n,
                minimium_dist=default_tolerance
            )
            assert (
                poly_default._get_winding_number(
                    p, tol=default_tolerance
                ) == 1
            ).all()


class TestRectilinearPolygon(TestGridPolygon):
    @fixture
    def transform_params(self):
        return {"angle": np.pi/4, "scalex": 2, "scaley": 2}

    @fixture
    def poly_default(self, vertices_default, transform_params):
        gp = GridPolygon(V=vertices_default, seed=0)
        return gp.transform(**transform_params)

    @fixture
    def poly_default_points_with_edf_value_1(
                self,
                poly_default_points_with_edf_value_05,
                transform_params
            ):
        old_values = poly_default_points_with_edf_value_05
        angle = transform_params["angle"]
        scalex, scaley = transform_params["scalex"], transform_params["scaley"]
        new_values = np.matmul(
            old_values,
            np.diag([scalex, scaley])
        )
        new_values = np.matmul(
            new_values,
            np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            ).T
        )
        return new_values

    @fixture
    def poly_double_square(self, vertices_double_square):
        gp = GridPolygon(V=vertices_double_square, seed=0)
        return gp.transform(angle=np.pi/4, scalex=2, scaley=2)

    @fixture
    def default_tolerance(self):
        return 1e-3


class TestRaycastMixin:
    @mark.parametrize("n_samples,n_rays,cls", list(itertools.product(
            [1, 2],
            [1, 8],
            [GridPolygon, RectilinearPolygon],
        )))
    def testShape(self, poly_default, n_samples, n_rays, cls):
        if cls is GridPolygon:
            polygon = poly_default
        elif cls is RectilinearPolygon:
            polygon = poly_default.transform()
        source = polygon.sample_interior(n_samples=n_samples)
        absolute_theta = np.array([0]*n_samples)
        n_rays = n_rays

        endpoints = polygon.raycast(
            source=source,
            absolute_theta=absolute_theta,
            n_rays=n_rays,
            total_angle=3*np.pi/4,
            tol=1e-2,
            maxiter=128,
        )

        assert endpoints.shape == (n_samples, n_rays, 2)
