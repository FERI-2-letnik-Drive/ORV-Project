import numpy as np

from app.similarity import (
    chi_square_distance,
    cosine_similarity,
    chi_square_similarity,
)


def test_chi_square_distance_identical_is_zero():
    h = np.array([0.1, 0.2, 0.3, 0.4])
    assert chi_square_distance(h, h) == 0.0


def test_chi_square_distance_increases_with_difference():
    h = np.array([1.0, 0.0, 0.0])
    near = np.array([0.9, 0.1, 0.0])
    far = np.array([0.0, 0.0, 1.0])
    assert chi_square_distance(h, near) < chi_square_distance(h, far)


def test_cosine_similarity_identical_is_one():
    h = np.array([0.1, 0.5, 0.4])
    assert np.isclose(cosine_similarity(h, h), 1.0)


def test_cosine_similarity_orthogonal_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 0.0)


def test_chi_square_similarity_in_unit_range_and_ordered():
    h = np.array([0.25, 0.25, 0.25, 0.25])
    near = np.array([0.30, 0.20, 0.25, 0.25])
    far = np.array([1.0, 0.0, 0.0, 0.0])
    s_near = chi_square_similarity(h, near)
    s_far = chi_square_similarity(h, far)
    assert 0.0 <= s_far <= s_near <= 1.0
    assert np.isclose(chi_square_similarity(h, h), 1.0)


def test_exp_chi_square_identical_is_one():
    from app.similarity import exp_chi_square_similarity
    h = np.array([0.2, 0.3, 0.5])
    assert exp_chi_square_similarity(h, h, n_cells=1) == 1.0


def test_exp_chi_square_in_unit_range_and_ordered():
    from app.similarity import exp_chi_square_similarity
    h = np.array([0.25, 0.25, 0.25, 0.25])
    near = np.array([0.30, 0.20, 0.25, 0.25])
    far = np.array([1.0, 0.0, 0.0, 0.0])
    s_near = exp_chi_square_similarity(h, near, n_cells=1)
    s_far = exp_chi_square_similarity(h, far, n_cells=1)
    assert 0.0 <= s_far <= s_near <= 1.0


def test_exp_chi_square_scale_makes_it_stricter():
    from app.similarity import exp_chi_square_similarity
    a = np.array([0.5, 0.3, 0.2])
    b = np.array([0.2, 0.3, 0.5])
    strict = exp_chi_square_similarity(a, b, n_cells=1, scale=0.1)
    loose = exp_chi_square_similarity(a, b, n_cells=1, scale=1.0)
    assert strict < loose
