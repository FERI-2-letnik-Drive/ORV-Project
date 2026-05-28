import numpy as np
import pytest

from app.lbp import compute_lbp, lbp_histogram


def test_compute_lbp_uniform_image_is_max_code():
    # Pri popolnoma enakomerni sliki je vsak sosed >= sredisce -> vsi biti 1 -> 255
    image = np.full((10, 10), 100, dtype=np.uint8)
    lbp = compute_lbp(image)
    # notranji del (brez 1px okvirja) mora biti 255
    assert np.all(lbp[1:-1, 1:-1] == 255)


def test_compute_lbp_shape_and_dtype():
    image = (np.random.rand(64, 64) * 255).astype(np.uint8)
    lbp = compute_lbp(image)
    assert lbp.shape == image.shape
    assert lbp.dtype == np.uint8


def test_compute_lbp_rejects_color_image():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_lbp(image)


def test_lbp_histogram_length_matches_grid():
    image = (np.random.rand(128, 128) * 255).astype(np.uint8)
    hist = lbp_histogram(image, grid=(8, 8))
    assert hist.shape == (8 * 8 * 256,)


def test_lbp_histogram_each_cell_is_normalized():
    image = (np.random.rand(128, 128) * 255).astype(np.uint8)
    hist = lbp_histogram(image, grid=(4, 4))
    # vsaka celica je L1-normalizirana -> vsota celotnega vektorja = stevilo celic
    assert np.isclose(hist.sum(), 4 * 4, atol=1e-3)


def test_lbp_histogram_identical_images_are_equal():
    image = (np.random.rand(128, 128) * 255).astype(np.uint8)
    assert np.allclose(lbp_histogram(image), lbp_histogram(image))
