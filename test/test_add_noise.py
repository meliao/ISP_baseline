import numpy as np

from helpers.add_noise import add_noise_to_d


class Test_add_noise_to_d:

    def test_0(self) -> None:
        d = np.array([[[1.0, 2.0, 3]]])
        noise_to_sig_ratio = 0.0
        res = add_noise_to_d(d, noise_to_sig_ratio)
        assert isinstance(res, np.ndarray)
        assert np.allclose(res, d)

    def test_1(self) -> None:
        """Tests that complex arrays get imag noise added to them."""
        d = np.array([[[1.0 + 0j, 2.0 + 0j, 3.0 + 0j]]])
        noise_to_sig_ratio = 0.1
        res = add_noise_to_d(d, noise_to_sig_ratio)
        assert res.dtype == d.dtype
        assert np.linalg.norm(res.imag) > 0

    def test_2(self) -> None:
        """Tests that the batching is done correctly."""
        d = np.random.normal(size=(10, 5, 3, 3))

        d[1, 1] = np.zeros_like(d[1, 1])
        noise_to_sig_ratio = 0.1
        res = add_noise_to_d(d, noise_to_sig_ratio)

        assert np.allclose(res[1, 1], d[1, 1])
