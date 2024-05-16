# Third party packages
import pytest
import os

# Local repo code
from WideBNetModel import WideBNet, morton
from src import models

from helpers.data_loading import load_data_from_dir, _load_eta_scatter_from_dir_Borong

DIR_TESTDATA = "/home/meliao/projects/Inverse_Scattering_ML_TF2/public-example/testdata"
testdatapresent = pytest.mark.skipif(
    not os.path.isdir(DIR_TESTDATA), reason="Test data not found"
)

DIR_OURDATA = "/net/projects/willettlab/meliao/recursive-linearization/dataset/"
ourdatapresent = pytest.mark.skipif(
    not os.path.isdir(DIR_OURDATA), reason="Our data not found"
)
DIR_OURDATA_FORMAT = "/net/projects/willettlab/meliao/recursive-linearization/dataset/test_measurements_nu_{}"


class Test_load_data_from_dir:
    @testdatapresent
    def test_0(self) -> None:
        L = 4
        s = 5
        n_pixels = (2**L) * s
        n_freqs = 3

        (scatter, eta, eta_mean, eta_std, scatter_means, scatter_stds) = (
            load_data_from_dir(DIR_TESTDATA, True, L, s)
        )

        n_samples = eta.shape[0]
        assert eta.shape == (n_samples, n_pixels, n_pixels)
        assert scatter.shape == (n_samples, 2, n_pixels * n_pixels, n_freqs)
        assert eta_mean.shape == ()
        assert eta_std.shape == ()
        assert scatter_means.shape == (n_freqs,)
        assert scatter_stds.shape == (n_freqs,)

    @ourdatapresent
    def test_1(self) -> None:
        L = 4
        s = 12
        n_pixels = (2**L) * s
        wavenumbers = ["2", "4", "8"]
        n_freqs = len(wavenumbers)
        (scatter, eta, eta_mean, eta_std, scatter_means, scatter_stds) = (
            load_data_from_dir(DIR_OURDATA_FORMAT, True, L, s, wavenumbers=wavenumbers)
        )
        n_samples = eta.shape[0]
        assert eta.shape == (n_samples, n_pixels, n_pixels)
        assert scatter.shape == (n_samples, 2, n_pixels * n_pixels, n_freqs)
        assert eta_mean.shape == ()
        assert eta_std.shape == ()
        assert scatter_means.shape == (n_freqs,)
        assert scatter_stds.shape == (n_freqs,)

    @ourdatapresent
    def test_2(self) -> None:
        L = 4
        s = 12
        n_pixels = (2**L) * s
        wavenumbers = ["2", "4", "8"]
        n_freqs = len(wavenumbers)
        truncate_num = 531
        (scatter, eta, eta_mean, eta_std, scatter_means, scatter_stds) = (
            load_data_from_dir(
                DIR_OURDATA_FORMAT,
                True,
                L,
                s,
                wavenumbers=wavenumbers,
                truncate_num=truncate_num,
            )
        )

        assert eta.shape == (truncate_num, n_pixels, n_pixels)
        assert scatter.shape == (truncate_num, 2, n_pixels * n_pixels, n_freqs)
        assert eta_mean.shape == ()
        assert eta_std.shape == ()
        assert scatter_means.shape == (n_freqs,)
        assert scatter_stds.shape == (n_freqs,)

    @ourdatapresent
    def test_3(self) -> None:
        L = 4
        s = 12
        n_pixels = (2**L) * s
        wavenumbers = ["2", "4", "8"]
        n_freqs = len(wavenumbers)
        truncate_num = 531
        (scatter, eta, eta_mean, eta_std, scatter_means, scatter_stds) = (
            load_data_from_dir(
                DIR_OURDATA_FORMAT,
                True,
                L,
                s,
                wavenumbers=wavenumbers,
                truncate_num=truncate_num,
                noise_to_sig_ratio=0.1,
            )
        )

        assert eta.shape == (truncate_num, n_pixels, n_pixels)
        assert scatter.shape == (truncate_num, 2, n_pixels * n_pixels, n_freqs)
        assert eta_mean.shape == ()
        assert eta_std.shape == ()
        assert scatter_means.shape == (n_freqs,)
        assert scatter_stds.shape == (n_freqs,)


if __name__ == "__main__":
    pytest.main()
