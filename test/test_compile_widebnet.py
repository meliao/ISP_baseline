# Third party packages
import pytest

# Local repo code
from WideBNetModel import WideBNet, morton
from src import models

from helpers.compile_widebnet import compile_widebnet


class Test_compile_widebnet:
    def test_0(self) -> None:
        L = 4
        s = 5
        r = 3
        n_pixels = (2**L) * s
        n_freqs = 3
        input_shape = (n_pixels, n_pixels, n_freqs)

        core_module, model = compile_widebnet(L, s, r, input_shape)
        assert type(core_module) == WideBNet.WideBNetModel
        assert type(model) == models.DeterministicModel


if __name__ == "__main__":
    pytest.main()
