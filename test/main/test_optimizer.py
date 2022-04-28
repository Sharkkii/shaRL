import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.network import PseudoMeasureNetwork
from src.optimizer import Optimizer


optimizer_factory = torch.optim.Adam

class TestOptimizer():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        optimizer = Optimizer(
            optimizer_factory = optimizer_factory
        )
        assert optimizer.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = PseudoMeasureNetwork
        optimizer = Optimizer(
            optimizer_factory = optimizer_factory,
            network = network
        )
        assert optimizer.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = PseudoMeasureNetwork
        optimizer = Optimizer(
            optimizer_factory = optimizer_factory
        )
        optimizer.setup(network)
        assert optimizer.is_available == True
