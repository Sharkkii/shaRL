import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.network import PseudoMeasureNetwork
from src.optimizer import BaseMeasureOptimizer
from src.optimizer import MeasureOptimizer
from src.optimizer import MetaMeasureOptimizer


factory = torch.optim.Adam

@pytest.mark.L5
class TestMetaMeasureOptimizer():

    @pytest.mark.unit
    def test_should_return_measure_optimizer(self):
        class C(metaclass=MetaMeasureOptimizer):
            factory = factory
        optimizer = C()
        assert isinstance(optimizer, BaseMeasureOptimizer)

    @pytest.mark.unit
    def test_should_be_metaclass_of_measure_optimizer(self):
        assert(MetaMeasureOptimizer in MeasureOptimizer.__metaclass__)

@pytest.mark.L5
class TestMeasureOptimizer():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        optimizer = MeasureOptimizer()
        assert optimizer.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        network = PseudoMeasureNetwork()
        optimizer = MeasureOptimizer(
            network = network
        )
        assert optimizer.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_setup(self):
        network = PseudoMeasureNetwork()
        optimizer = MeasureOptimizer()
        optimizer.setup(network)
        assert optimizer.is_available == True
