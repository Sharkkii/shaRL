import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.const import Interface
from src.network import MetaNetwork
from src.network import BaseNetwork
from src.network import DefaultNetwork

@pytest.mark.L5
class TestMetaNetwork():

    @pytest.mark.unit
    def test_should_return_network_with_valid_definition(self):
        
        class C(metaclass=MetaNetwork):
            spec = "default"
        interface = Interface(din = 1, dout = 1)
        network = C(interface = interface)
        assert isinstance(network, BaseNetwork)
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_invalid_definition(self):
        
        class C(metaclass=MetaNetwork):
            spec = "INVALID_SPEC"
        interface = Interface(din = 1, dout = 1)
        with pytest.raises(ValueError) as message:
            network = C(interface = interface)
    
    @pytest.mark.unit
    def test_should_be_metaclass_of_default_network(self):
        assert(MetaNetwork in DefaultNetwork.__metaclass__)
