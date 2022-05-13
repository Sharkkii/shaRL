import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.common import AgentInterface
from src.network import MetaNetwork
from src.network import BaseNetwork

default_agent_interface = AgentInterface(
    sin = 1,
    sout = 1
)

@pytest.mark.L5
class TestMetaNetwork():

    @pytest.mark.unit
    def test_should_return_network_with_valid_definition(self):
        class C(metaclass=MetaNetwork):
            spec = "default"
        interface = default_agent_interface
        network = C(interface = interface)
        assert isinstance(network, BaseNetwork)
    
    @pytest.mark.unit
    def test_should_raise_value_error_with_invalid_definition(self):
        class C(metaclass=MetaNetwork):
            spec = "INVALID_SPEC"
        interface = default_agent_interface
        with pytest.raises(ValueError) as message:
            network = C(interface = interface)
    
    @pytest.mark.unit
    def test_should_be_metaclass_of_default_network(self):
        class C(metaclass=MetaNetwork):
            spec = "default"
        assert(MetaNetwork in C.__metaclass__)
