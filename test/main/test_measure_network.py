import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.network import ValueNetwork, QValueNetwork, PolicyNetwork


class TestValueNetwork():

    @pytest.mark.unit
    def test_init(self):
        value_network = ValueNetwork()
        assert value_network.is_available == False

class TestQValueNetwork():

    @pytest.mark.unit
    def test_init(self):
        qvalue_network = QValueNetwork()
        assert qvalue_network.is_available == False

class TestPolicyNetwork():

    @pytest.mark.unit
    def test_init(self):
        policy_network = PolicyNetwork()
        assert policy_network.is_available == False
