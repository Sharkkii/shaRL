import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.value import Value, QValue


class TestValue():

    @pytest.mark.unit
    def test_init(self):
        value = Value()
        assert value.is_available == False

class TestQValue():

    @pytest.mark.unit
    def test_init(self):
        value = QValue()
        assert value.is_available == False
