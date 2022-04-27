import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.controller import Controller


class TestController():

    @pytest.mark.unit
    def test_init(self):
        controller = Controller()
        assert controller.is_available == False
