import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.actor import Actor


class TestActor():

    @pytest.mark.unit
    def test_init(self):
        actor = Actor()
        assert actor.is_available == False
