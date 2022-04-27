import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.policy import Policy


class TestPolicy():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        policy = Policy()
        assert policy.is_available == False
