import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.dataset import SA
from src.dataset import SARS
from src.dataset import SARSA
from src.dataset import SAG
from src.dataset import Dataset
from src.dataset import SarsDataset
from src.dataset import DataLoader


class TestSaData():

    @pytest.mark.unit
    def test_random_method_should_return_sa_objects(self):
        N = 3
        dataset = SA.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SA) for data in dataset ])

    @pytest.mark.unit
    def test_should_have_state_and_action(self):
        data = SA.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "action")


class TestSarsData():

    @pytest.mark.unit
    def test_random_method_should_return_sars_objects(self):
        N = 3
        dataset = SARS.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SARS) for data in dataset ])
    
    @pytest.mark.unit
    def test_should_have_state_and_action_and_reward_and_nextaction(self):
        data = SARS.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "action")
        assert hasattr(data, "reward")
        assert hasattr(data, "next_state")


class TestSarsaData():

    @pytest.mark.unit
    def test_random_method_should_return_sarsa_objects(self):
        N = 3
        dataset = SARSA.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SARSA) for data in dataset ])

    @pytest.mark.unit
    def test_should_have_state_and_action_and_reward_and_nextstate_and_nextaction(self):
        data = SARSA.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "action")
        assert hasattr(data, "reward")
        assert hasattr(data, "next_state")
        assert hasattr(data, "next_action")


class TestSagData():

    @pytest.mark.unit
    def test_random_method_should_return_sag_objects(self):
        N = 3
        dataset = SAG.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SAG) for data in dataset ])
    
    @pytest.mark.unit
    def test_should_have_state_and_action_and_goal(self):
        data = SAG.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "action")
        assert hasattr(data, "goal")


class TestDataset():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        dataset = Dataset()
        assert dataset.is_available == False

    @pytest.mark.unit
    def test_should_be_unavailable_on_nonempty_initialization(self):
        collection = [ 1, 2, 3 ]
        dataset = Dataset(
            collection = collection
        )
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        collection = [ 1, 2, 3 ]
        dataset = Dataset()
        dataset.setup(
            collection = collection
        )
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_unavailable_after_invalid_setup(self):
        dataset = Dataset()
        dataset.setup(
            collection = iter([ 1, 2, 3 ]) # invalid
        )
        assert dataset.is_available == False


class TestSarsDataset():

    @pytest.mark.unit
    def test_should_be_unavailable_on_initialization(self):
        dataset = SarsDataset()
        assert dataset.is_available == False

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        collection = SARS.random(n = 3)
        dataset = SarsDataset()
        dataset.setup(
            collection = collection
        )
        assert dataset.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TDataset", [ SA, SARSA, SAG ]
    )
    def test_should_raise_value_error_on_invalid_setup(self, TDataset):
        dataset = SarsDataset()
        collection = TDataset.random(n = 3)
        with pytest.raises(ValueError) as message:
            dataset.setup(
                collection = collection
            )


class TestDataLoader():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        dataloader = DataLoader()
        assert dataloader.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization(self):
        dataset = Dataset(
            collection = [ 1, 2, 3 ]
        )
        dataloader = DataLoader(
            dataset = dataset
        )
        assert dataloader.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        dataset = Dataset(
            collection = [ 1, 2, 3 ]
        )
        dataloader = DataLoader()
        dataloader.setup(
            dataset = dataset
        )
        assert dataloader.is_available == True

    @pytest.mark.unit
    def test_should_be_unavailable_after_invalid_setup(self):
        dataset = Dataset(collection = [ 1, 2, 3 ])
        dataloader = DataLoader(
            batch_size = -1, # invalid
            shuffle = True
        )
        dataloader.setup(
            dataset = dataset,
            batch_size = -1, # invalid
            shuffle = True
        )
        assert dataloader.is_available == False
