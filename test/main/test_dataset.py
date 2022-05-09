import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.dataset import Dataset
from src.dataset import DataLoader


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
        dataloader = DataLoader()
        dataloader.setup(
            dataset = dataset,
            batch_size = -1, # invalid
        )
        assert dataloader.is_available == True
