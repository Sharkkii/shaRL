import torch
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.common import UninitializedComponentException
from src.common import SA
from src.common import SARS
from src.common import SARSA
from src.common import SGASG
from src.common import SGARSG
from src.dataset import Dataset
from src.dataset import SarsDataset
from src.dataset import SgasgDataset
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
    def test_should_have_state_and_action_and_reward_and_nextstate(self):
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


class TestSgasgData():

    @pytest.mark.unit
    def test_random_method_should_return_sgasg_objects(self):
        N = 3
        dataset = SGASG.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SGASG) for data in dataset ])
    
    @pytest.mark.unit
    def test_should_have_state_and_goal_and_action_and_nextstate_and_nextgoal(self):
        data = SGASG.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "goal")
        assert hasattr(data, "action")
        assert hasattr(data, "next_state")
        assert hasattr(data, "next_goal")


class TestSgarsgData():

    @pytest.mark.unit
    def test_random_method_should_return_sgarsg_objects(self):
        N = 3
        dataset = SGARSG.random(n = N)
        assert len(dataset) == N
        assert all([ (type(data) is SGARSG) for data in dataset ])
    
    @pytest.mark.unit
    def test_should_have_state_and_goal_and_action_and_reward_and_nextstate_and_nextgoal(self):
        data = SGARSG.random(n = 1)[0]
        assert hasattr(data, "state")
        assert hasattr(data, "goal")
        assert hasattr(data, "action")
        assert hasattr(data, "reward")
        assert hasattr(data, "next_state")
        assert hasattr(data, "next_goal")


class TestDataset():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        dataset = Dataset()
        assert dataset.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization_with_empty_collection(self):
        dataset = Dataset(collection = [])
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_unavailable_on_nonempty_initialization_with_nonempty_collection(self):
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
    def test_should_raise_value_error_on_invalid_setup(self):
        collection = iter([ 1, 2, 3 ]) # invalid
        dataset = Dataset()
        with pytest.raises(ValueError) as message:
            dataset.setup(
                collection = collection
            )

    @pytest.mark.unit
    def test_add_collection_method_should_increase_dataset_size(self):
        dataset = Dataset(collection = [ 1, 2, 3 ])
        assert len(dataset) == 3
        dataset.add_collection(collection = [ 4, 5 ])
        assert len(dataset) == 5

    @pytest.mark.unit
    def test_add_item_method_should_increase_dataset_size(self):
        dataset = Dataset(collection = [ 1, 2, 3 ])
        assert len(dataset) == 3
        dataset.add_item(item = 4)
        assert len(dataset) == 4

    @pytest.mark.unit
    def test_add_method_should_be_alias_of_add_collection_method(self):
        dataset = Dataset(collection = [ 1, 2, 3 ])
        dataset.add([ 4, 5 ])
        assert len(dataset) == 5
        with pytest.raises(ValueError) as message:
            dataset.add(6) # invalid

    @pytest.mark.unit
    def test_remove_collection_method_should_decrease_dataset_size(self):
        dataset = Dataset(collection = [ 1, 2, 3, 4, 5 ])
        assert len(dataset) == 5
        dataset.remove_collection(n = 2)
        assert len(dataset) == 3

    @pytest.mark.unit
    def test_remove_item_method_should_decrease_dataset_size_by_one(self):
        dataset = Dataset(collection = [ 1, 2, 3, 4, 5 ])
        assert len(dataset) == 5
        dataset.remove_item()
        assert len(dataset) == 4

    @pytest.mark.unit
    def test_remove_method_should_be_alias_of_remove_collection_method(self):
        dataset = Dataset(collection = [ 1, 2, 3, 4, 5 ])
        assert len(dataset) == 5
        dataset.remove(n = 2)
        assert len(dataset) == 3

    @pytest.mark.unit
    def test_should_raise_value_error_if_collection_is_none(self):
        dataset = Dataset(collection = None)
        with pytest.raises(UninitializedComponentException) as message:
            _ = dataset[0]

    @pytest.mark.unit
    def test_should_return_none_if_collection_is_empty(self):
        dataset = Dataset(collection = [])
        for idx in [ 0, 1, 10, 100, 1000, 10000 ]:
            assert dataset[idx] is None

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "input_collection, expected_size",
        [
            ([], 0),
            (list(range(100)), 100)
        ]
    )
    def test_should_have_size(self, input_collection, expected_size):
        dataset = Dataset(collection = input_collection)
        assert dataset.size == expected_size

    @pytest.mark.unit
    def test_should_have_positive_max_size(self):
        dataset = Dataset(collection = [])
        assert dataset.max_size > 0

    @pytest.mark.unit
    def test_size_should_be_changed_on_addition(self):
        batch1 = [ i + 100 for i in range(10) ]
        batch2 = [ i + 200 for i in range(20) ]
        batch3 = [ i + 300 for i in range(30) ]
        dataset = Dataset(
            collection = [],
            max_size = 10000
        )

        dataset.add(batch1)
        assert dataset.size == 10
        dataset.add(batch2)
        assert dataset.size == 30
        dataset.add(batch3)
        assert dataset.size == 60

    @pytest.mark.unit
    def test_size_should_be_changed_on_removal(self):
        dataset = Dataset(
            collection = list(range(100)),
            max_size = 10000
        )
        dataset.remove(n = 10)
        assert dataset.size == 90
        dataset.remove(n = 20)
        assert dataset.size == 70
        dataset.remove(n = 30)
        assert dataset.size == 40

    @pytest.mark.unit
    def test_should_store_at_most_max_size_data_on_initialization(self):
        dataset = Dataset(
            collection = list(range(1000)),
            max_size = 100
        )
        assert dataset.size == 100

    @pytest.mark.unit
    def test_should_store_at_most_max_size_data_on_addition(self):
        batch1 = [ i + 100 for i in range(20) ]
        batch2 = [ i + 200 for i in range(30) ]
        batch3 = [ i + 300 for i in range(40) ]
        batch4 = [ i + 400 for i in range(50) ]
        batch5 = [ i + 500 for i in range(60) ]
        dataset = Dataset(
            collection = [],
            max_size = 100
        )

        assert dataset.size == 0
        dataset.add(batch1)
        assert dataset.size == 20
        dataset.add(batch2)
        assert dataset.size == 50
        dataset.add(batch3)
        assert dataset.size == 90
        dataset.add(batch4)
        assert dataset.size == 100
        dataset.add(batch5)
        assert dataset.size == 100


class TestSarsDataset():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        dataset = SarsDataset()
        assert dataset.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization_with_empty_collection(self):
        dataset = SarsDataset(collection = [])
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_unavailable_on_nonempty_initialization_with_nonempty_collection(self):
        collection = SARS.random(n = 3)
        dataset = Dataset(
            collection = collection
        )
        assert dataset.is_available == True

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
        "TDataset", [ SA, SARSA, SGASG, SGARSG ]
    )
    def test_should_raise_value_error_on_invalid_setup(self, TDataset):
        collection = TDataset.random(n = 3) # invalid
        dataset = SarsDataset()
        with pytest.raises(ValueError) as message:
            dataset.setup(
                collection = collection
            )

    @pytest.mark.unit
    def test_valid_collection_can_be_added_to_dataset(self):
        dataset = SarsDataset(collection = SARS.random(n = 3))
        assert len(dataset) == 3
        dataset.add(SARS.random(n = 2))
        assert len(dataset) == 5

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TDataset", [ SA, SARSA, SGASG, SGARSG ]
    )
    def test_invalid_collection_cannot_be_added_to_dataset(self, TDataset):
        dataset = SarsDataset(collection = SARS.random(n = 3))
        assert len(dataset) == 3
        with pytest.raises(ValueError) as message:
            dataset.add(TDataset.random(n = 2)) # invalid


class TestSgasgDataset():

    @pytest.mark.unit
    def test_should_be_unavailable_on_empty_initialization(self):
        dataset = SgasgDataset()
        assert dataset.is_available == False

    @pytest.mark.unit
    def test_should_be_available_on_nonempty_initialization_with_empty_collection(self):
        dataset = SgasgDataset(collection = [])
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_unavailable_on_nonempty_initialization_with_nonempty_collection(self):
        collection = SGASG.random(n = 3)
        dataset = SgasgDataset(
            collection = collection
        )
        assert dataset.is_available == True

    @pytest.mark.unit
    def test_should_be_available_after_valid_setup(self):
        collection = SGASG.random(n = 3)
        dataset = SgasgDataset()
        dataset.setup(
            collection = collection
        )
        assert dataset.is_available == True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TDataset", [ SA, SARS, SARSA, SGARSG ]
    )
    def test_should_raise_value_error_on_invalid_setup(self, TDataset):
        collection = TDataset.random(n = 3) # invalid
        dataset = SgasgDataset()
        with pytest.raises(ValueError) as message:
            dataset.setup(
                collection = collection
            )

    @pytest.mark.unit
    def test_valid_collection_can_be_added_to_dataset(self):
        dataset = SgasgDataset(collection = SGASG.random(n = 3))
        assert len(dataset) == 3
        dataset.add(SGASG.random(n = 2))
        assert len(dataset) == 5

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "TDataset", [ SA, SARS, SARSA, SGARSG ]
    )
    def test_invalid_collection_cannot_be_added_to_dataset(self, TDataset):
        dataset = SgasgDataset(collection = SGASG.random(n = 3))
        assert len(dataset) == 3
        with pytest.raises(ValueError) as message:
            dataset.add(TDataset.random(n = 2)) # invalid


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

    @pytest.mark.unit
    def test_should_allow_dataset_with_empty_collection(self):
        dataset = Dataset(collection = [])
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = False
        )
        assert dataloader.is_available == True

    @pytest.mark.unit
    def test_should_raise_value_error_if_collection_is_none(self):
        dataset = Dataset(collection = None)
        dataloader = DataLoader(
            dataset = dataset
        )
        with pytest.raises(UninitializedComponentException) as message:
            loader = iter(dataloader)
            _ = next(loader)

    @pytest.mark.unit
    def test_should_return_none_if_collection_is_empty(self):
        dataset = Dataset(collection = [])
        dataloader = DataLoader(
            dataset = dataset
        )
        loader = iter(dataloader)
        for _ in range(3):
            batch = next(loader)
            assert batch is None

    @pytest.mark.unit
    def test_should_yield_item_in_random_order(self):
        dataset = Dataset(collection = [ "a", "b", "c" ])
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = True
        )

        loader = iter(dataloader)
        collection = set()
        for _ in range(3):
            item = next(loader)
            collection.add(item[0])
        assert collection == { "a", "b", "c" }

    @pytest.mark.unit
    def test_should_yield_item_in_order(self):
        dataset = Dataset(collection = [ "a", "b", "c" ])
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = False
        )

        loader = iter(dataloader)
        item = next(loader)
        assert item[0] == "a"
        item = next(loader)
        assert item[0] == "b"
        item = next(loader)
        assert item[0] == "c"

    @pytest.mark.unit
    def test_should_yield_batch_in_order(self):
        dataset = Dataset(collection = [ "a", "b", "c", "d" ])
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = 2,
            shuffle = False
        )

        loader = iter(dataloader)
        batch = next(loader)
        assert batch[0] == "a"
        assert batch[1] == "b"
        batch = next(loader)
        assert batch[0] == "c"
        assert batch[1] == "d"
