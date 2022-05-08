#### Interface ####

from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

from ..const import SpaceType
from .helper import _is_tuple_of_int


class BaseInterface(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class AgentInterface(BaseInterface):

    def __init__(
        self,
        din: Union[int, Tuple[int, ...]],
        dout: Union[int, Tuple[int, ...]]
    ):
        if (type(din) is int):
            self.din = (din,)
        elif (_is_tuple_of_int(din)):
            self.din = din
        else:
            raise ValueError("din: Union[int, Tuple[int, ...]]")

        if (type(dout) is int):
            self.dout = (dout,)
        elif (_is_tuple_of_int(dout)):
            self.dout = dout
        else:
            raise ValueError("dout: Union[int, Tuple[int, ...]]")


class EnvironmentInterface(BaseInterface):

    def __init__(
        self,
        observation_type: SpaceType,
        action_type: SpaceType
    ):
        if (type(observation_type) is SpaceType):
            self.observation_type = observation_type
        else:
            raise ValueError("observation_type: SpaceType")

        if (type(action_type) is SpaceType):
                self.action_type = action_type
        else:
            raise ValueError("action_type: SpaceType")
