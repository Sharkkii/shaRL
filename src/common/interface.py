#### Interface ####

from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, Optional

from ..const import SpaceType
from .helper import is_tuple_of_int


class BaseInterface(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError


class AgentInterface(BaseInterface):

    # TO BE REMOVED: default value for compatibility
    def __init__(
        self,
        sin: Optional[Union[int, Tuple[int, ...]]],
        sout: Optional[Union[int, Tuple[int, ...]]],
        din: Optional[int] = None,
        dout: Optional[int] = None,
        tin: Optional[SpaceType] = None,
        tout: Optional[SpaceType] = None,
    ):
        if (type(sin) is int):
            self.sin = (sin,)
        elif (is_tuple_of_int(sin)):
            self.sin = sin
        else:
            raise ValueError("sin: Union[int, Tuple[int, ...]]")

        if (type(sout) is int):
            self.sout = (sout,)
        elif (is_tuple_of_int(sout)):
            self.sout = sout
        else:
            raise ValueError("sout: Union[int, Tuple[int, ...]]")

        if (din is None):
            self.din = None
        elif (type(din) is int):
            self.din = din
        else:
            raise ValueError("din: Optional[int]")

        if (dout is None):
            self.dout = None
        elif (type(dout) is int):
            self.dout = dout
        else:
            raise ValueError("dout: Optional[int]")

        if (tin is None):
            self.tin = None
        elif (type(tin) is SpaceType):
            self.tin = tin
        else:
            raise ValueError("tin: SpaceType")

        if (tout is None):
            self.tout = None
        elif (type(tout) is SpaceType):
            self.tout = tout
        else:
            raise ValueError("tout: SpaceType")


class EnvironmentInterface(BaseInterface):

    # TO BE REMOVED: default value for compatibility
    def __init__(
        self,
        observation_type: SpaceType,
        action_type: SpaceType,
        observation_shape: Optional[Tuple[int, ...]] = None, 
        action_shape: Optional[Tuple[int, ...]] = None,
        observation_ndim: Optional[int] = None,
        action_ndim: Optional[int] = None
    ):
        if (type(observation_type) is SpaceType):
            self.observation_type = observation_type
        else:
            raise ValueError("observation_type: SpaceType")

        if (type(action_type) is SpaceType):
            self.action_type = action_type
        else:
            raise ValueError("action_type: SpaceType")

        if (observation_shape is None):
            self.observation_shape = None
        elif (type(observation_shape) is int):
            self.observation_shape = (observation_shape,)
        elif (is_tuple_of_int(observation_shape)):
            self.observation_shape = observation_shape
        else:
            raise ValueError("observation_shape: Union[int, Tuple[int, ...]]")
        
        if (action_shape is None):
            self.action_shape = None
        elif (type(action_shape) is int):
            self.action_shape = (action_shape,)
        elif (is_tuple_of_int(action_shape)):
            self.action_shape = action_shape
        else:
            raise ValueError("action_shape: Union[int, Tuple[int, ...]]")

        self.observation_ndim = observation_ndim
        self.action_ndim = action_ndim
