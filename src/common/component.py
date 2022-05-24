from .error import UninitializedComponentException

class Component:

    def __init__(self):
        self._is_available = False

    @property
    def is_available(
        self
    ):
        return self._is_available

    def _become_available(
        self
    ):
        self._is_available = True

    def _become_unavailable(
        self
    ):
        self._is_available = False

    def check_whether_available(f):
        def wrapper(self, *args, **kwargs):
            if (not self.is_available):
                raise UninitializedComponentException(f"'{ self.__class__.__name__ }' object must be setup before using `{ self.__class__.__name__ }.{ f.__name__ }`")
            return f(self, *args, **kwargs)
        return wrapper
