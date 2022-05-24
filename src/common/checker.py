from .error import UninitializedComponentException

def check_whether_available(f):
    def wrapper(self, *args, **kwargs):
        if (not self.is_available):
            raise UninitializedComponentException(f"'{ self.__class__.__name__ }' object must be setup before using `{ self.__class__.__name__ }.{ f.__name__ }`")
        return f(self, *args, **kwargs)
    return wrapper