class ReferenceBase:
    
    def __init__(
        self,
        target = None
    ):
        self.setup(target = target)

    def setup(
        self,
        target
    ):
        self._target = target

    def __call__(
        self,
        *args,
        **kwargs
    ):
        return self._target.__call__(*args, **kwargs)


class ValueReference(ReferenceBase):
    pass


class QValueReference(ReferenceBase):
    pass


class AdvantageReference(ReferenceBase):
    pass


def dereference(reference):
    if (isinstance(reference, ReferenceBase)):
        return reference._target
    else:
        raise TypeError("`reference` must be `ReferenceBase.")
