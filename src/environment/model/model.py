#### Model ####

from .base import ModelBase
from .mixin import ModelMixin


class Model(ModelMixin, ModelBase):

    def __init__(
        self,
        env,
        configuration = None
    ):
        super().__init__(
            env,
            configuration = configuration
        )
