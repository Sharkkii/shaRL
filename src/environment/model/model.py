#### Model ####

from .mixin import ModelMixin
from .mixin import ApproximateForwardDynamicsModelMixin
from .mixin import ApproximateInverseDynamicsModelMixin


class Model(ModelMixin):

    def __init__(
        self,
        env,
        configuration = None
    ):
        super().__init__(
            env,
            configuration = configuration
        )


class ApproximateForwardDynamicsModel(ApproximateForwardDynamicsModelMixin):

    def __init__(
        self,
        env,
        configuration = None
    ):
        super().__init__(
            env,
            configuration = configuration
        )


class ApproximateInverseDynamicsModel(ApproximateInverseDynamicsModelMixin):

    def __init__(
        self,
        env,
        configuration = None
    ):
        super().__init__(
            env,
            configuration = configuration
        )
