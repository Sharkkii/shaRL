#### Helper ####

from ...value.helper import get_default_value
from ...value.helper import get_default_qvalue
from ..critic import Critic

def get_default_critic(
    env
):
    default_value = get_default_value(env)
    default_qvalue = get_default_qvalue(env)
    default_critic = Critic(
        value = default_value,
        qvalue = default_qvalue
    )
    return default_critic
