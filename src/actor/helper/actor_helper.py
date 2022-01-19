#### Helper ####

from ...policy.helper import get_default_policy
from ..actor import Actor

def get_default_actor(
    env
):
    default_policy = get_default_policy(env)
    default_actor = Actor(
        policy = default_policy
    )
    return default_actor
