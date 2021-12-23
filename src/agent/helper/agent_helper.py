#### Helper ####

from ...actor.helper import get_default_actor
from ...critic.helper import get_default_critic
from ..agent import Agent

def get_default_agent(
    env
):
    default_actor = get_default_actor(env)
    default_critic = get_default_critic(env)
    default_agent = Agent(
        actor = default_actor,
        critic = default_critic
    )
    return default_agent
