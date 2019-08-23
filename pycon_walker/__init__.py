from gym.envs.registration import register

register(
    id='pycon-v0',
    entry_point='pycon_walker.envs:PyconWalker',
    max_episode_steps=10000,
    reward_threshold=200,
)

register(
    id='pycon-v2',
    entry_point='pycon_walker.envs:PyconWalkerTwo',
    max_episode_steps=10000,
    reward_threshold=200,
)
