from gym.envs.registration import register

register(
    id='pycon-v0',
    entry_point='ml2_gym.envs:PyconWalker',
    max_episode_steps=10000,
    reward_threshold=200,
)

register(
    id='pycon-v2',
    entry_point='ml2_gym.envs:PyconWalkerTwo',
    max_episode_steps=10000,
    reward_threshold=200,
)
