from gym.envs.registration import register

# TODO: split environment.py to support different entry points
register(
    id='snaketh-v0',
    entry_point='ml2_gym.snaketh:Snaketh'
)
