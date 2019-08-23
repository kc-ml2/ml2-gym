import numpy as np
import keyboard

import gym
import gym_pycon

env = gym.make('pycon-v2')
#env = gym.make('pycon-v0')

s = env.reset(human=True)
score = 0
for _ in range(4000):
    #a = np.random.randint(env.action_space.n)
    a = np.zeros(4)
    while True:
        if keyboard.is_pressed('a'):
            break
    if keyboard.is_pressed('j'):
        a = a + np.array([0, 0, 1, -1])
    if keyboard.is_pressed('k'):
        a = a + np.array([0, 0, -1, 1])
    if keyboard.is_pressed('d'):
        a = a + np.array([-1, 1, 0, 0])
    if keyboard.is_pressed('f'):
        a = a + np.array([1, -1, 0, 0])
    #print(a)
    #a = 0.1 * a
    #a = np.random.choice([-1, 1], 4)
    #print(a)
    #a = env.action_space.sample()
    s, r, d, _ = env.step(a)
    env.step_human(a)
    score += r
    #print(r)
    env.render()

    if d:
        break

print(score)

