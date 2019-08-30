## Playable walker for Pycon2019

This is a modified version of BipedalWalker from OpenAI gym env.

### Install

To install, run
``` pip install -e . ```
on this directory

### Difference with the original bipedal walker

The placement of the legs, and the shape of the main body has been modified. 

The input action are continuous, but in application, will be inputted with discrete choice of comibination of 
`{[-1, 1], [1, -1], [0, 0]}` for each leg, resulting with 9 different combination of inputs.
