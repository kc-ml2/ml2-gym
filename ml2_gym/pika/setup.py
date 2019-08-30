from setuptools import setup

# FIXME: add dependencies
setup(name='pycon_walker',
      version='0.0.1',
      install_requires=['gym',
                        'box2d-py~=2.3.5',
                        'keyboard',
                        'torch',
                        ],
      )
