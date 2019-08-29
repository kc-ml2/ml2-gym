from setuptools import setup, find_packages

setup(name='ml2-gym',
      version='0.1.0',
      install_requires=['gym',
                        'box2d-py~=2.3.5',
                        'keyboard',
                        'torch',
                        'tqdm',
                        'termcolor',
                        'tensorboardX',
                        ],
      packages=[package for package in find_packages()
                if package.startswith('ml2_gym')]
      )
