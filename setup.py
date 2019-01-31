from setuptools import setup, find_packages
setup(
    name="BoneAge",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pip',
        'keras',
        'matplotlib',
        'numpy',
        'setuptools',
        'tensorboard',
        'tensorflow',
        'colorama',
        'pillow',
        'pillow-pip'
    ]
)