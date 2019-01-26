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
        'tensorboard==1.12.2',
        'tensorflow==1.12.0',
        'colorama',
        'pickle'
    ]
)