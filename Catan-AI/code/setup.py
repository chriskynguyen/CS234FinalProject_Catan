from setuptools import setup, find_packages

setup(
    name='catan',
    version='0.0.1',
    description='Custom Catan Environment for OpenAI Gym',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy','stable-baselines3','sb3_contrib'],
)