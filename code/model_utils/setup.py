from setuptools import setup

setup(
    name='model_utils',    
    version='0.1',
    author='Henry Dashwood',   
    author_email='hcndashwood@gmail.com',
    packages=['model_utils'],
    install_requires=[
        'torch',
        'fastai'
    ],
)