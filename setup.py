from setuptools import setup

setup(
    name = 'qchsleep',
    version = '0.0.1',
    packages = ['qchsleep'],
    entry_points = {
        'console_scripts': [
            'qchsleep = qchsleep.__main__:main'
        ]
    })
