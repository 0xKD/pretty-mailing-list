from setuptools import setup

# List of dependencies installed via `pip install -e .`
# by virtue of the Setuptools `install_requires` value below.
requires = [
    'pyramid',
    'gunicorn',
]

setup(
    name='pml',
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = pml:main'
        ],
    },
)
