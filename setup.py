from __future__ import print_function

from setuptools import setup

version = '0.1.0'

description = 'Python code from the book ML in Python'

def requirements(filename):
    with open(filename) as f:
        return f.readlines()


# Read this article before you start doing anything here:
#
#    https://caremad.io/2013/07/setup-vs-requirement/
#
# tl;dr: setup.py is for abstract (un-pinned), high-level dependencies.
#
# requirements.txt is for pinned dependencies and is used for repeatable
# builds.

setup(
    author='Timofey Asyrkin',
    author_email='tasyrkin@gmail.com',
    name='ml-in-python-raschka',
    description=description,
    install_requires=requirements("requirements.in"),
    tests_require=requirements('requirements-test.in'),
    license='closed',
    long_description=description,
    packages=['classification', 'plots'],
    url='https://github.com/tasyrkin/ml-in-python-raschka.git',
    version=version,
    zip_safe=False,
)
