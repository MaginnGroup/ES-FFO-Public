from setuptools import setup, find_packages

#####################################
VERSION = "0.0.0"
ISRELEASED = False
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev0"
#####################################

setup(
    name="utils",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    author="Montana Carlozo",
    author_email="mcarlozo@nd.edu",
    python_requires=">=3.6, <4",
)
