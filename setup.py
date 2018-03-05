import os
from setuptools import setup
from subprocess import CalledProcessError, call, check_call
from setuptools import Command
from abc import abstractmethod


class SimpleCommand(Command):
    """Make Command implementation simpler."""

    user_options = []

    def __init__(self, *args, **kwargs):
        """Store arguments so it's possible to call other commands later."""
        super().__init__(*args, **kwargs)
        self.__args = args
        self.__kwargs = kwargs

    @abstractmethod
    def run(self):
        """Run when command is invoked.

        Use *call* instead of *check_call* to ignore failures.
        """
        pass

    def run_command(self, command_class):
        """Run another command with same __init__ arguments."""
        command_class(*self.__args, **self.__kwargs).run()

    def initialize_options(self):
        """Set defa ult values for options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass


class Linter(SimpleCommand):
    """Lint Python source code."""

    description = 'lint Python source code'

    def run(self):
        """Run yala."""
        print('Yala is running. It may take several seconds...')
        try:
            check_call('yala iprocessor', shell=True)
            print('No linter error found.')
        except CalledProcessError:
            print('Linter check failed. Fix the error(s) above and try again.')
            exit(-1)


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "iprocessor",
    version = "0.0.1",
    author = "Macartur Sousa",
    author_email = "macartur.sc@gmail.com",
    description = ("A simple image processor."),
    setup_requires=['setuptools', 'sklearn', 'scikit-image'],
    extras_require={
        'dev':[
                    'ipython',
                    'yala',
                    'pylint',
                    'pytest-runner',
        ]
    },
    cmdclass={
        'lint': Linter
    },
    tests_require=['pytest'],
    license = "",
    keywords = "image processor ",
    url = "",
    packages=['iprocessor'],
    long_description=read('README.rst'),
    classifiers=[
        "Development Status :: 1 - Planning"
    ],
)
