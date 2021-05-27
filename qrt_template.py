import sys
import json

from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend


def program(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):
    """Function that does classical-quantum calculation."""
    # UserMessenger can be used to publish interim results.
    user_messenger.publish("This is an interim result.")
    return "final result"


def main(backend: ProgramBackend, user_messenger: UserMessenger, **kwargs):
    """This is the main entry point of a runtime program.

    The name of this method must not change. It also must have ``backend``
    and ``user_messenger`` as the first two positional arguments.

    Args:
        backend: Backend for the circuits to run on.
        user_messenger: Used to communicate with the program user.
        kwargs: User inputs.
    """
    # Massage the input if necessary.
    result = program(backend, user_messenger, **kwargs)
    # UserMessenger can be used to publish final results.
    user_messenger.publish(result, final=True)  