#!python
"""Foundation components of the behavior tree to control rover decisions"""

from abc import ABCMeta, abstractmethod
from enum import Enum

class Result(Enum):
    """Result of the operation"""
    Failure = 0
    Success = 1
    Continue = 2

#pylint: disable=too-few-public-methods

class Node(metaclass=ABCMeta):
    """Base class for all behavior tree nodes"""

    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return "[" + type(self).__name__ + "]"


    @abstractmethod
    def run(self, rover):
        """Runs the node job, returns the result of the operation"""
        raise NotImplementedError


class Decorator(Node):
    """Modifies result of a single child node"""

    def __init__(self, child):
        self._child = child


    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return super().name + self._child.name


    @abstractmethod
    def run(self, rover):
        """Runs the node job, returns the result of the operation"""
        raise NotImplementedError


class CompoundNode(Node):
    """Bass class for all nodes that contain multiple children"""

    def __init__(self):
        self._children = []
        self._idx = 0


    def append(self, child):
        """Append a tree child to this node"""
        self._children.append(child)


    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return super().name + self._children[self._idx].name


    @abstractmethod
    def run(self, rover):
        """Runs the node job, returns the result of the operation"""
        raise NotImplementedError


class UntilFail(Decorator):
    """Loops the child node until it returns Result.Failure"""

    def run(self, rover):
        result = self._child.run(rover)
        if Result.Failure == result:
            return Result.Failure

        return Result.Continue


class Not(Decorator):
    """Performs logical-not operation for a node passed in the constructor"""

    def run(self, rover):
        result = self._child.run(rover)
        if Result.Failure == result:
            return Result.Success
        elif Result.Success == result:
            return Result.Failure

        return Result.Continue


class Sequence(CompoundNode):
    """Calls all children nodes in a sequence, until all of them succeed or one
    of them fails"""

    def run(self, rover):
        for i in range(self._idx, len(self._children)):
            child = self._children[i]

            result = child.run(rover)

            if Result.Continue == result:
                self._idx = i
                return Result.Continue
            elif Result.Failure == result:
                break

        self._idx = 0
        return result


class Selection(CompoundNode):
    """Calls each child in the appended order until a child succeeds"""

    def run(self, rover):
        for i in range(self._idx, len(self._children)):
            child = self._children[i]

            result = child.run(rover)

            if Result.Continue == result:
                self._idx = i
                return Result.Continue
            elif Result.Success == result:
                break

        self._idx = 0
        return result
