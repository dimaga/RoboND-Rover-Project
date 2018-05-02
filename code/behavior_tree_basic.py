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


    def __init__(self):
        self._trace = ""


    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return "[" + type(self).__name__ + "]"


    def trace(self):
        """Returns the trace of the most recent activity as a string"""
        return self._trace


    def run(self, rover, depth=0):
        """Call this method to run the node job"""
        #pylint:disable=unused-argument

        result = self._run(rover)
        self._trace = self.name + ": " + str(result)
        return result


    @abstractmethod
    def _run(self, rover):
        """Override this method to implement specific node activity"""
        raise NotImplementedError


class Decorator(Node):
    """Modifies result of a single child node"""

    def __init__(self, child):
        super().__init__()
        self._child = child


    @property
    def name(self):
        """Returns the name of the node for debugging output"""
        return super().name + self._child.name


    def run(self, rover, depth=0):
        """Call this method to run the node job"""
        result = self._run_child(rover, depth)

        self._trace = (
            self.name
            + ": "
            + str(result)
            + "->"
            + self._child.trace())

        return result


    @abstractmethod
    def _run_child(self, rover, depth):
        """Runs a child of the decorator"""
        raise NotImplementedError


    def _run(self, rover):
        """For Decorator, override _run_child() method to update the trace
        correctly"""
        pass


class CompoundNode(Node):
    """Bass class for all nodes that contain multiple children"""

    def __init__(self, name):
        super().__init__()

        self._children = []
        self._idx = 0
        self.__name = name


    def append(self, child):
        """Append a tree child to this node"""
        self._children.append(child)


    @property
    def name(self):
        """Returns the name of the node for debugging output"""

        return (
            "["
            + type(self).__name__
            + "(\""
            + self.__name
            + "\")"
            + "]")


    @abstractmethod
    def run(self, rover, depth=0):
        """Call this method to run the node job. Override this method,
        supporting correct self._trace field update"""
        raise NotImplementedError


    def _run(self, rover):
        """For CompoundNode, override public run() method to update the trace
        correctly"""
        pass


class UntilFail(Decorator):
    """Loops the child node until it returns Result.Failure"""

    def _run_child(self, rover, depth):
        result = self._child.run(rover, depth)
        if Result.Failure == result:
            return Result.Failure

        return Result.Continue


class Not(Decorator):
    """Performs logical-not operation for a node passed in the constructor"""

    def _run_child(self, rover, depth):
        result = self._child.run(rover, depth)
        if Result.Failure == result:
            return Result.Success
        elif Result.Success == result:
            return Result.Failure

        return Result.Continue


class Sequence(CompoundNode):
    """Calls all children nodes in a sequence, until all of them succeed or one
    of them fails"""

    def run(self, rover, depth=0):

        child_trace = ""

        for i in range(self._idx, len(self._children)):
            child = self._children[i]

            result = child.run(rover, depth + 1)
            child_trace += "\n" + (" " * (depth + 1)) + child.trace()

            if Result.Continue == result:
                self._idx = i
                break
            elif Result.Failure == result:
                break

        if Result.Continue != result:
            self._idx = 0

        self._trace = self.name + ": " + str(result) + child_trace
        return result


class Selection(CompoundNode):
    """Calls each child in the appended order until a child succeeds"""

    def run(self, rover, depth=0):

        child_trace = ""

        for i in range(self._idx, len(self._children)):
            child = self._children[i]

            result = child.run(rover, depth + 1)
            child_trace += "\n" + (" " * (depth + 1)) + child.trace()

            if Result.Continue == result:
                self._idx = i
                break
            elif Result.Success == result:
                break

        if Result.Continue != result:
            self._idx = 0

        self._trace = self.name + ": " + str(result) + child_trace
        return result
