#!python
"""Unit tests for a behavior tree basic"""

import unittest

from behavior_tree_basic import \
    Result, \
    Node, \
    UntilFail, \
    Not, \
    Sequence, \
    Selection


class Stub(Node):
    """Runs a node that returns constant value"""

    # pylint: disable=too-few-public-methods

    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = 0


    def run(self, rover):
        self.calls += 1
        return self.return_value


class TestBehaviorTreeBasic(unittest.TestCase):
    """Tests cases to cover behavior tree basic blocks"""

    def test_sequence_success(self):
        """Tests Sequence node when it succeeds"""

        success1 = Stub(Result.Success)
        success2 = Stub(Result.Success)

        sequence = Sequence()
        sequence.append(success1)
        sequence.append(success2)

        sequence.run(None)
        result = sequence.run(None)

        self.assertEqual(Result.Success, result)
        self.assertEqual(2, success1.calls)
        self.assertEqual(2, success2.calls)


    def test_sequence_failure(self):
        """Tests Sequence node when it fails"""

        success1 = Stub(Result.Success)
        success2 = Stub(Result.Success)
        continue3 = Stub(Result.Continue)
        failure4 = Stub(Result.Failure)
        failure5 = Stub(Result.Failure)

        sequence = Sequence()
        sequence.append(success1)
        sequence.append(success2)
        sequence.append(continue3)
        sequence.append(failure4)
        sequence.append(failure5)

        sequence.run(None)
        continue3.return_value = Result.Success
        result = sequence.run(None)

        self.assertEqual(Result.Failure, result)
        self.assertEqual(1, success1.calls)
        self.assertEqual(1, success2.calls)
        self.assertEqual(2, continue3.calls)
        self.assertEqual(1, failure4.calls)
        self.assertEqual(0, failure5.calls)


    def test_selection_success(self):
        """Tests Selection node when it succeeds"""

        failure1 = Stub(Result.Failure)
        failure2 = Stub(Result.Failure)
        continue3 = Stub(Result.Continue)
        success4 = Stub(Result.Success)
        success5 = Stub(Result.Success)

        selection = Selection()
        selection.append(failure1)
        selection.append(failure2)
        selection.append(continue3)
        selection.append(success4)
        selection.append(success5)

        selection.run(None)
        continue3.return_value = Result.Failure
        result = selection.run(None)

        self.assertEqual(Result.Success, result)
        self.assertEqual(1, failure1.calls)
        self.assertEqual(1, failure2.calls)
        self.assertEqual(2, continue3.calls)
        self.assertEqual(1, success4.calls)
        self.assertEqual(0, success5.calls)


    def test_selection_failure(self):
        """Tests Selection node when it fails"""

        failure1 = Stub(Result.Failure)
        failure2 = Stub(Result.Failure)

        selection = Selection()
        selection.append(failure1)
        selection.append(failure2)

        result = selection.run(None)

        self.assertEqual(Result.Failure, result)
        self.assertEqual(1, failure1.calls)
        self.assertEqual(1, failure2.calls)


    def test_until_fail_fails(self):
        """Tests UntilFail node when a child returns failure"""

        failure = Stub(Result.Failure)
        result = UntilFail(failure).run(None)

        self.assertEqual(Result.Failure, result)
        self.assertEqual(1, failure.calls)


    def test_until_fail_continues1(self):
        """Tests UntilFail node when a child returns success"""

        success = Stub(Result.Success)
        result = UntilFail(success).run(None)

        self.assertEqual(Result.Continue, result)
        self.assertEqual(1, success.calls)


    def test_until_fail_continues2(self):
        """Tests UntilFail node when a child returns continue"""

        cont = Stub(Result.Continue)
        result = UntilFail(cont).run(None)

        self.assertEqual(Result.Continue, result)
        self.assertEqual(1, cont.calls)


    def test_not(self):
        """Tests Not node"""

        self.assertEqual(Result.Continue, Not(Stub(Result.Continue)).run(None))
        self.assertEqual(Result.Success, Not(Stub(Result.Failure)).run(None))
        self.assertEqual(Result.Failure, Not(Stub(Result.Success)).run(None))



if __name__ == '__main__':
    unittest.main()
