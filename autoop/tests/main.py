# flake8: noqa
import unittest

from autoop.tests.test_artifact import TestArtifactMethods  # noqa
from autoop.tests.test_database import TestDatabase  # noqa
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_metric import TestGetMetric  # noqa

# from autoop.tests.test_pipeline import TestPipeline # noqa
from autoop.tests.test_storage import TestStorage  # noqa


def test_get_metric():
    """Test the get_metric function."""
    suite = unittest.TestSuite()
    print("---Running tests on the Metric module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGetMetric))
    return suite


def suite():
    """
    Return a test suite.

    We create a test suite that only runs the test we want it to run.
    You add test with the addTest() method after initializing the suite.
    """
    suite = unittest.TestSuite()
    print("---Running tests on the Feature module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestFeatures))
    return suite


def artifact_suite():
    """Artifact test suite."""
    suite = unittest.TestSuite()
    print("---Running tests on the Artifact module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestArtifactMethods))
    return suite


if __name__ == "__main__":
    """
    Run the test suite.

    We first instantiate a unittest runner and then run the suite.
    """
    runner = unittest.TextTestRunner()
    runner.run(suite())
    runner.run(artifact_suite())
    runner.run(test_get_metric())
