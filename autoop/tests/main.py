# flake8: noqa
import unittest

from autoop.tests.test_artifact import TestArtifactMethods  # noqa
from autoop.tests.test_database import TestDatabase  # noqa
from autoop.tests.test_features import TestFeatures  # noqa
from autoop.tests.test_metric import TestGetMetric  # noqa
from autoop.tests.test_pipeline import TestPipeline  # noqa
from autoop.tests.test_storage import TestStorage  # noqa


def storage_suite():
    """Storage test suite."""
    suite = unittest.TestSuite()
    print("---Running tests on the Storage module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStorage))
    return suite


def pipleline_suite():
    """Pipeline test suite."""
    suite = unittest.TestSuite()
    print("---Running tests on the Pipeline module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestPipeline))
    return suite


def test_get_metric():
    """Test the get_metric function."""
    suite = unittest.TestSuite()
    print("---Running tests on the Metric module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestGetMetric))
    return suite


def database_suite():
    """Database test suite."""
    suite = unittest.TestSuite()
    print("---Running tests on the Database module---")
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDatabase))
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
    runner.run(storage_suite())
    runner.run(pipleline_suite())
    runner.run(test_get_metric())
    runner.run(database_suite())
    runner.run(artifact_suite())
