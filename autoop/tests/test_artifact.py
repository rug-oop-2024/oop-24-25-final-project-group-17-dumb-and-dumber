# flake8: noqa
import unittest

from autoop.core.ml.artifact import Artifact


class TestArtifact(Artifact):
    """Concrete test artifact class."""

    def read(self):
        """
        Read the data of the artifact.

        Needs to be noted that this returns the data part of the artifact
        and not all the information.
        """
        return self._data

    def save(self):
        """Save the data of the artifact."""
        return True


class TestArtifactMethods(unittest.TestCase):
    """Test case for the artifact class."""

    def setUp(self):
        """Set up the test case."""
        self.artifact = TestArtifact(
            name="test_artifact",
            asset_path="/path/to/asset",
            data=b"test_data",
            version="1.0.0",
        )

    def test_name_getter(self):
        """Test the name getter of the artifact."""
        self.assertEqual(self.artifact.name, "test_artifact")

    def test_asset_path_getter(self):
        """Test the asset path getter of the artifact."""
        self.assertEqual(self.artifact.asset_path, "/path/to/asset")

    def test_version_getter(self):
        """Test the version getter of the artifact."""
        self.assertEqual(self.artifact.version, "1.0.0")

    def test_name_setter(self):
        """Test the name setter of the artifact."""
        self.artifact.name = "new_name"
        self.assertEqual(self.artifact.name, "new_name")

    def test_read(self):
        """Test the read method of the artifact."""
        self.assertEqual(self.artifact.read(), b"test_data")

    def test_save(self):
        """Test the save method of the artifact."""
        self.assertTrue(self.artifact.save())


if __name__ == "__main__":
    unittest.main()
