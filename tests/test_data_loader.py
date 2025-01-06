# tests/test_data_loader.py

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from src.data_loader import LegoDataLoader


class TestLegoDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock connection that properly handles context management
        self.mock_connection = MagicMock()

        # Create a mock engine that returns our mock connection
        self.mock_engine = MagicMock()

        # Set up the context manager for connect()
        connect_context = MagicMock()
        connect_context.__enter__ = MagicMock(
            return_value=self.mock_connection
        )
        connect_context.__exit__ = MagicMock(return_value=None)
        self.mock_engine.connect.return_value = connect_context

        # Set up the context manager for begin()
        begin_context = MagicMock()
        begin_context.__enter__ = MagicMock(return_value=self.mock_connection)
        begin_context.__exit__ = MagicMock(return_value=None)
        self.mock_engine.begin.return_value = begin_context

        # Create patcher for create_engine
        self.engine_patcher = patch(
            "src.data_loader.create_engine", return_value=self.mock_engine
        )
        self.mock_create_engine = self.engine_patcher.start()

        # Create patcher for text
        self.text_patcher = patch("src.data_loader.text")
        self.mock_text = self.text_patcher.start()
        self.mock_text.side_effect = (
            lambda x: x
        )  # Simply return the input text

        # Initialize loader with mock database URL
        self.loader = LegoDataLoader("mock://db_url")

        # Mock the logger
        self.mock_logger = Mock()
        self.loader.logger = self.mock_logger

        # Sample test data
        self.sample_steps_data = {
            "inventory_id": [1, 1, 2],
            "booklet_number": [1, 1, 1],
            "step_number": [1, 2, 1],
            "page_number": [1, 1, 2],
        }

        self.sample_elements_data = {
            "inventory_id": [1, 1, 2],
            "booklet_number": [1, 1, 1],
            "step_number": [1, 1, 1],
            "element_id": ["123456", "789012", "123456"],
            "quantity": [1, 2, 1],
        }

    def tearDown(self):
        """Clean up patches after each test."""
        self.engine_patcher.stop()
        self.text_patcher.stop()

    def test_create_tables(self):
        """Test creation of set_steps and step_elements tables."""
        self.loader.create_tables()

        # Verify that execute was called for each SQL command
        execute_calls = self.mock_connection.execute.call_args_list
        self.assertEqual(len(execute_calls), 4)

        # Verify the SQL commands
        sql_commands = [str(call[0][0]) for call in execute_calls]
        self.assertTrue(
            any(
                "CREATE TABLE IF NOT EXISTS set_steps" in cmd
                for cmd in sql_commands
            )
        )
        self.assertTrue(
            any(
                "CREATE TABLE IF NOT EXISTS step_elements" in cmd
                for cmd in sql_commands
            )
        )
        self.assertTrue(
            any(
                "CREATE INDEX IF NOT EXISTS idx_set_steps_inventory" in cmd
                for cmd in sql_commands
            )
        )
        self.assertTrue(
            any(
                "CREATE INDEX IF NOT EXISTS idx_step_elements_element" in cmd
                for cmd in sql_commands
            )
        )

    @patch("pandas.read_sql")
    def test_validate_data_valid(self, mock_read_sql):
        """Test data validation with valid input."""
        # Mock the database query results
        mock_read_sql.side_effect = [
            pd.DataFrame({"id": [1, 2]}),  # inventories
            pd.DataFrame({"element_id": ["123456", "789012"]}),  # elements
        ]

        steps_df = pd.DataFrame(self.sample_steps_data)
        elements_df = pd.DataFrame(self.sample_elements_data)

        self.assertTrue(self.loader.validate_data(steps_df, elements_df))
        self.assertEqual(mock_read_sql.call_count, 2)

    @patch("pandas.read_sql")
    def test_validate_data_invalid_inventory(self, mock_read_sql):
        """Test data validation with invalid inventory_id."""
        # Mock the database query results
        mock_read_sql.side_effect = [
            pd.DataFrame({"id": [1]}),  # inventories (missing id 2)
            pd.DataFrame({"element_id": ["123456", "789012"]}),  # elements
        ]

        steps_df = pd.DataFrame(self.sample_steps_data)
        elements_df = pd.DataFrame(self.sample_elements_data)

        self.assertFalse(self.loader.validate_data(steps_df, elements_df))
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0][0]
        self.assertIn("Invalid inventory_ids found", args)

    @patch("pandas.read_sql")
    def test_validate_data_invalid_element(self, mock_read_sql):
        """Test data validation with invalid element_id."""
        # Mock the database query results
        mock_read_sql.side_effect = [
            pd.DataFrame({"id": [1, 2]}),  # inventories
            pd.DataFrame(
                {"element_id": ["123456"]}
            ),  # elements (missing 789012)
        ]

        steps_df = pd.DataFrame(self.sample_steps_data)
        elements_df = pd.DataFrame(self.sample_elements_data)

        self.assertFalse(self.loader.validate_data(steps_df, elements_df))
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0][0]
        self.assertIn("Invalid element_ids found", args)

    @patch("pandas.read_csv")
    def test_load_data_success(self, mock_read_csv):
        """Test successful data loading."""
        # Create mock DataFrames with to_sql method
        steps_df = MagicMock(spec=pd.DataFrame)
        elements_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.side_effect = [steps_df, elements_df]

        # Mock validate_data to return True
        with patch.object(self.loader, "validate_data", return_value=True):
            self.loader.load_data("mock_steps.csv", "mock_elements.csv")

        # Verify to_sql was called for both DataFrames
        steps_df.to_sql.assert_called_once_with(
            "set_steps", self.mock_connection, if_exists="append", index=False
        )
        elements_df.to_sql.assert_called_once_with(
            "step_elements",
            self.mock_connection,
            if_exists="append",
            index=False,
        )

    @patch("pandas.read_csv")
    def test_load_data_validation_failure(self, mock_read_csv):
        """Test data loading with validation failure."""
        # Mock the CSV reading
        steps_df = MagicMock(spec=pd.DataFrame)
        elements_df = MagicMock(spec=pd.DataFrame)
        mock_read_csv.side_effect = [steps_df, elements_df]

        # Mock validate_data to return False
        with patch.object(self.loader, "validate_data", return_value=False):
            self.loader.load_data("mock_steps.csv", "mock_elements.csv")
            self.mock_logger.error.assert_called_once_with(
                "Data validation failed"
            )

    @patch("pandas.read_csv")
    def test_load_data_exception(self, mock_read_csv):

        # Mock read_csv to raise an exception
        error_message = "Mock error"
        mock_read_csv.side_effect = Exception(error_message)

        # Test the exception is raised
        with self.assertRaises(Exception):
            self.loader.load_data("mock_steps.csv", "mock_elements.csv")

        # Verify the error was logged
        self.mock_logger.error.assert_called_once()
        args = self.mock_logger.error.call_args[0][0]
        self.assertIn("Error loading data", args)
        self.assertIn(error_message, args)


if __name__ == "__main__":
    unittest.main()
