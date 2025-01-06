import pandas as pd
from sqlalchemy import create_engine, text
from typing import Dict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LegoDataLoader:
    def __init__(self, db_url: str):
        """Initialize the data loader with database connection."""
        self.engine = create_engine(db_url)
        self.logger = logging.getLogger(__name__)

    def create_tables(self):
        """Create the new tables for instruction steps."""
        sql = [
            """
                CREATE TABLE IF NOT EXISTS set_steps (
                    inventory_id INT,
                    booklet_number INT,
                    step_number INT,
                    page_number INT,
                    PRIMARY KEY (inventory_id, booklet_number, step_number),
                    FOREIGN KEY (inventory_id) REFERENCES inventories(id)
                );
            """,
            """
                CREATE TABLE IF NOT EXISTS step_elements (
                    inventory_id INT,
                    booklet_number INT,
                    step_number INT,
                    element_id VARCHAR(10),
                    quantity INT,
                    PRIMARY KEY (inventory_id, booklet_number, step_number, element_id),
                    FOREIGN KEY (inventory_id, booklet_number, step_number) 
                        REFERENCES set_steps(inventory_id, booklet_number, step_number),
                    FOREIGN KEY (element_id) REFERENCES elements(element_id)
                );
            """,
            "CREATE INDEX IF NOT EXISTS idx_set_steps_inventory ON set_steps(inventory_id);",
            "CREATE INDEX IF NOT EXISTS idx_step_elements_element ON step_elements(element_id);",
        ]
        with self.engine.connect() as conn:
            for cmd in sql:
                conn.execute(text(cmd))
                conn.commit()
        self.logger.info("Created tables and indexes")

    def validate_data(
        self, steps_df: pd.DataFrame, elements_df: pd.DataFrame
    ) -> bool:
        """
        Validate the input data against existing database records.
        Returns True if validation passes, False otherwise.
        """

        with self.engine.connect() as conn:
            existing_inventories = pd.read_sql(
                "SELECT id FROM inventories", conn
            )
            existing_elements = pd.read_sql(
                "SELECT element_id FROM elements", conn
            )

        invalid_inventories = ~steps_df["inventory_id"].isin(
            existing_inventories["id"]
        )
        if invalid_inventories.any():
            self.logger.error(
                f"Invalid inventory_ids found: {steps_df[invalid_inventories]\
                                                ['inventory_id'].unique()}"
            )
            return False

        invalid_elements = ~elements_df["element_id"].isin(
            existing_elements["element_id"]
        )
        if invalid_elements.any():
            self.logger.error(
                f"Invalid element_ids found: {elements_df[invalid_elements]\
                                              ['element_id'].unique()}"
            )
            return False

        return True

    def load_data(self, steps_csv: str, elements_csv: str):
        """Load data from CSV files into the database."""

        try:
            steps_df = pd.read_csv(steps_csv)
            elements_df = pd.read_csv(elements_csv)

            if not self.validate_data(steps_df, elements_df):
                self.logger.error("Data validation failed")
                return

            with self.engine.begin() as conn:
                steps_df.to_sql(
                    "set_steps", conn, if_exists="append", index=False
                )
                elements_df.to_sql(
                    "step_elements", conn, if_exists="append", index=False
                )

            self.logger.info(
                f"Successfully loaded {len(steps_df)} steps and \
                    {len(elements_df)} elements"
            )

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
