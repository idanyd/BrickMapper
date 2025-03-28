import pandas as pd
import logging
from sqlalchemy import create_engine, text
from pathlib import Path

logger = logging.getLogger(__name__)


class LegoDataLoader:
    def __init__(
        self,
        db_url: str,
        manual_mapping_csv: str,
        inventories_csv: str,
        manual_id: int,
    ):
        """Initialize the data loader with database connection."""
        self.engine = create_engine(db_url)
        self.logger = logging.getLogger(__name__)
        self.manual_mapping_csv = manual_mapping_csv
        self.inventories_csv = inventories_csv
        self.manual_id = manual_id
        self.batch_size = 1000  # Batch size for inserting step elements

    def __del__(self):
        """Dispose of the database connection engine"""
        self.engine.dispose()

    def create_tables(self):
        """Create the new tables for instruction steps"""
        sql = [
            """
                CREATE TABLE IF NOT EXISTS set_steps (
                    step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inventory_id INT,
                    booklet_number INT,
                    page_number INT,
                    step_number INT,
                    UNIQUE (inventory_id, booklet_number, page_number, step_number),
                    FOREIGN KEY (inventory_id) REFERENCES inventories(id)
                );
            """,
            """
                CREATE TABLE IF NOT EXISTS step_elements (
                    step_id INTEGER,
                    element_id VARCHAR(10),
                    PRIMARY KEY (step_id, element_id),
                    FOREIGN KEY (step_id) REFERENCES set_steps(step_id),
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
                f"Invalid inventory_ids found: {steps_df[invalid_inventories]['inventory_id'].unique()}"
            )
            return False

        invalid_elements = ~elements_df["element_id"].isin(
            existing_elements["element_id"]
        )
        if invalid_elements.any():
            self.logger.error(
                f"Invalid element_ids found: {elements_df[invalid_elements]['element_id'].unique()}"
            )
            return False

        return True

    def load_reference_data(self):
        """
        Load manual mapping and inventories data from CSV files.

        Args:
            None

        Returns:
            tuple: (manual_mapping DataFrame, inventories DataFrame) or (None, None) if error
        """
        try:
            manual_mapping = pd.read_csv(self.manual_mapping_csv)
            inventories = pd.read_csv(self.inventories_csv)

            # set nums are saved as "{set_num}-{n}", so we split on '-' and extract the first part
            inventories["pure_set_num"] = (
                inventories["set_num"].str.split("-").str[0]
            )

            # Log sample data for debugging
            self.logger.debug("Manual mapping sample:")
            self.logger.debug(manual_mapping.head())
            self.logger.debug("Inventories sample:")
            self.logger.debug(inventories.head())

            return manual_mapping, inventories
        except Exception as e:
            self.logger.error(f"Error loading CSV files: {str(e)}")
            return None, None

    def get_inventory_info(self, manual_id, manual_mapping, inventories):
        """
        Get inventory information for a manual.

        Args:
            manual_id: ID of the manual
            manual_mapping: DataFrame with manual to set mapping
            inventories: DataFrame with inventory information

        Returns:
            tuple: (inventory_id, booklet_number) or (None, None) if not found
        """
        # Get set_num and booklet_number from mapping
        if manual_id not in manual_mapping["manual_id"].values:
            self.logger.warning(
                f"Manual ID {manual_id} not found in mapping. Skipping."
            )
            return None, None

        manual_info = manual_mapping[
            manual_mapping["manual_id"] == manual_id
        ].iloc[0]
        set_num = manual_info["set_num"]
        booklet_number = manual_info["booklet_number"]

        # Get inventory_id from inventories
        inventory_row = inventories[
            inventories["pure_set_num"] == str(set_num)
        ]
        if inventory_row.empty:
            self.logger.warning(
                f"Set number {set_num} not found in inventories. Skipping."
            )
            return None, None

        inventory_id = inventory_row.iloc[0]["id"]
        return inventory_id, booklet_number

    def extract_unique_steps(
        self, manual_data_dict, manual_mapping, inventories
    ):
        """
        Extract unique steps from manual data dictionary.

        Args:
            manual_data_dict: Dictionary with manual data
            manual_mapping: DataFrame with manual to set mapping
            inventories: DataFrame with inventory information

        Returns:
            DataFrame: DataFrame with unique steps
        """
        set_steps_data = []

        # Process the dictionary to collect unique steps
        inventory_id, booklet_number = self.get_inventory_info(
            self.manual_id, manual_mapping, inventories
        )
        if inventory_id is None:
            return None

        # Collect unique steps
        unique_steps = (
            {}
        )  # (inventory_id, booklet_number, step_number) -> page_number

        # Process element data to identify unique steps
        for pages in manual_data_dict.values():
            for page_num, steps_info in pages.items():
                for step_info in steps_info:
                    step_number = step_info["step"]
                    step_key = (inventory_id, booklet_number, step_number)
                    unique_steps[step_key] = page_num

        # Add unique steps to set_steps_data
        for (inv_id, book_num, step_num), page_num in unique_steps.items():
            set_steps_data.append([inv_id, book_num, step_num, page_num])

        # Convert to dataframe
        set_steps_df = pd.DataFrame(
            set_steps_data,
            columns=[
                "inventory_id",
                "booklet_number",
                "step_number",
                "page_number",
            ],
        )
        return set_steps_df.drop_duplicates()

    def _get_step_id_from_set_steps(
        self,
        connection,
        inventory_id,
        booklet_number,
        page_number,
        step_number,
    ):
        query = text(
            """
            SELECT step_id FROM set_steps 
            WHERE inventory_id = :inv_id AND booklet_number = :book_num AND page_number = :page_num AND step_number = :step_num 
        """
        )
        return connection.execute(
            query,
            {
                "inv_id": inventory_id,
                "book_num": booklet_number,
                "page_num": page_number,
                "step_num": step_number,
            },
        ).fetchone()

    def insert_steps_and_get_ids(self, set_steps_df):
        """
        Insert steps into the database and get back the assigned step_ids.

        Args:
            set_steps_df: DataFrame with step data
            engine: SQLAlchemy engine

        Returns:
            dict: Dictionary mapping (inventory_id, booklet_number, step_number) to step_id
        """
        step_id_map = (
            {}
        )  # Maps (inventory_id, booklet_number, step_number) to step_id

        try:
            with self.engine.begin() as conn:
                # Insert set_steps data
                for _, row in set_steps_df.iterrows():

                    params = {
                        "inventory_id": int(row["inventory_id"]),
                        "booklet_number": int(row["booklet_number"]),
                        "page_number": int(row["page_number"]),
                        "step_number": int(row["step_number"]),
                    }
                    # Check if step already exists
                    result = self._get_step_id_from_set_steps(conn, **params)

                    if result:
                        # Step already exists, use existing step_id
                        step_id = result[0]
                    else:
                        # Insert new step
                        insert_query = text(
                            """
                            INSERT INTO set_steps (inventory_id, booklet_number, page_number, step_number)
                            VALUES (:inventory_id, :booklet_number, :page_number, :step_number)
                            """
                        )
                        self.logger.debug(
                            f"Inserting: inv_id={row['inventory_id']}, book_num={row['booklet_number']}, "
                            f"page_num={row['page_number']}, step_num={row['step_number']}"
                        )

                        result = conn.execute(insert_query, params)

                        # Get the last inserted ID
                        last_id_query = text("SELECT last_insert_rowid()")
                        step_id = conn.execute(last_id_query).fetchone()[0]

                    # Store in map
                    step_key = (
                        row["inventory_id"],
                        row["booklet_number"],
                        row["page_number"],
                        row["step_number"],
                    )
                    step_id_map[step_key] = step_id

            self.logger.info(
                f"Successfully inserted or retrieved {len(step_id_map)} steps"
            )
            return step_id_map
        except Exception as e:
            self.logger.error(f"Error inserting set_steps data: {str(e)}")
            return {}

    def extract_step_elements(
        self, manual_data_dict, manual_mapping, inventories, step_id_map
    ):
        """
        Extract step elements from manual data dictionary.

        Args:
            manual_data_dict: Dictionary with manual data
            manual_mapping: DataFrame with manual to set mapping
            inventories: DataFrame with inventory information
            step_id_map: Dictionary mapping step keys to step_ids

        Returns:
            DataFrame: DataFrame with step elements
        """
        step_elements_data = []

        # Process each manual again to collect element data
        inventory_id, booklet_number = self.get_inventory_info(
            self.manual_id, manual_mapping, inventories
        )
        if inventory_id is None:
            return None

        # Process element data
        for element_id, pages in manual_data_dict.items():
            for page_num, steps_info in pages.items():
                for step_info in steps_info:
                    step_number = step_info["step"]
                    step_key = (
                        inventory_id,
                        booklet_number,
                        page_num,
                        step_number,
                    )

                    if step_key in step_id_map:
                        step_id = step_id_map[step_key]
                        step_elements_data.append([step_id, element_id])

        # Convert to dataframe
        step_elements_df = pd.DataFrame(
            step_elements_data, columns=["step_id", "element_id"]
        )
        return step_elements_df.drop_duplicates()

    def insert_step_elements(self, step_elements_df):
        """
        Insert step elements into the database.

        Args:
            step_elements_df: DataFrame with step elements data
            engine: SQLAlchemy engine

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.engine.begin() as conn:
                inserted, skipped = 0, 0
                # Insert in batches to avoid potential memory issues
                for i in range(0, len(step_elements_df), self.batch_size):
                    batch = step_elements_df.iloc[i : i + self.batch_size]

                    # For each row in the batch
                    for _, row in batch.iterrows():
                        # Check if element already exists for this step
                        query = text(
                            """
                            SELECT 1 FROM step_elements 
                            WHERE step_id = :step_id AND element_id = :element_id
                        """
                        )
                        result = conn.execute(
                            query,
                            {
                                "step_id": row["step_id"],
                                "element_id": row["element_id"],
                            },
                        ).fetchone()

                        if not result:
                            self.logger.debug(
                                f"Inserting: step_id={row['step_id']}, element_id={row['element_id']}"
                            )
                            # Insert new element
                            insert_query = text(
                                """
                                INSERT INTO step_elements (step_id, element_id)
                                VALUES (:step_id, :element_id)
                                """
                            )
                            conn.execute(
                                insert_query,
                                {
                                    "step_id": row["step_id"],
                                    "element_id": row["element_id"],
                                },
                            )
                            inserted += 1
                        else:
                            self.logger.debug(
                                f"Skipping: step_id={row['step_id']}, element_id={row['element_id']}"
                            )
                            skipped += 1

            self.logger.info(
                f"Successfully processed {len(step_elements_df)} step elements (inserted: {inserted}. Skipped: {skipped})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error inserting step_elements data: {str(e)}")
            return False

    def process_and_load_manual_data(self, manual_data_dict):
        """
        Process manual data dictionary and load it into the database using the schema.

        Args:
            manual_data_dict: Dictionary with format {manual_id: {element_id: {page_num:[{'step': step_num, ...}]}}}
            inventories_csv: Path to inventories.csv
            db_url: Database connection URL
        """

        self.create_tables()

        # Load reference data
        manual_mapping, inventories = self.load_reference_data()
        if manual_mapping is None or inventories is None:
            return

        # Extract unique steps
        set_steps_df = self.extract_unique_steps(
            manual_data_dict, manual_mapping, inventories
        )
        logger.debug("Set steps data sample:")
        logger.debug(set_steps_df.head())

        # Insert steps and get step_ids
        step_id_map = self.insert_steps_and_get_ids(set_steps_df)
        if not step_id_map:
            return

        # Extract step elements
        step_elements_df = self.extract_step_elements(
            manual_data_dict, manual_mapping, inventories, step_id_map
        )
        logger.debug("Step elements data sample:")
        logger.debug(step_elements_df.head())

        # Insert step elements
        self.insert_step_elements(step_elements_df)


def main():
    from utils.logger import setup_logging

    setup_logging(log_level=logging.DEBUG)

    # Example dictionary (replace with your actual data)
    example_dict = {
        "6285778": {
            1: [{"step": 1, "piece": 1, "similarity": 0.95}],
            2: [{"step": 2, "piece": 3, "similarity": 0.92}],
        },
        "4520632": {
            2: [
                {"step": 2, "piece": 1, "similarity": 0.98},
                {"step": 3, "piece": 1, "similarity": 0.98},
            ]
        },
    }
    example_manual_id = 6560320
    # Example paths and connection string
    db_url = "sqlite:///test_db.db"
    manual_mapping_csv = Path("data/training/manuals/manuals_to_sets.csv")
    inventories_csv = Path("data/inventories.csv")

    # Uncomment to run with example data
    # Create data loader and ensure tables exist
    loader = LegoDataLoader(
        manual_mapping_csv, inventories_csv, db_url, example_manual_id
    )
    loader.process_and_load_manual_data(example_dict)


if __name__ == "__main__":
    main()
