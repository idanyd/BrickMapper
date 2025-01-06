import pandas as pd
import sqlite3
from pathlib import Path


def create_database():
    """Create the SQLite database and tables"""
    conn = sqlite3.connect("brickmapper.db")

    # Enable foreign key support
    conn.execute("PRAGMA foreign_keys = ON")

    # Create tables in order based on dependencies
    tables = {
        "themes": """
            CREATE TABLE IF NOT EXISTS themes (
                id INTEGER PRIMARY KEY,
                name VARCHAR(40),
                parent_id INTEGER,
                FOREIGN KEY (parent_id) REFERENCES themes(id)
            )
        """,
        "part_categories": """
            CREATE TABLE IF NOT EXISTS part_categories (
                id INTEGER PRIMARY KEY,
                name VARCHAR(200)
            )
        """,
        "parts": """
            CREATE TABLE IF NOT EXISTS parts (
                part_num VARCHAR(20) PRIMARY KEY,
                name VARCHAR(250),
                part_cat_id INTEGER,
                part_material VARCHAR(20),
                FOREIGN KEY (part_cat_id) REFERENCES part_categories(id)
            )
        """,
        "colors": """
            CREATE TABLE IF NOT EXISTS colors (
                id INTEGER PRIMARY KEY,
                name VARCHAR(200),
                rgb VARCHAR(6),
                is_trans BOOLEAN
            )
        """,
        "sets": """
            CREATE TABLE IF NOT EXISTS sets (
                set_num VARCHAR(20) PRIMARY KEY,
                name VARCHAR(256),
                year INTEGER,
                theme_id INTEGER,
                num_parts INTEGER,
                img_url VARCHAR(200),
                FOREIGN KEY (theme_id) REFERENCES themes(id)
            )
        """,
        "inventories": """
            CREATE TABLE IF NOT EXISTS inventories (
                id INTEGER PRIMARY KEY,
                version INTEGER,
                set_num VARCHAR(20),
                FOREIGN KEY (set_num) REFERENCES sets(set_num)
            )
        """,
        "inventory_parts": """
            CREATE TABLE IF NOT EXISTS inventory_parts (
                inventory_id INTEGER,
                part_num VARCHAR(20),
                color_id INTEGER,
                quantity INTEGER,
                is_spare BOOLEAN,
                img_url VARCHAR(200),
                FOREIGN KEY (inventory_id) REFERENCES inventories(id),
                FOREIGN KEY (part_num) REFERENCES parts(part_num),
                FOREIGN KEY (color_id) REFERENCES colors(id),
                PRIMARY KEY (inventory_id, part_num, color_id)
            )
        """,
        "elements": """
            CREATE TABLE IF NOT EXISTS elements (
                element_id VARCHAR(10) PRIMARY KEY,
                part_num VARCHAR(20),
                color_id INTEGER,
                design_id INTEGER,
                FOREIGN KEY (part_num) REFERENCES parts(part_num),
                FOREIGN KEY (color_id) REFERENCES colors(id)
            )
        """,
    }

    for _, create_statement in tables.items():
        conn.execute(create_statement)

    conn.commit()
    return conn


def load_csv_data(conn, data_dir):
    """Load data from CSV files into the database"""
    # Temporarily disable foreign key constraints
    conn.execute("PRAGMA foreign_keys = OFF")

    # Define loading order based on dependencies
    load_order = [
        "themes",
        "part_categories",
        "parts",
        "colors",
        "sets",
        "inventories",
        "inventory_parts",
        "elements",
    ]

    for table in load_order:
        print(f"Loading {table}...")
        csv_path = data_dir / f"{table}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.fillna("")  # Replace NaN values with empty strings
            try:
                df.to_sql(table, conn, if_exists="append", index=False)
            except sqlite3.Error as e:
                print(f"Error loading {table}: {e}")
        else:
            print(f"Warning: {csv_path} not found")

    # Re-enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")
    print("Data loading complete")


if __name__ == "__main__":
    # Create data directory path
    data_dir = Path("data")

    # Create and populate database
    print("Creating database...")
    conn = create_database()

    print("Loading data...")
    load_csv_data(conn, data_dir)

    conn.close()
    print("Database initialization complete")
