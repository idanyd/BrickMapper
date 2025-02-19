import sqlite3
from pathlib import Path
from typing import List, Dict

class DatabaseInterface:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def store_step_images(self, inventory_id: int, booklet_number: int, steps: List[Dict]):
        """Store step image information in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for step in steps:
                cursor.execute("""
                    UPDATE set_steps 
                    SET image_path = ? 
                    WHERE inventory_id = ? 
                    AND booklet_number = ? 
                    AND step_number = ? 
                    AND page_number = ?
                """, (
                    step['image_path'],
                    inventory_id,
                    booklet_number,
                    step['step_number'],
                    step['page_number']
                ))

            conn.commit()