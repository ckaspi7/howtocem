import sqlite3
import json
import os

# Function to create a database connection with proper settings
def get_db_connection():
    """Creates a new database connection and enables multi-threading."""
    db_path = os.path.join("data", "user_data.db")
    conn = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    return conn

# Function to fetch user data safely
def get_user_data():
    """Fetches user data safely while preventing database lock issues."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE full_name = ?", ("Cem Kaspi",))
        row = cursor.fetchone()

        if not row:
            return json.dumps({"error": "User not found"}, indent=4)

        column_names = [description[0] for description in cursor.description]
        user_data = dict(zip(column_names, row))

        cursor.close()
        conn.close()

        return json.dumps(user_data, indent=4)

    except sqlite3.OperationalError as e:
        return json.dumps({"error": f"Database error: {str(e)}"}, indent=4)

# Test JSON output
if __name__ == "__main__":
    print(get_user_data())
