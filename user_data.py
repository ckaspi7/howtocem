import sqlite3
import json

# Function to create a database connection with proper settings
def get_db_connection():
    """Creates a new database connection and enables multi-threading."""
    conn = sqlite3.connect("user_data.db", timeout=10, check_same_thread=False)
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




# import sqlite3
# import json

# def get_db_connection():
#     """Creates a new database connection and enables multi-threading."""
#     conn = sqlite3.connect("user_data.db", timeout=10, check_same_thread=False)
#     conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging
#     return conn

# # Ensure table exists
# def setup_database():
#     conn = get_db_connection()
#     cursor = conn.cursor()

#     cursor.execute("""
#     CREATE TABLE IF NOT EXISTS users (
#         id INTEGER PRIMARY KEY,
#         full_name TEXT,
#         age INTEGER,
#         gender TEXT,
#         date_of_birth TEXT,
#         place_of_birth TEXT,
#         mother_name TEXT,
#         mother_birth_year INTEGER,
#         father_name TEXT,
#         father_birth_year INTEGER,
#         sibling_count INTEGER,
#         sibling_name TEXT,
#         sibling_birth_year INTEGER,
#         sibling_gender TEXT,
#         marital_status TEXT,
#         email TEXT,
#         phone_number TEXT,
#         city TEXT,
#         hometown TEXT,
#         languages_spoken TEXT,
#         favorite_cuisines TEXT,
#         shoe_size INTEGER,
#         height REAL,
#         weight REAL,
#         eye_color TEXT,
#         hair_color TEXT,
#         hobbies TEXT
#     )
#     """)
    
#     # Check if user already exists
#     cursor.execute("SELECT COUNT(*) FROM users WHERE full_name = ?", ("Cem Kaspi",))
#     if cursor.fetchone()[0] == 0:
#         cursor.execute("""
#         INSERT INTO users (full_name, age, gender, date_of_birth, place_of_birth, mother_name, mother_birth_year, father_name, father_birth_year, sibling_count, sibling_name,
#                             sibling_birth_year, sibling_gender, marital_status, email, phone_number, city, hometown, languages_spoken, favorite_cuisines, shoe_size, height, weight, 
#                             eye_color, hair_color, hobbies)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, (
#             "Cem Kaspi", 28, "male", "1997-03-12", "Istanbul", "Elda", 1966, "Sami", 1963, 1, "Elsa", 
#             1991, "female", "single", "cem.kaspi97@gmail.com", "+1 (778) 839-6610", "Vancouver", "Istanbul", "Turkish, English, Spanish", "Turkish, Indian, Thai, Italian, Japanese", 8, 1.80, 75,
#             "Brown", "Brown", "Football, basketball, swimming, gaming, guitar, cooking, working out"
#         ))
#         conn.commit()  # Save changes

#     cursor.close()  # Close cursor
#     conn.close()  # Close connection

# # Fetch user data safely
# def get_user_data():
#     """Fetch user data safely while preventing database lock issues."""
#     try:
#         conn = get_db_connection()
#         cursor = conn.cursor()

#         cursor.execute("SELECT * FROM users WHERE full_name = ?", ("Cem Kaspi",))
#         row = cursor.fetchone()

#         if not row:
#             return json.dumps({"error": "User not found"}, indent=4)

#         column_names = [description[0] for description in cursor.description]
#         user_data = dict(zip(column_names, row))

#         cursor.close()
#         conn.close()

#         return json.dumps(user_data, indent=4)
    
#     except sqlite3.OperationalError as e:
#         return json.dumps({"error": f"Database error: {str(e)}"}, indent=4)

# # def get_user_data():
# #     conn = get_db_connection()
# #     cursor = conn.cursor()

# #     cursor.execute("SELECT * FROM users WHERE full_name = ?", ("Cem Kaspi",))
# #     row = cursor.fetchone()
    
# #     column_names = [description[0] for description in cursor.description]
# #     user_data = dict(zip(column_names, row)) if row else {}

# #     cursor.close()
# #     conn.close()
# #     return json.dumps(user_data, indent=4)

# # Run setup
# setup_database()

# # Test JSON output
# if __name__ == "__main__":
#     print(get_user_data())