import sqlite3

# Create the database connection
conn = sqlite3.connect("user_data.db", check_same_thread=False)
cursor = conn.cursor()

# Enable Write-Ahead Logging (WAL mode) for multi-threading
cursor.execute("PRAGMA journal_mode=WAL;")

# Create the users table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    full_name TEXT,
    age INTEGER,
    gender TEXT,
    date_of_birth TEXT,
    place_of_birth TEXT,
    mother_name TEXT,
    mother_birth_year INTEGER,
    father_name TEXT,
    father_birth_year INTEGER,
    sibling_count INTEGER,
    sibling_name TEXT,
    sibling_birth_year INTEGER,
    sibling_gender TEXT,
    marital_status TEXT,
    email TEXT,
    phone_number TEXT,
    city TEXT,
    hometown TEXT,
    languages_spoken TEXT,
    favorite_cuisines TEXT,
    shoe_size INTEGER,
    height REAL,
    weight REAL,
    eye_color TEXT,
    hair_color TEXT,
    hobbies TEXT
)
""")

# Check if Cem Kaspi exists, insert only if missing
cursor.execute("SELECT COUNT(*) FROM users WHERE full_name = ?", ("Cem Kaspi",))
if cursor.fetchone()[0] == 0:
    cursor.execute("""
    INSERT INTO users (full_name, age, gender, date_of_birth, place_of_birth, mother_name, mother_birth_year, father_name, father_birth_year, sibling_count, sibling_name,
                        sibling_birth_year, sibling_gender, marital_status, email, phone_number, city, hometown, languages_spoken, favorite_cuisines, shoe_size, height, weight, 
                        eye_color, hair_color, hobbies)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "Cem Kaspi", 28, "male", "1997-03-12", "Istanbul", "Elda", 1966, "Sami", 1963, 1, "Elsa", 
        1991, "female", "single", "cem.kaspi97@gmail.com", "+1 (778) 839-6610", "Vancouver", "Istanbul", "Turkish, English, Spanish", "Turkish, Indian, Thai, Italian, Japanese", 8, 1.80, 75,
        "Brown", "Brown", "Football, basketball, swimming, gaming, guitar, cooking, working out"
    ))
    conn.commit()  # Save changes only if a new user was added

cursor.close()
conn.close()

print("Database setup completed successfully!")
