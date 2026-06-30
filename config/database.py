# config/database.py
import mysql.connector
from mysql.connector import Error

# --- CONFIGURATION ENGINE CREDENTIALS ---
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dbo',
    'user': 'root',
    'password': 'NameisRoot909'
}

def init_db():
    """
    Initializes and verifies the database schema structure at application boot time.
    Ensures the 'users' table exists with appropriate column types and unique indexes.
    """
    create_table_query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Username VARCHAR(255) NOT NULL UNIQUE,
        Email VARCHAR(255) NOT NULL UNIQUE,
        Password VARCHAR(255) NOT NULL,
        dob VARCHAR(20) DEFAULT NULL
    );
    """
    connection = None
    cursor = None
    try:
        # Establish structural pipeline link to MySQL Server instance
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute(create_table_query)
            connection.commit()
            print("🚀 MySQL Database Engine: 'users' table verified successfully.")
            
    except Error as e:
        print(f"❌ Critical Database Core Initialization Fault: {e}")
        
    finally:
        # Prevent memory leaks by safely wrapping up execution blocks
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()