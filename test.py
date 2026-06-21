import mysql.connector
from mysql.connector import Error

def test_db_connection():
    connection = None
    try:
        # Establish the connection
        connection = mysql.connector.connect(
            host='localhost',          # e.g., '127.0.0.1' or your server IP
            database='dbo',  
            user='root',      
            password='NameisRoot909'   
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"Successfully connected to MySQL Server version {db_info}")
           
            # Optional: Verify you can query the database
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            record = cursor.fetchone()
            print(f"You are currently connected to database: {record[0]}")

    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

    finally:
        # Always ensure the connection is closed properly
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")

if __name__ == "__main__":
    test_db_connection()