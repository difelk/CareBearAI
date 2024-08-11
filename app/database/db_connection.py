import mysql.connector
from mysql.connector import Error

def create_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='1234',
            database='bcnstock'
        )
        if conn.is_connected():
            print("Successfully connected to the database.")
            return conn
    except Error as e:
        print(f"Error: {e}")
        return None

def test_connection():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT DATABASE();")
            db_name = cursor.fetchone()
            print(f"Connected to database: {db_name[0]}")

        except Error as e:
            print(f"Error while executing query: {e}")
        finally:
            cursor.close()
            connection.close()

if __name__ == "__main__":
    test_connection()
