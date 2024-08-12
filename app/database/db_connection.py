# import csv
#
# import mysql.connector
# from mysql.connector import Error
# from datetime import datetime
#
#
# def create_connection():
#     try:
#         conn = mysql.connector.connect(
#             host='localhost',
#             user='root',
#             password='1234',
#             database='foodpriceslk'
#         )
#         if conn.is_connected():
#             print("Successfully connected to the database.")
#             return conn
#     except Error as e:
#         print(f"Error: {e}")
#         return None
#
#
# def execute_query(query, params=None):
#     connection = create_connection()
#     if connection:
#         try:
#             cursor = connection.cursor()
#             cursor.execute(query, params)
#             results = cursor.fetchall()
#             return results
#         except Error as e:
#             print(f"Error while executing query: {e}")
#         finally:
#             cursor.close()
#             connection.close()
#     return None
#
#
# def sanitize_column_name(col_name):
#     return f"`{col_name.replace('`', '``')}`"
#
#
# def convert_date(date_str):
#     try:
#         return datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
#     except ValueError:
#         return None
#
#
# def handle_empty_values(value, column_type):
#     if column_type in ['latitude', 'longitude', 'price', 'usdprice', 'usd_rate']:
#         return None if value.strip() == '' else value
#     return value
#
#
# def insert_csv_to_db(csv_file_path, table_name):
#     connection = create_connection()
#     if connection:
#         try:
#             cursor = connection.cursor()
#
#             with open(csv_file_path, mode='r') as file:
#                 csv_reader = csv.reader(file)
#
#                 header = next(csv_reader)
#                 sanitized_header = [sanitize_column_name(col) for col in header]
#
#                 if '`id`' in sanitized_header:
#                     sanitized_header.remove('`id`')
#
#                 placeholders = ', '.join(['%s'] * len(sanitized_header))
#                 query = f"INSERT INTO {table_name} ({', '.join(sanitized_header)}) VALUES ({placeholders})"
#
#                 for row in csv_reader:
#                     if 'date' in header:
#                         date_index = header.index('date')
#                         row[date_index] = convert_date(row[date_index])
#
#                     row = [handle_empty_values(value, col) for value, col in zip(row, header)]
#
#                     cursor.execute(query, tuple(row))
#
#             connection.commit()
#             print(f"Data from {csv_file_path} has been successfully inserted into {table_name}.")
#             return "successful"
#         except Error as e:
#             print(f"Error while inserting data: {e}")
#             connection.rollback()
#             return "failed: " + str(e)
#         finally:
#             cursor.close()
#             connection.close()
# # def get_data_in_date_range(start_date, end_date):
# #     query = "SELECT * FROM your_table WHERE date_column BETWEEN %s AND %s"
# #     return execute_query(query, (start_date, end_date))
#
# # def get_limited_data(limit, offset=0):
# #     query = "SELECT * FROM your_table LIMIT %s OFFSET %s"
# #     return execute_query(query, (limit, offset))
#
# if __name__ == "__main__":
#     insert_csv_to_db('data.csv', 'your_table')
#
#     # date_range_data = get_data_in_date_range('2023-01-01', '2023-12-31')
#     # print("Data in Date Range:", date_range_data)
#
#     # limited_data = get_limited_data(10, 20)
#     # print("Limited Data:", limited_data)
