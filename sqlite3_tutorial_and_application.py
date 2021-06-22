import sqlite3

# db = sqlite3.connect("beans.sqlite")

CREATE_TABLE = "CREATE TABLE IF NOT EXISTS students_table (id INTEGER PRIMARY KEY, firstname TEXT, lastname TEXT, marks INTEGER)"
INSERT_STUDENT = "INSERT INTO students_table (firstname , lastname , marks) VALUES (?,?,?)"
GET_ALL_INFO = "SELECT * FROM students_table"
SELECT_STUDENT_BY_LASTNAME = "SELECT * FROM students_table WHERE (lastname,marks) = (?,?)"

def connect():
    return sqlite3.connect("beans.sqlite")

def create_table(connection):
    with connection:
        connection.execute(CREATE_TABLE)

def add_student(connection,fname,lname,marks):
    with connection:
        connection.execute(INSERT_STUDENT,(fname,lname,marks))

def select_student_by_lastname(connection,lstname,marks):
    with connection:
        return connection.execute(SELECT_STUDENT_BY_LASTNAME,(lstname,marks))

def select_all_from_db(connection):
    with connection:
        return len(connection.execute(GET_ALL_INFO).fetchall())
