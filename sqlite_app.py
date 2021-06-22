import sqlite3_tutorial_and_application as database
import sys

menu_prompt = """

--Students marks management App--

Please choose an option:
1) Add a student?
2) Retrieve all students?
3) Get student by lastname?
4) Exit?

Your selection: """

def start():
    connection = database.connect()
    database.create_table(connection)

    while (user_input:=input(menu_prompt)) !="5":
        # print(user_input)
        if user_input=="1":
            fname = input(" Student Firstname: ")
            lname = input(" Student Lastname: ")
            marks = int(input(" Student Marks: "))
            database.add_student(connection,fname,lname,marks)
            print("Student record was added successfully...")

        elif user_input=="2":
            students = database.select_all_from_db(connection)
            # for student in students:
            print(students)
        elif user_input=="3":
            lastname = input("What is the lastname? ")
            markss = input("what are the marks?")
            student_info = database.select_student_by_lastname(connection,lastname,markss)
            for n in student_info:
                print(n)
        else:
            print("Error ,please check your select options!!!")
            sys.exit(0)


start()

