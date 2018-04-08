import sqlite3


def initialize():
    db = sqlite3.connect('names.db')
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE Names(id INTEGER PRIMARY KEY AUTOINCREMENT , name TEXT)''')
    db.commit()


def add_new_entry(name):
    db = sqlite3.connect('names.db')
    cursor = db.cursor()
    cursor.execute('''INSERT INTO Names VALUES(NULL,?)''', (name,))
    db.commit()


def get_name(value):
    db = sqlite3.connect('names.db')
    cursor = db.cursor()
    cursor.execute('''SELECT name FROM Names WHERE id =?''', (value,))
    name = cursor.fetchone()
    count = cursor.fetchall()
    if count != 0:
        return name[0]
    return 'No Name'


def get_max_value():
    db = sqlite3.connect('names.db')
    cursor = db.cursor()
    cursor.execute('''SELECT max(id) from Names''')
    id = cursor.fetchone()
    count = cursor.fetchall()
    try:
        return id[0]
    except:
        return 0


def name_present(name):
    db = sqlite3.connect('names.db')
    cursor = db.cursor()
    cursor.execute('''Select id from Names where name = ?''', (name,))
    id = cursor.fetchone()
    count = cursor.fetchall().__len__()
    if count != 0:
        return id[0]
    return False
