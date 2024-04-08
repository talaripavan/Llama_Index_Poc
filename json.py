import psycopg

hostname = 'localhost'
database = 'Db_name'
username = 'postgres'
pwd = 'my_password'
port_id = 5432

conn = None
cur = None

try:
    conn = psycopg.connect(
        host = hostname,
        dbname = database,
        user = username,
        password = pwd,
        port = port_id
    )
    
    cur = conn.cursor()
    
    cur.execute('DROP TABLE IF EXISTS employee')

    create_table = '''CREATE TABLE TakeOrder (
	id serial NOT NULL PRIMARY KEY,
	info json NOT NULL
    )'''
 
    cur.execute(create_table)

except Exception as error:
    print(error)
    
finally:
    if cur is not None:
        cur.close()
    
    if conn is not None:
        conn.close()