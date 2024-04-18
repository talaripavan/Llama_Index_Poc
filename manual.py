import psycopg

hostname = 'localhost'
database = 'Demo_connect'
username = 'postgres'
pwd = '<your password>'
port_id = 5432

conn = None
cur = None

try:
    conn = psycopg.connect(
        host = hostname,
        dbname = database,
        user = username,
        password = pwd,
        port = port_id )

    cur = conn.cursor()

    cur.execute('DROP TABLE IF EXISTS Test_Orders')
    
    create_script = '''CREATE TABLE Test_Orders (
        id serial NOT NULL PRIMARY KEY,
	    testFile text ,
        deployment boolean
     ); '''
     
    cur.execute(create_script)
    
    insert_script = 'INSERT INTO Test_Orders (testFile,deployment) VALUES (%s,%s)'
    insert_values = ('test_unit.py','False')
    
    cur.execute(insert_script,insert_values)
    '''
    for json_data in insert_values:
        cur.execute(insert_script, (json_data,))
    '''
    
    conn.commit()
    
except Exception as error:
    print(error)
    
finally:
    if cur is not None:
        cur.close()
    
    if conn is not None:
        conn.close()
        
        
