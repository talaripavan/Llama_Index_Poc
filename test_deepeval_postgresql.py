import json
import psycopg2

with open('test1.json', 'r') as file:
    json_data = json.load(file)


hostname = 'localhost'
database = 'Demo_connect'
username = '<user name>'
pwd = '<your password>'
port_id = 5432

conn = psycopg2.connect(
    host=hostname,
    dbname=database,
    user=username,
    password=pwd,
    port=port_id
)


cur = conn.cursor()


create_script = '''
    CREATE TABLE deepeval(
        id UUID DEFAULT gen_random_uuid(),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        testfile TEXT,
        deployment BOOLEAN,
        testcase JSON
    );
'''


cur.execute(create_script)


insert_statement = '''
    INSERT INTO deepeval(testfile, deployment, testcase)
    VALUES (%s, %s, %s);
'''


testfile = json_data['testFile']
deployment = json_data['deployment']
testcase = json.dumps(json_data['testCases'])  

cur.execute(insert_statement, (testfile, deployment, testcase))

conn.commit()

cur.close()
conn.close()
