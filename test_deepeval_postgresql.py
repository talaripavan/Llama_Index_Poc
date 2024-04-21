import psycopg2 , json

class DeepEvalDatabase:
    def __init__(self,hostname,database,username,password,port):
        self.hostname = hostname
        self.database = database
        self.username = username
        self.password = password
        self.port = port
        
    def connection(self):
        try:
            self.conn = psycopg2.connect(
                host = self.hostname,
                dbname = self.database,
                user = self.username,
                password = self.password,
                port = self.port
            )
            self.cur = self.conn.cursor()
            print("Connected to the database")
        except psycopg2.Error as e:
            print("Error in connecting ",e)
            
    def insert_data(self):
        # First we are reading the json file.
        with open('test_10.json','r') as file:
            json_data = json.load(file)
            
        testfile = json_data['testFile']
        deployment = json_data['deployment']
        testcase = json.dumps(json_data['testCases'])
        
        insert_statements = '''
            INSERT INTO test_schema.deepeval1(testfile, deployment, testcase)
            VALUES (%s,%s,%s);
        '''
        try:
            self.cur.execute(insert_statements,(testfile,deployment,testcase))
            self.conn.commit()
            print("Data Inserted Successfully")
        except psycopg2.Error as e:
            print("Error in inserting data : ",e)
            
    def read_data(self):
        try:
            self.cur.execute('SELECT * FROM test_schema.deepeval1')
            rows = self.cur.fetchall()
            print("Data retrived from the table :-")
            for row in rows:
                print(row)
        except psycopg2.Error as e:
            print("Can not retrive the data from table", e)
            


if __name__ == '__main__':
    
    hostname = 'localhost'
    database = '<Database>'
    username = '<username>'
    password = '<password>'
    port = 5432
    
    deepeval_db = DeepEvalDatabase(hostname,database,username,password,port)
    
    # Calling the methods.
    deepeval_db.connection()
    deepeval_db.insert_data()
    deepeval_db.read_data()
