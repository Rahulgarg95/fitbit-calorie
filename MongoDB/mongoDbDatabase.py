import pandas as pd
import pymongo
import json
from application_logging import logger

class mongoDBOperation:
    def __init__(self):
        #self.connstr=os.getenv('Mongo_Key')
        self.connstr='mongodb://localhost:27017/'
        self.file = 'mongoOperations'

    def connectMongoClient(self):
        """
            Method: connectMongoClient
            Description: Establishes Connection with MongoDB
        """
        try:
            client = pymongo.MongoClient(self.connstr)
            return client
        except Exception as e:
            raise Exception('Exception Occurred')
            #self.logger.log(self.file, 'Function => connectMongoClient Error Occurred while establishing connection with mongodb: ' + str(e))


    def isDbPresent(self,db_name):
        """
        Method: isDBPresent
        Description: Check if db is present.
        :param db_name: Name of database to be searched.
        :return: True if connection is established else return False.
        """
        try:
            client=self.connectMongoClient()
            if db_name in client.list_database_names():
                client.close()
                return True
            else:
                client.close()
                return False
        except Exception as e:
            raise Exception('Exception Occurred')
            #self.logger.log(self.file, 'Function => isDbPresent Error Occurred while establishing connection with mongodb in: ' + str(e))


    def createOrGetCollection(self,db_name,collection_name):
        """
        Method: CreateOrGetCollection
        Description: To get required collection given by the user.
        :param db_name: Database Name
        :param collection_name: Collection Name
        :return: Collection
        """
        try:
            client = self.connectMongoClient()
            db = client[db_name]
            collection = db[collection_name]
            return collection
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => createOrGetCollection, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def isCollectionPresent(self,db_name,collection_name):
        """
        Method: isCollectionPresent
        Description: Check if the provided collection is present.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :return: True if Collection Found otherwise False
        """
        try:
            client=self.connectMongoClient()
            db=client[db_name]
            if collection_name in db.list_collection_names():
                client.close()
                return True
            else:
                client.close()
                return False
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => isCollectionPresent, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def insertOneRecord(self,db_name,collection_name,record):
        """
        Method: insertOneRecord
        Description: Inserts a single record in given collection.
        :param db_name: DataBase Name
        :param collection_name: Collection Name
        :param record: Dictionary to be inserted.
        :return: None
        """
        try:
            collection=self.createOrGetCollection(db_name,collection_name)
            collection.insert_one(record)
        except Exception as e:
            print('Exception', str(record))
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => insertOneRecord, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def insertManyRecords(self,db_name,collection_name,records):
        """
        Method: insertManyRecords
        Description: Inserts Multiple records in given collection.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :param records: List of records(dictionary)
        :return: None
        """
        try:
            temp_l = list(records)
            collection=self.createOrGetCollection(db_name,collection_name)
            collection.insert_many(temp_l)
        except Exception as e:
            print('Exception: ',e)
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => insertManyRecords, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def isRecordPresent(self,db_name,collection_name,records):
        """
        Method: isRecordPresent
        Description: Check if a record is present in provided collection.
        :param db_name: Database name
        :param collection_name: Collection Name
        :param records: Record to be searched
        :return: True if Record Found else False
        """
        try:
            collection=self.createOrGetCollection(db_name,collection_name)
            record_data=collection.find(records)
            if record_data.count() > 0:
                return False
            else:
                return True
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => isRecordPresent, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def dataframeToRecords(self,db_name,collection_name,data):
        """
        Method: dataframeToRecords
        Description: Insert a Dataframe to collection.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :param data: Dataframe to be inserted
        :return: None
        """
        try:
            records = list(json.loads(data.T.to_json()).values())
            self.insertManyRecords(db_name,collection_name,records)
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => dataframeToRecords, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def recordsToDataFrame(self,db_name,collection_name):
        """
        Method: recordsToDataFrame
        Description: Extract Records and insert to dataframe.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :return: Dataframe created
        """
        try:
            collection=self.createOrGetCollection(db_name,collection_name)
            tmp_df=pd.DataFrame(list(collection.find()))
            if '_id' in tmp_df:
                tmp_df=tmp_df.drop('_id',axis=1)
            return tmp_df
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => recordsToDataFrame, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def dropCollection(self,db_name,collection_name):
        """
        Method: dropCollection
        Description: Delete the provided collection.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :return: None
        """
        try:
            if self.isCollectionPresent(db_name,collection_name):
                collection=self.createOrGetCollection(db_name,collection_name)
                collection.drop()
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => dropCollection, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))


    def getRecords(self,db_name,collection_name):
        """
        Method: getRecords
        Description: Get all records of a collection.
        :param db_name: DB Name
        :param collection_name: Collection Name
        :return: List of records.
        """
        try:
            if self.isCollectionPresent(db_name, collection_name):
                collection = self.createOrGetCollection(db_name, collection_name)
                records=collection.find()
                if records.count()==1:
                    for record in records:
                        return record
                elif records.count()>1:
                    record_l=[]
                    for record in records:
                        record_l.append(record)
                    return record_l
                else:
                    return None
        except Exception as e:
            raise Exception('Exception Occurred')
            message = 'Exception Occurred: Function => getRecords, DB Name => ' + db_name + 'Collection => ' + collection_name
            #self.logger.log(self.file, message + ' : ' + str(e))