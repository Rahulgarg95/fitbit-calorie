from datetime import datetime
from MongoDB.mongoDbDatabase import mongoDBOperation

class App_Logger:
    def __init__(self):
        self.dbObj = mongoDBOperation()

    def log(self, collection, log_message):
        now = datetime.now()
        date = now.date()
        current_time = now.strftime("%H:%M:%S")
        log_dict = {}
        log_dict['Insert_Date'] = str(date) + ' ' + str(current_time)
        log_dict['message'] = str(log_message)
        self.dbObj.insertOneRecord('fitbitDB',collection,log_dict)