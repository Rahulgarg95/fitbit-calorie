from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger
from AzureBlobStorage.azureBlobStorage import AzureBlobStorage
from MongoDB.mongoDbDatabase import mongoDBOperation

class dataTransformPredict:

     """
          This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.
     """

     def __init__(self):
          self.goodDataPath = "Prediction_Good_Raw_Files_Validated"
          self.logger = App_Logger()
          self.azureObj = AzureBlobStorage()
          self.dbObj = mongoDBOperation()


     def addQuotesToStringValuesInColumn(self):

          """
              Method Name: addQuotesToStringValuesInColumn
              Description: This method replaces the missing values in columns with "NULL" to
                           store in the table. We are using substring in the first column to
                           keep only "Integer" data for ease up the loading.
                           This column is anyways going to be removed during prediction.
          """

          try:
               log_file = 'dataTransformLog'
               onlyfiles = self.azureObj.listDirFiles(self.goodDataPath)
               for file in onlyfiles:
                    data = self.azureObj.csvToDataframe(self.goodDataPath, file)

                    data['Id'] = data["Id"].apply(lambda x: "'" + str(x) + "'")
                    data['ActivityDate'] = data["ActivityDate"].apply(lambda x: "'" + str(x) + "'")

                    self.azureObj.saveDataframeToCsv(self.goodDataPath,file,data)
                    self.logger.log(log_file, " %s: Quotes added successfully!!" % file)

          except Exception as e:
               log_file = 'dataTransformLog'
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               raise e