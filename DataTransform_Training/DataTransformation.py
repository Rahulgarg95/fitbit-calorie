from application_logging.logger import App_Logger
import pandas as pd
from AzureBlobStorage.azureBlobStorage import AzureBlobStorage

class dataTransform:

     """
          This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.
     """

     def __init__(self):
          self.goodDataPath = 'Training_Good_Raw_Files_Validated'
          self.logger = App_Logger()
          self.azureObj = AzureBlobStorage()


     def addQuotesToStringValuesInColumn(self):
          """
             Method Name: addQuotesToStringValuesInColumn
             Description: This method converts all the columns with string datatype such that
                         each value for that column is enclosed in quotes. This is done
                         to avoid the error while inserting string values in table as varchar.
          """

          log_file = 'addQuotesToStringValuesInColumn'
          try:
               onlyfiles = self.azureObj.listDirFiles(self.goodDataPath)
               for file in onlyfiles:
                    data = self.azureObj.csvToDataframe(self.goodDataPath, file)

                    data['Id'] = data["Id"].apply(lambda x: "'" + str(x) + "'")
                    data['ActivityDate'] = data["ActivityDate"].apply(lambda x: "'" + str(x) + "'")

               self.azureObj.saveDataframeToCsv(self.goodDataPath,file,data)
               self.logger.log(log_file," %s: Quotes added successfully!!" % file)
          except Exception as e:
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)