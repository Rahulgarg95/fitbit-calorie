import pickle
import os
import shutil
from AzureBlobStorage.azureBlobStorage import AzureBlobStorage

class File_Operation:
    """
        This class shall be used to save the model after training
        and load the saved model for prediction.
    """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory='models'
        self.azureObj = AzureBlobStorage()

    def save_model(self,model,filename):
        """
            Method Name: save_model
            Description: Save the model file to directory
            Outcome: File gets saved
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        try:
            self.azureObj.saveObject(self.model_directory, filename + '.sav', model)
            print('Model Saved')
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()

    def load_model(self,filename):
        """
            Method Name: load_model
            Description: load the model file to memory
            Output: The Model file loaded in memory
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        try:
            f = self.azureObj.loadObject(self.model_directory, filename + '.sav')
            print('Model Loading Done!')
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
            return f
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()

    def find_correct_model_file(self,cluster_number):
        """
            Method Name: find_correct_model_file
            Description: Select the correct model based on cluster number
            Output: The Model file
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the find_correct_model_file method of the File_Operation class')
        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = self.azureObj.listDirFiles(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split('.')[0]
            self.logger_object.log(self.file_object,
                                   'Exited the find_correct_model_file method of the Model_Finder class.')
            print('Model Name: ', self.model_name)
            return self.model_name
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in find_correct_model_file method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Exited the find_correct_model_file method of the Model_Finder class with Failure')
            raise Exception()