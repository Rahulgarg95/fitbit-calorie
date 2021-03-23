import os
import io
import dill
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from application_logging.logger import App_Logger
import pandas as pd

class AzureBlobStorage:
    def __init__(self):
        self.conn_str=os.getenv('FitBit_Key')
        self.logger = App_Logger()
        self.azure_client = BlobServiceClient.from_connection_string(self.conn_str)
        self.file = 'AzureManagementLogs'


    def listDirFiles(self,folder_name):
        """
        Method: listDirFiles
        Description: List all the files in a directory.
        :param folder_name: Folder Name
        :return: List of files.
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            container_client = self.azure_client.get_container_client(folder_name)
            return [x.name for x in container_client.list_blobs()]
        except Exception as e:
            message = 'Exception Found: Function => listDirFiles, Folder Name: ' + folder_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def isFolderPresent(self,folder_name):
        """
        Method: isFolderPresent
        Description: Check if the given folder is present on Azure
        :param folder_name: Folder to be checked for presence.
        :return: True if Present else False
        """
        try:
            folder_name = folder_name.replace('_','-').lower().strip()
            folder_list = [f.name for f in self.azure_client.list_containers()]

            if folder_name in folder_list:
                return True
            else:
                return False
        except Exception as e:
            message = 'Exception Found: Function => isFolderPresent, Folder Name: ' + folder_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def deleteFolder(self,folder_name):
        """
        Method: deleteFolder
        Description: Enables user to delete a user.
        :param folder_name: Folder to be delted
        :return: None
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if self.isFolderPresent(folder_name):
                self.azure_client.delete_container(folder_name)
        except Exception as e:
            message = 'Exception Found: Function => deleteFolder, Folder Name: ' + folder_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e

    def isFilePresent(self,folder_name,file_name):
        """
        Method: isFilePresent
        Description: Check if given file is present in provided folder.
        :param folder_name: Folder Name where file should be present.
        :param file_name: File Name to search.
        :return: True if File Present else False
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if file_name in self.listDirFiles(folder_name):
                return True
            else:
                return False
        except Exception as e:
            message = 'Exception Found: Function => createFolder, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def deleteFile(self,folder_name,file_name=''):
        """
        Method: deleteFile
        Description: Enables user to Delete the given file.
        :param folder_name: Folder name where file to be present.
        :param file_name: File to be deleted.
        :return: None
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if self.isFolderPresent(folder_name):
                file_names = self.listDirFiles(folder_name)
                if file_name == '':
                    for file_n in file_names:
                        blob_client = self.azure_client.get_blob_client(folder_name, file_n)
                        blob_client.delete_blob()
                else:
                    if file_name in file_names:
                        blob_client = self.azure_client.get_blob_client(folder_name, file_name)
                        blob_client.delete_blob()
        except Exception as e:
            message = 'Exception Found: Function => deleteFile, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def createFolder(self,folder_name):
        """
        Method: createFolder
        Description: Enables user to create a folder/directory on Azure.
        :param folder_name: Folder Name to be created
        :return: None
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if self.isFolderPresent(folder_name):
                self.deleteFolder(folder_name)
            else:
                self.azure_client.create_container(folder_name)
        except Exception as e:
            message = 'Exception Found: Function => createFolder, Folder Name: ' + folder_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def uploadFiles(self,source_dir,dest_dir,file_name=''):
        """
        Method: uploadFiles
        Description: Upload files from local to azure.
        :param source_dir: Folder on local PC
        :param dest_dir: Folder on Azure where files to be uploaded
        :param file_name: Files to be uploaded.
        :return: None
        """
        try:
            dest_dir = dest_dir.replace('_', '-').lower().strip()
            print('Dest Dir: ',dest_dir)
            if not self.isFolderPresent(dest_dir):
                self.createFolder(dest_dir)

            container_client = self.azure_client.get_container_client(dest_dir)
            if file_name=='':
                for file_n in os.listdir(source_dir):
                    self.deleteFile(dest_dir,file_n)
                    with open(os.path.join(source_dir, file_n), "rb") as data:
                        blob_client = container_client.upload_blob(name=file_n, data=data)
            else:
                self.deleteFile(dest_dir,file_name)
                with open(os.path.join(source_dir, file_name), "rb") as data:
                    blob_client = container_client.upload_blob(name=file_name, data=data)
        except Exception as e:
            message = 'Exception Found: Function => uploadFiles, Source Dir: ' + source_dir + ', Dest Dir: ' + dest_dir
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def csvToDataframe(self, folder_name, file_name=''):
        """
        Method: csvToDataframe
        Description: Enables user to get csv files from Azure and load to a dataframe.
        :param folder_name: Folder to be looked for csv file
        :param file_name: Files to be loaded
        :return: Dataframe
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            container_client = self.azure_client.get_container_client(folder_name)
            if file_name == '':
                request_files = self.listDirFiles(folder_name)
                cnt=0
                for file in request_files:
                    if '.csv' in file:
                        #print(file['Key'])
                        obj = container_client.download_blob(file)
                        tmp_df = pd.read_csv(io.StringIO(obj.content_as_text()))
                        print(tmp_df.shape)
                        if cnt==0:
                            obj_df=tmp_df.copy()
                            cnt+=1
                        else:
                            obj_df=pd.concat([obj_df,tmp_df])
                return obj_df
            else:
                obj = container_client.download_blob(file_name)
                obj_df = pd.read_csv(io.StringIO(obj.content_as_text()))
                return obj_df
        except Exception as e:
            message = 'Exception Found: Function => csvToDataframe, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def saveDataframeToCsv(self,folder_name,file_name,df):
        """
        Method: saveDataframeToCsv
        Description: Loads dataframe to a csv file on Azure.
        :param folder_name: Folder Name
        :param file_name: File Name
        :param df: Dataframe to be loaded.
        :return: None
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if not self.isFolderPresent(folder_name):
                self.createFolder(folder_name)
            container_client = self.azure_client.get_container_client(folder_name)
            csv_buffer = io.StringIO()
            print(df.shape)
            df.to_csv('tmp.csv', index=False)
            self.deleteFile(folder_name,file_name)
            df.to_csv(csv_buffer,index=False)
            container_client.upload_blob(name=file_name,data=csv_buffer.getvalue())
        except Exception as e:
            message = 'Exception Found: Function => saveDataframeToCsv, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def saveObject(self,folder_name,file_name,object_name):
        """
        Method: saveObject
        Description: Load model to object on cloud.
        :param folder_name: Folder Name
        :param file_name: File Name
        :param object_name: Object Name
        :return: None
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            if not self.isFolderPresent(folder_name):
                self.createFolder(folder_name)
            self.deleteFile(folder_name, file_name)
            container_client = self.azure_client.get_container_client(folder_name)
            if '.PNG' in file_name:
                container_client.upload_blob(name=file_name, data=object_name)
            else:
                container_client.upload_blob(name=file_name, data=dill.dumps(object_name))
        except Exception as e:
            message = 'Exception Found: Function => saveObject, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def loadObject(self,folder_name,file_name):
        """
        Method: loadObject
        Description: Loads object from Azure and dumps to a variable
        :param folder_name: Folder Name
        :param file_name: Fle Name
        :return: Model
        """
        try:
            folder_name = folder_name.replace('_', '-').lower().strip()
            container_client = self.azure_client.get_container_client(folder_name)
            blobstring = container_client.download_blob(file_name)
            object_name = dill.loads(blobstring.readall())
            return object_name
        except Exception as e:
            message = 'Exception Found: Function => loadObject, Folder Name: ' + folder_name + ', File Name: ' + file_name
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def copyFileToFolder(self,source_folder,target_folder,file_name=''):
        """
        Method: copyFileToFolder
        Description: Copy files from source to target folder.
        :param source_folder: Source folder
        :param target_folder: Target Folder
        :param file_name: File To be copied
        :return: None
        """
        try:
            source_folder = source_folder.replace('_', '-').lower().strip()
            target_folder = target_folder.replace('_', '-').lower().strip()
            account_name = self.azure_client.account_name
            if file_name=='':
                file_list=self.listDirFiles(source_folder)
                for file_n in file_list:
                    print(file_n)
                    source_blob = (f"https://{account_name}.blob.core.windows.net/{source_folder}/{file_n}")
                    self.deleteFile(target_folder, file_n)
                    copied_blob = self.azure_client.get_blob_client(target_folder,file_n)
                    copied_blob.start_copy_from_url(source_blob)
            else:
                source_blob = (f"https://{account_name}.blob.core.windows.net/{source_folder}/{file_name}")
                self.deleteFile(target_folder, file_name)
                copied_blob = self.azure_client.get_blob_client(target_folder, file_name)
                copied_blob.start_copy_from_url(source_blob)
        except Exception as e:
            message = 'Exception Found: Function => copyFileToFolder, Source Dir: ' + source_folder + ', Target Dir: ' + target_folder
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e


    def moveFileToFolder(self, source_folder, target_folder, file_name=''):
        """
        Method: moveFileToFolder
        Description: Move files from source to target folder.
        :param source_folder: Source Folder
        :param target_folder: Target Folder
        :param file_name: File to be moved
        :return:
        """
        try:
            source_folder = source_folder.replace('_', '-').lower().strip()
            target_folder = target_folder.replace('_', '-').lower().strip()
            if file_name == '':
                file_list = self.listDirFiles(source_folder)
                for file_n in file_list:
                    self.copyFileToFolder(source_folder, target_folder, file_n)
                    source_client = self.azure_client.get_blob_client(source_folder,file_n)
                    source_client.delete_blob()
            else:
                self.copyFileToFolder(source_folder, target_folder, file_name)
                source_client = self.azure_client.get_blob_client(source_folder, file_name)
                source_client.delete_blob()
        except Exception as e:
            message = 'Exception Found: Function => moveFileToFolder, Source Dir: ' + source_folder + ', Target Dir: ' + target_folder
            self.logger.log(self.file, message + ' : ' + str(e))
            raise e