import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from AzureBlobStorage.azureBlobStorage import AzureBlobStorage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from Email_Trigger.send_email import email
from datetime import datetime


class prediction:

    def __init__(self,path):
        self.file_object = 'Prediction_Log'
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)
        self.azureObj = AzureBlobStorage()
        self.emailObj = email()

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            data = preprocessor.dropUnnecessaryColumns(data,['Id','ActivityDate','TotalDistance','TrackerDistance'])

            # replacing 'na' values with np.nan as discussed in the EDA part

            data = preprocessor.replaceInvalidValuesWithNull(data)

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            #scale the prediction data
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)

            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            kmeans=file_loader.load_model('KMeans')

            ##Code changed
            clusters=kmeans.predict(data_scaled)#drops the first column for cluster prediction
            data_scaled['clusters']=clusters
            clusters=data_scaled['clusters'].unique()
            result=[] # initialize blank list for storing predicitons

            for i in clusters:
                cluster_data= data_scaled[data_scaled['clusters']==i]
                cluster_data = cluster_data.drop(['clusters'],axis=1)
                model_name = file_loader.find_correct_model_file(i)
                model = file_loader.load_model(model_name)
                for val in (model.predict(cluster_data.values)):
                    result.append(val)
            result = pandas.DataFrame(result,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            self.azureObj.saveDataframeToCsv('Prediction_Output_File', 'Predictions.csv', result)
            #result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')

            # Triggering mail with confirmation
            msg = MIMEMultipart()
            msg['Subject'] = 'FitBit Calories - Prediction Done | ' + str(datetime.now())
            body = 'Model Prediction Done Successfully... <br><br> Thanks and Regards, <br> Rahul Garg'
            msg.attach(MIMEText(body, 'html'))
            to_addr = ['rahulgarg366@gmail.com']
            self.emailObj.trigger_mail(to_addr, [], msg)

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path