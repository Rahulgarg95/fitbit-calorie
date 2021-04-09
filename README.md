# FIT BIT Calorie Predictor
A regression model to predict calories burnt using multiple sensor readings.

### Highlights
---
  1. Object Oriented Design.
  2. Individual Train and Prediction Pipelines.
  3. User Friendly UI as well as API support to easily initial train/prediction pipelines.
  4. Azure Blob Storage For Files/Models and all intermediary data.
  5. End to End Log Capture in MongoDB.
  6. Combining Clustering and Regression Techniques For Better Results.
  7. Multiple Performance Metrics Capture in MongoDB to compare Model performance.
  8. End to End ML pipeline deployment with production ready code.
  
 
 ### ML Models
 ---
  1. K Means Clustering.
  2. DecisionTree Regressor.
  3. RandomForest Regressor.
  4. K Nearest Neighbors Regressor.
  5. SGD Regressor.
  6. XGBoost Regressor.
  7. Support Vector Regressor.
  

### Performance Metrics
---
  1. Meadian Absolute Error.
  2. Mean Squared Error.
  3. R2 Error. (Main Metrics To Compare Model Performances and Select Best Model For Each Cluster.)
  4. Explained Variance Ratio.
  5. Mean Absolute Error.
 
  
## Training  

### Input Data
---

Description Of Input File Attributes:
  1. Id: The customer ID
  2. ActivityDate: The date for which the activity is getting tracked.
  3. TotalSteps:  Total Steps taken on that day.
  4. TotalDistance: Total distance covered.
  5. TrackerDistance: Distance as per the tracker
  6. LoggedActivitiesDistance: Logged 
  7. VeryActiveDistance: The distance for which the user was the most active. 
  8. ModeratelyActiveDistance: The distance for which the user was moderately active.
  9. LightActiveDistance: The distance for which the user was the least active.
  10.	SedentaryActiveDistance: The distance for which the user was almost inactive.
  11.	VeryActiveMinutes: The number of minutes for the most activity.
  12.	FairlyActiveMinutes: The number of minutes for moderately activity.
  13.	LightlyActiveMinutes: The number of minutes for the least activity
  14.	SedentaryMinutes: The number of minutes for almost no activity
  15.	Calories(Target): The calories burnt.
  
Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:
Name of the files, Length of Date value in FileName, Length of Time value in FileName, Number of Columns, Name of the Columns, and their datatype.


### Data Validation
---

List of data validation performed before data preprocessing stage:
  1.  Name Validation - We validate the name of the files based on the given name in the schema file. We have created a regex pattern as per the name given in the schema file to use for validation. After validating the pattern in the name, we check for the length of date in the file name as well as the length of time in the file name. If all the values are as per requirement, we move such files to "Good_Data_Folder" else we move such files to "Bad_Data_Folder."
  2.  Number of Columns - We validate the number of columns present in the files, and if it doesn't match with the value given in the schema file, then the file is moved to "Bad_Data_Folder."
  3.  Name of Columns - The name of the columns is validated and should be the same as given in the schema file. If not, then the file is moved to "Bad_Data_Folder".
  4.  The datatype of columns - The datatype of columns is given in the schema file. This is validated when we insert the files into Database. If the datatype is wrong, then the file is moved to "Bad_Data_Folder".
  5.  Null values in columns - If any of the columns in a file have all the values as NULL or missing, we discard such a file and move it to "Bad_Data_Folder".


### Data Insertion
---

  1. After initial set of validation, data is inserted to a single table 'Good_Data'
  2. Mongo Atlas is used to store all the data.
  
  
### Model Training
---

  1.  Data Export from Db - The data in a stored database is exported as a CSV file to be used for model training.
  2.  Data Preprocessing   
    a.  Drop columns not useful for training the model. Such columns were selected while doing the EDA.  
    b.  Replace the invalid values with numpy “nan” so we can use imputer on such values.  
    c.  Check for null values in the columns. If present, impute the null values.   
    d.  Scale the training and test data separately.   
  3.  Clustering - KMeans algorithm is used to create clusters in the preprocessed data. The optimum number of clusters is selected by plotting the elbow plot, and for the dynamic selection of the number of clusters, we are using "KneeLocator" function. The idea behind clustering is to implement different algorithms.  
    To train data in different clusters. The Kmeans model is trained over preprocessed data and the model is saved for further use in prediction.
  4.  Model Selection - After clusters are created, we find the best model for each cluster. We are using 5 algorithm, "RandomForest Regressor", "XGBoost Regressor", "DecisionTree Regressor", "K-Nearest Neighbors" and "SGDRegressor". For each cluster, all five algorithms are passed with the best parameters derived from GridSearch. We calculate the Rsquared scores for both models and select the model with the best score. Similarly, the model is selected for each cluster. All the models for every cluster are saved for use in prediction.


## Prediction
---

1.  Data Export from Db - The data in the stored database is exported as a CSV file to be used for prediction.
2.  Data Preprocessing   
  a.  Drop columns not useful for training the model. Such columns were selected while doing the EDA.  
  b.  Replace the invalid values with numpy “nan” so we can use imputer on such values.  
  c.  Check for null values in the columns. If present, impute the null values.  
  d.  Scale the training data.  
3.  Clustering - KMeans model created during training is loaded, and clusters for the preprocessed prediction data is predicted.
4.  Prediction - Based on the cluster number, the respective model is loaded and is used to predict the data for that cluster.
5.  Once the prediction is made for all the clusters, the predictions along with the original names before label encoder are saved in a CSV file at a given location and the location is returned to the client
