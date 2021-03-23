from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics  import explained_variance_score,accuracy_score,mean_squared_error,mean_absolute_error,r2_score,median_absolute_error
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from datetime import datetime

class Model_Finder:
    """
        This class shall  be used to find the model with best accuracy and AUC score.
    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestRegressor()
        self.DecisionTreeReg = DecisionTreeRegressor()
        self.xgb = XGBRegressor()
        self.sgdreg = SGDRegressor()
        self.knn = KNeighborsRegressor()
        self.svr = SVR()
        self.best_param_list = []
        self.perf_data = []
        self.model_list = []
        self.model_acc = []



    def get_best_params_for_random_forest(self,train_x,train_y):
        """
            Method Name: get_best_params_for_random_forest
            Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['mse', 'mae'],
                               "max_depth": range(2, 10, 2), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            tmp_dict = {'Model_Name': 'RandomForest Classifier', 'criterion': self.criterion,
                        'max_depth': self.max_depth,
                        'max_features': self.max_features, 'n_estimators': self.n_estimators}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = RandomForestRegressor(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        """
            Method Name: get_best_params_for_DecisionTreeRegressor
            Description: get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_decisionTree = {"criterion": ["mse", "friedman_mse", "mae"],
                              "splitter": ["best", "random"],
                              "max_features": ["auto", "sqrt", "log2"],
                              'max_depth': range(2, 16, 2),
                              'min_samples_split': range(2, 16, 2)
                              }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.DecisionTreeReg, self.param_grid_decisionTree, verbose=3,cv=5, n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_features = self.grid.best_params_['max_features']
            self.max_depth  = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']

            tmp_dict = {'Model_Name': 'DecisionTree Classifier', 'criterion': self.criterion,
                        'max_depth': self.max_depth, 'splitter': self.splitter,
                        'max_features': self.max_features, 'min_samples_split': self.min_samples_split}
            self.best_param_list.append(tmp_dict)

            # creating a new model with the best parameters
            self.decisionTreeReg = DecisionTreeRegressor(criterion=self.criterion,splitter=self.splitter,max_features=self.max_features,max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'DecisionTree Regression best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
            return self.decisionTreeReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_DecisionTreeRegressor method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'DecisionTree Regression tuning  failed. Exited the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_xgboost(self,train_x,train_y):

        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBRegressor(objective='reg:linear'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            tmp_dict = {'Model_Name': 'XGBRegressor', 'learning_rate': self.learning_rate,
                        'max_depth': self.max_depth,
                        'n_estimators': self.n_estimators}
            self.best_param_list.append(tmp_dict)

            # creating a new model with the best parameters
            self.xgb = XGBRegressor(objective='reg:linear',learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_knn(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_knn method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
               'leaf_size' : [5,10,15,20,24,30],
               'n_neighbors' : [3,5,7,9,10,11]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.knn, param_grid=self.param_grid, cv=5, verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.algo = self.grid.best_params_['algorithm']
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.neigh = self.grid.best_params_['n_neighbors']

            tmp_dict = {'Model_Name': 'KNN Regression','algorithm': self.algo,'leaf_size': self.leaf_size,'n_neighbors':self.neigh}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = KNeighborsRegressor(algorithm = self.algo, leaf_size =self.leaf_size, n_neighbors =self.neigh)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'K-Nearest Neighbors Regression best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_knn method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'K-Nearest Neighbors Regression Parameter tuning  failed. Exited the get_best_params_for_knn method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_svm(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'C':[0.1,1,10,50],'gamma':[1,0.5,0.1,0.01]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.svr, param_grid=self.param_grid, cv=5, verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.C_value = self.grid.best_params_['C']
            self.gamma_value = self.grid.best_params_['gamma']

            tmp_dict = {'Model_Name':'Support Vector Regressor','C': self.C_value,'gamma': self.gamma_value}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = SVR(C=self.C_value, gamma=self.gamma_value)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVR best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Support Vector Regression Parameter tuning  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_sgd_reg(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_sgd_reg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'loss':['squared_loss','huber'],'penalty':['l2','l1','elasticnet'],'alpha':[0.0001,0.001,0.01,0.1,1,10]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.sgdreg, param_grid=self.param_grid, cv=5,  verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.loss = self.grid.best_params_['loss']
            self.penalty = self.grid.best_params_['penalty']
            self.alpha = self.grid.best_params_['alpha']

            tmp_dict = {'Model_Name': 'SGD Regression','loss': self.loss,'penalty': self.penalty, 'alpha': self.alpha}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = SGDRegressor(loss=self.loss, penalty=self.penalty, alpha=self.alpha)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SGD Regression best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_sgd_reg method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_sgd_reg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SGD Regression Parameter tuning  failed. Exited the get_best_params_for_sgd_reg method of the Model_Finder class')
            raise Exception()


    def get_performance_parameters(self,train_x,train_y,test_x,test_y,model_name,model,cluster_no):
        try:
            pref_dict={}

            now = datetime.now()
            date = now.date()
            current_time = now.strftime("%H:%M:%S")
            insert_date = str(date) + ' ' + str(current_time)
            pref_dict['Insert_Date']=str(insert_date)
            pref_dict['Cluster_No']=int(cluster_no)
            pref_dict['Model_Name']=model_name

            train_pred=model.predict(train_x)

            #Test Accuracy
            test_pred=model.predict(test_x)

            # Median Abs Error
            mae=median_absolute_error(test_y,test_pred)
            pref_dict['Median_Abs_Error'] = round(mae, 2)
            self.logger_object.log(self.file_object, 'Median Abs Error for ' + model_name + ' : ' + str(mae))

            # Mean Squared Error
            mse = mean_squared_error(test_y, test_pred)
            pref_dict['Mean_Squared_Error'] = round(mse, 2)
            self.logger_object.log(self.file_object, 'Mean Squared Error for ' + model_name + ' : ' + str(mse))

            # R2 Error
            r2_val = r2_score(test_y, test_pred)
            pref_dict['R2_Error'] = round(r2_val, 2)
            self.logger_object.log(self.file_object, 'R2 Error for ' + model_name + ' : ' + str(r2_val))

            #explained_variance_score
            evs = explained_variance_score(test_y, test_pred)
            pref_dict['Explained_Variance_Ratio'] = round(evs, 2)
            self.logger_object.log(self.file_object, 'Explained variance Error for ' + model_name + ' : ' + str(evs))

            #Mean abs error
            mae = mean_absolute_error(test_y, test_pred)
            pref_dict['Mean_Absolute_Error'] = round(mae, 2)
            self.logger_object.log(self.file_object, 'Mean Abs Error for ' + model_name + ' : ' + str(mae))

            self.model_acc.append(r2_val)
            self.model_list.append(model_name)
            self.perf_data.append(pref_dict)

        except Exception as e:
            print('Exception Occurred: ', e)
            raise e

    def get_best_model(self, train_x, train_y, test_x, test_y, cluster_no):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            model_list1 = []

            # create best model for GaussianNB
            self.decision_tree = self.get_best_params_for_DecisionTreeRegressor(train_x, train_y)
            model_list1.append(self.decision_tree)
            print('Setting Performance Parameters Decision Tree: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'DecisionTree', self.decision_tree, cluster_no)

            # create best model for XGBoost
            print('Training XgBoost Model: ')
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            model_list1.append(self.xgboost)
            print('Setting Performance Parameters XGBoost: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'XGBoost', self.xgboost, cluster_no)

            # create best model for Random Forest
            print('Training Random Forest Model: ')
            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            model_list1.append(self.random_forest)
            print('Setting Performance Parameters RandomForest: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'RandomForest', self.random_forest,
                                            cluster_no)

            # create best model for SVM
            self.support_vector = self.get_best_params_for_svm(train_x, train_y)
            model_list1.append(self.support_vector)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'SVR', self.support_vector, cluster_no)

            # create best model for SGD Regressor
            self.sgd_reg = self.get_best_params_for_sgd_reg(train_x, train_y)
            model_list1.append(self.sgd_reg)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'SGDRegressor', self.sgd_reg,
                                            cluster_no)

            # create best model for KNN Classification
            self.knearest_neigh = self.get_best_params_for_knn(train_x, train_y)
            model_list1.append(self.knearest_neigh)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'KNN', self.knearest_neigh, cluster_no)

            print('Best Param List: ', self.best_param_list)
            print('Model Names: ', self.model_list)
            print('Model Accuracy: ', self.model_acc)

            best_acc_score = max(self.model_acc)
            temp_idx = self.model_acc.index(best_acc_score)
            best_model_name = self.model_list[temp_idx]
            print(best_model_name, best_acc_score)

            return best_model_name, model_list1[temp_idx]

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()