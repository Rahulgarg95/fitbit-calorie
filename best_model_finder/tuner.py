from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics  import r2_score
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics  import roc_auc_score,accuracy_score,confusion_matrix
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


class Model_Finder:
    """
        This class shall  be used to find the model with best accuracy and AUC score.
    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestRegressor()
        self.DecisionTreeReg = DecisionTreeRegressor()
        self.xgb = XGBRegressor(objective='reg:squarederror')
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
                        'max_features': self.max_features, 'n_estimators': self.n_estimators}
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

            tmp_dict = {'Model_Name': 'Linear Regression','loss': self.loss,'penalty': self.penalty, 'alpha': self.alpha}
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


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:

            self.decisionTreeReg= self.get_best_params_for_DecisionTreeRegressor(train_x, train_y)
            self.prediction_decisionTreeReg = self.decisionTreeReg.predict(test_x) # Predictions using the decisionTreeReg Model
            self.decisionTreeReg_error = r2_score(test_y,self.prediction_decisionTreeReg)



         # create best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)  # Predictions using the XGBoost Model
            self.prediction_xgboost_error = r2_score(test_y,self.prediction_xgboost)


            #comparing the two models
            if(self.decisionTreeReg_error <  self.prediction_xgboost_error):
                return 'XGBoost',self.xgboost
            else:
                return 'DecisionTreeReg',self.decisionTreeReg

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

