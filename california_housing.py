import os 
import joblib 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score 

MODEL_FILE = "model.pkl"
PIPELINE_FILE ="pipeline.pkl"

def build_pipeline(num_attributes , categorical_attributes):
    #  making pipelines for numerical coulmns  and categorical columns 
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median" )), 
        ("scaler" , StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("one hot" , OneHotEncoder(handle_unknown= "ignore" ))
    ])

    # constructing the main pipeline 

    full_pipeline = ColumnTransformer([
        ("numerical" , num_pipeline , num_attributes) ,
        ("categorical" , categorical_pipeline , categorical_attributes)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE) :
    # lets train the model 
    
    #  loading the csv data from pandas 
    housing = pd.read_csv("housing.csv")
    
    # creating stratified  test set 
    housing["income_cat"] = pd.cut(housing["median_income"] , 
                                            bins=[0.0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf],
                                            labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1 , test_size= 0.2  , random_state=42)

    for train_index , test_index in split.split(housing , housing["income_cat"]):
        housing_train = housing.iloc[train_index ].drop("income_cat" , axis= 1)
        housing_test = housing.iloc[test_index ].drop("income_cat" , axis= 1)
        housing_test.to_csv("input.csv" , index= False)
        
    #  seprating features and labels ( seprating the predicting values "target variable" from the datasets  )
    housing_labels = housing_train["median_house_value"].copy()
    housing_features = housing_train.drop("median_house_value" , axis = 1 )

    num_attributes = housing_features.drop("ocean_proximity" , axis = 1).columns.tolist()
    categorical_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributes , categorical_attributes )
    housing_prepared = pipeline.fit_transform(housing_features)
    

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared , housing_labels)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline , PIPELINE_FILE)
    print("Model is trained . congrats !! ")

else :
    # lets do inference 
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data['median_house_value'] = predictions
    
    input_data.to_csv("output.csv" , index= False)
    print("Inference is completed  \n The result is at output.csv ")


