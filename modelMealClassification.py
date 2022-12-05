import pandas as pd
from defines import *
from model import *
from xml_read import *
from xml_write import *
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std


def fillmealcarbs(data:pd.DataFrame,ts):
    ts2=pd.Timestamp(ts)+pd.Timedelta(5,"m")
    print(ts2)
    bool=False
    for x in data['meal']['ts']:
        if(ts2==x):
            bool=True
    if(bool==True):
        return 1
    else: return 0


def model(data: dict[str,pd.DataFrame]):
    scaler=MinMaxScaler(feature_range=(0,1))
    split_fraction = 0.8
    train_split = int(split_fraction * int(data['glucose_level'].shape[0]))
    step = 1
    past = 18
    future = 18
    learning_rate = 0.001
    batch_size = 256
    epochs = 10
    dataX = data['glucose_level']
    dataX['ts'] = pd.to_numeric(pd.to_datetime(dataX['ts']))
    dataX = scaler.fit_transform(dataX)
    dataX=pd.DataFrame(dataX)


    train_data=dataX.loc[0:train_split-1]
    val_data=dataX.loc[train_split:]

    #Training dataset
    start=past+future
    end=start+train_split

    x_train = train_data.values
    #y_train = .iloc[start:end][[1]]

    sequence_length = int(past / step)





    #dataMeal=data['meal']
    #dataExercise=data['exercise']
    #dataWork=data['work']
    #dataX=pd.concat([dataGlucoseLevel,dataMeal,dataWork,dataExercise])
    #dataX=dataX.sort_values(by='ts',ignore_index=True)



    dataY=data['meal']['carbs']

if __name__ == "__main__":
    data, patient_data = load(TRAIN2_540_PATH)
    clean_data = data_preparation(data, pd.Timedelta(5, "m"), 30, 3)
    model(clean_data)
    mealdata=clean_data['meal']
    mealdata['carbs']=mealdata['carbs'].apply(lambda x:1 )
    glucosedata=clean_data['glucose_level']
    glucosedata['carbs']=""
    glucosedata['carbs'] = glucosedata['carbs'].apply(lambda x: 0)
    glucosedata=glucosedata.drop(['value'],axis=1)
    meal=pd.concat([glucosedata,mealdata])
    meal=meal.sort_values(by='ts',ignore_index=True)

    print(fillmealcarbs(clean_data,"22-05-2027 09:45:00"))