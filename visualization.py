
import pandas as pd
import numpy as np
import yaml
import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from main import visualize_model

if __name__ == '__main__':
    np.random.seed(seed=42)

    df = pd.read_csv('./resources/flughafen/data.csv', sep=";")
    config = yaml.load(open('./resources/flughafen/config.yml'))
    time_regex = config['config']['timestamp_regex']
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, time_regex))

    x = df.drop('value', axis=1)
    y = df['value']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # create a 2d array to save y
    y_array = np.array(y_test, ndmin=2)
    for i in range(999):
        y_array = np.concatenate([np.array(y_test, ndmin=2), y_array])

    # initialize a 2d array to save y^
    initializer = np.random.normal(100, 1, 6790)
    prediction_array = np.array(initializer, ndmin=2)

    # Select the sigma randomly in the range of (std*0.75, std*1.25)
    std = np.sqrt(np.var(y_train))
    sigma = np.random.uniform(std * 0.9, std * 1.1, 1000).tolist()

    model = []
    y = []
    MSE = []
    counter = 1

    # iterate mu and sigma 1000 times to simulate 1000 models and calculate the mse
    for mu, sigma in zip(y_test, sigma):
        prediction = np.random.normal(mu, sigma, 6790)
        # save mse
        mse = mean_squared_error(y_test, prediction)
        MSE.append(mse)
        prediction = np.array(prediction, ndmin=2)
        # save y^ to a 2d array
        prediction_array = np.concatenate([prediction_array, prediction])
        # save model name
        model.append('Model-%d' % (counter))
        counter = counter + 1

    # delete the initializer
    prediction_array = prediction_array[1:]

    MSE_2d = np.array([1], ndmin=2)
    for i in MSE:
        MSE_2d_element = np.array(i, ndmin=2)
        MSE_2d = np.concatenate([MSE_2d, MSE_2d_element])

    MSE_2d = MSE_2d[1:]

    models = {'model': model, 'y': y_array, 'y^': prediction_array, 'MSE': MSE}


    # Visualization
    i = 300
    name = models['model'][i]+' '+str(models['MSE'][i])
    data = pd.DataFrame(data={'y': models['y'][i],'y_pred':models['y^'][i]})
    visualize_model(model_name=name, data=data.iloc[:200])





    X = np.concatenate((y_array, prediction_array), axis=1)
    y = MSE_2d
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X=X, y=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
