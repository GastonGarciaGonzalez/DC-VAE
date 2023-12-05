import pandas as pd
import os

class InfluxdbSeries:

    def __init__(self, path=''):
        self.path = path
        # list of filenames in path
        self.filenames = os.listdir(path)
    
    def getseries(self, concatenation=False):
        # Read all files
        self.dfs = []
        for filename in self.filenames:
            # read data csv and use as columns names the values of the row number 2
            data_new = pd.read_csv(self.path+filename, header=3)
            #remove all columns except _time _value and _measurment
            data_new = data_new[['_time', '_value', '_measurement']]
            #rename column _value to the first value of _mesaurment and delete this last column
            data_new = data_new.rename(columns={'_value': data_new['_measurement'][0]}) 
            data_new = data_new.drop(columns=['_measurement'])
            #convert _time column to datetime format and set it as index
            data_new['_time'] = pd.to_datetime(data_new['_time'])
            data_new = data_new.set_index('_time')
            #append the new dataframe to the list
            self.dfs.append(data_new)

        if concatenation:
            #concatenate all dataframes
            return pd.concat(self.dfs, axis=1)
        
        return self.dfs
        