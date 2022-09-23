from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler,LabelEncoder
import numpy as np
import pandas as pd

class process_data():
    def __init__(self,data,automatic_procedure=True):
        self.raw_data = data
        self.prep_data = None
        if automatic_procedure:
            self._initiat_preprocessing()

    def encode_data(self,exclude: list = [],one_hot_lim_threshold = 10,data=None):
        """This methode will indicate string data and will one hot encode it except for what we tell it to exclude from the process.
        Args:
            exclude: list inputs contain the column names what we don't want to process.
            one_hot_lim_threshold: If number of unique values exceeds that limit it will automatically label_encode it,
                                 make 0 if you want to label all categories.
        return:
            prep_data: the one hot encoded data after preprocessing.
        """
        if data:
            prep_data = data
        else:
            if len(exclude) > 0:
                prep_data = self.raw_data[self.raw_data!=exclude]
            else:
                prep_data = self.prep_data
        
        for col in prep_data.columns:
            if prep_data[col].dtypes == object:
                if len(prep_data[col].unique()) > one_hot_lim_threshold:
                    prep_data[col] = self._label_encode(prep_data[col])
                else:
                    encoded_data = self._onehotencode(prep_data[col])
                    prep_data = prep_data.drop(col,axis=1)
                    prep_data = pd.concat([prep_data, encoded_data],axis=1)

        if len(exclude) > 0: # Checks if we wanted to exclude any columns
            prep_data[exclude] = self.raw_data[exclude]

        self.prep_data = prep_data
        return self.prep_data
        # ---------------

    def _onehotencode(self,data):
        encoder = OneHotEncoder(sparse=False)
        return pd.DataFrame(encoder.fit_transform(np.array(data).reshape(-1,1)))
        # --------

    def _label_encode(self,data):
        encoder = LabelEncoder()
        return encoder.fit_transform(data)
        # --------

    def scale_data(self,data=None,scaler: str = "MinMaxScaler"):
        if not scaler in ["StandardScaler","MinMaxScaler"]:
            raise ValueError(f"{scaler} isn't a StandardScaler or MinMaxScaler,"+ 
                                "please pick one.. or leave as the defualt MinMaxScaler")

        if data:
            prep_data = data
        else:
            prep_data = self.raw_data
        for col in prep_data.columns:
            if prep_data[col].dtypes != object:
                if scaler == "StandardScaler":
                    prep_data[col] = self._standard_scaler(prep_data[col])
                else:
                    prep_data[col] = self._minmaxscaler(prep_data[col])
        self.prep_data = prep_data
        return self.prep_data

    def _standard_scaler(self,data):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(np.array(data).reshape(-1,1)))
    
    def _minmaxscaler(self,data):
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(np.array(data).reshape(-1,1)))

    def _initiat_preprocessing(self):
        """This function automates the processes of ecnoding and scaling all data
        return: 
            self.prep_data: the final processed data
        """
        self.scale_data()
        self.encode_data()       
        return self.prep_data

    def get_preprocessed_data(self):
        """This function return the preprocessed data"""
        return self.prep_data