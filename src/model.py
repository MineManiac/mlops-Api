import pickle
# Modeling
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

def load_model():
    model = pickle.load(open('../models/model.pkl','rb'))
    return model


def load_encoder():
    one_hot_enc = pickle.load(open("../models/ohe.pkl","rb"))
    return one_hot_enc