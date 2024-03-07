import os 
import sys 
from dataclasses import dataclass 

import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer      #<--it is use to create pipline for model training we use  col trans , OHE , standarscaler 
from sklearn.impute import SimpleImputer           # <- if we have missing val we gonna treat it bu using simpleimputer  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder , StandardScaler  

from src.exception import  CustomException 
from src.logger import logging 