# DATA3402
Overview
FIYIN OLANIYI
HORSE HEALTH PREDICTIONS PROJECT
LINK https://www.kaggle.com/competitions/playground-series-s3e22/data?select=train.csv

Given various medical indicators, I am asked to predict the health outcomes of horses. At the end to print for each id in the test set, predicting the corresponding outcome (Lived, Died or euthanized) using a machine learning model.
While reading about this dataset, I found that it is synthetic (created by using algorithms and simulations based on generative artificial intelligence technologies) which explains certain inconsistencies.

The Predict Health Outcomes of Horses project is dedicated to a multi-class classification challenge, where the objective is to predict whether a horse can survive based on past medical conditions. The target variable is the horse's outcome, which can fall into different classes.

Dataset Features

Here's a brief overview of the features in the dataset:

surgery: Nominal variable ('yes' or 'no') indicating whether surgery was performed.

age: Nominal variable ('adult' or 'young') representing the age of the horse.

hospital_number: Numeric variable indicating how many times a horse has gone to the hospital.

rectal_temp: Ratio data representing the horse's rectal temperature.

pulse: Ratio data indicating the horse's pulse rate.

respiratory_rate: Ratio data showing the horse's respiratory rate.

temp_of_extremitiess: Nominal data indicating the temperature of extremities.

peripheral_pulse: Nominal data indicating peripheral pulse conditions.

mucous_membrane: Ordinal data representing the condition of mucous membranes.

capillary_refill_time: Ordinal data indicating capillary refill time.

pain: Nominal data indicating the horse's pain level.

peristalsis: Ordinal data indicating the activity in the horse's gut.

abdominal_distention: Ordinal data indicating abdominal distention.

nasogastric_tube: Ordinal data representing nasogastric tube condition.

nasogastric_reflux: Ordinal data indicating nasogastric reflux.

nasogastric_reflux_ph: Ratio data representing nasogastric reflux pH.

rectal_exam_feces: Ordinal data representing rectal examination findings related to feces.

abdomen: Ordinal data representing the condition of the abdomen.

packed_cell_volume: Ratio data representing packed cell volume in the blood.

total_protein: Ratio data representing total protein levels in the blood.

abdomo_appearance: Nominal data representing abdominal appearance.

abdomo_protein: Ratio data indicating abdominal protein levels.

surgical_lesion: Nominal data indicating whether a surgical lesion is present.

lesion_1, lesion_2, lesion_3: Ratio data related to lesions (some of which may be dropped).

cp_data: Nominal data indicating the presence of central nervous system disorders.

outcome: Target variable representing the horse's health outcome.


**LIBRARIES I USED**
import numpy as np
import pandas as pd

# Data Visualization Libraries Import
import matplotlib.pyplot as plt
import seaborn as sns

# Splitting the data
from sklearn.model_selection import StratifiedShuffleSplit

#Feature Engineering and Encoding
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

#Scaling
from sklearn.preprocessing import MinMaxScaler

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')


**OVERVIEW**
I had no numerical columns with missing values
categorical columns with missing values: ['temp_of_extremities', 'peripheral_pulse', 'mucous_membrane', 'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention', 'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen', 'abdomo_appearance']
