import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#Goal: To create a predictive model using Forbes data from 2013, 2015, 2017, 2020, 2021, and 2023 to see what companies will most likely be in the top 10 in 2024

#Initialize all datasets
df_global_2000_2013 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/forbes2013.csv")
df_global_2000_2015 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/Forbes2015.csv")
df_global_2000_2017 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/Forbes Top2000 2017.csv")
df_global_2000_2020 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/forbes_top_2000_world_largest_public_companies.csv")
df_global_2000_2021 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/fortune_2000_in_2021.csv")

#We are going to set aside this dataset for prediction later
df_global_2000_2023 = pd.read_csv("/Users/eshan/Downloads/ForbesTop2000DataSets/forbes_the_global_2000_2023update.csv")

#Cleaning up all datasets 
df_global_2000_2013 = df_global_2000_2013.sort_values(by = 'Rank') 
df_global_2000_2015 = df_global_2000_2015.sort_values(by = 'Rank')
column_to_move = df_global_2000_2015.pop('Rank')
df_global_2000_2015.insert(0, 'Rank', column_to_move)
df_global_2000_2017 = df_global_2000_2017.drop("Unnamed: 0", axis = 1)
df_global_2000_2020['rank'] = df_global_2000_2020['rank'].str.replace('#', '')
df_global_2000_2020.rename(columns={'rank': 'Rank'}, inplace=True)
df_global_2000_2023.rename(columns={'rank': 'Rank'}, inplace=True)
df_global_2000_2017.rename(columns={' Rank': 'Rank'}, inplace=True)

#Now that data is cleaned and we have an idea of the top 10 for each year, we can start working on finding correlations
#We're going to make a new column in each dataframe called "top_10" that will contain the amount of times any company has been in the top 10

new_column_name = 'top_10'
df_global_2000_2013[new_column_name] = 0
df_global_2000_2015[new_column_name] = 0 
df_global_2000_2017[new_column_name] = 0
df_global_2000_2020[new_column_name] = 0
df_global_2000_2021[new_column_name] = 0 

#Remember we do want to keep 2023 separate from the other data for prediction purposes
df_global_2000_2023[new_column_name] = 0
df_global_2000_2013.at[182, 'top_10'] = 1
df_global_2000_2013.at[183, 'top_10'] = 1
df_global_2000_2013.at[1031, 'top_10'] = 1
df_global_2000_2013.at[1032, 'top_10'] = 1
df_global_2000_2013.at[1033, 'top_10'] = 1
df_global_2000_2013.at[936, 'top_10'] = 1
df_global_2000_2013.at[529, 'top_10'] = 1
df_global_2000_2013.at[184, 'top_10'] = 1
df_global_2000_2013.at[185, 'top_10'] = 1
df_global_2000_2013.at[1034, 'top_10'] = 1

df_global_2000_2015.at[159, 'top_10'] = 1
df_global_2000_2015.at[160, 'top_10'] = 1
df_global_2000_2015.at[161, 'top_10'] = 1
df_global_2000_2015.at[1433, 'top_10'] = 1
df_global_2000_2015.at[1434, 'top_10'] = 1
df_global_2000_2015.at[1435, 'top_10'] = 1
df_global_2000_2015.at[1436, 'top_10'] = 1
df_global_2000_2015.at[1437, 'top_10'] = 1
df_global_2000_2015.at[162, 'top_10'] = 1
df_global_2000_2015.at[163, 'top_10'] = 1

df_global_2000_2017.at[0, 'top_10'] = 1
df_global_2000_2017.at[1, 'top_10'] = 1
df_global_2000_2017.at[2, 'top_10'] = 1
df_global_2000_2017.at[3, 'top_10'] = 1
df_global_2000_2017.at[4, 'top_10'] = 1
df_global_2000_2017.at[5, 'top_10'] = 1
df_global_2000_2017.at[6, 'top_10'] = 1
df_global_2000_2017.at[7, 'top_10'] = 1
df_global_2000_2017.at[8, 'top_10'] = 1
df_global_2000_2017.at[9, 'top_10'] = 1

df_global_2000_2020.at[0, 'top_10'] = 1
df_global_2000_2020.at[1, 'top_10'] = 1
df_global_2000_2020.at[2, 'top_10'] = 1
df_global_2000_2020.at[3, 'top_10'] = 1
df_global_2000_2020.at[4, 'top_10'] = 1
df_global_2000_2020.at[5, 'top_10'] = 1
df_global_2000_2020.at[6, 'top_10'] = 1
df_global_2000_2020.at[7, 'top_10'] = 1
df_global_2000_2020.at[8, 'top_10'] = 1
df_global_2000_2020.at[9, 'top_10'] = 1

df_global_2000_2021.at[0, 'top_10'] = 1
df_global_2000_2021.at[1, 'top_10'] = 1
df_global_2000_2021.at[2, 'top_10'] = 1
df_global_2000_2021.at[3, 'top_10'] = 1
df_global_2000_2021.at[4, 'top_10'] = 1
df_global_2000_2021.at[5, 'top_10'] = 1
df_global_2000_2021.at[6, 'top_10'] = 1
df_global_2000_2021.at[7, 'top_10'] = 1
df_global_2000_2021.at[8, 'top_10'] = 1
df_global_2000_2021.at[9, 'top_10'] = 1

df_global_2000_2023.at[0, 'top_10'] = 1
df_global_2000_2023.at[1, 'top_10'] = 1
df_global_2000_2023.at[2, 'top_10'] = 1
df_global_2000_2023.at[3, 'top_10'] = 1
df_global_2000_2023.at[4, 'top_10'] = 1
df_global_2000_2023.at[5, 'top_10'] = 1
df_global_2000_2023.at[6, 'top_10'] = 1
df_global_2000_2023.at[7, 'top_10'] = 1
df_global_2000_2023.at[8, 'top_10'] = 1
df_global_2000_2023.at[9, 'top_10'] = 1

#Now that we have the top_10 columns done, what we want to do is make a master dataframe
#This is obviously because we don't want to be training a model across 5 different dataframes, that would be too much cause for error
#We're going to want to make sure that we also have a separate year value in every dataframe now so that later, after merging, we don't get any of the data confused
#To make this dataframe, we'll need to make all of the columns the exact same
#And we must remember to keep 2023 separate from the master dataframe, that is what we will do our predictions on

new_column_name = 'publish_year'
df_global_2000_2013[new_column_name] = 2013
df_global_2000_2015[new_column_name] = 2015
df_global_2000_2017[new_column_name] = 2017
df_global_2000_2020[new_column_name] = 2020
df_global_2000_2021[new_column_name] = 2021
df_global_2000_2013.rename(columns={'Sales($billion)': 'Sales'}, inplace=True)
df_global_2000_2013.rename(columns={'Profits($billion)': 'Profits'}, inplace=True)
df_global_2000_2015 = df_global_2000_2015.drop("Sector", axis = 1)
df_global_2000_2015 = df_global_2000_2015.drop("Industry", axis = 1)
df_global_2000_2015 = df_global_2000_2015.drop("Continent", axis = 1)
df_global_2000_2015 = df_global_2000_2015.drop("Forbes Webpage", axis = 1)
df_global_2000_2017 = df_global_2000_2017.drop("Industry", axis = 1)
df_global_2000_2017 = df_global_2000_2017.drop("Sector", axis = 1)
df_global_2000_2020.rename(columns={'company': 'Company'}, inplace=True)
df_global_2000_2020.rename(columns={'contry/territory': 'Country'}, inplace=True)
df_global_2000_2020.rename(columns={'sales': 'Sales'}, inplace=True)
df_global_2000_2020.rename(columns={'profits': 'Profits'}, inplace=True)
df_global_2000_2020.rename(columns={'assets': 'Assets'}, inplace=True)
df_global_2000_2020.rename(columns={'market_value': 'Market Value'}, inplace=True)
df_global_2000_2020['Sales'] = df_global_2000_2020['Sales'].str.replace('[$BM]', '', regex=True)
df_global_2000_2020['Profits'] = df_global_2000_2020['Profits'].str.replace('[$BM]', '', regex=True)
df_global_2000_2020['Assets'] = df_global_2000_2020['Assets'].str.replace('[$BM]', '', regex=True)
df_global_2000_2020['Market Value'] = df_global_2000_2020['Market Value'].str.replace('[$BM]', '', regex=True)
df_global_2000_2020['Sales'] = pd.to_numeric(df_global_2000_2020['Sales'], errors='coerce')
df_global_2000_2020['Rank'] = pd.to_numeric(df_global_2000_2020['Rank'], errors='coerce')
df_global_2000_2020['Profits'] = pd.to_numeric(df_global_2000_2020['Profits'], errors='coerce')
df_global_2000_2020['Assets'] = pd.to_numeric(df_global_2000_2020['Assets'], errors='coerce')
df_global_2000_2020['Market Value'] = pd.to_numeric(df_global_2000_2020['Market Value'], errors='coerce')
df_global_2000_2021.rename(columns={'Name': 'Company'}, inplace=True)
df_global_2000_2021.rename(columns={'Profit': 'Profits'}, inplace=True)
df_global_2000_2021['Sales'] = df_global_2000_2021['Sales'].str.replace('[$BM]', '', regex=True)
df_global_2000_2021['Profits'] = df_global_2000_2021['Profits'].str.replace('[$BM]', '', regex=True)
df_global_2000_2021['Assets'] = df_global_2000_2021['Assets'].str.replace('[$BM]', '', regex=True)
df_global_2000_2021['Market Value'] = df_global_2000_2021['Market Value'].str.replace('[$BM]', '', regex=True)
df_global_2000_2023.rename(columns={'company': 'Company'}, inplace=True)
df_global_2000_2023.rename(columns={'country': 'Country'}, inplace=True)
df_global_2000_2023.rename(columns={'sales': 'Sales'}, inplace=True)
df_global_2000_2023.rename(columns={'profit': 'Profits'}, inplace=True)
df_global_2000_2023.rename(columns={'asset': 'Assets'}, inplace=True)
df_global_2000_2023.rename(columns={'market_value': 'Market Value'}, inplace=True)
df_global_2000_2023['Sales'] = df_global_2000_2023['Sales'] / 1000
df_global_2000_2023['Profits'] = df_global_2000_2023['Profits'] / 1000
df_global_2000_2023['Assets'] = df_global_2000_2023['Assets'] / 1000
df_global_2000_2023['Market Value'] = df_global_2000_2023['Market Value'] / 1000

#Now we can make the actual master dataframe WITHOUT 2023 data
#We're also going to want to drop every row with NaN in it because sklearn can't train models over NaN

df_master = pd.concat([df_global_2000_2013, df_global_2000_2015, df_global_2000_2017, df_global_2000_2020, df_global_2000_2021])
df_master = df_master.dropna()

#There are a ton of values with commas, and we don't want that since those are strings! Strings can't be used in .corr(), so we make them into floats!

def clean_and_convert(value):
    try:
        cleaned_value = ''.join(char for char in str(value) if char.isdigit() or char == '.')
        return float(cleaned_value)
    except (ValueError, TypeError):
        return None

df_master['Assets'] = df_master['Assets'].apply(clean_and_convert)
df_master['Market Value'] = df_master['Market Value'].apply(clean_and_convert)

#We're also going to check quickly which companies are actually in the top 10, some will have been removed due to NaN values
df_master_top_10 = df_master[df_master['top_10'] == 1]

#Now that we have a master dataframe, we're gonna look for a correlation between numerical values and being top 10 or not
#This is what we will use to see correlations - numerical values only

df_master_numbers = df_master.drop("Company", axis = 1)
df_master_numbers = df_master_numbers.drop("Country", axis = 1)
df_master_numbers = df_master_numbers.drop("publish_year", axis = 1)
df_master_numbers = df_master_numbers.drop("Rank", axis = 1)
column_to_move = df_master_numbers.pop('top_10')
df_master_numbers.insert(0, 'top_10', column_to_move)
sns.heatmap(df_master_numbers.corr(), annot = True, cmap = "rocket")
plt.show()

#In terms of placing in the top 10, Assets and Market Value seem to be the two most important factors, with Sales coming in at a close third
#Profits for some reason seem to actually have a negative correlation with placing in the top 10
#Want to make a Logistic Regression model with Assets, Market Value, and Sales
#Want to be Binary classification. (Will be top 10 or not? Binary)
#Now we can create a model trained on this database

X_train, X_test, y_train, y_test = train_test_split(df_master_numbers[['Sales','Assets', 'Market Value']],df_master_numbers.top_10,test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

#Accuracy has a lot of leeway, anything above 0.95 is still statistically significant!
print("With " + str(model.score(X_test, y_test)) + " accuracy:")
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

#What really matters here is that the model actually does have mistakes
#Why do we want it to have mistakes? To prove that it's actually learning, a model can't learn anything without mistakes

sns.heatmap(cm, annot= True)
plt.xlabel("Prediction")
plt.ylabel("Truth")
plt.show()

#Now, we have to go back to the 2023 data and alter it in a manner that can be used for prediction
df_global_2000_2023_predict = df_global_2000_2023.drop("Company", axis = 1)
df_global_2000_2023_predict = df_global_2000_2023_predict.drop("Country", axis = 1)
df_global_2000_2023_predict = df_global_2000_2023_predict.drop("publish_year", axis = 1)
df_global_2000_2023_predict = df_global_2000_2023_predict.drop("Rank", axis = 1)
df_global_2000_2023_predict = df_global_2000_2023_predict.drop("Profits", axis = 1)
df_global_2000_2023_predict = df_global_2000_2023_predict.drop("top_10", axis = 1)

#Now we can get the predictions from the 2023 data and the probabilities of those predictions
predictions = np.array(model.predict(df_global_2000_2023_predict))
prediction_probabilities = np.array(model.predict_proba(df_global_2000_2023_predict))
indices_of_ones = np.where(predictions == 1)
count_of_ones = len(indices_of_ones[0])

#Here we can see how many companies are predicted to be within the top 10
#print(str(count_of_ones) + " companies are predicted to be within the top 10.")

#Here we can see were those companies are in the 2023 database
#print(indices_of_ones[0])

#We can then see the exact probabilities of these companies actually being in the top 10
#print(prediction_probabilities[0])
#print(prediction_probabilities[1])
#print(prediction_probabilities[2])
#print(prediction_probabilities[3])
#print(prediction_probabilities[4])
#print(prediction_probabilities[5])
#print(prediction_probabilities[6])
#print(prediction_probabilities[8])
#print(prediction_probabilities[9])
#print(prediction_probabilities[11])
#print(prediction_probabilities[19])
#print(prediction_probabilities[35])
#print(prediction_probabilities[90])
#print(prediction_probabilities[349])
#print(prediction_probabilities[366])
#print(df_global_2000_2023.iloc[[0,1,2,3,4,5,6,8,9,11,19,35,90,349,366]])

#Now that we have the probabilities all in order, we can make a dataframe of them
df_most_likely_2024 = pd.DataFrame({'Company': ['JPMorgan Chase', 'Saudi Arabian Oil Company (Saudi Aramco)', 'ICBC', 'China Construction Bank', 'Agricultural Bank of China', 'Bank of America', 'Alphabet', 'Microsoft', 'Apple', 'Bank of China', 'HSBC Holdings', 'Amazon', 'Mitsubishi UFJ Financial', 'Fannie Mae', 'Freddie Mac'],
                                    'Probability': [0.97734562, 0.99613625, 0.99981410, 0.99668024, 0.99834411, 0.81363702, 0.62373803, 0.99192603, 0.99934153, 0.98208165, 0.61404634, 0.58270177, 0.50401387, 0.95595821, 0.59169407]
                                   })
df_most_likely_2024 = df_most_likely_2024.sort_values(by = 'Probability', ascending = False) 
print(df_most_likely_2024)

#In conclusion:
#The above database shows the 15 companies most likely to be among the top 10 Forbes Global 2000 in 2024 and the probabilities that each company places in the top 10
