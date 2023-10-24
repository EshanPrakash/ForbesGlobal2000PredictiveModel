# A Computational Approach to Understanding the Future Economic Performance of Corporations

By: Eshan Prakash

## Abstract:

The Forbes Global 2000 is an annual ranking of hte top 2000 public companies in the world, published by Forbes magazine. This paper attempts to utilize machine learning and data science in order to determine the most important factors when ranking these companies and to ultimately predict the top 10 Forbes Global 2000 companies in 2024. Multiple factors that I have found to contribute to top 10 placement include Sales, Assets, and Market Value. These factors can then be put into a logistic regression model, which will in turn predict what companies will rank in the top 10. This results of this study can be used to predict future top 10 contenders of the Forbes Global 2000 and give insight into the factors that affect a corporation's chances of being in the top 10.

## Introduction:

Work in Progress

## Methods:

Work in Progress

### Data Collection

The first step was determining an efficient way to collect Forbes Global 2000 data from previous years. Instead of manually going through the datasets uploaded by Forbes onto their website, I instead chose to collect data through user-uploaded databases on Kaggle. Due to the nature of Kaggle databases being user-uploaded, it was not possible to get the data from every year from 2013 to 2023. Instead, the only datasets available were from years 2013, 2015, 2017, 2020, 2021, and 2023. I downloaded these datasets off of Kaggle in the csv file format, which made each of these datasets compatible with the Pandas Python library. Use of the Pandas library allowed for specific information on corporations, which made the processes of data organization and anlaysis easier later down the line. Through preliminary data analysis, I discovered that a number of corporations consistently placed in the top 10 of the Global 2000 over a period of multiple years. This was useful information since the end goal of this project was to predict what corporations would be most likely to place in the top 10 of the Global 2000 in 2024. As a result, I could utilize 2023 Global 2000 data to make predictions for 2024. This meant that, for the purposes of training a model, I could only use data up until 2021.

### Data Cleaning

Next, I took the data for each Forbes Global 2000 from 2013, 2015, 2017, 2020, and 2021 and merged them all into one master dataframe. I did not include the year 2023 in this master dataframe as the intention was to use 2023 for the purposes of prediction, not training. After the creation of the master dataframe, data cleaning had to be done. Many rows from the master dataframe either contained string values, or contained the special number NaN. When training a regression model, the model is unable to process NaN or string values, meaning that these values had to be cleaned up. I used the to_numeric function of Pandas in order to change all of the strings to floats, and used the dropna() function of Pandas in order to drop every row containing a NaN value. I then had to remove the "Company", "Country", "publish_year", "Rank", "Sector", "Industry", and "Continent" columns from the dataframe, as these were all columns containing data otherwise irrelevant. I then added a column "top_10" that was a categorical value (0 or 1) that stated whether a company had ever been in the top 10 or not.

### Data Analysis and Predictions

My next step was actually creating a logistic regression model in Python. I used SciKitLearn, a software library written for Python for machine learning, to read the data from the master dataframe and create a logistic regression model based off of it. First, I created a heatmap using the Seaborn Python library to determine the most important factors when deciding whta corporations would be in the top 10. Based on the heatmap, overall Sales, Assets, and Market Value were the three most important attributees. The correlation coefficient between Sales and being top 10 was 0.13, the correlation coefficient between Assets and being top 10 was 0.43, and the correlation coefficient between Market Value was 0.26. Next, before creating the logistic regression model, I made sure to drop the "Profits" column as there was a negative correlation between Profits and being in the top 10. Afterwards, I began to create the machine learning model, setting the testing size to 0.2 (or 20% of the entire master dataset). According to the .score() function of SciKitLearn, the model created had a 99.6% accuracy rate. The final step was using the model to then predict possible top 10 candidates for the Forbes Global 2000 of 2024. I loaded in the previous dataset of 2023's data and plugged it into the logistic regression model. The model predicted the following corporations in decreasing probability of being in the top 10 Forbes Global 2000 of 2024: ICBC, Apple, Agricultural Bank of China, China Construction Bank, Saudi Arabian Oil Company (Saudi Aramco), Microsoft, Bank of China, JPMorgan Chase, Fannie Mae, Bank of America, Alphabet, HSBC Holdings, Freddie Mac, Amazon, and Mitsubishi UFJ Financial.

## Results:

Work in Progress

## Conclusion:

Work in Progress

## Images:

![ForbesGlobal2000PredictiveModelHeatMap](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/a788eb13-1b2d-4683-b59e-7c06d90a7f24)

![ForbesGlobal2000PredictiveModelConfusionMatrix](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/e85d7695-429d-4f18-9d2e-8c524b860ad0)

![ForbesGlobal2000PredictiveModelMasterDataFrame](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/dd842c38-8cfe-4036-886b-e68fc226fe22)
