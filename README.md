# Computational Insights on Future Corporate Economic Performance

By: Eshan Prakash

## Abstract:

The Forbes Global 2000 is an annual ranking of the top 2000 public companies in the world, published by Forbes magazine. This paper attempts to utilize machine learning and data science in order to determine the most important factors when ranking these companies and to ultimately predict the top 10 Forbes Global 2000 companies in 2024. Multiple factors that I have found to contribute to top 10 placement include Sales, Assets, and Market Value. These factors can then be put into a logistic regression model, which will in turn predict what companies will rank in the top 10. This results of this study can be used to predict future top 10 contenders of the Forbes Global 2000 and give insight into the factors that affect a corporation's chances of being in the top 10.

## Introduction:

Logistic regression is a predictive analysis that cana be used to describe data and explain the relationship between a single binary variable and one or more normal, ordinal, ratio-level, or interval variables. Regarding the placement corporations in the top 10 of the Forbes Global 2000, the use of logistic regression may be useful.

This paper focuses on the Forbes Global 2000, an annual ranking of the top 2000 corporations in the entire world that has been established in 2003. The Global 2000 ranks the largest companies in the world based on Sales, Profits, Assets, and Market Value. Four different lists are created, one for each main parameter, and then companies are put into those lists based on how high their Sales, Profits, Assets, and Market Values are. Each company then recieves a separate score for each metric based on where the company places within each metric's individual list. The scores of each company are then added up with equal weightage to compile a composite score for each company based on all four metrics. The companies are then sorted by descending order by their composite scores and then applied to the Forbes Global 2000 ranking.

By training a logistic model on the same four parameters used for the Forbes Global 2000, it is possible to make highly accurate predictions of future company placements based on those companies' projected Sales, Profits, Assets, and Market Value. Due to the mathematical and unbiased nature of logistic regression, the model I create should be fairly accurate.

This paper aims to analyze a practical use for logistic regression and machine learning. It focuses on creating a logistic regression model, which will then be applied to determine the future top 10 candidates of the Forbes Global 2000 in 2024 will be. It will do this by looking at multiple factors that have a clear correlation with being in the top 10 of previous Forbes Global 2000 rankings.

## Methods:

### Data Collection

The first step was determining an efficient way to collect Forbes Global 2000 data from previous years. Instead of manually going through the datasets uploaded by Forbes onto their website, I instead chose to collect data through user-uploaded databases on Kaggle. The databases I took came from users Ashwini Swain, Raphael Fontes, Shivam Bansal, and Viola Kwong. Due to the nature of Kaggle databases being user-uploaded, it was not possible to get the data from every year from 2013 to 2023. Instead, the only datasets available were from years 2013, 2015, 2017, 2020, 2021, and 2023. I downloaded these datasets off of Kaggle in the csv file format, which made each of these datasets compatible with the Pandas Python library. Use of the Pandas library allowed for specific information on corporations, which made the processes of data organization and anlaysis easier later down the line. Through preliminary data analysis, I discovered that a number of corporations consistently placed in the top 10 of the Global 2000 over a period of multiple years. This was useful information since the end goal of this project was to predict what corporations would be most likely to place in the top 10 of the Global 2000 in 2024. As a result, I could utilize 2023 Global 2000 data to make predictions for 2024. This meant that, for the purposes of training a model, I could only use data up until 2021.

### Data Cleaning

Next, I took the data for each Forbes Global 2000 from 2013, 2015, 2017, 2020, and 2021 and merged them all into one master dataframe. I did not include the year 2023 in this master dataframe as the intention was to use 2023 for the purposes of prediction, not training. After the creation of the master dataframe, data cleaning had to be done. Many rows from the master dataframe either contained string values, or contained the special number NaN. When training a regression model, the model is unable to process NaN or string values, meaning that these values had to be cleaned up. I used the to_numeric function of Pandas in order to change all of the strings to floats, and used the dropna() function of Pandas in order to drop every row containing a NaN value. I then had to remove the "Company", "Country", "publish_year", "Rank", "Sector", "Industry", and "Continent" columns from the dataframe, as these were all columns containing data otherwise irrelevant. I then added a column "top_10" that was a categorical value (0 or 1) that stated whether a company had ever been in the top 10 or not.

### Data Analysis and Predictions

My next step was actually creating a logistic regression model in Python. I used SciKitLearn, a software library written for Python for machine learning, to read the data from the master dataframe and create a logistic regression model based off of it. First, I created a heatmap using the Seaborn Python library to determine the most important factors when deciding whta corporations would be in the top 10. Based on the heatmap, overall Sales, Assets, and Market Value were the three most important attributees. The correlation coefficient between Sales and being top 10 was 0.13, the correlation coefficient between Assets and being top 10 was 0.43, and the correlation coefficient between Market Value was 0.26. Next, before creating the logistic regression model, I made sure to drop the "Profits" column as there was a negative correlation between Profits and being in the top 10. Afterwards, I began to create the machine learning model, setting the testing size to 0.2 (or 20% of the entire master dataset). According to the .score() function of SciKitLearn, the model created had a 99.6% accuracy rate. The final step was using the model to then predict possible top 10 candidates for the Forbes Global 2000 of 2024. I loaded in the previous dataset of 2023's data and plugged it into the logistic regression model. The model predicted the following corporations in decreasing probability of being in the top 10 Forbes Global 2000 of 2024: ICBC, Apple, Agricultural Bank of China, China Construction Bank, Saudi Arabian Oil Company (Saudi Aramco), Microsoft, Bank of China, JPMorgan Chase, Fannie Mae, Bank of America, Alphabet, HSBC Holdings, Freddie Mac, Amazon, and Mitsubishi UFJ Financial.

## Results:

The results were similar to what was expected. I determined that the most correlated factors to determining top 10 placement were Assets, Sales, and Market Value. This mostly matches up with Forbes' own statements on their methodology for determining the Global 2000, with one discrepancy being that Profits actually had a negative correlation with placing in the top 10, contrary to Forbes' claims. My results to the top 10 of 2024 consist of 15 mian corporations. Many of these corporations have previously and consistently appeared on the Forbes Global 2000, adding credibility to the model's results.

The majority of the corporations in my results have had major recent economic successes, furthering the confirmation that the model works. These corporations were chosen based on the previous 3 factors that I found were the greatest factors: Assets, Sales, and Market Value. In terms of the data obtained through logistic regression, the model itself and therefore the predictions of the model returned a high probability of accuracy, 99.6%. While this value is not perfect, it is enough to label the model's predictions as statistically significant. Understanding the potential such logistical models can have when predicting the Forbes Global 2000 top 10 candidates is key to understanding economics because of its application to the real world. With the ability to predict the success of companies in the future, one could gain an advantage in numerous things, most notably investing in these companies. These results provide insight into the opportunities for logistic models which measure the effects that multiple factors can have in the future.

## Conclusion:

Based on the numeric results from the logistic regression model, ICBC, Apple, Agricultural Bank of China, China Construction Bank, Saudi Arabian Oil Company (Saudi Aramco), Microsoft, Bank of China, JPMorgan Chase, Fannie Mae, and Bank of America are the ten most likely corporations to place in the top 10 of the Forbes Global 2000 in 2024. Since, with the exception of Bank of America, the difference in probability of placing in the top 10 between each corporation is so minimal, it is not possible to predict the exact placement that each corporation will have, only that they will be in the top 10. In addition, I determined that the most important factors when determining if a corporation will be in the top 10 of the Forbes Global 2000 are: Assets, Sales, and Market Value.

### Table:

| Company Name  | Top 10 Probability |
| ------------- | ------------- |
| ICBC  | 99.9814%  |
| Apple  | 99.9342%  |
| Agricultural Bank of China | 99.8344% |
| China Construction Bank | 99.6680% |
| Saudi Arabian Oil Company (Saudi Aramco) | 99.6136% |
| Microsoft | 99.1926% |
| Bank of China | 98.2082% |
| JPMorgan Chase | 97.7346% |
| Fannie Mae | 95.5958% |
| Bank of America | 81.3637% |
| Alphabet | 62.3738% |
| HSBC Holdings | 61.4046% |
| Freddie Mac | 59.1694% |
| Amazon | 58.2702% |
| Mitsubishi UFJ Financial | 50.4014% |

### Heatmap:

![ForbesGlobal2000PredictiveModelHeatMap](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/a788eb13-1b2d-4683-b59e-7c06d90a7f24)

### Confusion Matrix:

![ForbesGlobal2000PredictiveModelConfusionMatrix](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/e85d7695-429d-4f18-9d2e-8c524b860ad0)

### Master Dataframe:

![ForbesGlobal2000PredictiveModelMasterDataFrame](https://github.com/EshanPrakash/ForbesGlobal2000PredictiveModel/assets/148392140/dd842c38-8cfe-4036-886b-e68fc226fe22)

## References:

https://www.forbes.com/lists/global2000/?sh=2bd11cb25ac0

https://www.forbes.com/sites/andreamurphy/2023/05/16/the-global-2000s-20th-anniversary-how-weve-crunched-the-numbers-for-the-past-two-decades/

https://www.kaggle.com/datasets/ash316/forbes-top-2000-companies

https://www.kaggle.com/datasets/unanimad/forbes-2020-global-2000-largest-public-companies

https://www.kaggle.com/datasets/shivamb/fortune-global-2000-companies-till-2021

https://www.kaggle.com/datasets/kwongmeiki/forbes-the-global-2000-rankings-2023

