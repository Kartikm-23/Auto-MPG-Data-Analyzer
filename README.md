# Auto-MPG-Data-Analyzer

This project focuses on predicting the fuel efficiency (Miles Per Gallon - MPG) of automobiles using key performance indicators such as horsepower, weight, and acceleration. 
The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, and training a Multiple Linear Regression model using Python and the Scikit-learn library.
The project uses the Auto MPG Dataset, which contains 392 records of various cars manufactured between 1970 and 1982.
The analysis follows a standard Machine Learning workflow:
Data Loading and Cleaning: The dataset was loaded and inspected for missing values, Missing values, initially represented by '?', were converted to np.nan.
Rows containing missing values were dropped, and the horsepower column was correctly cast to a float data type.
Exploratory Data Analysis (EDA):A Pair Plot was generated to visually inspect the relationships between the target variable (mpg) and the selected features (horsepower, weight, acceleration). This step confirmed the strong inverse linear relationship between weight/horsepower and mpg.
Model Training:The data was split into 80% Training and 20% Testing sets.A LinearRegression model was initialized and trained on the training data.
Evaluation:The model's performance was assessed using standard regression metrics on the test set.
