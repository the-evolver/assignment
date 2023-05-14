For the task of predicting wine variety, I have developed a machine learning model using the Random Forest algorithm. I have preprocessed the data by removing missing values and converting categorical variables into numeric format using one-hot encoding. The model achieved an accuracy score of 75.3% on the test data.

To serve predictions, I have built an API using Flask, which takes in a JSON input containing the wine characteristics and returns the predicted wine variety. The input schema includes the following fields: country, province, region_1, winery, price, and points.

Assumptions made:

The dataset is representative of the overall wine market, and the insights and predictions derived from it are generalizable to other wine products.
The review ratings are a reliable indicator of wine quality and are not biased by personal preferences or external factors.

To use the API, send a POST request to the URL http://localhost:5000/predict with a JSON input containing the wine characteristics:

The API will return a JSON response containing the predicted wine variety

Note that the input data must include all the categorical variables used to train the model in one-hot encoded format. If any of these variables are missing, the prediction will fail.
