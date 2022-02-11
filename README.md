# Medical-Cost-Personal-Datasets

# Dataset [link](https://www.kaggle.com/mirichoi0218/insurance)

# Introduction
```Health insurance or medical insurance is a type of insurance that covers the whole or a part of the risk of a person incurring medical expenses. By estimating the overall risk of health risk and health system expenses over the risk pool, an insurer can develop a routine finance structure, such as a monthly premium or payroll tax, to provide the money to pay for the health care benefits specified in the insurance agreement. ```— [Wikipedia](https://en.wikipedia.org/wiki/Health_insurance)

In this blog, the cost borne by insurance companies for their customers who took insurance is predicted. Different ML Regression algorithms like Linear Regression, kNN Regression, SVM Regression, Random Forest Regression, XGBoost Regression were used for prediction.


# [Link](https://binginagesh.medium.com/estimating-medical-insurance-charges-4373979cd421) to medium blog.


# Results
| Model        | test MAPE           |
| ------------- |:-------------:|
| Linear Regression with L1 and L2 regularization    | 0.9429 |
| kNN Regressor      | 0.3386      |
| **SVM Regressor** | **0.2112**     |
|  Random Forest Regressor    | 0.3064 |
| XGBoost Regressor      | 0.2663      |


# Conclusion
In this blog, different experiments are done to predict the medical charges. The SVM Regression model worked well with MAPE of **0.2112**. In insurance industry, it is very important to know which features increase our costs. Even though SVM gave least MAPE, I would like to go with **XGBoost** as we can get **feature importance** with XGBoost model.
