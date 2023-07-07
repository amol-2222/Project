# Project

Problem Statement: Telecom Customer Churn Prediction

In this project, our goal is to develop a predictive model that can accurately identify customers who are at risk of churning in a telecom company. Customer churn refers to the phenomenon where customers terminate their subscription or switch to a competitor's service. Churn prediction is crucial for telecom companies as it allows them to take proactive measures to retain valuable customers and minimize revenue loss.

The problem at hand involves using historical customer data, including various features such as demographics, usage patterns, service plans, and customer interactions, to build a machine learning model that can effectively predict customer churn. The model will be trained on a labeled dataset, where each instance represents a customer and is labeled as either churned or non-churned.

The specific objectives of this project are as follows:

Data Preparation: Collect and preprocess the telecom customer data, including cleaning, handling missing values, and performing feature engineering to extract relevant information.

Exploratory Data Analysis (EDA): Conduct a comprehensive analysis of the dataset to gain insights into the customer churn patterns, identify key features, and understand the relationships between different variables.

Feature Selection: Select the most relevant features that have a significant impact on customer churn prediction, considering both statistical significance and domain expertise.

Model Development: Develop and train machine learning models on the prepared dataset, utilizing appropriate algorithms such as logistic regression, decision trees, random forests, support vector machines,k nearest neighbors or gradient boosting algorithms.

Model Evaluation: Assess the performance of the trained models using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC). Compare the performance of different models to identify the most effective one.

Hyperparameter Tuning: Fine-tune the selected model by optimizing hyperparameters using techniques such as grid search, random search, or Bayesian optimization to improve model performance.

Model Deployment: Once the final model is selected, prepare it for deployment by saving the trained model, necessary preprocessing steps, and feature transformations for future use. Create an API or application interface that allows for real-time or batch predictions on new customer data.

Recommendations and Insights: Based on the trained model and analysis, provide actionable recommendations to the telecom company to reduce churn rates. Identify key factors driving customer churn and suggest strategies to improve customer retention.

The successful completion of this project will enable the telecom company to proactively identify customers who are likely to churn and take appropriate measures to retain them, ultimately reducing customer attrition and improving customer satisfaction and profitability.

Data information¶

customerID: Customer ID

MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service)

InternetService: Customer’s internet service provider (DSL, Fiber optic, No)

OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service)

OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service)

DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service)

TechSupport: bold text Whether the customer has tech support or not (Yes, No, No internet service)

StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service)

StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service)

gender: gender (female, male)

SeniorCitizen: Whether the customer is a senior citizen or not (1, 0)

PartnerWhether: the customer has a partner or not (Yes, No)

Dependents: Whether the customer has dependents or not (Yes, No)

tenure: Number of months the customer has stayed with the company

PhoneService: Whether the customer has a phone service or not (Yes, No)

Contract: The contract term of the customer (Month-to-month, One year, Two year)

PaperlessBilling: Whether the customer has paperless billing or not (Yes, No)

PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))

MonthlyCharges: The amount charged to the customer monthly

TotalCharges: The total amount charged to the customer

Churn: Whether the customer churned or not (Yes or No)

#model building
-- machine learning model to predict customer churn in a telecom company, the recall value is an important metric to consider.
Recall, also known as sensitivity or true positive rate,measures the model's ability to correctly identify positive instances (in this case, customers who are likely to churn) from the total actual positive instances.

Here's why recall is important in this scenario:

1.Identifying potential churners: Telecom companies are concerned about identifying customers who are likely to churn so that they can take proactive measures
to retain them. Recall helps in capturing as many actual churn cases as possible. A high recall value implies that the model is successful in identifying 
a large portion of customers who are likely to churn, reducing the chances of false negatives (missed churn cases).

2.Cost of false negatives: False negatives occur when the model predicts a customer will not churn(-ve) when they actually do churn (positive).
This can be costly for telecom companies as they may lose valuable customers without taking any preventive actions. By optimizing for higher recall, 
the model can minimize false negatives and ensure that as few actual churners as possible are missed.

3.Trade-off with precision: Recall and precision are often inversely related. Precision measures the proportion of true positive predictions among all
positive predictions made by the model. It focuses on the accuracy of positive predictions rather than capturing all actual positives. In the context
of customer churn, precision represents the percentage of correctly identified churners among all predicted churners. While precision is important,
a balance needs to be struck with recall. Maximizing recall may result in more false positives (customers predicted to churn but who actually don't),
but it ensures that a significant number of true churners are not overlooked.

Overall, recall is important in customer churn prediction models because it helps capture as many true churners as possible,
minimizing the chances of missing valuable customers who are likely to churn. By achieving a high recall value, telecom companies can take proactive
steps to retain those customers and mitigate revenue loss.


#Balanced Data
When it comes to customer churn prediction in a telecom company, the balance of the target column (churn or non-churn) is important, particularly if there
is a bias in the data. Here's why:

1.Model performance: Imbalanced data can lead to biased model performance. If the majority class (non-churn) heavily dominates the dataset, a model trained 
on such data may become excessively biased towards predicting the majority class. As a result, the model's ability to identify the minority class (churn) 
accurately may be compromised. Balancing the target column helps ensure that the model is exposed to an adequate number of churn instances, enabling it to
learn patterns and make better predictions for both classes.

2.Accurate churn prediction: The primary goal of a churn prediction model is to identify customers who are likely to churn. In an imbalanced dataset, a model
that is biased towards the majority class may incorrectly classify churners as non-churners, leading to missed opportunities for intervention. By balancing 
the target column, you provide equal representation to churners, allowing the model to learn from both classes and improve its ability to accurately predict
churn.

3.Decision-making and resource allocation: Telecom companie rely on churn prediction models to make informed decisions and allocate resources effectively.
If the model is biased towards the majority class, it may lead to misallocation of resources. For instance, the company might end up targeting non-churners
with retention efforts, while actual churners are left unnoticed. Balancing the target column helps in achieving a more accurate prediction of churn,
enabling the company to allocate resources efficiently and focus on customers who are genuinely at risk of churning.

4.Evaluation metrics: Balanced data is crucial for reliable evaluation of model performance. Common evaluation metrics like accuracy can be misleading in 
the presence of class imbalance. For example, if the non-churn class constitutes 90% of the data, a model that simply predicts non-churn for all instances 
will achieve 90% accuracy. However, this model fails to capture any churn cases. By balancing the target column, you can evaluate the model using more 
appropriate metrics like precision, recall, F1-score, or area under the ROC curve (AUC-ROC) that account for class imbalance and provide a more comprehensive
assessment.

In summary, balancing the target column is essential in classification problems, especially when predicting customer churn in a telecom company.
It helps address bias in the data, improves model performance, facilitates accurate churn prediction, aids decision-making and resource allocation,
and enables reliable evaluation of the model's effectiveness.



#Final model
--1.In logistic regression model get 90% recall value which is best value till now with 83 % accuracy and without overfitting

--2.Here this result is with only 12 important feature which is really affecting churn of customer

In summary, for telecom customer churn scenarios, where interpretability, resource efficiency, and a balanced trade-off between accuracy and 
simplicity are crucial, the logistic regression model with 12 independent features (90% arecall) would be a suitable choice for deployment.
