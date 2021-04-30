# Overview
The Consumer Financial Protection Bureau (CFPB) is a federal U.S. agency that acts as a mediator when disputes arise between financial institutions and consumers. Via a web form, consumers can send the agency a narrative of their dispute.

This project developed Natural Language Processing (NLP) machine learning models to process the narratives' text and categorize the complaints into one of five classes.    

**Business case**: An NLP model would make the classification of complaints and their routing to the appropriate teams more efficient than manually tagged complaints.  

# About the Data

A data file was downloaded directly from the [CFPB website](https://www.consumerfinance.gov/data-research/consumer-complaints/
) for training and testing the model. It included one year's worth of data (March 2020 to March 2021). Later in the project, I used an API to download up-to-the-minute data to verify the model's performance. 

Each submission was tagged with one of nine financial product classes. Because of similarities between certain classes as well some class imbalances, I consolidated them into five classes:

1. credit reporting
2. debt collection
3. mortgages and loans (includes car loans, payday loans, student loans, etc.)
4. credit cards
5. retail banking (includes checking/savings accounts, as well as money transfers, Venmo, etc.)

After data cleaning, the dataset consisted of around 162,400 consumer submissions containing narratives. The dataset was still imbalanced, with 56% in the credit reporting class, and the remainder roughly equally distributed (between 8% and 14%) among the remaining classes.

![class_imbalances](https://github.com/halpert3/flatiron-capstone-project/blob/main/notebooks/exported_images/class_imbalances.png) 

# Process

## Exploratory Data Analysis

- Instantiated various vectorizing techniques—TF-IDF and CountVectorizer with both unigrams and bigrams—to explore word frequency.
- Lemmatized the corpora, since words such as "payment" and "payments" appeared as separate entries
- Using the lemmatized words, made pie charts of the top word per class how it compared to other classes

For example, "card" was the top word in the credit-card class, appearing in 67.6% of associated narratives. That word, however, appeared only in 1.7% of mortgages-and-loan narratives. I assumed word-frequency imbalances like this one would be useful for the model to categorize the narratives. 

![Credit Card Pie](https://github.com/halpert3/flatiron-capstone-project/blob/main/notebooks/exported_images/credit%20card%20pie.png)

## Data Preparation

Created a dataframe with two columns: a product class and narrative per line. The narrative string consisted of space-separated lemmatized words with stopwords (such as "the" and "if") removed. 

## Baseline Modeling

Prep:

- Replaced class names with numbers
- Performed train-test split
- Vectorized data with TF-IDF
- Created function to score baseline models

Ran six baseline models with mostly default parameters:

- Multinomial NB 
- Random Forest
- Decision Tree
- KNN
- Gradient Boosting
- XGBoost 

When scoring the models, I relied mostly on the "macro recall" scores, since the recall metric accounts for false negatives and doesn't favor an imbalanced class. I also took into account the difference between the recall scores of training and test sets (as closer they were, the better the model was at not overfitting). 

The best baseline models were Multinomial NB, Decision Tree, and Gradient Boosting.

## Model Refinement

I use Grid Search and implemented different ways of vectorizing—TF-IDF and CountVectorizer with different maximum features—and various parameters for the three modeling techniques.

**Multinomial NB** did the best, with a recall of 86%, much improved from the baseline of 58%. I experimented with using SMOTE to correct for class imbalances, but the model actually did better without it.

With **Gradient Boosting**, I found parameters that yielded a recall score not far behind Multinomial NB at 83%. Both models had only a small problem with overfitting (a 3% discrepancy between training and test sets).   

### Downloaded New Data with an API

As mentioned, the CFPB also allows the downloading of data via an API. 

I developed functions to immediately process up to 1,000 lines of downloaded data into a useable form (consolidating classes, lemmatizing words, removing stopwords, etc.) and then to vectorize and run the data through the Multinomial NB model. 

I achieved classification results similar to the original testing data for the model.  

## Post-Modeling EDA

Even though I considered Multinomial NB to be my "winning" model, I used the not-far-behind Gradient Boosting model to check for feature importances, since it's an intrinsic capability of Gradient Boosting.

This chart shows the ten most important features (words) for classifying texts and how prevalent they were in each class.

![Importance by Class](https://github.com/halpert3/flatiron-capstone-project/blob/main/notebooks/exported_images/Importance%20by%20Class.png)

Clearly, some features were far more prevalent in particular classes such as "card" in credit cards, "Experian" in credit reporting , and "bank" in retail banking. Other words such as "account" and "credit" had more mixed frequencies across the classes, and I assume the model used these features in conjunction with other features to classify narratives. 

# Next Steps

**Improve Business Case**

Since consumers classified their own complaints, ask CFPB employees to double-check narratives’ classes, particularly those that the model misclassified. I'd seek to understand how the CFPB internally routes and processes consumer complaints and develop further modeling capabilities for **sub-product**, **issue**, and **sub-issue**. 

**Refine Models**

- Use more than one year’s worth of data and further refine parameters

- Create Latent Dirichlet Allocation (LDA) model to develop new classification categories and learn if they might be useful to CFPB

