# APS360---Predictive-Modelling-for-Sports-Analytics

# Basketball 3-Pointers Prediction

This repository contains code for predicting the number of 3-pointers in future basketball games using deep learning techniques. The project is motivated by the evolving landscape of sports analytics and the increasing importance of predictive modeling in basketball decision-making.

## Project Overview

Traditional methods in sports analytics have limitations in capturing the dynamics of gameplay, especially in predicting specific outcomes like the number of 3-pointers scored. Therefore, the primary goal of this project is to leverage deep learning techniques to provide accurate predictions that inform strategic decisions, optimize player performance, and deepen fan engagement.

## Key Features

- **Data Scraping**: The `data_scraping` script retrieves player career statistics from the NBA API for the period between 2018 and 2023. It collects relevant data for analysis and prediction.

- **Feature Selection**: The `feature_selection` script performs feature selection to identify the most relevant variables for predicting the number of 3-pointers. It utilizes correlation analysis and Recursive Feature Elimination (RFE) to select the top features.

- **Baseline Model**: The `baseline_model` script builds a baseline model using Random Forest Regression. It trains the model on the selected features and evaluates its performance on a validation set.


## Requirements

- Python 3.x
- Access to the NBA API (pip install npa_api)

## Results

The project aims to revolutionize basketball analysis by providing accurate predictions of 3-pointers in future games. Deep learning techniques uncover hidden patterns within basketball statistics, enriching our understanding of player performance and game dynamics. Through accurate predictions, stakeholders can make informed decisions and deepen engagement with the sport.

For more detailed insights and results, refer to the individual scripts in the repository.
