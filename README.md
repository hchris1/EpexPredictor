# EPEX day-ahead price prediction for Germany

This is a simple statistical model to predict EPEX day-ahead prices for Germany based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
This repository includes
- The self-training prediction model itself
- A simple Flask app to get a REST API up
- A Dockerfile to easily have it running wherever with gunicorn


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample a few sample points all over Germany and fetch hourly weather data for those for the past n days (default n=60).
This serves as the main data source.

Parameters:

- Wind sum
- Temperature average
- Expected solar output (computed via OpenMeteoSolarForecast)
- Hour of day
- Whether it is a Holiday/Sunday
- Whether it is a Saturday

Output:
- Electricity price

## How it works
- First, we create a simple multi-linear-regression to get an idea how important each of
the training parameters is.
- We then multiply each parameter with its weight.
- This is not enough, because electricity prices are not linear.
I.e. low wind&solar leads to gas power plants being turned on, and due to merit order
pricing, electricity prices explode.
- Therefore, in the next step, we use KNN (k=3) approach to find hours in the past with similar properties and use that to determine the final price

## Model performance
No scientific evaluation. I just looked at the result and they mostly seem to be within 1-10%.
