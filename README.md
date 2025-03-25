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


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Configuration:
```
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices?fixedPrice=13.70084&taxPercent=19"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    unit_of_measurement: ct/kWh
    value_template: "{{ value_json.prices[0].total }}"
    json_attributes:
      - prices
```

Display, e.g. via Plotly Graph Card:
```
type: custom:plotly-graph
time_offset: 26h
entities:
  - entity: sensor.epex_price_prediction
    attribute: dummy
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({meta}) => ({
              xs: meta.prices.map(p => new Date(p.startsAt)),
              ys: meta.prices.map(p => p.total)
          })
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y":
      - -1000
      - 1000
hours_to_show: 30
refresh_interval: 10
```