# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
This repository includes
- The self-training prediction model itself
- A simple Flask app to get a REST API up
- A Dockerfile to easily have it running wherever with gunicorn

Supported Countries:
- Germany (default)
- Austria


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample a few sample points all over the country and fetch hourly [Weather data from Open-Meteo.com](https://open-meteo.com/) for those for the past n days (default n=90).
This serves as the main data source.



Parameters:

- Wind for each location
- Temperature for each location
- Expected solar irradiance for each location
- Hour of day
- Day of the week from monday to saturday
- Whether it is a Holiday/Sunday

Output:
- Electricity price

## How it works
- First, we create a simple multi-linear-regression to get an idea how important each of
the training parameters is.
- This alone is not enough, because electricity prices are not linear.
E.g. low wind&solar leads to gas power plants being turned on, and due to merit order pricing, electricity prices explode.
- Therefore, we then multiply each parameter with its weight (LinReg factor) to get a "normalized" data set.
- In the next step, we use a KNN (k=3) approach to find hours in the past with similar properties and use that to determine the final price.

## Model performance
For performance testing, we used historical weather data with a 90%/10% split for a training/testing data set. See `predictor/model/performance_testing.py`.

Results:\
DE: Mean squared error ~4.53 ct/kWh, mean absolute error ~1.35 ct/kWh\
AT: Mean squared error ~6.86 ct/kWh, mean absolute error ~1.78 ct/kWh

Some observations:
- At night, predictions are usually within 1-2ct/kWh
- Morning/Evening peaks are usually within 3-4ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challange. E.g.
the model might predict 75ct, while in reality it's only 60ct or vice versa
- High PV noons are usually correctly detected. Sometimes it will return 3ct instead of -1ct, but the ballpark is usually correct.

This graph compares the actual prices to the ones returned by the model for a random two week time period in early 2025.

Note that this was created for a time range in the past with historic weather data, rather than forecasted weather data,
so actual performance might be a bit worse if the weather forecast is not correct.

```mermaid
---
config:
    xyChart:
        width: 1700
        height: 900
        plotReservedSpacePercent: 80
        xAxis:
            showLabel: false
---
xychart-beta
    title "Performance comparison"
    line [13.5,10.1,6.8,5.1,3.1,2.8,5.0,7.0,9.9,14.3,17.7,17.0,14.2,11.9,10.8,9.3,9.2,9.0,9.2,8.9,9.5,11.1,13.9,13.7,10.7,7.2,2.5,0.1,-0.1,-0.0,0.6,5.0,9.6,14.1,17.1,16.7,14.2,12.3,11.6,10.3,10.7,10.2,10.1,10.2,10.7,12.3,15.8,14.8,10.8,8.0,2.7,-0.0,-0.1,-0.0,0.1,6.7,11.0,14.8,24.8,22.0,17.7,13.8,12.4,11.1,11.7,10.3,10.0,10.0,10.3,12.0,14.5,14.1,10.0,7.9,3.8,0.3,0.0,0.1,1.7,7.9,12.1,15.5,28.0,25.4,16.4,13.7,12.5,11.8,12.5,11.6,11.3,10.6,10.9,12.7,15.6,14.8,11.1,7.3,4.3,0.7,0.1,0.0,0.2,1.2,4.6,10.0,12.1,8.3,7.0,4.7,2.7,1.8,2.7,1.3,0.9,0.7,1.0,1.3,1.3,1.3,0.3,0.0,-0.2,-0.6,-1.5,-1.5,-0.7,-0.2,0.1,3.5,6.0,5.1,4.2,3.6,3.7,3.6,1.4,1.9,2.4,2.3,2.5,2.5,2.5,2.4,1.9,2.3,1.4,1.0,1.3,0.5,1.1,4.2,8.4,12.4,14.0,14.3,14.1,13.0,12.6,11.4,11.0,10.7,10.2,9.9,10.2,12.0,14.2,16.4,16.2,13.5,11.1,10.7,10.1,10.1,10.7,11.6,13.1,15.4,21.2,19.3,16.2,13.9,12.6,11.5,10.2,10.2,10.0,10.3,10.2,10.7,13.6,14.2,13.3,10.6,9.2,7.9,6.8,6.9,8.2,9.6,10.9,13.9,16.1,16.6,14.9,12.2,11.5,10.2,10.1,9.8,9.7,9.7,9.6,9.9,12.9,14.2,13.7,11.2,10.0,9.6,9.5,8.4,9.1,9.4,10.1,13.5,16.2,17.0,15.3,12.9,12.0,10.9,10.5,9.9,9.8,9.9,9.7,10.5,13.0,14.0,12.3,9.0,7.8,4.5,2.5,0.7,3.0,7.5,9.4,12.7,15.8,17.1,14.2,12.4,11.3,10.1,9.0,7.9,7.9,7.9,8.0,10.4,12.9,12.2,10.2,7.6,2.6,0.2,-0.0,-0.0,1.4,7.0,10.4,13.2,16.6,16.6,14.8,12.5,11.8,11.0,11.4,10.4,9.7,9.6,9.7,9.3,9.7,9.8,9.2,7.6,6.6,6.5,6.5,0.2,0.4,2.5,6.6,10.0,14.7,15.2,12.1,10.0,8.3,6.8,4.6,1.6,0.5,0.1,0.0,0.0,0.0,-0.0,-0.0,-0.3,-1.1,-1.9,-2.6,-2.6,-1.3,-0.4,-0.0,1.4,6.1,5.9,5.0,6.2,5.6,6.6,6.3,6.4,6.7,6.9,8.8,10.9,14.4,17.4]
    line [14.6,11.3,4.0,1.9,1.0,7.1,4.6,6.5,9.0,12.8,16.8,16.7,14.8,12.4,12.0,10.6,10.7,10.3,10.0,10.6,10.7,11.7,15.4,15.7,17.0,6.8,5.5,1.7,1.0,0.0,0.8,6.5,9.6,15.2,18.4,16.7,14.6,12.4,11.7,10.6,11.2,11.2,11.0,10.8,11.3,12.6,15.3,18.5,16.5,7.9,4.0,0.2,-0.0,0.0,1.1,7.2,10.8,14.3,22.9,22.0,17.1,14.0,12.9,11.5,12.6,11.7,11.0,10.1,9.9,11.0,14.0,14.7,11.0,7.7,4.4,0.5,0.0,0.0,1.5,7.4,10.6,15.1,21.9,19.1,14.4,13.5,11.9,11.1,11.2,10.7,10.3,9.9,10.1,11.8,14.7,14.9,10.4,7.6,3.6,0.4,0.0,0.0,1.1,6.1,9.7,8.9,13.0,8.6,7.8,6.6,6.3,5.6,4.6,4.2,4.0,4.0,4.3,4.7,4.9,1.9,1.9,0.0,1.2,-0.3,-0.6,1.7,-0.1,3.5,6.7,8.0,9.4,5.8,4.5,5.0,4.5,7.3,6.7,6.6,6.8,6.8,7.0,7.1,7.1,9.8,3.6,2.4,1.6,0.4,0.4,2.8,3.3,4.8,8.7,13.3,14.3,14.2,12.1,13.1,11.3,10.0,10.3,10.1,10.5,10.1,10.2,11.8,14.9,16.3,15.3,13.2,10.6,9.9,10.8,11.2,10.5,11.1,12.5,15.3,22.2,19.9,15.6,13.3,11.6,10.1,9.9,9.8,9.5,9.6,9.9,10.9,13.9,14.2,11.8,8.8,10.4,8.5,3.0,2.8,8.8,9.0,10.8,11.8,15.9,15.3,15.6,11.8,10.7,10.5,10.0,10.2,10.1,10.1,10.2,10.8,13.6,15.5,15.3,13.0,11.7,11.3,9.8,8.6,10.0,10.7,12.2,14.5,20.3,19.0,15.9,14.0,12.9,12.2,12.2,11.6,11.4,11.3,11.3,11.3,13.5,15.6,11.1,9.3,7.3,8.4,3.5,0.5,2.7,7.5,11.0,14.4,21.8,19.1,15.4,13.5,11.9,11.0,9.5,8.5,8.6,8.4,9.2,10.6,12.5,14.1,10.4,7.6,3.6,0.4,0.0,0.0,1.5,7.4,10.6,14.6,20.7,15.8,14.7,11.4,11.3,10.1,12.2,10.3,10.2,10.2,8.5,10.9,10.1,10.8,9.3,8.9,7.1,7.7,8.4,3.6,6.5,7.2,8.0,10.8,12.8,13.9,11.9,11.0,11.3,9.1,7.8,6.5,6.2,6.2,3.7,3.3,3.2,-0.0,-0.0,-0.5,2.3,4.7,5.6,4.7,6.0,6.9,6.2,8.2,6.7,3.6,5.9,6.2,4.9,6.1,5.7,4.9,5.9,5.2,6.2,8.1,12.4,15.9]
```


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.



### Configuration:
```yaml
# Make sure you change the parameters fixedPrice and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    scan_interval: 500
    unit_of_measurement: ct/kWh
    value_template: "{{ value_json.prices[0].total }}"
    json_attributes:
      - prices

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices?country=DE&fixedPrice=13.15&taxPercent=19&#evaluation=true"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  scan_interval: 3600
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: ct/kWh
  #  value_template: "{{ value_json.prices[0].total }}"
  #  json_attributes:
  #    - prices
```

### Display, e.g. via Plotly Graph Card:
```yaml
type: custom:plotly-graph
time_offset: 26h
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.epex_price_prediction
    name: EPEX Price History
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
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
      - 0
      - 1
hours_to_show: 30
refresh_interval: 10
```
