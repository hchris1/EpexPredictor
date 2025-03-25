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
- This alone is not enough, because electricity prices are not linear.
E.g. low wind&solar leads to gas power plants being turned on, and due to merit order pricing, electricity prices explode.
- Therefore, we then multiply each parameter with its weight (LinReg factor) to get a "normalized" data set.
- In the next step, we use a KNN (k=3) approach to find hours in the past with similar properties and use that to determine the final price.

## Model performance
No scientific evaluation. I just looked at the result and they mostly seem to be within 1-10%.
Some observations:
- At night, predictions are usually within 1-2ct/kWh
- Morning/Evening peaks are usually within 3-4ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challange. E.g.
the model might predict 75ct, while in reality it's only 60ct or vice versa
- High PV noons are usually correctly detected. Sometimes it will return 3ct instead of -1ct, but the ballpark is usually correct.

This graph compares the actual prices to the ones returned by the model for a random two week time period in january 2025.

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
    x-axis [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335]
    line [5.8,4.5,4.1,5.0,7.3,8.5,10.9,11.6,9.6,8.9,7.8,7.6,7.5,7.5,7.5,7.5,8.3,8.9,9.2,8.5,8.1,9.2,8.6,5.9,6.5,6.4,6.7,7.7,7.7,7.9,7.7,8.9,8.7,8.6,8.8,8.3,8.2,8.9,11.1,13.3,14.6,15.1,14.8,13.8,13.5,13.2,12.2,12.6,11.6,11.1,10.7,10.5,10.8,11.1,11.8,12.4,12.2,10.6,10.3,9.7,9.0,9.3,11.2,12.6,13.3,11.9,9.0,7.9,6.8,4.9,2.4,4.0,3.5,2.6,1.4,1.0,2.2,3.4,5.7,7.2,6.3,4.6,4.0,3.6,3.8,5.4,8.7,10.2,11.9,11.9,10.0,9.4,8.5,8.2,6.9,7.7,7.2,5.7,6.9,7.7,7.8,9.8,12.5,14.8,13.5,12.1,11.6,11.5,11.7,12.1,14.0,14.7,15.0,13.5,12.9,9.5,8.7,8.0,6.4,8.8,8.3,7.8,7.6,7.9,8.7,9.3,14.1,14.3,13.0,10.5,8.9,8.5,8.6,8.8,10.3,13.3,15.0,15.0,14.6,13.2,12.0,11.2,9.9,9.2,9.2,9.3,9.3,9.2,10.9,14.0,15.3,16.0,15.2,13.5,12.3,11.6,11.3,12.9,15.3,16.1,18.0,17.7,16.1,15.0,13.4,12.6,12.0,12.0,11.7,11.4,11.3,11.6,12.1,13.2,15.8,16.4,15.5,13.0,12.6,12.0,12.1,12.0,15.0,16.8,18.5,18.4,17.9,16.3,15.3,14.4,13.1,13.7,12.9,12.4,11.9,11.9,12.1,12.7,14.5,15.3,14.2,12.8,12.0,11.0,10.3,11.4,12.9,15.4,16.8,16.8,16.3,15.7,14.5,14.2,13.2,13.3,12.7,12.3,12.1,12.2,12.3,12.8,13.6,13.7,13.8,13.0,12.9,12.2,11.2,11.6,12.8,15.1,16.4,17.2,17.4,16.6,15.6,14.8,13.3,13.0,12.7,12.5,12.0,12.3,12.8,15.9,20.8,24.3,17.8,15.0,12.6,11.7,11.5,12.4,14.6,17.1,21.6,20.1,20.1,16.8,15.3,14.4,13.6,13.2,12.6,12.3,12.4,12.3,12.8,15.5,19.2,21.2,16.3,13.7,12.7,11.7,11.3,12.0,12.7,16.5,16.9,17.5,16.2,14.5,13.2,12.9,12.1,11.3,11.0,10.9,10.8,11.5,12.2,14.0,16.5,18.1,15.9,13.2,12.0,11.5,11.9,12.9,14.4,16.1,17.7,19.0,19.0,17.0,15.5,14.4,13.3,13.4,13.0,12.6,12.5,12.5,13.5,15.6,18.0,21.9,17.9,16.0,15.3,14.8,14.5,13.8,15.1,16.5,17.2,17.0,16.3,14.1,13.1,13.4,12.2,12.6]
    line [6.9,7.7,7.4,6.9,7.9,9.2,13.2,13.5,12.6,10.9,8.7,6.6,6.6,7.2,8.8,10.3,11.7,10.3,9.1,8.6,7.9,8.1,7.2,6.2,6.2,4.0,6.0,6.5,6.8,7.0,9.7,11.8,11.8,10.4,9.4,9.8,9.4,9.9,11.5,12.5,13.3,12.9,14.4,12.1,12.4,12.1,11.1,11.8,11.3,10.5,10.2,10.2,9.7,9.9,11.7,12.8,14.5,12.7,12.2,11.5,10.5,12.0,12.8,14.5,13.6,14.1,13.1,11.8,9.9,7.5,5.7,6.9,5.9,5.0,4.3,6.4,7.1,8.0,7.4,9.0,11.5,8.0,6.9,6.6,6.6,7.2,8.8,10.3,12.7,10.3,9.1,8.6,7.9,8.2,7.3,7.5,6.3,6.0,6.2,6.9,7.9,9.2,11.0,14.2,12.9,13.1,11.9,9.8,11.3,13.0,14.8,13.7,14.4,13.2,13.3,11.0,9.3,8.1,7.2,6.6,5.9,6.0,6.2,6.9,7.9,9.2,10.7,14.1,12.1,8.0,6.9,6.6,6.6,7.2,8.8,10.3,13.4,15.1,13.3,11.4,10.0,11.2,11.6,9.6,9.6,8.3,7.0,8.0,9.0,12.4,15.4,17.2,15.2,14.6,13.2,11.3,11.4,12.0,14.8,16.3,16.8,17.3,16.3,15.1,11.2,11.1,10.3,9.5,9.6,8.3,11.1,9.1,12.0,14.3,15.4,17.5,16.4,14.3,12.8,12.2,10.5,11.0,15.3,16.7,17.2,17.2,16.7,15.1,13.2,13.5,12.8,12.0,11.8,11.2,10.7,11.6,10.8,11.4,14.6,14.8,16.5,14.5,13.2,11.0,10.4,11.4,13.4,16.0,15.9,16.7,16.0,14.6,13.4,13.3,12.5,13.0,11.4,11.1,10.6,9.8,10.7,11.0,15.2,17.2,14.7,13.2,12.0,11.3,10.8,11.7,13.1,15.5,22.6,22.2,15.8,15.2,15.2,15.6,12.7,12.1,11.8,11.6,11.4,11.5,12.5,14.9,19.5,22.6,18.7,13.8,12.0,11.3,11.1,11.7,14.9,17.9,24.6,25.5,21.4,17.9,15.5,14.6,13.7,13.6,13.1,12.1,11.4,12.9,12.0,15.8,19.3,22.6,18.5,13.7,12.0,11.3,11.1,11.7,13.0,16.6,17.7,18.7,17.0,16.1,13.5,11.8,10.6,11.9,10.1,10.1,10.8,10.6,12.2,14.0,19.0,20.2,16.1,14.0,12.9,11.5,11.3,12.2,13.4,16.5,17.2,17.5,17.4,16.4,13.8,14.6,12.2,12.6,11.3,11.0,11.0,11.4,12.3,14.8,16.4,20.1,16.1,14.2,12.9,11.9,15.3,15.1,15.3,16.9,16.5,16.3,15.1,15.5,12.7,13.0,11.7,11.2]
    
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