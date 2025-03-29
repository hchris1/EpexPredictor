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
We sample a few sample points all over Germany and fetch hourly [Weather data from Open-Meteo.com](https://open-meteo.com/) for those for the past n days (default n=90).
This serves as the main data source.



Parameters:

- Wind for each location
- Temperature for each location
- Expected solar output for an arbitrary fake solar installation (computed via OpenMeteoSolarForecast)
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
    x-axis [600,601,602,603,604,605,606,607,608,609,610,611,612,613,614,615,616,617,618,619,620,621,622,623,624,625,626,627,628,629,630,631,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,656,657,658,659,660,661,662,663,664,665,666,667,668,669,670,671,672,673,674,675,676,677,678,679,680,681,682,683,684,685,686,687,688,689,690,691,692,693,694,695,696,697,698,699,700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,738,739,740,741,742,743,744,745,746,747,748,749,750,751,752,753,754,755,756,757,758,759,760,761,762,763,764,765,766,767,768,769,770,771,772,773,774,775,776,777,778,779,780,781,782,783,784,785,786,787,788,789,790,791,792,793,794,795,796,797,798,799,800,801,802,803,804,805,806,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856,857,858,859,860,861,862,863,864,865,866,867,868,869,870,871,872,873,874,875,876,877,878,879,880,881,882,883,884,885,886,887,888,889,890,891,892,893,894,895,896,897,898,899,900,901,902,903,904,905,906,907,908,909,910,911,912,913,914,915,916,917,918,919,920,921,922,923,924,925,926,927,928,929,930,931,932,933,934,935]
    line [10.5,10.5,11.2,14.2,16.4,15.0,13.4,11.5,11.1,10.7,11.0,11.2,12.7,14.3,14.9,14.4,12.7,10.0,9.9,10.0,8.1,8.5,8.6,8.7,9.0,9.1,9.1,9.5,11.1,9.3,7.9,6.9,6.8,8.0,9.0,11.8,14.1,15.0,16.0,16.1,15.6,14.9,14.4,13.6,13.5,13.5,13.0,11.9,11.7,11.7,11.7,13.3,13.7,12.4,11.5,10.6,9.9,9.6,11.0,12.0,14.2,14.9,16.2,15.6,15.1,14.3,14.3,12.8,11.9,11.2,10.9,10.6,10.6,11.2,14.5,16.2,18.1,16.3,13.6,11.5,10.9,11.0,12.3,12.9,13.4,13.4,14.2,11.5,9.4,9.1,8.9,8.7,7.7,7.1,7.2,7.5,7.7,8.7,10.0,13.5,15.6,14.1,9.7,9.3,9.8,11.1,12.3,14.4,14.4,16.0,16.8,16.7,16.3,14.9,14.3,12.8,12.9,12.8,12.7,12.7,12.9,13.1,15.1,16.4,18.0,17.5,16.2,14.4,13.7,14.2,14.9,16.0,17.3,18.5,19.7,19.3,17.4,15.6,15.0,14.2,14.0,13.8,14.1,13.7,14.0,14.2,16.0,19.0,22.2,22.0,20.4,18.5,17.5,17.0,16.7,17.0,18.2,21.3,20.7,19.7,17.7,16.3,15.3,13.9,13.9,13.3,12.8,12.8,13.0,14.0,16.3,22.7,26.6,21.5,18.1,16.3,15.0,14.9,15.6,17.5,20.0,29.9,29.3,22.6,18.4,16.4,15.3,14.0,13.5,13.0,12.0,11.8,11.8,12.0,13.4,14.9,15.1,13.7,12.5,11.8,11.2,11.0,11.6,12.9,14.6,16.5,17.5,16.5,15.0,12.8,13.2,12.5,13.4,12.9,12.4,12.0,12.0,12.0,12.0,12.8,13.0,13.0,12.8,12.4,12.5,11.9,12.1,13.0,13.8,15.4,17.0,16.0,15.4,13.9,13.8,13.2,12.8,12.1,12.0,12.6,12.9,13.2,15.3,20.5,21.0,17.1,14.0,11.7,10.9,10.7,11.2,13.2,15.9,22.3,27.1,21.5,17.8,15.5,13.9,13.0,13.3,12.5,12.4,11.7,11.6,12.2,14.4,19.0,19.4,14.4,12.4,11.0,9.6,8.9,9.8,11.0,12.8,17.8,17.4,15.6,14.2,12.9,12.2,11.4,10.5,9.9,10.0,9.7,9.4,9.7,11.5,14.9,16.6,12.9,10.8,8.9,8.2,8.3,8.5,9.1,11.0,13.1,14.3,14.4,12.1,11.4,10.7,10.0,8.6,8.5,8.5,8.4,8.2,8.4,8.6,11.0,12.8,10.6,9.4,8.7,8.7,8.5,8.5,9.0,9.8,11.0,11.2,12.2,10.6,9.2,9.1,8.3,7.6,7.5,8.0,8.3]
    line [6.5,7.1,8.0,13.5,14.1,13.4,10.9,10.3,9.9,10.2,10.7,11.0,11.7,13.6,13.5,13.0,13.0,10.2,8.8,9.6,7.6,8.4,8.2,10.1,7.3,9.0,9.7,12.4,12.0,10.8,11.6,7.7,9.0,10.1,10.0,10.5,11.9,13.9,14.7,14.7,13.8,13.1,12.4,10.9,10.3,11.2,12.6,11.1,11.2,11.0,9.9,13.4,12.3,12.7,10.8,10.5,9.7,9.2,9.7,12.6,12.0,13.1,14.5,15.1,12.9,12.4,12.2,10.7,10.2,10.0,9.6,9.5,9.8,10.2,14.0,16.3,15.4,15.3,12.1,11.9,10.9,10.9,11.9,12.8,12.3,14.6,15.1,13.5,11.4,10.0,9.8,8.2,7.6,7.1,4.8,7.2,6.4,11.0,13.2,13.0,15.1,16.0,13.2,10.5,11.5,12.1,12.1,13.3,12.3,13.4,14.1,15.5,16.1,14.1,13.7,11.6,10.6,10.6,10.3,10.3,10.4,10.8,12.8,14.6,17.3,14.5,15.3,13.9,11.5,13.8,14.0,15.6,16.6,21.2,30.7,17.8,19.0,16.0,14.7,13.5,13.1,13.1,12.8,12.6,12.6,13.5,15.5,22.5,29.9,22.9,17.5,14.7,15.0,12.7,15.0,15.6,16.6,18.2,17.5,16.8,15.4,14.4,13.7,11.5,12.3,13.0,12.3,12.6,12.8,12.8,15.4,20.0,24.4,20.4,16.3,15.9,15.1,14.3,15.7,19.5,16.2,23.4,24.8,21.4,21.3,15.2,13.9,13.5,13.4,12.7,12.0,11.7,11.6,11.8,11.9,14.8,14.8,13.9,12.6,11.7,10.8,10.4,11.0,12.3,14.2,15.9,16.7,16.0,15.2,13.4,13.3,12.5,13.0,11.4,10.9,11.1,11.0,11.0,11.1,12.8,13.5,13.1,12.4,12.0,11.5,10.9,11.8,12.6,14.4,14.5,15.4,16.3,13.8,15.0,14.3,12.6,12.2,12.8,12.8,12.6,12.8,12.1,15.2,19.2,22.3,16.4,12.9,11.5,10.6,10.2,12.2,13.9,16.2,23.4,24.8,20.5,18.6,15.5,14.4,13.2,13.6,12.4,12.3,12.3,11.5,12.1,14.6,19.1,22.3,17.7,12.4,9.2,9.6,9.3,9.8,12.2,15.1,19.1,17.4,16.3,14.4,12.0,12.1,11.2,10.4,11.1,11.6,11.7,10.9,11.5,12.7,13.8,17.0,12.2,12.3,10.5,9.8,9.5,9.8,11.0,12.3,12.7,13.5,13.0,11.1,11.8,9.8,10.5,9.9,10.6,10.3,10.3,10.4,10.8,12.5,15.5,18.5,16.4,14.3,11.3,10.7,10.1,10.8,11.4,13.5,14.8,14.4,14.2,13.4,11.5,10.7,10.3,8.7,7.6,8.8,9.0]
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
    scan_interval: 500
    unit_of_measurement: ct/kWh
    value_template: "{{ value_json.prices[0].total }}"
    json_attributes:
      - prices

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices?fixedPrice=13.70084&taxPercent=19&#evaluation=true"
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
