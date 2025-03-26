#!/usr/bin/python3

import math
from typing import Dict, List, Tuple, cast
import urllib.request
from open_meteo_solar_forecast import OpenMeteoSolarForecast
import pandas as pd
import numpy as np
import datetime
import dateutil.parser
import json
import logging
import os
import asyncio
import holidays
import time
import pytz

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

log = logging.getLogger(__name__)


LATITUDES = [48.4,
    49.7,
    51.3,
    52.8,
    53.8,
    54.1]
LONGITUDES = [9.3,
    11.3,
    8.6,
    12.0,
    8.1,
    11.6
    ]


class PricePredictor:
    weather : pd.DataFrame | None = None
    solar : pd.DataFrame | None = None
    prices: pd.DataFrame | None = None

    fulldata : pd.DataFrame | None = None

    testdata : bool = False
    learnDays : int = 60
    forecastDays : int

    predictor : KNeighborsRegressor | None = None

    def __init__(self, testdata : bool = False, learnDays=60, forecastDays=7):
        self.testdata = testdata
        self.learnDays = learnDays
        self.forecastDays = forecastDays

    
    def train(self) -> None:
        # To determine the importance of each parameter, we first weight them using linreg, because knn is treating difference in each parameter uniformly
        self.fulldata = self.prepare_dataframe()
        if self.fulldata is None:
            return
        

        learnset = self.fulldata.dropna()
        learn_params = learnset.drop(columns=["time", "price"])
        learn_output = learnset["price"]
        linreg = LinearRegression().fit(learn_params, learn_output)
        param_scaling_factors = linreg.coef_

        # Do the scaling on all params, then prepare knn
        learn_params *= param_scaling_factors

        self.predictor = KNeighborsRegressor(n_neighbors=3).fit(learn_params, learn_output)

        # Update self.fulldata with scaled parameters
        orig_time_price = self.fulldata[["time", "price"]]
        self.fulldata = self.fulldata.drop(columns=["time", "price"])
        self.fulldata *= param_scaling_factors
        self.fulldata["time"] = orig_time_price["time"]
        self.fulldata["price"] = orig_time_price["price"]
        
        

    def is_trained(self) -> bool:
        return self.predictor is not None

    def predict(self, estimateAll : bool = False) -> Dict[datetime.datetime, float]:
        """
        if estimateAll is true, you will get an estimation for the full time range, even if the prices are known already (for performance evaluation).
        if false, you will get known data as is, and only estimations for unknown data
        """
        if self.predictor is None:
            self.train()
        assert self.fulldata is not None
        assert self.predictor is not None

        predictionDf = self.fulldata.copy()
        predictionDf["price"] = self.predictor.predict(predictionDf.drop(columns=["time", "price"]))

        predDict = self._to_price_dict(predictionDf)

        if not estimateAll:
            knownDict = self._to_price_dict(self.fulldata)
            predDict.update(knownDict)

      
        return predDict

    def _to_price_dict(self, df : pd.DataFrame) -> Dict[datetime.datetime, float]:
        result = {}
        for _, row in df.iterrows():
            ts = cast(pd.Timestamp, row["time"]).to_pydatetime()
            price = row["price"]
            if math.isnan(price):
                continue
            result[ts] = row["price"]
        return result


    def prepare_dataframe(self) -> pd.DataFrame | None:
        if self.weather is None or self.solar is None:
            self.refresh_forecasts()
        if self.prices is None:
            self.refresh_prices()
        assert self.weather is not None
        assert self.solar is not None
        assert self.prices is not None
        
        df = pd.concat([self.solar, self.weather], axis=1).dropna()
        df = pd.concat([df, self.prices], axis=1).reset_index()
        df = df.dropna(subset=["solar", "wind", "temp"])


        holis = holidays.country_holidays("DE")
        df["holiday"] = df["time"].apply(lambda t: 1 if t.weekday() == 6 or t.date() in holis else 0)
        df["saturday"] = df["time"].apply(lambda t: 1 if t.weekday() == 5 else 0)
        for h in range(0, 24):
            df[f"h_{h}"] = df["time"].apply(lambda t: 1 if t.hour == h else 0)
        return df


    def refresh_prices(self) -> None:
        log.info("Updating prices...")
        try:
            self.prices = self.fetch_prices()
            last_price = self.get_last_known_price()
            log.info("Price update done. Prices available until " + last_price[0].isoformat() if last_price is not None else "UNEXPECTED NONE")
        except Exception as e:
            log.warning(f"Failed to update prices : {str(e)}")
    
    def refresh_forecasts(self) -> None:
        log.info("Updating weather forecast...")
        try:
            self.solar = self.fetch_solar()
            self.weather = self.fetch_weather()
            log.info("Weather update done")
        except Exception as e:
            log.warning(f"Failed to update forecast : {str(e)}")
        

    def fetch_solar(self) -> pd.DataFrame | None:
        if self.testdata and os.path.exists("solar.json"):
            solar = pd.read_json("solar.json")
            solar.index = solar.index.tz_localize("UTC") # type: ignore
            solar.index.set_names("time", inplace=True)
            return solar

        result = OpenMeteoSolarForecast(azimuth=[0.0]*len(LATITUDES), declination=[0.0]*len(LATITUDES), dc_kwp=[1.0]*len(LATITUDES), latitude=LATITUDES, longitude=LONGITUDES, past_days=self.learnDays, forecast_days=self.forecastDays)
        estimate = asyncio.run(result.estimate())
        data = pd.DataFrame(estimate.watts.items(), columns=["time", "solar"]) # type: ignore
        data.time = data.reset_index().time.map(lambda dt: dt.replace(minute=0))
        data.time = pd.DatetimeIndex(data.time).tz_convert("UTC")
        data = data.groupby("time").agg("mean")
        #data.rename(columns={"fc_time": "time", "watts": "solar"}, inplace=True)

        if self.testdata:
            data.to_json("solar.json")

        assert isinstance(data, pd.DataFrame)
        return data
    
    def fetch_weather(self) -> pd.DataFrame | None:
        if self.testdata and os.path.exists("weather.json"):
            weather = pd.read_json("weather.json")
            weather.index = weather.index.tz_localize("UTC") # type: ignore
            weather.index.set_names("time", inplace=True)
            return weather

        lats = ",".join(map(str, LATITUDES))
        lons = ",".join(map(str, LONGITUDES))
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&past_days={self.learnDays}&forecast_days={self.forecastDays}&hourly=wind_speed_80m,temperature_2m&timezone=UTC"
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode("utf-8")

            data = json.loads(data)
            frames = []
            for fc in data:
                df = pd.DataFrame(columns=["time", "wind", "temp"]) # type: ignore
                times = fc["hourly"]["time"]
                winds = fc["hourly"]["wind_speed_80m"]
                temps = fc["hourly"]["temperature_2m"]
                df["time"] = times
                df["wind"] = winds
                df["temp"] = temps
                df.dropna(inplace=True)
                frames.append(df)

            df = pd.concat(frames).groupby("time").mean().reset_index()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df.set_index("time", inplace=True)

            if self.testdata:
                df.to_json("weather.json")
            
            return df

    def fetch_prices(self) -> pd.DataFrame | None:
        if self.testdata and os.path.exists("prices.json"):
            prices = pd.read_json("prices.json")
            prices.index = prices.index.tz_localize("UTC") # type: ignore
            prices.index.set_names("time", inplace=True)
            return prices

        filter = 4169 # marktpreis DE/LU
        region = "DE"
        filterCopy = filter
        regionCopy = region
        resolution = "hour"

        # Get available timestamps
        url = f"https://www.smard.de/app/chart_data/{filter}/{region}/index_{resolution}.json"
        with urllib.request.urlopen(url) as resp:
            data = resp.read().decode("utf-8")
            timestamps : List[int] = json.loads(data)["timestamps"]
            timestamps.sort()

        startTs = 1000 * (int(time.time()) - self.learnDays * 24 * 60 * 60)
      
        startIndex = len(timestamps) - 1
        for i, timestamp in enumerate(timestamps):
            if timestamp > startTs:
                startIndex = i - 1
                break
        timestamps = timestamps[startIndex:]

        pricesDict = {}

        for timestamp in timestamps:
            url = f"https://www.smard.de/app/chart_data/{filter}/{region}/{filterCopy}_{regionCopy}_{resolution}_{timestamp}.json"
            with urllib.request.urlopen(url) as resp:
                data = resp.read().decode("utf-8")
                series : List = json.loads(data)["series"]
                for entry in series:
                    price = entry[1]
                    if price is None:
                        continue
                    dt = datetime.datetime.fromtimestamp(entry[0] / 1000, tz=pytz.timezone("Europe/Berlin"))
                    dt = dt.astimezone(pytz.UTC)
                    pricesDict[dt] = price
        
        data = pd.DataFrame.from_dict(pricesDict, orient="index", columns=["price"]).reset_index() # type: ignore
        data.rename(columns={"index": "time"}, inplace=True)
        data["time"] = pd.to_datetime(data["time"], utc=True)
        data.set_index("time", inplace=True)

        if self.testdata:
            data.to_json("prices.json")

        return data

    def get_last_known_price(self) -> Tuple[datetime.datetime, float] | None:
        if self.prices is None:
            return None
        lastrow = self.prices.dropna().reset_index().iloc[-1]
        return lastrow["time"].to_pydatetime(), float(lastrow["price"])




def main():
    import sys
    pd.set_option("display.max_rows", None)

    root = log.getLogger()
    root.setLevel(log.DEBUG)
    handler = log.StreamHandler(sys.stdout)

    pred = PricePredictor(testdata=True, learnDays=60)
    pred.train()

    actual = pred.predict()
    predicted = pred.predict(estimateAll=True)

    #xdt : List[datetime.datetime] = list(actual.keys())
    #x = map(lambda k : k.isoformat(), xdt)
    x = map(str, range(0, len(actual)))
    actuals = map(lambda p: str(round(p/10, 1)), actual.values())
    preds = map(lambda p: str(round(p/10, 1)), predicted.values())

    x = list(x)[0:14*24]
    actuals = list(actuals)[0:14*24]
    preds = list(preds)[0:14*24]

    print(f"""
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
    x-axis [{",".join(x)}]
    line [{",".join(actuals)}]
    line [{",".join(preds)}]
    """)
    

    
    """prices = pred.predict()
    prices = {
        k.isoformat(): v for k, v in prices.items()
    }
    print(json.dumps(prices))"""

if __name__ == "__main__":
    main()
