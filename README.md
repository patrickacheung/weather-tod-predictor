# CMPT 318 - Project (Webcams, Predictions, and Weather)

This project attempts to use ML to predict an image's weather and time of day.

# Required Libraries

  - Pandas
  - Scipy
  - Sklearn
  - Pyplot
  - Seaborn

# Execution

1. Extract the included images (katkat-secret-location.zip) and weather data (weather.zip).

2. Run *clean_weather.py* to clean and combine weather data.

    ```
    $ python3 clean_weather.py <weather data folder> <image folder>
    ```

    Cleaned csv data *weather_data.csv* which will be produced and used for our ML models.

3. Run *weather_prediction.py* to see ML score and produce graphs for analysis.

    ```
    $ python3 weather_prediction.py <weather data folder> <image folder>
    ```

    Two graphs *ctgy_count.png* and *tod_count.png* will be created.
 