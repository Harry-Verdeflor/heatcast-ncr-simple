# The project in plain words

## What are we doing?

A weather app tells you the temperature. But on a humid day 33 °C *feels* hotter. The **heat index** is the "feels-like"
temperature (it combines temperature and humidity). We want to know **tonight**: *how hot will tomorrow feel in my city?*

We do not predict the exact number. We predict a **level**:

| Level | Tomorrow's highest heat index | Rows in our data (16 cities × 1,823 days) |
|---|---|---|
| Not Hazardous | below 27 °C | 40 (almost never) |
| Caution | 27 to below 33 °C | 3,954 (14 %) |
| Extreme Caution | 33 to below 42 °C | 22,994 (79 %), the usual Metro Manila day |
| Danger | 42 to below 52 °C | 2,180 (7.5 %), the highest level that happens here |
| Extreme Danger | 52 °C and above | 0 (never) |

These are the five levels of PAGASA. Two of them are almost empty, so a model can only really learn the middle three.

Predicting a level (a category) is called **classification**.

## What is in the table?

One row = one city on one day. The city's name, five numbers about **today's** weather in that city, and one label: **tomorrow's level** in that city.

```
Date        City      HeatIndex_Max_Today  Temp_Min  Rainfall_Total  Pressure_Mean  SolarRadiation_Total   HeatLevelTomorrow
2021-01-01  Caloocan        29.88            24.4        15.3          1012.95            959.0              Caution
```

The model learns from many rows: "when today looked like this in this city, tomorrow was usually that."

The inputs (the city and the five numbers) are called **features**. The label is the **target**. The model is the "recipe" that turns
features into a guess.

## Why is the table built this way?

* **16 rows per day, one per city, nothing averaged.** The heat index is not the same in every city. On a typical day the hottest and
  coolest city differ by about 1.8 °C, and that is enough to change the level: Malabon, Navotas and Valenzuela have *Danger* tomorrow on 17 % of days, Muntinlupa and San Juan on only 2 %.
  Averaging the cities into one row would hide that. The `City` column lets the model learn it.
* **A word of care about the 16 rows.** The weather data comes from a weather model with big grid cells, and the 16 cities fall in just 2 cells.
  So rain, air pressure and sunshine are shared by many cities (the heat index and the temperature still differ). The 16 rows of one day are therefore *not* 16 independent
  pieces of information, and the table is not as big as 29,168 rows sounds. That is why the split and the practice rounds always keep a whole day together.
* **The last 5 years only (2021 to 2025).** Recent years describe today's climate and keep the table to a size that trains quickly. The price: less data (1,823 days), and the first practice round learns from only 246 days.
* **The five official PAGASA levels, but only three can be learned.** *Not Hazardous* happens on only 40 rows in 5 years and *Extreme Danger* never happens.
  A model cannot learn a level it almost never sees. We keep all five so the labels match PAGASA, and we judge the models mainly on
  Caution, Extreme Caution and Danger (the score `macro_f1_core3`).
* **Five weather inputs.** Today's heat index is by far the strongest clue. The other four each add a different clue: last night's temperature,
  rain, air pressure and sunshine. More columns mostly repeat each other.
* **No "yesterday" columns.** Easy to explain, and the app only needs today's numbers.

## The three models, in one line each

* **Logistic Regression**: gives each input a weight, adds them up and picks the most likely level. Simple and easy to explain.
* **Random Forest**: hundreds of small "yes/no question" trees (is the heat index above 38? was it rainy?) that vote.
* **Gradient Boosting**: also trees, but built one after another, each fixing the mistakes of the ones before.

## Words you will see

| Word | Meaning |
|---|---|
| train / test | the model learns from the **train** days; the **test** days are kept hidden to give an honest score at the very end |
| split by date | we train on the past and test on the future, like a real forecast. Shuffling rows would let the model "peek" at the neighbours of the day it is tested on, and at the other 15 cities of the same day |
| cross-validation (folds) | practice rounds *inside* the training days: learn from the early days, check on the next block, repeat 5 times, each time with more learning days. We choose settings with these, never with the test days |
| baseline ("copy today") | a dumb rule: tomorrow's level = today's level in the same city. Any model has to beat it |
| macro F1 | one score between 0 and 1 that averages how well each level is predicted, so the small *Danger* level counts as much as the big one. `macro_f1_core3` does the same for only Caution, Extreme Caution and Danger, and is steadier because it ignores the two almost-empty levels |
| accuracy | the share of rows guessed right. It can fool us: always guessing *Extreme Caution* is right 79 % of the time and never finds a Danger day |
| Danger recall | of the real Danger rows, how many did the model catch? Missing one is the costly mistake |
| confusion matrix | a grid: real level on one side, guessed level on the other |
| hyperparameters | the settings of a model (for example how many trees). Tuning = trying settings and keeping the best, using the folds |
| leakage | the model accidentally sees information from the future. Scores look great and mean nothing. Splitting by date prevents it |
| scaling | putting the numbers on the same footing (average 0, spread 1) so that Logistic Regression is not misled by big units such as the 1,000 hPa pressure. The trees do not need it |

## What do I do next?

Open `README.md`: it lists the 7 steps and who does what. Everyone uses `src/common.py`, so the three models are compared in exactly the same way.
