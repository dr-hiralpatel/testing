# Student Marks Prediction

A small Python machine-learning project that explores student performance data and uses linear regression to predict marks from study hours.

## Project Overview

The main program loads `student_data.csv`, trains a supervised linear regression model, evaluates it on a held-out test set, and predicts the marks for a student who studies for 6 hours.

The workflow in `overall_unit2_3.py` includes:

- Inspecting the dataset shape, columns, and sample feature and label values
- Splitting the data into training and test sets (80/20)
- Fitting a scikit-learn `LinearRegression` model
- Reporting mean absolute error (MAE) and mean squared error (MSE)
- Predicting a mark for a new study-hours value

## Project Files

| File | Description |
| --- | --- |
| `overall_unit2_3.py` | Active regression example using `student_data.csv` |
| `student_data.csv` | Study hours and marks used by the active example |
| `p1_regression.py` | Loads and previews `student.csv`; its regression and classification examples are currently commented out |
| `student.csv` | Student performance dataset with study, attendance, assignment, and score fields |

## Requirements

- Python 3
- pandas
- scikit-learn

To run `p1_regression.py`, matplotlib must also be installed because it is imported by that script.

Install the packages with:

```bash
python -m pip install pandas scikit-learn matplotlib
```

## Run

Open a terminal in the project directory and run:

```bash
python overall_unit2_3.py
```

On Windows, you can also use:

```powershell
py overall_unit2_3.py
```

The script prints the dataset, train/test record counts, actual and predicted test marks, MAE, MSE, and the prediction for a student studying 6 hours. Keep `student_data.csv` in the same directory as the script when running it.

To preview the other dataset, run:

```bash
python p1_regression.py
```

That script currently prints the first rows of `student.csv`. The modeling code in it is commented out and is not executed.

## Dataset Fields

`student_data.csv` contains:

| Column | Meaning |
| --- | --- |
| `StudyHours` | Number of hours studied |
| `Marks` | Student marks used as the prediction target |

`student.csv` contains additional student performance fields, including study hours, attendance, assignment score, previous score, participation, internet usage, result, and final marks.
