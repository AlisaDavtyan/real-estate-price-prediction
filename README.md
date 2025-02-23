# House Price Predictor

This project is a simple machine learning pipeline for predicting house prices based on real estate data of Armenian market.
Data is extracted from https://myrealty.am/ 
It includes data preprocessing, feature selection, model training, and evaluation. Additionally, there is a web scraping script to collect real estate data.

## Features
- **Data Preprocessing**: Handles missing values, renames columns, and converts categorical variables.
- **Feature Selection**: Uses Lasso regression to select important features.
- **Model Training**: Trains multiple regression models (Linear Regression, Ridge, ElasticNet, KNN) and selects the best one based on Mean Squared Error (MSE).
- **Predictions & Visualization**: Generates predictions and plots actual vs. predicted values for both train and test datasets.
- **Web Scraping**: Includes a script to extract real estate data for model training.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/house-price-predictor.git
   cd house-price-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Data Preprocessing & Model Training
1. Import the `HousePricePredictor` class and load your dataset:
   ```python
   from house_price_predictor import HousePricePredictor
   import pandas as pd

   df = pd.read_csv("data.csv")
   predictor = HousePricePredictor()
   df = predictor.preprocessing(df)
   ```
2. Train the model:
   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop(columns=['Price'])
   y = df['Price']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   predictor.setup_pipeline(X_train)
   predictor.select_features(X_train, y_train)
   predictor.train_models(X_train, y_train, X_test, y_test)
   ```

3. Make predictions:
   ```python
   predictions = predictor.predict(X_test)
   ```

4. Plot results:
   ```python
   from visualization import plot_predictions
   plot_predictions(predictor, y_train)
   ```

### Web Scraping
To scrape real estate data, run:
```bash
python Scrapping.py
```
This will extract and save property details for further analysis.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Requests (for scraping)
- BeautifulSoup (for scraping)

