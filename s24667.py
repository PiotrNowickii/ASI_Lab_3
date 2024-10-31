import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import logging

if __name__ == "__main__":
    logging.basicConfig(filename='log.txt',filemode='w',level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    df = pd.read_csv("./CollegeDistance.csv")
    df = df.dropna()
    df = df.drop(columns='rownames')
    rows = len(df)
    logger.info('Amount of rows: ' + str(rows))
    num_columns = ['unemp', 'wage', 'distance', 'tuition', 'education']
    cat_columns = ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'income', 'region']
    logger.info('Began the preperation of data')
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_columns),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_columns)])
    X = df.drop(columns='score')
    y = df['score']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    random_forest = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor())])
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    logger.info("MAE: " + str(mean_absolute_error(y_test, y_pred)))
    logger.info("R^2: " + str(r2_score(y_test, y_pred)))

    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(random_forest, param_grid, scoring='r2')
    grid_search.fit(X_train, y_train)

    logger.info("MAE: " + str(mean_absolute_error(y_test, grid_search.predict(X_test))))
    logger.info("R^2: " + str(grid_search.best_score_))
