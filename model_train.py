from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def train_and_evaluate(model, df, target_col, feature_cols):
    """
    Función base para entrenar y evaluar modelos.
    Parámetros:
        model: Modelo de machine learning a entrenar.
        df (DataFrame): Conjunto de datos preprocesado.
        target_col (str): Columna objetivo.
        feature_cols (list): Columnas de características.
    Retorna:
        model: Modelo entrenado.
        metrics (dict): Métricas de evaluación.
    """
    X = df[feature_cols]
    y = df[target_col]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

    return model, metrics

def train_decision_tree(df, target_col, feature_cols):
    model = DecisionTreeRegressor(random_state=42)
    return train_and_evaluate(model, df, target_col, feature_cols)

def train_random_forest(df, target_col, feature_cols):
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    return train_and_evaluate(model, df, target_col, feature_cols)

def train_linear_regression(df, target_col, feature_cols):
    model = LinearRegression()
    return train_and_evaluate(model, df, target_col, feature_cols)

def train_svr(df, target_col, feature_cols):
    model = SVR()
    return train_and_evaluate(model, df, target_col, feature_cols)

def train_knn(df, target_col, feature_cols, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    return train_and_evaluate(model, df, target_col, feature_cols)

