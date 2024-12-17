from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

# Función para entrenar un modelo de Árbol de Decisión
def train_decision_tree(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,  # RMSE calculado manualmente
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para entrenar un modelo de Random Forest
def train_random_forest(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,  # RMSE calculado manualmente
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para entrenar un modelo de Regresión Lineal
def train_linear_regression(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para entrenar un modelo de Support Vector Regressor (SVR)
def train_svr(df, target_col, feature_cols):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = SVR()
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para entrenar un modelo de K-Vecinos más Cercanos (KNN)
def train_knn(df, target_col, feature_cols, n_neighbors=5):
    X = df[feature_cols]
    y = df[target_col]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred)**0.5,
        "R2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

# Función para realizar validación cruzada
def cross_validate_model(model, df, target_col, feature_cols, cv=5):
    """Realiza validación cruzada y devuelve la media y desviación estándar del MSE."""
    X = df[feature_cols]
    y = df[target_col]
    
    # Usamos la métrica negativa de MSE porque cross_val_score minimiza las métricas
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=cv, scoring=mse_scorer)
    
    # Convertimos MSE a valores positivos
    mse_scores = -scores
    mean_mse = mse_scores.mean()
    std_mse = mse_scores.std()

    return mean_mse, std_mse
