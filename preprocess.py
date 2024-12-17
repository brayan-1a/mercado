import pandas as pd

def load_and_select_data(supabase_client, table_name, selected_columns):
    response = supabase_client.table(table_name).select(",".join(selected_columns)).execute()
    return pd.DataFrame(response.data)

def clean_data(df):
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Nulos en numéricos
    df.fillna("Sin datos", inplace=True)  # Nulos en texto
    if "nombre_cliente" in df.columns:
        df.drop(columns=["nombre_cliente"], inplace=True)
    return df

def normalize_data(df, columns):
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def add_features(df):
    df["diferencia_inventario"] = df["inventario_inicial"] - df["inventario_final"]
    df["porcentaje_desperdicio"] = df.apply(
        lambda row: (row["desperdicio"] / row["inventario_inicial"]) * 100 
        if row["inventario_inicial"] > 0 else 0, axis=1
    )
    df["dia_semana"] = pd.to_datetime(df["fecha"]).dt.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].apply(lambda x: 1 if x >= 5 else 0)
    return df
