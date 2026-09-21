import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Função para calcular todas as métricas
def calculate_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    R     = np.corrcoef(y_test, y_pred)[0, 1]  # Correlation Coefficient (R)
    R2    = r2_score(y_test, y_pred)          # R² Score
    RMSE  = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
    MAE   = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    MAPE  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    return R, R2, RMSE, MAE, MAPE

# Função para percorrer as pastas e arquivos JSON
def process_results(base_dir):
    # Criar um DataFrame vazio para armazenar os resultados
    columns = ['Model', 'Run', 'Seed', 'R', 'R2', 'RMSE', 'MAE', 'MAPE']
    df_results = pd.DataFrame(columns=columns)

    # Percorrer as pastas dentro de base_dir
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        
        # Verificar se é um diretório
        if os.path.isdir(model_path):
            for json_file in os.listdir(model_path):
                if json_file.endswith('.json'):
                    json_path = os.path.join(model_path, json_file)
                    
                    # Abrir o arquivo JSON
                    with open(json_path, 'r') as file:
                        data = json.load(file)
                        
                        # Extrair as informações necessárias
                        run    = data.get('run')
                        seed   = data.get('seed')
                        model  = data.get('Model')
                        y_test = data.get('y_test')
                        y_pred = data.get('y_pred')
                        
                        # Calcular as métricas
                        R, R2, RMSE, MAE, MAPE = calculate_metrics(y_test, y_pred)
                        
                        # Criar um DataFrame com a linha de resultados
                        result_df = pd.DataFrame([{
                            'Model': model,
                            'Run'  : run,
                            'Seed' : seed,
                            'R'    : R,
                            'R2'   : R2,
                            'RMSE' : RMSE,
                            'MAE'  : MAE,
                            'MAPE' : MAPE
                        }])
                        
                        # Concatenar a linha ao DataFrame principal
                        df_results = pd.concat([df_results, result_df], ignore_index=True)
    
    return df_results

# Caminho para a pasta 'results'
base_dir = './results'

# Processar os resultados
df_metrics = process_results(base_dir)

# Agrupar por modelo e calcular médias e desvios padrões
df_grouped = df_metrics.groupby('Model').agg(
    {'R'   : ['mean', 'std'],
     'R2'  : ['mean', 'std'],
     'RMSE': ['mean', 'std'],
     'MAE' : ['mean', 'std'],
     'MAPE': ['mean', 'std']}).reset_index()

# Renomear as colunas para o formato desejado (ex: 'R_mean', 'R_std', etc.)
df_grouped.columns = [
    'Model',
    'R'    , 'R std', 
    'R2'   , 'R2 std', 
    'RMSE' , 'RMSE std',
    'MAE'  , 'MAE std',
    'MAPE' , 'MAPE std'
]



# Caminho para salvar o arquivo CSV
output_csv_path  = './analysis/metrics_summary.csv'

# Garantir que a pasta existe
output_directory = os.path.dirname(output_csv_path)
os.makedirs(output_directory, exist_ok=True)

# Salvar o DataFrame no arquivo CSV
df_grouped.to_csv(output_csv_path, index=False)


# Exibir o DataFrame com médias e desvios padrão
print(df_grouped)
