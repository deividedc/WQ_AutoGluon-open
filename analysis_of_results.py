#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 09:09:27 2026

@author: deivid
"""

import os
import json
import numpy             as np
import matplotlib.pyplot as plt
import sympy             as sp
import pandas            as pd
import seaborn           as sns
import re


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from collections     import defaultdict
from scipy.stats     import pearsonr  
from collections     import defaultdict

#%%

def resultados(modelo):
    pasta_resultado = os.path.join("results", modelo)  
    
    y_test = []
    y_pred = []

    for i in range(1, 100):
        filename = f"result_run_{i}.json"
        filepath = os.path.join(pasta_resultado, filename)

        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                data = json.load(file)

                y_test.append(np.array(data["y_test"]))
                y_pred.append(np.array(data["y_pred"]))

    return y_test, y_pred

#%%

def listar_modelos():
    pasta_principal = "results"
    
    # Verifica se a pasta existe
    if not os.path.exists(pasta_principal):
        print("A pasta 'results' não existe.")
        return []
    
    # Lista apenas os diretórios dentro da pasta 'results'
    pastas = [nome for nome in os.listdir(pasta_principal) if os.path.isdir(os.path.join(pasta_principal, nome))]
    
    return pastas

#%%

def calculate_metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    
    R     = np.corrcoef(y_test, y_pred)[0, 1]  # Correlation Coefficient (R)
    R2    = r2_score(y_test, y_pred)          # R² Score
    RMSE  = np.sqrt(mean_squared_error(y_test, y_pred))  # Root Mean Squared Error
    MAE   = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    MAPE  = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    return R, R2, RMSE, MAE, MAPE

#%%

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

#%%

# Caminho para a pasta 'results'
base_dir = './results'

# Processar os resultados
df_metrics = process_results(base_dir)

# Agrupar por modelo e calcular médias e desvios padrões
df_metrics = df_metrics.groupby('Model').agg(
    {'R'   : ['mean', 'std'],
     'R2'  : ['mean', 'std'],
     'RMSE': ['mean', 'std'],
     'MAE' : ['mean', 'std'],
     'MAPE': ['mean', 'std']}).reset_index()

# Renomear as colunas para o formato desejado (ex: 'R_mean', 'R_std', etc.)
df_metrics.columns = [
    'Model',
    'R'    , 'R_std', 
    'R2'   , 'R2_std', 
    'RMSE' , 'RMSE_std',
    'MAE'  , 'MAE_std',
    'MAPE' , 'MAPE_std'
]

df_metrics.to_latex("model_performance.tex", index=False, float_format="%.4f")

#%%

# Filtrar as colunas necessárias
metrics=[
        'R', 
        #'WI',
        'R2', 
        #'RRMSE',
        #'RMSELKX', 'RMSELQ', 
        #'RMSE$(K_x<100)$', 'RMSE$(B/H<50)$', 
        'RMSE', #'NDEI', 
        'MAE', #'Accuracy', 
        'MAPE',
        #'NSE', #'LNSE', 
        #'KGE',
        #'MARE', 
        #'MSE',
        #'VAF', 'MAE (MJ/m$^2$)', 'R',  'RMSE (MJ/m$^2$)',
        ]
    
metrics_max =  ['NSE', 'VAF', 'R', 'Accuracy','R2', 'KGE', 'WI'] 
df = df_metrics[metrics]

# Normalizar os dados (inverter RMSE, MAE e MAPE para melhor visualização no radar)
normalized_data = df.copy()
for metric in metrics:
    if metric in metrics_max:
        normalized_data[metric] = (normalized_data[metric] / normalized_data[metric].max())
    else:
        normalized_data[metric] = (normalized_data[metric].min() / normalized_data[metric])

# Preparar os dados para o gráfico
labels = metrics
num_vars = len(labels)

# Adicionar o primeiro elemento ao final para fechar o gráfico
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Criar o gráfico de radar
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for i, row in normalized_data.iterrows():
    values = row.tolist()
    values += values[:1]  # Fechar o gráfico
    ax.plot(angles, values, label=df_metrics['Model'][i])
    ax.fill(angles, values, alpha=0.01)

# Configurar os eixos
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["20", "40", "60", "80", "100"], color="gray", size=10)
ax.set_ylim(0, 1)

# Legenda e título
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)  # Ajuste do posicionamento
plt.title("Desempenho dos Modelos - Radar Chart", size=14, weight="bold", pad=20)

# Ajustar o layout para evitar que a legenda seja cortada
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajuste o retângulo da figura para acomodar a legenda
plt.savefig('plots/Radar Chart.png', dpi=300, bbox_inches='tight')  # Evita cortes ao salvar
plt.show()

#%%
def plot_resultados(model):
    y_test, y_pred = resultados(model)

    # Evaluation metrics
    mae  = []
    rmse = []
    r    = []
    r2   = []
    mape = []
    for i in range(len(y_test)):
        mae.append(mean_absolute_error(y_test[i], y_pred[i]))
        rmse.append(np.sqrt(mean_squared_error(y_test[i], y_pred[i])))
        r.append(np.corrcoef(y_test[i], y_pred[i])[0, 1])
        r2.append(r2_score(y_test[i], y_pred[i]))
        mape.append(np.mean(np.abs((y_test[i] - y_pred[i]) / y_test[i])))

    r_mean = np.mean(r)
    r_std  = np.std(r)
    
    r2_mean = np.mean(r2)
    r2_std  = np.std(r2)
    
    rmse_mean = np.mean(rmse)
    rmse_std  = np.std(rmse)
    
    mape_mean = np.mean(mape) * 100
    mape_std  = np.std(mape) * 100

    # Combine arrays
    y_test = np.concatenate(y_test) if y_test else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])

    if y_test.size == 0 or y_pred.size == 0:
        print("No data available for plotting.")
        return

    # Sort values
    sorted_indices = np.argsort(y_test)
    y_test_sorted = y_test[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    # Group values for mean/std
    unique_y_test = np.unique(y_test_sorted)
    mean_y_pred = np.array([np.mean(y_pred_sorted[y_test_sorted == y]) for y in unique_y_test])
    std_y_pred  = np.array([np.std( y_pred_sorted[y_test_sorted == y]) for y in unique_y_test])

    # --- FONTES EM SIZE 14 ---
    plt.rcParams.update({
        "text.usetex": True,
        #"font.family": "Times New Roma",
        "font.size": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 16
    })

    # Criar subpasta plots
    save_dir = "plots/scatter_plots"
    os.makedirs(save_dir, exist_ok=True)

    # --- PLOT ---
    plt.figure(figsize=(8, 6))
    plt.errorbar(unique_y_test, mean_y_pred, yerr=std_y_pred, fmt='o',
                 capsize=5, mfc='#0979b0', mec='black', label="Mean ± Std Dev")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             '--', label="Ideal Line", color='#0979b0')

    plt.text(
        0.95, 0.15,
        f'R = {r_mean:.2f} ± {r_std:.2f}\n$R^2$ = {r2_mean:.2f} ± {r2_std:.2f}',
        transform=plt.gca().transAxes,
        fontsize=14,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5')
    )

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Salvar na nova subpasta
    plt.savefig(f"{save_dir}/{model}.png", dpi=300)

    plt.show()


modelos = listar_modelos()

for modelo in modelos:
    plot_resultados(modelo)
    


process_results(base_dir)

#%%

def feature_importance_analysis(modelo, save_plots=True):
    """
    Lê os arquivos da pasta results/{modelo}, analisa as importâncias das features e gera gráficos.
    """

    # --- FONTES EM SIZE 14 ---
    plt.rcParams.update({
        "text.usetex": True,
        #"font.family": "Times New Roma",
        "font.size": 20,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16
    })

    pasta_resultado = os.path.join("results", modelo)
    feature_importance_data = []

    for i in range(1, 100):
        filename = f"result_run_{i}.json"
        filepath = os.path.join(pasta_resultado, filename)

        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

                if "Feature importance" in data:
                    for feature, valores in data["Feature importance"].items():
                        feature_importance_data.append({
                            "Feature": feature,
                            "Importance": valores["importance"],
                            "StdDev": valores["stddev"],
                            "P_Value": valores["p_value"],
                            "N": valores["n"],
                            "P99_High": valores["p99_high"],
                            "P99_Low": valores["p99_low"],
                            "Normalized_Importance": valores["normalized_importance"]
                        })

    # Criar DataFrame e calcular estatísticas
    df_importance = pd.DataFrame(feature_importance_data)
    
    if df_importance.empty:
        print("Nenhum dado encontrado!")
        return None

    df_summary = df_importance.groupby("Feature").agg(
        Mean_Importance=("Importance", "mean"),
        StdDev_Mean=("StdDev", "mean"),
        P_Value_Mean=("P_Value", "mean"),
        N_Samples=("N", "sum"),
        P99_High_Mean=("P99_High", "mean"),
        P99_Low_Mean=("P99_Low", "mean"),
        Normalized_Importance_Mean=("Normalized_Importance", "mean")
    ).reset_index()

    # Criar subpastas para cada tipo de gráfico
    if save_plots:
        base_path = "feature_importance_plots"
        subpastas = ["barplot", "boxplot", "kdeplot"]

        for sub in subpastas:
            os.makedirs(os.path.join(base_path, sub), exist_ok=True)

    # -------- BARPLOT --------
    plt.figure(figsize=(6, 4))
    sns.barplot(data=df_summary, x="Mean_Importance", y="Feature", palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    #plt.title(f"Mean Feature Importance - {modelo}")
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{base_path}/barplot/{modelo}_barplot.png", dpi=300)
    plt.show()

    # -------- BOXPLOT --------
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_importance, x="Importance", y="Feature", palette="coolwarm")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    #plt.title(f"Feature Importance Distribution - {modelo}")
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{base_path}/boxplot/{modelo}_boxplot.png", dpi=300)
    plt.show()

    # -------- KDE PLOT --------
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df_importance, x="Normalized_Importance", hue="Feature",
                fill=True, common_norm=False)
    plt.xlabel("Normalized Importance")
    #plt.title(f"Distribution of Normalized Importance - {modelo}")
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{base_path}/kdeplot/{modelo}_kdeplot.png", dpi=300)
    plt.show()

    return None


for modelo in modelos:
    feature_importance_analysis(modelo)