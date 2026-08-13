import pandas as pd
import numpy as np

def limpar_dados_vendas(df, coluna_alvo='vendas_unidades'):
    """
    Função para limpar um dataset de vendas de dispositivos médicos.
    """
    print(f"Dimensões originais do dataset: {df.shape}")
    
    # 1. Remover dados duplicados
    # Remove linhas que são cópias exatas de outras
    df_limpo = df.drop_duplicates()
    print(f"Dimensões após remover duplicatas: {df_limpo.shape}")
    
    # 2. Tratar valores ausentes (Missing Values)
    # Removemos as linhas onde a variável principal (ex: volume de vendas) está vazia, 
    # pois não servem para treinar o modelo preditivo.
    df_limpo = df_limpo.dropna(subset=[coluna_alvo])
    
    # Para outras colunas numéricas (ex: densidade populacional, preço), 
    # podemos preencher os valores ausentes com a mediana da respectiva coluna.
    colunas_numericas = df_limpo.select_dtypes(include=[np.number]).columns
    df_limpo[colunas_numericas] = df_limpo[colunas_numericas].fillna(df_limpo[colunas_numericas].median())
    
    print(f"Dimensões após tratar valores ausentes: {df_limpo.shape}")

    # 3. Remover Outliers (Valores Discrepantes) usando o método IQR
    # Ideal para dados de vendas, pois lida bem com distribuições assimétricas.
    Q1 = df_limpo[coluna_alvo].quantile(0.25)
    Q3 = df_limpo[coluna_alvo].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definindo os limites de corte
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Filtrando o dataframe para manter apenas valores dentro de limites aceitáveis
    # Isso evita que o modelo "aprenda" com uma compra atípica (ex: um hospital comprou um estoque de 5 anos de uma vez)
    df_limpo = df_limpo[(df_limpo[coluna_alvo] >= limite_inferior) & (df_limpo[coluna_alvo] <= limite_superior)]
    
    print(f"Dimensões finais após remover outliers: {df_limpo.shape}")
    
    return df_limpo

# ==========================================
# Exemplo de como usar a função na prática:
# ==========================================

# 1. Carregar os dados (Simulação)
# df_vendas = pd.read_csv('vendas_dispositivos_sustentaveis.csv')

# 2. Aplicar a limpeza
# df_vendas_pronto = limpar_dados_vendas(df_vendas, coluna_alvo='volume_vendas')

# 3. Exportar os dados limpos para a próxima etapa (Feature Engineering)
# df_vendas_pronto.to_csv('vendas_limpas.csv', index=False)