# Algoritimo de Evolução Diferencial (DE) para o Problema do Caixeiro Viajante (TSP)

import os
import numpy as np

def ler_dados_txt(nome_arquivo):
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    caminho_completo = os.path.join(diretorio_do_script, nome_arquivo)
    
    coordenadas = []
    with open(caminho_completo, 'r') as f:
        linhas = f.readlines()
        
    for linha in linhas[1:]:
        partes = linha.strip().split()
        if len(partes) == 2:
            try:
                x = float(partes[0])
                y = float(partes[1])
                coordenadas.append([x, y])
            except ValueError:
                continue
                
    return np.array(coordenadas)

def calcular_matriz_distancias(coords):
    n = len(coords)
    matriz = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            matriz[i, j] = dist
            matriz[j, i] = dist
    return matriz

def calcular_distancia_caminho(rota, matriz_dist):
    distancia = 0.0
    n = len(rota)
    for i in range(n):
        origem = rota[i]
        destino = rota[(i + 1) % n]
        distancia += matriz_dist[origem, destino]
    return distancia

def mutacao_path(alvo, r1, F):
    mutante = r1.copy()
    n = len(mutante)
    if np.random.rand() < F:
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        mutante[idx1], mutante[idx2] = mutante[idx2], mutante[idx1]
    return mutante

def cruzamento_ordem(pai, mutante, CR):
    if np.random.rand() > CR:
        return pai.copy()
        
    n = len(pai)
    filho = np.full(n, -1, dtype=int)
    
    start, end = sorted(np.random.choice(n, 2, replace=False))
    filho[start:end] = mutante[start:end]
    
    pos_filho = end % n
    pos_pai = end % n
    
    while -1 in filho:
        cidade_pai = pai[pos_pai]
        if cidade_pai not in filho:
            filho[pos_filho] = cidade_pai
            pos_filho = (pos_filho + 1) % n
        pos_pai = (pos_pai + 1) % n
        
    return filho

def evolucao_diferencial_path_tsp(coords, tam_pop=50, geracoes=150, F=0.6, CR=0.9):
    n_cidades = len(coords)
    matriz_dist = calcular_matriz_distancias(coords)
    
    populacao = [np.random.permutation(n_cidades) for _ in range(tam_pop)]
    fitness = np.array([calcular_distancia_caminho(ind, matriz_dist) for ind in populacao])
    
    melhor_idx = np.argmin(fitness)
    melhor_caminho = populacao[melhor_idx].copy()
    melhor_distancia = fitness[melhor_idx]
    
    for g in range(geracoes):
        for i in range(tam_pop):
            indices = [idx for idx in range(tam_pop) if idx != i]
            r1_idx, r2_idx = np.random.choice(indices, 2, replace=False)
            
            r1 = populacao[r1_idx]
            alvo = populacao[i]
            
            mutante = mutacao_path(alvo, r1, F)
            experimental = cruzamento_ordem(alvo, mutante, CR)
            
            fit_experimental = calcular_distancia_caminho(experimental, matriz_dist)
            if fit_experimental < fitness[i]:
                populacao[i] = experimental
                fitness[i] = fit_experimental
                
                if fit_experimental < melhor_distancia:
                    melhor_distancia = fit_experimental
                    melhor_caminho = experimental.copy()
                    
        if (g + 1) % 50 == 0 or g == 0:
            print(f"Geração {g+1:03d}/{geracoes} -> Melhor Distância: {melhor_distancia:.4f}")
            
    return melhor_caminho, melhor_distancia

if __name__ == "__main__":
    nome_arquivo = "instancia.txt"
    
    try:
        coordenadas = ler_dados_txt(nome_arquivo)
        print(f"Sucesso: {len(coordenadas)} cidades carregadas.\n")
        
        caminho_otimo, dist_otima = evolucao_diferencial_path_tsp(
            coordenadas, tam_pop=50, geracoes=150, F=0.6, CR=0.9
        )
        
        print("\n--- Resultado Final ---")
        print(f"Melhor ordem: {' -> '.join(map(str, caminho_otimo))} -> {caminho_otimo[0]}")
        print(f"Menor distância: {dist_otima:.4f}")
        
    except FileNotFoundError:
        diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
        esperado = os.path.join(diretorio_do_script, nome_arquivo)
        print(f"Erro: O arquivo não foi encontrado em: {esperado}")