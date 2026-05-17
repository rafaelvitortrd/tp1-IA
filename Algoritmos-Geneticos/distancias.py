import numpy as np
from deap import base, creator, tools, algorithms
import random
import numpy as np
import plotly.graph_objects as go


# Função de avaliação para o algoritmo genético, mede a distancia total do caminho
# Esse é o fitness usado
def avaliacao(individuo):

    distancia_total = 0

    # Tamanho do individuo da escolha genetica (Possivel caminho)
    for i in range(len(individuo) - 1):
        distancia_total += matriz[individuo[i]][individuo[i + 1]]

    # Adiciona a distancia de volta para o ponto de partida 
    # individuo[-1] é o ultimo ponto do caminho
    distancia_total += matriz[individuo[-1]][individuo[0]]

    # Deve retornar uma tupla, mesmo que seja apenas um valor
    # Padrao do DEAP
    return (distancia_total,)

# Leitura do arquivo de entrada dos pontos
with open("instancia.txt", "r") as f:
    quantidade = int(f.readline())
    pontos = np.loadtxt(f)

#inicialiaza a matriz de distancias
matriz = np.zeros((quantidade, quantidade))

for i in range(quantidade):

    # Como a matriz é simetrica, so precisamos percorrer metade dela
    for j in range(i):

        distancia = np.linalg.norm(pontos[i] - pontos[j])

        matriz[i][j] = distancia
        matriz[j][i] = distancia


# Criação do algoritmo genetico usando DEAP

# Criacao de um tipo
creator.create(
    "FitnessMin",
    # Classe do DEAP para usar no fitness, armazena e compara o valor
    base.Fitness,
    # Quanto menor o valor do fitness, melhor o individuo
    weights=(-1.0,)
)

# Criação do individuo (Possivel melhor caminho)
creator.create(
    "Individual",
    # É uma lista com o caminho dos pontos (ex: [0, 2, 1, 3])
    list,
    # Conecta o tipo ao fitness
    fitness=creator.FitnessMin
)

# Criação da toolbox do DEAP, para registrar as funções que vão ser utilizadas
# no algoritmo genético e informações sobre o individuo e a população
toolbox = base.Toolbox()

# Criação do caminho, escolhendo pontos aleatorios
toolbox.register(
    "indices",
    # Escolha dos elementos SEM REPETIÇÃO
    random.sample,
    # Tamanho do caminho, que é a quantidade de pontos
    range(quantidade),
    quantidade
)

# Integra a função de gerar caminhos para gerar o individuo
# O individuo é o que possui o fitness para ser testado.
toolbox.register(
    "individual",
    tools.initIterate,
    creator.Individual,
    toolbox.indices
)

# Cria a população que sera usada no algoritmo genetico
toolbox.register(
    "population",
    # Repete a criação de individuos para criar a população
    tools.initRepeat,
    list,
    toolbox.individual
)

# Define a função de avaliação para o algoritmo genético
# Sendo aquela que mede a distancia total do caminho
toolbox.register("evaluate", avaliacao)

# Define a função Crossover
# "cxOrdered" é um tipo de crossover que impede a repetição de pontos no caminho
toolbox.register(
    "mate",
    tools.cxOrdered
)

# Define a função Mutação
# "mutShuffleIndexes" é um tipo de mutação que garante que não tenha repetição de pontos
toolbox.register(
    "mutate",
    tools.mutShuffleIndexes,
    # Probabilidade de cada gene ser mutado (20%)
    indpb=0.2
)

# Define a função seleção, usando o metodo de torneio para escolher o melhor individuo
# que tera chance de sofrer crossover
toolbox.register(
    "select",
    tools.selTournament,
    tournsize=3
)

# Define o tamanho da população
pop = toolbox.population(n=100)

# Pegar o melhor individuo dentre todas as gerações
hof = tools.HallOfFame(1)

# Pegar as estatisticas de cada geração para grafico.
stats = tools.Statistics(lambda ind: ind.fitness.values[0])

stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("max", np.max)

# Excutando o algoritmo genetico
pop, logbook = algorithms.eaSimple(
    pop,
    toolbox,
    cxpb=0.7,
    mutpb=0.2,
    ngen=100,
    stats=stats,
    halloffame=hof,
    verbose=True
)

geracoes = logbook.select("gen")
melhores = logbook.select("min")
medias = logbook.select("avg")
piores = logbook.select("max")

fig = go.Figure()

fig.add_scatter(
    x=geracoes,
    y=melhores,
    mode='lines',
    name='Melhor'
)

fig.add_scatter(
    x=geracoes,
    y=medias,
    mode='lines',
    name='Média'
)

fig.update_layout(
    title="Convergência do Algoritmo Genético",
    xaxis_title="Geração",
    yaxis_title="Fitness"
)

fig.show()

# Imprime o melhor caminho encontrado e a distancia total do caminho
print("Melhor:", hof[0])
print("Fitness:", hof[0].fitness.values)

