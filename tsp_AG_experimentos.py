import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from copy import deepcopy

USA13 = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]

qtd_cidades = len(USA13)

def gerar_rota_aleatoria(qtd_cidades):
    rota = list(range(qtd_cidades))
    random.shuffle(rota)
    return rota

def calcular_distancia(rota, matriz_cidades):
    distancia = 0
    qtd_cidades = len(rota)

    for cidade in range(qtd_cidades - 1):
        origem = rota[cidade]
        destino = rota[(cidade + 1) % qtd_cidades] # voltando à inicial
        distancia += matriz_cidades[origem][destino]

    distancia += matriz_cidades[rota[-1]][rota[0]]

    return distancia

def rota_valida(rota, qtd_cidades):
    eh_valido = False
    if len(rota) == qtd_cidades and len(set(rota)) == qtd_cidades: 
        eh_valido = True
    return eh_valido

def torneio_selecao(populacao, distancias, k=3):
    participantes = random.sample(range(len(populacao)), k)
    melhor = participantes[0] #primeiro
    
    for i in participantes[1:]: #segundo em diante
        if distancias[i] < distancias[melhor]:
            melhor = i
    
    return populacao[melhor].copy()

#ox crossover para permutação retornando dois filhos
def order_crossover(pai1, pai2):
    n = len(pai1)
    a, b = sorted(random.sample(range(n), 2))

    filho1 = [-1] * n
    filho2 = [-1] * n
    filho1[a:b+1] = pai1[a:b+1]
    filho2[a:b+1] = pai2[a:b+1]

    for filho, pai in [(filho1, pai2), (filho2, pai1)]:
        pos = (b + 1) % n
        for gene in pai[b+1:] + pai[:b+1]:
            if gene not in filho:
                filho[pos] = gene
                pos = (pos + 1) % n

    return filho1, filho2

def mutacao_swap(individuo, taxa=0.05):
    ind = individuo.copy()
    n = len(ind)
    for i in range(n):
        if random.random() < taxa:
            j = random.randint(0, n-1)
            ind[i], ind[j] = ind[j], ind[i]
    return ind

# execução do AG
def executar_ag_tsp(
    matriz,
    seed,
    tamanho_pop=50,
    geracoes=400,
    torneio_k=3,
    taxa_crossover=0.9,
    taxa_mutacao=0.05,
    elitismo=5
):
    random.seed(seed)
    np.random.seed(seed)

    n = len(matriz)
    populacao = [gerar_rota_aleatoria(n) for pop in range(tamanho_pop)]
    distancias = [calcular_distancia(ind, matriz) for ind in populacao]
    melhor_historico = []

    for geracao in range(geracoes):
        idx_ordenado = np.argsort(distancias)
        elites = []
        for i in idx_ordenado[:elitismo]:
            if rota_valida(populacao[i], n):
                elites.append(deepcopy(populacao[i]))
            else:
                print("Elite inválida!", populacao[i])

        nova_pop = elites.copy()

        while len(nova_pop) < tamanho_pop:
            pai1 = torneio_selecao(populacao, distancias, k=torneio_k)
            pai2 = torneio_selecao(populacao, distancias, k=torneio_k)

            if random.random() < taxa_crossover:
                filho1, filho2 = order_crossover(pai1, pai2)
            else:
                filho1, filho2 = pai1.copy(), pai2.copy()

            filho1 = mutacao_swap(filho1, taxa_mutacao)
            filho2 = mutacao_swap(filho2, taxa_mutacao)

            if rota_valida(filho1, n):
                nova_pop.append(filho1)
            else:
                print("Filho1 inválido gerado!", filho1)

            if len(nova_pop) < tamanho_pop and rota_valida(filho2, n):
                nova_pop.append(filho2)
            else:
                if not rota_valida(filho2, n):
                    print("Filho2 inválido gerado!", filho2)

        populacao = nova_pop[:tamanho_pop]
        distancias = [calcular_distancia(ind, matriz) for ind in populacao]
        melhor_historico.append(min(distancias))

    melhor_idx = int(np.argmin(distancias))
    melhor_rota = populacao[melhor_idx]
    melhor_dist = distancias[melhor_idx]

    return {
        'melhor_distancia': melhor_dist,
        'melhor_rota': melhor_rota,
        'historico': melhor_historico
    }


# 30rusn 
def experimento_runs(
    runs=30,
    tamanho_pop=50,
    geracoes=400,
    torneio_k=3,
    taxa_crossover=0.9,
    taxa_mutacao=0.05,
    elitismo=5
):
    resultados_finais = []
    historicos = []

    for run in range(runs):
        res = executar_ag_tsp(
            matriz=USA13,
            tamanho_pop=tamanho_pop,
            geracoes=geracoes,
            torneio_k=torneio_k,
            taxa_crossover=taxa_crossover,
            taxa_mutacao=taxa_mutacao,
            elitismo=elitismo,
            seed=run
        )
        resultados_finais.append(res['melhor_distancia'])
        historicos.append(res['historico'])
        print(f"Run {run+1}/{runs} -> melhor distância: {res['melhor_distancia']:.2f}")

    resultados_finais = np.array(resultados_finais)
    historicos = np.array(historicos)  

    #estatísticas
    media = resultados_finais.mean()
    desvio = resultados_finais.std(ddof=1)
    print("\n=== Estatísticas sobre as 30 execuções ===")
    print(f"Média das distâncias resultados_finais: {media:.3f}")
    print(f"Desvio padrão: {desvio:.3f}")

    np.savetxt('resultados_resultados_finais_tsp.txt', resultados_finais, fmt='%.6f')

    medias_gen = historicos.mean(axis=0)
    desvios_gen = historicos.std(axis=0, ddof=1)

    plt.figure(figsize=(10,6))
    plt.plot(medias_gen, label='Média do melhor por geração')
    plt.fill_between(range(len(medias_gen)), medias_gen - desvios_gen, medias_gen + desvios_gen, alpha=0.2)
    plt.title('Convergência do AG (TSP) - média do melhor por geração')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Convergencia_TSP.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.boxplot(resultados_finais, labels=['TSP - 13 cidades'], patch_artist=True)
    plt.title('Boxplot das distâncias resultados_finais (30 runs)')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Boxplot_TSP.png')
    plt.show()

    return {
        'resultados_finais': resultados_finais,
        'historicos': historicos,
        'media': media,
        'desvio': desvio
    }

def diversidade_populacional(historicos_pop):
    unicos_por_ger = [len({tuple(ind) for ind in ger}) for ger in historicos_pop]
    return unicos_por_ger

def experimento_populacao():
    tamanhos = [20, 50, 100]
    print("\nExperimento 1: Tamanho da População ")
    result = {}
    for tamanho in tamanhos:
        print(f"\n--- Tamanho da população: {tamanho} ---")
        t0 = time.time()
        out = experimento_runs(
            runs=30,
            tamanho_pop=tamanho,
            geracoes=400,
            torneio_k=3,
            taxa_crossover=0.9,
            taxa_mutacao=0.05,
            elitismo=5
        )
        t1 = time.time()
        result[tamanho] = {"result": out, "tempo": t1-t0}
        print(f"Tempo total para {tamanho}: {t1-t0:.1f}s")

    plt.figure(figsize=(10,7))
    for tamanho in tamanhos:
        medias = result[tamanho]['result']['historicos'].mean(axis=0)
        plt.plot(medias, label=f'Pop {tamanho}')
    plt.title('Experimento 1: Tamanho da população')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp1_Populacao_Convergencia.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.boxplot(
        [result[t]['result']['resultados_finais'] for t in tamanhos],
        labels=[f'Pop {t}' for t in tamanhos],
        patch_artist=True
    )
    plt.title('Boxplot - Tamanho da População')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp1_Populacao_Boxplot.png')
    plt.show()

    for tamanho in tamanhos:
        print(f"População: {tamanho} - Média: {result[tamanho]['result']['media']:.2f} - Tempo: {result[tamanho]['tempo']:.1f}s")
    return result

def experimento_mutacao():
    taxas = [0.01, 0.05, 0.10, 0.20]
    print("\nExperimento 2: Taxa de Mutação")
    result = {}
    for taxa in taxas:
        print(f"\n Taxa de Mutação: {int(100*taxa)}%")
        out = experimento_runs(
            runs=30,
            tamanho_pop=50,
            geracoes=400,
            torneio_k=3,
            taxa_crossover=0.9,
            taxa_mutacao=taxa,
            elitismo=5
        )
        result[taxa] = out
    plt.figure(figsize=(10,7))
    for taxa in taxas:
        medias = result[taxa]['historicos'].mean(axis=0)
        plt.plot(medias, label=f'Mutação {int(taxa*100)}%')
    plt.title('Experimento 2: Taxa de mutação')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp2_Mutacao_Convergencia.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.boxplot(
        [result[t]['resultados_finais'] for t in taxas],
        labels=[f'{int(100*t)}%' for t in taxas],
        patch_artist=True
    )
    plt.title('Boxplot - Taxa de Mutação')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp2_Mutacao_Boxplot.png')
    plt.show()

    return result

def experimento_torneio():
    tamanhos = [2, 3, 5, 7]
    print("\nExperimento 3: Tamanho do Torneio")
    result = {}
    diversidades = {}
    for k in tamanhos:
        print(f"\n Torneio k={k} ---")
        historicos_pop = []
        best_out = experimento_runs(
            runs=30,
            tamanho_pop=50,
            geracoes=400,
            torneio_k=k,
            taxa_crossover=0.9,
            taxa_mutacao=0.05,
            elitismo=5
        )

        random.seed(0)
        np.random.seed(0)
        n = qtd_cidades
        populacao = [gerar_rota_aleatoria(n) for pop in range(50)]
        distancias = [calcular_distancia(ind, USA13) for ind in populacao]
        pops = [deepcopy(populacao)]
        for ger in range(400):
            idx_ordenado = np.argsort(distancias)
            elites = [deepcopy(populacao[i]) for i in idx_ordenado[:5]]
            nova_pop = elites.copy()
            while len(nova_pop) < 50:
                pai1 = torneio_selecao(populacao, distancias, k=k)
                pai2 = torneio_selecao(populacao, distancias, k=k)
                if random.random() < 0.9:
                    filho1, filho2 = order_crossover(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                filho1 = mutacao_swap(filho1, 0.05)
                filho2 = mutacao_swap(filho2, 0.05)
                if rota_valida(filho1, n):
                    nova_pop.append(filho1)
                if len(nova_pop) < 50 and rota_valida(filho2, n):
                    nova_pop.append(filho2)
            populacao = nova_pop[:50]
            distancias = [calcular_distancia(ind, USA13) for ind in populacao]
            pops.append(deepcopy(populacao))
        diversidades[k] = diversidade_populacional(pops)
        result[k] = best_out
    #plot convergência
    plt.figure(figsize=(10,7))
    for k in tamanhos:
        medias = result[k]['historicos'].mean(axis=0)
        plt.plot(medias, label=f'Torneio {k}')
    plt.title('Experimento 3: Tamanho do torneio')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp3_Torneio_Convergencia.png')
    plt.show()
    #diversidadae
    plt.figure(figsize=(10,7))
    for k in tamanhos:
        plt.plot(diversidades[k], label=f'Torneio {k}')
    plt.title('Experimento 3: Diversidade Populacional')
    plt.xlabel('Geração')
    plt.ylabel('Qtde Indivíduos Únicos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp3_Torneio_Diversidade.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.boxplot(
        [result[tk]['resultados_finais'] for tk in tamanhos],
        labels=[f'k={tk}' for tk in tamanhos],
        patch_artist=True
    )
    plt.title('Boxplot - Tamanho do Torneio')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp3_Torneio_Boxplot.png')
    plt.show()

    return result, diversidades

def experimento_elitismo():
    percentuais = [0, 1, 5, 10]
    print("\n Experimento 4: Elitismo")
    result = {}
    for perc in percentuais:
        elit = int(np.floor(perc * 50 / 100))
        print(f"\n Elitismo: {perc}% = {elit} individuos ")
        out = experimento_runs(
            runs=30,
            tamanho_pop=50,
            geracoes=400,
            torneio_k=3,
            taxa_crossover=0.9,
            taxa_mutacao=0.05,
            elitismo=elit
        )
        result[perc] = out
    plt.figure(figsize=(10,7))
    for perc in percentuais:
        medias = result[perc]['historicos'].mean(axis=0)
        plt.plot(medias, label=f'Elitismo {perc}%')
    plt.title('Experimento 4: Elitismo')
    plt.xlabel('Geração')
    plt.ylabel('Distância')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp4_Elitismo_Convergencia.png')
    plt.show()

    plt.figure(figsize=(8,6))
    plt.boxplot(
        [result[p]['resultados_finais'] for p in percentuais],
        labels=[f'{p}%' for p in percentuais],
        patch_artist=True
    )
    plt.title('Boxplot - Elitismo')
    plt.ylabel('Distância')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Exp4_Elitismo_Boxplot.png')
    plt.show()

    return result

# chamando todos os experimentos
if __name__ == "__main__":
    #experimento 1: Tamanho da população
    resultados_pop = experimento_populacao()
    #experimento 2: Taxa de mutação
    resultados_mut = experimento_mutacao()
    #experimento 3: Tamanho do torneio e diversidade
    resultados_torneio, diversidades = experimento_torneio()
    #experimento 4: Elitismo
    resultados_elit = experimento_elitismo()