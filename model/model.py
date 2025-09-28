# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÃO
# =============================================================================
import numpy as np
import nltk
try:
    nltk.data.find('stemmers/rslp')
except nltk.downloader.DownloadError:
    nltk.download('rslp')

# Agora, o resto do seu código
from nltk.stem import RSLPStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÃO
# =============================================================================


model = SentenceTransformer("all-MiniLM-L6-v2")
stemmer = RSLPStemmer()

def calcular_fator_salarial(candidato, vaga):
    """Calcula fator salarial (0-1)"""
    pretencao = candidato['pretencao_salarial']
    min_vaga = vaga['orcamento_salario']['min']
    max_vaga = vaga['orcamento_salario']['max']
    
    if min_vaga <= pretencao <= max_vaga:
        return 1.0
    elif pretencao < min_vaga * 0.9 or pretencao > max_vaga * 1.1:
        return 0.75
    elif pretencao < min_vaga * 0.8 or pretencao > max_vaga * 1.2:
        return 0.5
    elif pretencao < min_vaga * 0.7 or pretencao > max_vaga * 1.3:
        return 0.25
    else:
        return 0.0


def calcular_fator_experiencia_final(candidato, vaga, max_anos=20):
    """
    Calcula o fator de experiência do candidato para a vaga,
    considerando área de atuação (com stemming), tempo de experiência e nível da vaga.
    Retorna um valor entre 0 e 1.
    """
    # Stem das áreas de atuação
    cand_areas = candidato.get('areas_atuacao', [])
    vaga_area = vaga.get('area_atuacao', [])
    
    if not cand_areas or not vaga_area:
        sim_score = 0.0
    else:
        cand_areas_stem = [stemmer.stem(str(w)) for w in cand_areas]
        vaga_area_stem = [stemmer.stem(str(w)) for w in vaga_area]
        
        cand_vecs = model.encode(cand_areas_stem)
        vaga_vecs = model.encode(vaga_area_stem)
        scores = []
        for v_vec in vaga_vecs:
            sims = cosine_similarity([v_vec], cand_vecs)[0]
            scores.append(max(sims))
        sim_score = float(np.mean(scores))

    # Ajuste pelo tempo de experiência e nível da vaga
    tempo = candidato.get('tempo_experiencia', 0)
    nivel_vaga = vaga.get('nivel_profissional', "")

    # Força a ser string sempre
    if isinstance(nivel_vaga, list):
        nivel_vaga = nivel_vaga[0] if nivel_vaga else ""
    nivel_vaga = str(nivel_vaga).lower().strip()

    intervalos = {
        "estagio": (0, 0),
        "junior": (1, 2),
        "pleno": (3, 5),
        "senior": (5, 10),
        "especialista": (10, 100)
    }

    min_anos, max_anos_nivel = intervalos.get(nivel_vaga, (0, max_anos))

    if nivel_vaga == "estagio":
        nivel_score = 1.0
    elif tempo + 1 <= min_anos:
        nivel_score = 0.0
    elif tempo + 1 >= max_anos_nivel:
        nivel_score = 1.0
    else:
        nivel_score = (tempo + 1 - min_anos) / (max_anos_nivel - min_anos)
    return sim_score * nivel_score


def calcular_fator_engajamento(candidato, vaga):
    """Calcula fator de engajamento (0-3)"""
    score = 0
    
    # Modelo de trabalho (1 ponto)
    if candidato['modelo_trabalho'] == vaga['modelo_trabalho']:
        score += 1
    
    # Tipo de contrato (1 ponto)
    if candidato['tipo_contrato'] == vaga['tipo_contratacao']:
        score += 1
    
    # Viagens (0.5 ponto se compatível)
    if candidato['disponibilidade_viagens'] == vaga['disponibilidade_viagens']:
        score += 0.5
    
    return score / 2.5  # Normalizar para 0-1


def calcular_fator_cultural(candidato, vaga):
    """Calcula similaridade cultural usando stemming"""
    cand_comp = candidato.get('hab_comportamentais', [])
    vaga_comp = vaga.get('hab_comportamentais', [])
    
    if not cand_comp or not vaga_comp:
        return 0.0

    # Stem das habilidades
    cand_stem = [stemmer.stem(w) for w in cand_comp]
    vaga_stem = [stemmer.stem(w) for w in vaga_comp]

    # Embeddings
    cand_vecs = model.encode(cand_stem)
    vaga_vecs = model.encode(vaga_stem)
    
    scores = []
    for v_vec in vaga_vecs:
        sims = cosine_similarity([v_vec], cand_vecs)[0]
        best_match = np.max(sims)   # melhor similaridade do candidato para cada skill da vaga
        scores.append(best_match)

    # Score final = média dos melhores matches
    return float(np.mean(scores))


def calcular_fator_tecnico(candidato, vaga, threshold=0.6):
    """Calcula similaridade técnica usando embeddings com stemming"""
    cand_skills = candidato.get('hab_tecnicas', [])
    vaga_skills = vaga.get('hab_tecnicas', [])
    
    if not cand_skills or not vaga_skills:
        return 0.0

    # Stem das skills
    cand_stem = [stemmer.stem(w) for w in cand_skills]
    vaga_stem = [stemmer.stem(w) for w in vaga_skills]

    # Embeddings
    cand_vecs = model.encode(cand_stem)
    vaga_vecs = model.encode(vaga_stem)
    
    scores = []
    for v_vec in vaga_vecs:
        sims = cosine_similarity([v_vec], cand_vecs)[0]
        best_match = np.max(sims)   # melhor similaridade do candidato para cada skill da vaga
        scores.append(best_match)

    # Score final = média dos melhores matches
    return float(np.mean(scores))


def calcular_fator_idioma(candidato, vaga):
    """Calcula fator de idioma"""
    score = 0
    
    # Inglês
    niveis = {"Nenhum": 0, "Básico": 1, "Intermediário": 2, "Avançado": 3, "Fluente": 4}
    req_ingles = vaga['nivel_ingles_min']
    cand_ingles = candidato['nivel_ingles']
    
    if req_ingles != "Não necessário":
        if niveis.get(cand_ingles, 0) >= niveis.get(req_ingles, 0):
            score += 1
    else:
        score += 1
    
    # Espanhol
    req_espanhol = vaga['nivel_espanhol_min']
    cand_espanhol = candidato['nivel_espanhol']
    
    if req_espanhol != "Não necessário":
        if niveis.get(cand_espanhol, 0) >= niveis.get(req_espanhol, 0):
            score += 1
    else:
        score += 1
    
    return score / 2  # Normalizar para 0-1

def calcular_match_score(candidato, vaga, pesos):
    """Calcula score final de match considerando todos os fatores"""
    
    fatores = {
        'salarial': calcular_fator_salarial(candidato, vaga),
        'engajamento': calcular_fator_engajamento(candidato, vaga),
        'cultural': calcular_fator_cultural(candidato, vaga),
        'tecnico': calcular_fator_tecnico(candidato, vaga),
        'idioma': calcular_fator_idioma(candidato, vaga),
        'experiencia': calcular_fator_experiencia_final(candidato, vaga)
    }
    
    # Garantir que só use os pesos definidos
    score_final = sum(pesos.get(k, 0) * fatores[k] for k in fatores)
    return score_final
