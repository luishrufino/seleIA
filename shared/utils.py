# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÃO
# =============================================================================

import os
import re
import json
import glob
import docx
import nltk
import string
import unicodedata
import numpy as np
import pandas as pd
from pypdf import PdfReader
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timedelta
from botocore.exceptions import NoCredentialsError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


VAGAS_PATH = 'vagas/'
model = SentenceTransformer("all-MiniLM-L6-v2")


# =============================================================================
# TOKENIZAÇÃO E NORMALIZAÇÃO
# =============================================================================

def normalize_accents(text):
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")

def normalize_str(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = normalize_accents(text)
    text = re.sub(re.compile(r" +"), " ",text)
    return " ".join([w for w in text.split()])

def remove_punctuation(text):
    punctuations = string.punctuation
    table = str.maketrans({key: " " for key in punctuations})
    text = text.translate(table)
    return text


def tokenizer(text):
    stop_words = nltk.corpus.stopwords.words("portuguese")
    if isinstance(text, str):
        text = normalize_str(text)
        text = "".join([w for w in text if not w.isdigit()])
        text = word_tokenize(text)
        text = [x for x in text if x not in stop_words]
        text = [y for y in text if len(y) > 1]
        return [t for t in text]
    elif isinstance(text, list):
        return [token for item in text if item for token in tokenizer(str(item))]
    else:
        return []

# =============================================================================
# FUNÇÕES PRINCIPAIS STREAMLIT
# =============================================================================

def parse_date_safe(s):
    if not s:
        return datetime.min
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min

def encerrar_vaga(vaga_id):
    """Encerra uma vaga (muda status para 'encerrada')"""
    vagas = carregar_dados(VAGAS_PATH)
    
    for vaga in vagas:
        if vaga['id'] == vaga_id:
            vaga['status'] = 'encerrada'         
            vaga['data_fechamento'] = datetime.now().strftime("%Y-%m-%d")
            salvar_dados(VAGAS_PATH, f"vaga_{vaga_id}.json", vaga)
            return True, "Vaga encerrada com sucesso"
    
    return False, "Vaga não encontrada"

def reabrir_vaga(vaga_id):
    vagas = carregar_dados(VAGAS_PATH)
    
    for vaga in vagas:
        if vaga['id'] == vaga_id:
            vaga['status'] = 'ativa'
            vaga['data_fechamento'] = None
            salvar_dados(VAGAS_PATH, f"vaga_{vaga_id}.json", vaga)
            return True, "Vaga reaberta com sucesso"
    
    return False, "Vaga não encontrada"


def carregar_dados(pasta):
    """Carrega todos os JSONs de uma pasta"""
    dados = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith('.json'):
            with open(os.path.join(pasta, arquivo), 'r', encoding='utf-8') as f:
                try:
                    dados.append(json.load(f))
                except json.JSONDecodeError:
                    continue
    return dados

def salvar_dados(pasta, nome_arquivo, dados):
    """Salva dados em JSON"""
    with open(os.path.join(pasta, nome_arquivo), 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

# =============================================================================
# CALCULO DOS FATORES DE MATCH
# =============================================================================

def calcular_fator_salarial(candidato, vaga):
    """Calcula fator salarial (0-1)"""
    pretencao = candidato['pretencao_salarial']
    min_vaga = vaga['orcamento_salario']['min']
    max_vaga = vaga['orcamento_salario']['max']
    
    if min_vaga <= pretencao <= max_vaga:
        return 1.0
    elif pretencao < min_vaga * 0.8 or pretencao > max_vaga * 1.1:
        return 0.0
    else:
        return 0.5

stemmer = RSLPStemmer()

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


# =============================================================================
# LEITURA DE CURRÍCULOS E JSONS
# =============================================================================

def ler_jsons(pasta):
    arquivos = glob.glob(os.path.join(pasta, "*.json"))
    dados = []
    for arq in arquivos:
        with open(arq, "r", encoding="utf-8") as f:
            conteudo = json.load(f)
            # Se o arquivo já for lista de objetos
            if isinstance(conteudo, list):
                dados.extend(conteudo)
            # Se for só um dicionário
            elif isinstance(conteudo, dict):
                dados.append(conteudo)
    return pd.DataFrame(dados)


def processar_curriculos(pasta_curriculos):
    """
    Lê arquivos .pdf e .docx de uma pasta, extrai o texto e o código do candidato.

    Args:
        pasta_curriculos (str): O caminho para a pasta contendo os arquivos de currículo.

    Returns:
        pd.DataFrame: Um DataFrame com as colunas 'codigo_candidato' e 'cv_pt'.
    """
    dados_curriculos = []
    
    # Pega todos os arquivos .pdf e .docx da pasta
    arquivos_pdf = glob.glob(os.path.join(pasta_curriculos, "*.pdf"))
    arquivos_docx = glob.glob(os.path.join(pasta_curriculos, "*.docx"))
    todos_arquivos = arquivos_pdf + arquivos_docx

    print(f"Encontrados {len(todos_arquivos)} currículos para processar.")

    for arquivo_path in todos_arquivos:
        try:
            # Extrai o nome do arquivo do caminho completo
            nome_arquivo = os.path.basename(arquivo_path)
            
            # --- Extração do código do candidato com Expressão Regular ---
            # O padrão (CAND\d+) busca pela palavra "CAND" seguida por um ou mais dígitos
            match = re.search(r"(CAND\d+)", nome_arquivo)
            if not match:
                print(f"AVISO: Não foi possível extrair o código do candidato do arquivo: {nome_arquivo}")
                continue # Pula para o próximo arquivo
            
            codigo_candidato = match.group(1)
            
            # --- Leitura do conteúdo do arquivo ---
            texto_completo = ""
            if nome_arquivo.endswith(".pdf"):
                reader = PdfReader(arquivo_path)
                for page in reader.pages:
                    texto_completo += page.extract_text() or "" # Adiciona o texto da página
            
            elif nome_arquivo.endswith(".docx"):
                documento = docx.Document(arquivo_path)
                for paragrafo in documento.paragraphs:
                    texto_completo += paragrafo.text + "\n" # Adiciona o texto do parágrafo
            
            # Armazena o resultado
            dados_curriculos.append({
                "codigo_candidato": codigo_candidato,
                "cv_pt": texto_completo.strip() # .strip() para remover espaços extras no início/fim
            })

        except Exception as e:
            print(f"ERRO ao processar o arquivo {nome_arquivo}: {e}")

    return pd.DataFrame(dados_curriculos)



# =============================================================================
# NOVAS FUNÇÕES PARA S3
# =============================================================================
import boto3
import json
import streamlit as st
import io

# Carregar credenciais do Streamlit secrets
def get_s3_client():
    if "s3_client" not in st.session_state:
        try:
            st.session_state.s3_client = boto3.client(
                's3',
                aws_access_key_id=st.secrets["s3"]["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=st.secrets["s3"]["AWS_SECRET_ACCESS_KEY"]
            )
        except Exception as e:
            st.error(f"Erro ao conectar ao S3: {e}")
            st.stop()
    return st.session_state.s3_client

S3_BUCKET_NAME = "meu-projeto-seleai-dados" # Substitua pelo nome do seu bucket
S3_VAGAS_PATH = "vagas/"
S3_CANDIDATOS_PATH = "candidatos/"
S3_CURRICULOS_PATH = "curriculos/"


def carregar_dados_s3(prefix):
    """Carrega todos os JSONs de um prefixo no S3"""
    s3_client = get_s3_client()
    dados = []
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith('.json'):
                    obj_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj['Key'])
                    file_content = obj_content['Body'].read().decode('utf-8')
                    try:
                        dados.append(json.loads(file_content))
                    except json.JSONDecodeError:
                        print(f"Erro de decodificação no arquivo S3: {obj['Key']}")
                        continue
    except Exception as e:
        st.error(f"Erro ao carregar dados do S3: {e}")
    return dados

def salvar_dados_s3(pasta, nome_arquivo, dados):
    """Salva dados em JSON no S3."""
    s3_client = get_s3_client()
    try:
        file_path = f"{pasta}{nome_arquivo}"
        file_content = json.dumps(dados, ensure_ascii=False, indent=2)
        s3_client.put_object(
            Bucket=st.secrets["s3"]["S3_BUCKET_NAME"],
            Key=file_path,
            Body=file_content.encode('utf-8'),
            ContentType='application/json'
        )
        return True
    except Exception as e:
        # AQUI É ONDE VOCÊ ADICIONA MAIS INFORMAÇÕES
        st.error(f"Erro ao salvar dados no S3: {e}")
        # A linha abaixo irá mostrar o erro técnico completo
        st.exception(e) 
        return False

def processar_curriculos_s3(prefix):
    """Lê arquivos .pdf e .docx de um prefixo no S3"""
    s3_client = get_s3_client()
    dados_curriculos = []
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith(('.pdf', '.docx')):
                    # Extrai o código do candidato da chave do objeto
                    match = re.search(r"(CAND\d+)", key)
                    if not match:
                        print(f"AVISO: Não foi possível extrair o código do candidato do arquivo: {key}")
                        continue
                    
                    codigo_candidato = match.group(1)
                    
                    obj_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
                    file_stream = io.BytesIO(obj_content['Body'].read())
                    
                    texto_completo = ""
                    if key.endswith(".pdf"):
                        reader = PdfReader(file_stream)
                        for page in reader.pages:
                            texto_completo += page.extract_text() or ""
                    elif key.endswith(".docx"):
                        # docx2txt não funciona bem com streams
                        # Salvar temporariamente ou usar outra lib
                        pass # AQUI VOCÊ PODE PRECISAR DE UMA SOLUÇÃO ALTERNATIVA
                    
                    dados_curriculos.append({
                        "codigo_candidato": codigo_candidato,
                        "cv_pt": texto_completo.strip()
                    })

    except Exception as e:
        st.error(f"ERRO ao processar currículos do S3: {e}")
    
    return pd.DataFrame(dados_curriculos)

# Atualize estas funções no seu utils.py
def encerrar_vaga_s3(vaga_id):
    """Encerra uma vaga e salva a alteração no S3."""
    vaga = carregar_vaga_s3(VAGAS_PATH, f"vaga_{vaga_id}.json")
    
    if vaga:
        vaga['status'] = 'encerrada'
        vaga['data_fechamento'] = datetime.now().strftime("%Y-%m-%d")
        
        # Salva apenas a vaga específica que foi modificada
        success = salvar_dados_s3(VAGAS_PATH, f"vaga_{vaga_id}.json", vaga)
        if success:
            return True, "Vaga encerrada com sucesso"
        else:
            return False, "Erro ao salvar a vaga no S3"
    
    return False, "Vaga não encontrada"

def reabrir_vaga_s3(vaga_id):
    """Reabre uma vaga e salva a alteração no S3."""
    vaga = carregar_vaga_s3(VAGAS_PATH, f"vaga_{vaga_id}.json")
    
    if vaga:
        vaga['status'] = 'ativa'
        vaga['data_fechamento'] = None
        
        # Salva apenas a vaga específica que foi modificada
        success = salvar_dados_s3(VAGAS_PATH, f"vaga_{vaga_id}.json", vaga)
        if success:
            return True, "Vaga reaberta com sucesso"
        else:
            return False, "Erro ao salvar a vaga no S3"
    
    return False, "Vaga não encontrada"


# Adicione esta função ao seu utils.py
def carregar_vaga_s3(pasta, nome_arquivo):
    """Carrega dados de um único arquivo JSON do S3."""
    s3_client = get_s3_client()
    try:
        file_path = f"{pasta}{nome_arquivo}"
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_path)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except s3_client.exceptions.NoSuchKey:
        st.warning(f"Arquivo não encontrado no S3: {file_path}")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar a vaga do S3: {e}")
        return None
    
def ler_jsons_s3(prefix):
    """
    Lê todos os arquivos JSON de um prefixo no S3 e retorna um DataFrame.
    
    Args:
        prefix (str): O "caminho da pasta" no S3 (ex: "vagas/", "candidatos/").
    """
    s3_client = get_s3_client()
    dados = []
    
    try:
        # Lista todos os objetos com o prefixo
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)
        
        # Verifica se há conteúdo no bucket para o prefixo especificado
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                # Pula subdiretórios ou objetos que não são JSON
                if file_key.endswith('.json'):
                    obj_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
                    file_content = obj_content['Body'].read().decode('utf-8')
                    
                    try:
                        conteudo = json.loads(file_content)
                        # Anexa os dados à lista se for um dicionário ou cada item se for uma lista
                        if isinstance(conteudo, dict):
                            dados.append(conteudo)
                        elif isinstance(conteudo, list):
                            dados.extend(conteudo)
                    except json.JSONDecodeError:
                        st.warning(f"Erro de decodificação no JSON: {file_key}")
                        continue
                        
    except NoCredentialsError:
        st.error("Credenciais da AWS não encontradas. Verifique o arquivo secrets.toml.")
    except Exception as e:
        st.error(f"Erro ao carregar dados do S3: {e}")
        st.exception(e) # Mostra o erro completo para debugging

    # Retorna o DataFrame, que estará vazio se não houver dados
    return pd.DataFrame(dados)