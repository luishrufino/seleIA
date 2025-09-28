
# 🧠 SeleAI: Sistema de Triagem e Matchmaking 

**SeleAI** é uma aplicação web interativa desenvolvida para o **processo de recrutamento**, automatizando a **triagem de candidatos e o matchmaking com vagas de emprego**. Utiliza algoritmos de Inteligência Artificial para calcular a compatibilidade entre o perfil do candidato e os requisitos da vaga, oferecendo um sistema ágil e eficiente para os recrutadores.

A aplicação é construída inteiramente em Python e Streamlit, oferecendo uma experiência unificada sem a necessidade de uma API externa.

---

## ✨ Visão Geral

**Este projeto foi desenhado como uma ferramenta completa de recrutamento, oferecendo:**

- 🤖 Sistema de Matchmaking Inteligente: Calcula um score de compatibilidade entre candidatos e vagas com base em múltiplos fatores, como habilidades técnicas, comportamentais, experiência, pretensão salarial e idiomas.

- 📁 Armazenamento em Nuvem (AWS S3): Substitui o armazenamento local por um serviço de nuvem robusto, garantindo que os dados de vagas, candidatos e currículos fiquem seguros e acessíveis de qualquer lugar.

- 📊 Dashboard de Gestão de Vagas: Permite ao gestor visualizar e gerenciar vagas ativas, candidatos, e resultados de matchmaking em um painel interativo.

- 📥 Exportação de Dados: Oferece a funcionalidade de exportar todos os dados de triagem para um arquivo Excel para análises e relatórios.


---

## 🧱 Estrutura do Projeto

```bash
seleAI/
├── model/               # Módulos de cálculo e match
│   └── model.py
│
├── shared/             # Funções de utilidade e I/O
│   └── utils.py
│
├── dados_app/          # Pasta de dados (para desenvolvimento local)
│   ├── candidatos/
│   ├── curriculos/
│   └── vagas/
│
├── .streamlit/         # Configurações de segredos para deploy
│   └── secrets.toml
│
├── appS3.py            # Aplicação Streamlit S3
├── appLocal.py            # Aplicação Streamlit para uso e testes local
├── requirements.txt    
└── README.md
```

---

## 🧠 Indicadores Calculados

O score de compatibilidade final entre o candidato e a vaga é a soma ponderada de seis fatores principais. O peso de cada fator é definido pelo gestor no momento da criação da vaga, permitindo adaptar o algoritmo às necessidades específicas de cada posição.

1. **Fator Técnico (`calcular_fator_tecnico`)**
Utiliza o modelo `SentenceTransformer("all-MiniLM-L6-v2")` para converter habilidades técnicas (como Python, SQL, AWS) em vetores numéricos. Em seguida, a similaridade de cosseno é usada para comparar as habilidades do candidato com as da vaga, resultando em um score de 0 a 1 (1 = match perfeito). Para garantir a precisão, a técnica de stemming (`RSLPStemmer`) é aplicada para reduzir palavras a suas raízes, como "programação" para "programa".

2. **Fator Cultural (`calcular_fator_cultural`)**
Funciona de forma similar ao fator técnico. Ele compara as habilidades comportamentais do candidato (como proatividade, liderança, comunicação) com as da vaga, utilizando `SentenceTransformer` e `stemming` para calcular a similaridade de cosseno.

3. **Fator de Idioma (`calcular_fator_idioma`)**
Avalia o nível de proficiência em inglês e espanhol. Para cada idioma, o candidato recebe 1 ponto se o seu nível de proficiência for igual ou superior ao requisito mínimo da vaga. O score final é a média dos pontos de cada idioma, variando de 0 a 1.

4. **Fator Salarial (`calcular_fator_salarial`)**
Compara a pretensão salarial do candidato com a faixa salarial oferecida na vaga. O cálculo é feito em faixas, premiando a compatibilidade exata com um score de 1.0 e diminuindo o score progressivamente conforme a pretensão se afasta do intervalo ideal.

5. **Fator de Engajamento (`calcular_fator_engajamento`)**
Avalia a compatibilidade de preferências e condições de trabalho. Considera o modelo de trabalho (remoto, presencial, híbrido) e o tipo de contrato (CLT, PJ), concedendo pontos por cada correspondência. O score de viagens é 0.5 se for compatível. O score total é normalizado para um valor entre 0 e 1.

6. **Fator de Experiência (`calcular_fator_experiencia_final`)**
Este é um **fator composto que combina a similaridade entre a área de atuação do candidato e a área da vaga com o tempo de experiência do candidato e o nível exigido** (Júnior, Pleno, Sênior). Ele usa uma combinação de `similaridade de cosseno` para a área e um cálculo de pontuação baseado no tempo de experiência para criar um score único e abrangente.


Para o seu projeto SeleAI, os pesos por fator são a forma de você controlar a importância de cada critério na hora de calcular o score de compatibilidade do candidato.

Em vez de todos os fatores (técnico, cultural, salarial, etc.) terem o mesmo valor, você pode definir pesos de 0 a 10. O algoritmo soma todos esses pesos e normaliza os valores para que a soma total seja 100%. **Essa determinação de pesos é personalizada para cada tipo de vaga, de acordo com as necessidades de cada cliente**.

---

## ⚙️ Como Executar Localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/luishrufino/seleAI.git
cd seleAI
```

### 2. Configurar o Ambiente Virtual e Instalar Dependências

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar o ambiente (Windows)
venv\Scripts\activate
# Ativar o ambiente (Linux/Mac)
source venv/bin/activate

# Instalar as dependências
pip install -r requirements.txt
```

O modelo será salvo em `models/obesity_model.joblib`.

### 3. Configurar o Acesso à AWS S3

```bash
# .streamlit/secrets.toml
PASSOWORD = "SUA SENHA DE ACESSO AO DASHBOARD, RELATÓRIO E CRIAÇÃO DE VAGAS"
[s3]
AWS_ACCESS_KEY_ID = "SUA_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "SUA_SECRET_ACCESS_KEY"
S3_BUCKET_NAME = "seu-nome-do-bucket-aqui"
```

### 4. Rodar a Aplicação Streamlit

- Para salvar arquivos na sua S3:
  ```bash
  streamlit run appS3.py
  ```
- Para testar app localmente:
  ```bash
  streamlit run appLocal.py
  ```


---
## 🌐 Deploy no Streamlit Community Cloud

Acesso online: https://obsesityfastcheck-gbiqph9l9czs3hg3krvpeu.streamlit.app/

O deploy desta aplicação é muito simples:

1. Faça o commit e push de todas as suas alterações para o seu repositório no GitHub.
2. Acesse o Streamlit Community Cloud: https://streamlit.io/cloud.
3. Clique em "New app" e conecte seu repositório do GitHub.
4. Selecione o repositório seleAI e o arquivo principal appS3.py.
5. Vá para a seção "Advanced settings..." e adicione seus "Secrets". O conteúdo será o mesmo do seu arquivo secrets.toml.
```bash
PASSOWORD = "SUA SENHA DE ACESSO AO DASHBOARD, RELATÓRIO E CRIAÇÃO DE VAGAS"
[s3]
AWS_ACCESS_KEY_ID = "SUA_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "SUA_SECRET_ACCESS_KEY"
S3_BUCKET_NAME = "seu-nome-do-bucket-aqui"
```
6. Clique em "Deploy!". Sua aplicação estará online em poucos minutos.

---

## 👨‍💻 Autor

Desenvolvido por **Luis Rufino**, Analista de Dados.  
Este projeto integra conceitos de ciência de dados, embeddings, e engenharia de software para otimizar processos de recrutamento.

