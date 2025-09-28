# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÕES
# =============================================================================

import os
import io
import json
import glob
import uuid
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from shared.utils import (tokenizer, 
                          carregar_dados_s3,
                          salvar_dados_s3,
                          processar_curriculos_s3,
                          get_s3_client,
                          encerrar_vaga_s3,
                          reabrir_vaga_s3,
                          ler_jsons_s3)

from model.model import (calcular_fator_tecnico,
                         calcular_fator_idioma,
                         calcular_fator_salarial,
                         calcular_fator_engajamento,
                         calcular_fator_cultural,
                         calcular_fator_experiencia_final,
                         calcular_match_score)


# Configuração da página
st.set_page_config(page_title="SeleAI - Sistema de Triagem", page_icon="🤖", layout="wide")


# Sistema de arquivos
VAGAS_PATH = "vagas/"
CANDIDATOS_PATH = "candidatos/"
CURRICULOS_PATH = "curriculos/"
BUCKET_NAME = "meu-projeto-seleai-dados"

# =============================================================================
# LOGINS E AUTENTICAÇÃO
# =============================================================================

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.sidebar.subheader("🔑 Login")
    pwd = st.sidebar.text_input("Digite a senha:", type="password")
    if st.sidebar.button("Entrar"):
        if pwd == st.secrets["PASSWORD"]:
            st.session_state["authenticated"] = True
            st.sidebar.success("Login bem-sucedido!")
        else:
            st.sidebar.error("Senha incorreta")

# =============================================================================
# MENUS E PÁGINAS
# =============================================================================

def main():
    st.sidebar.title("🎯 SeleAI Navigation")
    page = st.sidebar.radio("Navegação", ["🏠 Dashboard", "📊 Resultados", "📋 Nova Vaga", "👤 Novo Candidato"])

    # Páginas restritas
    restricted_pages = ["🏠 Dashboard", "📋 Nova Vaga", "📊 Resultados"]

    if page in restricted_pages and not st.session_state["authenticated"]:
        st.warning("🔒 Área restrita. Faça login para acessar.")
        login()
        return

    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📋 Nova Vaga":
        criar_vaga()
    elif page == "📊 Resultados":
        mostrar_resultados()
    elif page == "👤 Novo Candidato":
        cadastrar_candidato()

# =============================================================================
# CONFIGURAÇÕES DE DASHBOARD
# =============================================================================

def show_dashboard():
    st.title("🏠 Dashboard SeleAI")
    
    # Carregar dados
    vagas = carregar_dados_s3(VAGAS_PATH)
    candidatos = carregar_dados_s3(CANDIDATOS_PATH)
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        vagas_ativas = len([v for v in vagas if v.get('status') == 'ativa'])
        st.metric("Vagas Ativas", vagas_ativas)
    with col2:
        vagas_encerradas = len([v for v in vagas if v.get('status') == 'encerrada'])
        st.metric("Vagas Encerradas", vagas_encerradas)
    with col3:
        st.metric("Total Candidatos", len(candidatos))
    with col4:
        st.metric(
        "Matches Realizados",
        len([c for c in candidatos if c.get('score_match') and c['score_match'] >= 0.7])
)

    # Lista de vagas recentes (mais seguro converter datas)
    st.subheader("📋 Vagas Recentes")
    vagas_ordenadas = sorted(
        vagas,
        key=lambda x: (
            0 if x.get("status") != "ativa" else 1,
            x.get("data_abertura", "")
        ),
        reverse=True
    )[:5]

    for vaga in vagas_ordenadas:
        status = vaga.get('status', 'ativa')
        if status == 'encerrada':
            status_icon = "❌"
            status_color = "red"
            status_text = "Encerrada"
        else:
            status_icon = "✅"
            status_color = "green"
            status_text = "Ativa"
        
        
        with st.expander(f"{status_icon} #{vaga['id']} - {vaga.get('titulo_vaga','(sem título)')} - {status_text}"):

            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Empresa:** {vaga.get('empresa_contratante', 'N/A')}")
                st.write(f"**Nível:** {vaga.get('nivel_profissional', 'N/A')}")
                st.write(f"**Data Abertura:** {vaga.get('data_abertura', 'N/A')}")
                st.write(f"**Data Fechamento:** {vaga.get('data_fechamento', 'N/A')}")
                st.write(f"**Consultor:** {vaga.get('consultor_responsavel', 'N/A')}")
            
            with col2:
                candidatos_vaga = [c for c in candidatos if c.get('id_vaga') == vaga['id']]
    
                total_candidatos = len(candidatos_vaga)
                total_matches = len([c for c in candidatos_vaga if c.get('score_match') and c['score_match'] >= 0.7])
                
                melhor_candidato = None
                if candidatos_vaga:
                    melhor_candidato = max(candidatos_vaga, key=lambda c: c.get('score_match', 0))
                
                st.write(f"**Candidatos:** {total_candidatos}")
                st.write(f"**Matches:** {total_matches}")
                st.write(f"**Melhor candidato:** {melhor_candidato['codigo_candidato'] + ': ' + melhor_candidato['nome'] if melhor_candidato else 'N/A'}")
                st.markdown(
                    f"**Status:** <span style='color:{status_color}; font-weight:bold'>{status_text}</span>",
                    unsafe_allow_html=True
                )

            if status != 'encerrada':
                if st.button("❌ Encerrar Vaga", key=f"encerrar_{vaga['id']}"):
                    success, message = encerrar_vaga_s3(vaga['id'])
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)
            else:
                if st.button("🔄 Reabrir Vaga", key=f"reabrir_{vaga['id']}"):
                    success, message = reabrir_vaga_s3(vaga['id'])
                    if success:
                        st.success(message)
                        st.experimental_rerun()
                    else:
                        st.error(message)

# =============================================================================
# CRIAR VAGA
# =============================================================================

def criar_vaga():
    st.title("📋 Criar Nova Vaga")
    
    with st.form("form_vaga"):
        # Informações básicas
        col1, col2 = st.columns(2)
        with col1:
            empresa = st.text_input("Empresa Contratante*")
            titulo_vaga = st.text_input("Título da Vaga*")
            nivel = st.selectbox("Nível Profissional*", ["Estágio","Júnior", "Pleno", "Sênior", "Especialista"])
            modelo_trabalho = st.selectbox("Modelo de Trabalho*", ["Presencial", "Híbrido", "Remoto"])
            area_atuacao = st.text_input("Área de Atuação*")
        with col2:
            consultor = st.text_input("Consultor Responsável*")
            email = st.text_input("Email do consultor*")
            solicitante = st.text_input("Solicitante*")
            contato = st.text_input("Contato*")
            prazo = st.text_input("Prazo de Contratação", "Indeterminado")
        
        # Tipo de contratação
        col3, col4 = st.columns(2)
        with col3:
            pcd = st.checkbox("Vaga específica para PCD")

        with col4:
            tipo_contratacao = st.radio("Tipo de Contratação*", ["CLT", "PJ"])
            
        
        # Localização
        st.subheader("📍 Localização")
        col5, col6, col7 = st.columns(3)
        with col5:
            pais = st.text_input("País*", "Brasil")
        with col6:
            estado = st.text_input("Estado*")
        with col7:
            cidade = st.text_input("Cidade*")
        
        # Requisitos
        st.subheader("🎯 Requisitos da Vaga")
        col8, col9 = st.columns(2)
        with col8:
            nivel_academico = st.selectbox("Nível Acadêmico Mínimo*", 
                ["Ensino Médio", "Técnico", "Graduação", "Pós-graduação", "Mestrado", "Doutorado"])
            ingles = st.selectbox("Inglês Mínimo", ["Não necessário", "Básico", "Intermediário", "Avançado", "Fluente"])
            espanhol = st.selectbox("Espanhol Mínimo", ["Não necessário", "Básico", "Intermediário", "Avançado", "Fluente"])
            viagens = st.checkbox("Disponibilidade para viagens")
        with col9:
            hab_comportamentais = st.text_area(
                "Habilidades Comportamentais* (separadas por vírgula)",
                placeholder="Exemplo: Trabalho em equipe, Proatividade, Comunicação, Liderança, Resiliência, Flexibilidade, Pensamento crítico"
            )
            hab_tecnicas = st.text_area(
                "Habilidades Técnicas* (separadas por vírgula)",
                placeholder="Exemplo: Python, SQL, Machine Learning, Power BI, Java, React, AWS, Docker"
            )
        
        # Orçamento e benefícios
        st.subheader("💰 Remuneração e Benefícios")
        salario_min = st.number_input("Salário Mínimo (R$)*", min_value=0, value=3000)
        salario_max = st.number_input("Salário Máximo (R$)*", min_value=salario_min, value=6000)
        beneficios = st.text_area(
            "Benefícios Oferecidos (separados por vírgula, evite abreviações)",
            placeholder="Exemplo: Vale-refeição, Vale-alimentação, Plano de saúde, Plano odontológico, Vale-transporte, Gympass"
        )
        
        # Datas
        st.subheader("📅 Datas Importantes")
        data_fechamento = st.date_input("Data de Fechamento*")

        st.subheader("⚖️ Pesos dos Fatores (0 a 10)")
        peso_tecnico = st.slider("Peso Técnico", 0, 10, 5)
        peso_cultural = st.slider("Peso Cultural", 0, 10, 2)
        peso_engajamento = st.slider("Peso Engajamento", 0, 10, 1)
        peso_idioma = st.slider("Peso Idioma", 0, 10, 1)
        peso_salarial = st.slider("Peso Salarial", 0, 10, 1)
        peso_experiencia = st.slider("Peso Experiência", 0, 10, 1)

        # Normalizar para que somem 100%
        soma_pesos = peso_tecnico + peso_cultural + peso_engajamento + peso_idioma + peso_salarial + peso_experiencia
        pesos = {
            "tecnico": peso_tecnico / soma_pesos,
            "cultural": peso_cultural / soma_pesos,
            "engajamento": peso_engajamento / soma_pesos,
            "idioma": peso_idioma / soma_pesos,
            "experiencia": peso_experiencia / soma_pesos,
            "salarial": peso_salarial / soma_pesos
            
}
        
        if st.form_submit_button("💾 Salvar Vaga"):
            nova_vaga = {
                "id": str(uuid.uuid4())[:8],
                "data_abertura": datetime.now().strftime("%Y-%m-%d"),
                "data_fechamento": data_fechamento.strftime("%Y-%m-%d"),
                "consultor_responsavel": consultor,
                'email': email,
                "empresa_contratante": empresa,
                "informacao_nome_solicitante": solicitante,
                "contato_solicitante": contato,
                "titulo_vaga": titulo_vaga,
                "nivel_profissional": tokenizer(nivel),
                "tipo_contratacao": tipo_contratacao,
                "prazo_contratacao": prazo,
                "vaga_especifica_pcd": pcd,
                "area_atuacao": tokenizer(area_atuacao),
                "pais_vaga": pais,
                "estado_vaga": estado,
                "cidade_vaga": cidade,
                "modelo_trabalho": modelo_trabalho,
                "disponibilidade_viagens": viagens,
                "nivel_academico_min": nivel_academico,
                "nivel_ingles_min": ingles,
                "nivel_espanhol_min": espanhol,
                "hab_comportamentais": tokenizer(hab_comportamentais),
                "hab_tecnicas": tokenizer(hab_tecnicas),
                "beneficios": tokenizer(beneficios),
                "orcamento_salario": {"min": salario_min, "max": salario_max},
                "status": "ativa",
                "pesos": pesos
            }

            salvar_dados_s3(VAGAS_PATH, f"vaga_{nova_vaga['id']}.json", nova_vaga)
            st.success("✅ Vaga criada com sucesso!")

# =============================================================================
# CADADASTRO DE CANDIDATO
# =============================================================================

def cadastrar_candidato():
    st.title("👤 Cadastrar Candidato")
    
    vagas = carregar_dados_s3(VAGAS_PATH)
    vagas_ativas = [v for v in vagas if v.get('status') == 'ativa']
    
    if not vagas_ativas:
        st.warning("❌ Não há vagas ativas para candidatura.")
        return
    
    vaga_selecionada = st.selectbox(
        "Selecione a Vaga*", 
        options=vagas_ativas, 
        format_func=lambda x: f"#{x['id']} - {x['titulo_vaga']} - {x['empresa_contratante']}"
    )
    
    with st.form("form_candidato"):
        # Informações pessoais
        st.subheader("👤 Informações Pessoais")
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome Completo*")
            email = st.text_input("Email*")
            contato = st.text_input("Contato*")
        with col2:
            modelo_preferido = st.selectbox("Modelo de Trabalho Preferido*", ["Presencial", "Híbrido", "Remoto"])
            tipo_contrato_pref = st.selectbox("Tipo de Contrato Preferido*", ["CLT", "PJ"])
            possui_def = st.checkbox("Possui deficiência")
        
        # Localização
        st.subheader("📍 Localização")
        col3, col4, col5 = st.columns(3)
        with col3:
            pais = st.text_input("País*", "Brasil")
        with col4:
            estado = st.text_input("Estado*")
        with col5:
            cidade = st.text_input("Cidade*")
        
        # Formação e experiências
        st.subheader("🎓 Formação e Experiência")
        col6, col7 = st.columns(2)
        with col6:
            nivel_academico = st.selectbox("Nível Acadêmico*", ["Ensino Médio", "Técnico", "Graduação", "Pós-graduação", "Mestrado", "Doutorado"])
            area_atuacao = st.text_input("Áreas de Atuação* (separadas por vírgula)")
            tempo_experiencia = st.number_input("Tempo de Experiência (anos)*", min_value=0)
        with col7:
            nivel_ingles = st.selectbox("Nível de Inglês*", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            nivel_espanhol = st.selectbox("Nível de Espanhol", ["Nenhum", "Básico", "Intermediário", "Avançado", "Fluente"])
            viagens = st.checkbox("Disponibilidade para viagens")
        
        # Habilidades
        st.subheader("⚡ Habilidades")
        hab_comportamentais = st.text_area(
            "Habilidades Comportamentais* (separadas por vírgula)",
            placeholder="Exemplo: Trabalho em equipe, Proatividade, Comunicação, Liderança, Resiliência, Flexibilidade, Pensamento crítico"
        )
        hab_tecnicas = st.text_area(
            "Habilidades Técnicas* (separadas por vírgula)",
            placeholder="Exemplo: Python, SQL, Machine Learning, Power BI, Java, React, AWS, Docker"
        )
        
        # Remuneração
        st.subheader("💰 Remuneração")
        col8, col9 = st.columns(2)
        with col8:
            ultimo_salario = st.number_input("Último Salário (R$)", min_value=0)
            pretencao_salarial = st.number_input("Pretensão Salarial (R$)*", min_value=0)
        with col9:
            ultimos_beneficios = st.text_area(
                "Benefícios Oferecidos (separados por vírgula, evite abreviações)",
                placeholder="Exemplo: Vale-refeição, Vale-alimentação, Plano de saúde, Plano odontológico, Vale-transporte, Gympass"
            )
            
        # Upload de currículo
        uploaded_cv = st.file_uploader("📎 Anexar Currículo (PDF até 2MB)", type=["pdf"], key="cv")

        # BOTÃO DE ENVIO -> dentro do form
        submit = st.form_submit_button("📤 Enviar Candidatura")

        if submit:
            novo_candidato = {
                "id_vaga": vaga_selecionada['id'],
                "nome": nome,
                "email": email.lower(),
                "contato": contato,
                "pais": pais,  
                "estado": estado,
                "cidade": cidade,
                "possui_def": possui_def,
                "modelo_trabalho": modelo_preferido,  
                "tipo_contrato": tipo_contrato_pref,
                "disponibilidade_viagens": viagens,
                "nivel_academico": nivel_academico,  
                "areas_atuacao": tokenizer(area_atuacao),
                "tempo_experiencia": tempo_experiencia,
                "nivel_ingles": nivel_ingles,  
                "nivel_espanhol": nivel_espanhol,  
                "hab_comportamentais": tokenizer(hab_comportamentais),
                "hab_tecnicas": tokenizer(hab_tecnicas),
                "ultimo_salario": ultimo_salario,
                "ultimo_beneficio": tokenizer(ultimos_beneficios),
                "pretencao_salarial": pretencao_salarial,
                "data_candidatura": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            novo_candidato['codigo_candidato'] = f"CAND{hash(email.lower()) % 10000:04d}"

            # Salvar currículo
            if uploaded_cv is not None:
                if uploaded_cv.size > 2 * 1024 * 1024:  # 2MB
                    st.error("❌ O arquivo é muito grande. Máximo 2MB.")
                    return
                cv_filename = f"curriculo_{novo_candidato['codigo_candidato'].replace('@','_')}_{vaga_selecionada['id']}.pdf"
                
                s3_client = get_s3_client()
                try:
                    s3_client.upload_fileobj(uploaded_cv, st.secrets["S3_BUCKET_NAME"], CURRICULOS_PATH + cv_filename)
                except Exception as e:
                    st.error(f"Erro ao fazer upload do currículo: {e}")
                    return
                
                novo_candidato["cv_file"] = cv_filename
            else:
                novo_candidato["cv_file"] = ""

            # Calcular score
            score = calcular_match_score(novo_candidato, vaga_selecionada, vaga_selecionada['pesos'])
            novo_candidato['score_match'] = score

            fatores = {
                'salarial': calcular_fator_salarial(novo_candidato, vaga_selecionada),
                'engajamento': calcular_fator_engajamento(novo_candidato, vaga_selecionada),
                'cultural': calcular_fator_cultural(novo_candidato, vaga_selecionada),
                'tecnico': calcular_fator_tecnico(novo_candidato, vaga_selecionada),
                'idioma': calcular_fator_idioma(novo_candidato, vaga_selecionada),
                'experiencia': calcular_fator_experiencia_final(novo_candidato, vaga_selecionada)
            }

            novo_candidato['fatores'] = fatores

            salvar_dados_s3(
                CANDIDATOS_PATH, 
                f"candidato_{novo_candidato['codigo_candidato']}_{vaga_selecionada['id']}.json", 
                novo_candidato
            )
            st.success(f"✅ Candidatura enviada! Score de match: {score:.2%}, {fatores}")

# =============================================================================
# RESULTADOS E MATCHMAKING
# =============================================================================

def mostrar_resultados():
    st.title("📊 Resultados e Matchmaking")
    
    # Carregue os dados de todas as vagas e candidatos (uma única vez)
    # Certifique-se de que carregar_dados_s3() retorna a lista completa
    vagas = carregar_dados_s3(VAGAS_PATH)
    candidatos = carregar_dados_s3(CANDIDATOS_PATH)
    
    vagas_ordenadas = sorted(
        vagas, 
        key=lambda x: 0 if x.get('status') == 'ativa' else 1
    )

    vaga_selecionada = st.selectbox(
        "Selecione a Vaga para Análise", 
        options=vagas_ordenadas, 
        format_func=lambda x: f"#{x['id']} - {x['titulo_vaga']} - {'❌ Encerrada' if x.get('status') != 'ativa' else '✅ Ativa'}"
    )
        
    def atualizar_e_salvar_candidato(candidato_data, novo_status, comentario):
        if 'historico_status' not in candidato_data:
            candidato_data['historico_status'] = []
            
        filename = f"candidato_{candidato_data['codigo_candidato']}_{candidato_data['id_vaga']}.json"

        candidato_data['historico_status'].append({
            'data': datetime.now().isoformat(),
            'status': novo_status,
            'comentario': comentario,
            'vaga_id': candidato_data['id_vaga']
        })
        
        candidato_data['status_atual'] = novo_status 
        
        # Chame a função de salvamento do S3
        salvar_dados_s3(CANDIDATOS_PATH, filename, candidato_data)
        
        st.experimental_rerun()
        
    if vaga_selecionada:
        candidatos_vaga = [c for c in candidatos if c.get('id_vaga') == vaga_selecionada['id']]
        
        st.subheader(f"Candidatos para Vaga #{vaga_selecionada['id']}")
        st.write(f"**Total de candidatos:** {len(candidatos_vaga)}")
        
        if candidatos_vaga:
            candidatos_vaga.sort(key=lambda x: x.get('score_match', 0), reverse=True)
            
            for i, candidato in enumerate(candidatos_vaga, 1):
                status_atual = candidato.get('status_atual', 'Pendente')
                status_icon = '🟡' if status_atual == 'Pendente' else ('✅' if status_atual == 'Qualificado' else '❌')
                
                expander_label = (
                    f"#{i} - {candidato['nome']} - Score: {candidato.get('score_match', 0):.2%} - Status: **{status_icon} {status_atual}**"
                )
                
                with st.expander(expander_label):
                    # ... (resto das informações do candidato) ...
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Email:** {candidato['email']}")
                        st.write(f"**Contato:** {candidato['contato']}")
                        st.write(f"**Área:** {candidato['areas_atuacao']}")
                        st.write(f"**Experiência:** {candidato['tempo_experiencia']} anos")
                    with col2:
                        st.write(f"**Inglês:** {candidato['nivel_ingles']}")
                        st.write(f"**Pretensão:** R$ {candidato['pretencao_salarial']:,.2f}")
                        st.write(f"**Modelo Trabalho:** {candidato['modelo_trabalho']}")
                        st.write(f"**Fatores de avaliação:** {candidato['fatores']}")
                        
                    st.markdown("---")
                    
                    comentario_key = f"comentario_{candidato['codigo_candidato']}_{vaga_selecionada['id']}"
                    comentario = st.text_area("Adicionar Comentários", key=comentario_key)
                    
                    st.markdown("""
                        <style>
                        div[data-testid="column"] button {
                            width: 100% !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    col_b1, col_b2 = st.columns(2)

                    with col_b1:
                        if st.button("✅ Qualificar", key=f"qualify_{i}"):
                            if comentario.strip() == "":
                                st.warning("Por favor, adicione um comentário antes de qualificar.")
                            else:
                                atualizar_e_salvar_candidato(candidato, "Qualificado", comentario)

                    with col_b2:
                        if st.button("❌ Desqualificar", key=f"disqualify_{i}"):
                            if comentario.strip() == "":
                                st.warning("Por favor, adicione um comentário antes de desqualificar.")
                            else:
                                atualizar_e_salvar_candidato(candidato, "Desqualificado", comentario)
                    
                    if 'historico_status' in candidato:
                        st.markdown("**Histórico de Avaliações:**")
                        historico_vaga = [h for h in candidato['historico_status'] if h.get('vaga_id') == vaga_selecionada['id']]
                        for hist in historico_vaga:
                            data = datetime.fromisoformat(hist['data']).strftime('%d/%m/%Y %H:%M') 
                            st.info(f"[{data}] Status: **{hist['status']}** | Comentário: *{hist['comentario']}*")

    # BLOCO DE EXPORTAÇÃO GERAL
    st.markdown("---")
    
    # Use a função ler_jsons mais robusta
    df_candidatos = ler_jsons_s3(CANDIDATOS_PATH)
    df_vagas = ler_jsons_s3(VAGAS_PATH)

    # Certifique-se de que os DataFrames não estão vazios antes do merge
    if not df_candidatos.empty and not df_vagas.empty:
        df_final = pd.merge(
            df_candidatos,
            df_vagas,
            left_on="id_vaga",
            right_on="id",
            how="left",
            suffixes=("_candidato", "_vaga")
        )
        
        # Processar currículos e fazer merge
        df_curriculos = processar_curriculos_s3(CURRICULOS_PATH)
        if not df_curriculos.empty:
            df_final = pd.merge(
                df_final, 
                df_curriculos, 
                on="codigo_candidato", 
                how="left" 
            )
        
        # Converta o DataFrame para um objeto em memória
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, index=False, sheet_name='Resultados')
        processed_data = output.getvalue()

        # Crie o botão de download
        st.download_button(
            label="📥 Baixar dados para Excel",
            data=processed_data,
            file_name="dados_matchmaking.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_button"
        )
    else:
        st.warning("Não há dados de vagas ou candidatos para exportar.")
                

if __name__ == "__main__":
    main()