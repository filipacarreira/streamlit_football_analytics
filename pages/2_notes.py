import streamlit as st

st.markdown("<h1 style='text-align: center; color: white;'>🔮 Trabalho Futuro - coisas que gostava de ter feito se tivesse mais tempo</h1>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: white;'>Ferramenta de Procura de Jogadores</h2>", unsafe_allow_html=True)
st.write(
    """
        Uma possível evolução da dashboard seria criar uma funcionalidade de procura de jogadores personalizada: o utilizador podia
        introduzir as características desejadas (ex: muitos remates, boa pressão, presença na área) e a ferramenta indicaria a que 
        cluster esse perfil corresponderia, e alguns exemplos de jogadores que pertencessem a esse grupo.
    """
)

st.markdown("<h2 style='text-align: center; color: white;'>Análise Aprofundada de Clusters com Baixa Participação</h2>", unsafe_allow_html=True)
st.write(
    """
        Alguns clusters têm métricas ofensivas e defensivas muito baixas. Um passo seguinte seria investigar com mais detalhe estes grupos 
        para perceber se os valores baixos se devem a poucos minutos jogados, ao contexto da equipa ou ao perfil dos próprios jogadores.
    """
)

st.markdown("<h2 style='text-align: center; color: white;'>Expansão a Outros Perfis de Jogadores</h2>", unsafe_allow_html=True)
st.write(
    """
        Esta análise focou-se apenas em avançados e defesas. Um desenvolvimento natural seria incluir médios, que têm características
        híbridas, e guarda-redes, o que permitiria ter uma visão mais completa do plantel e identificar perfis únicos também noutras posições.

    """
)

st.markdown("<h2 style='text-align: center; color: white;'>Clustering Separado por Género</h2>", unsafe_allow_html=True)
st.write(
    """
        Outra abordagem interessante seria realizar a análise separadamente para jogadores masculinos e femininos. Isto permitiria 
        avaliar se os padrões de desempenho e os clusters identificados são consistentes entre géneros ou se existem diferenças estruturais.

    """
)

st.markdown("<h2 style='text-align: center; color: white;'>Épocas e Ligas Diferentes</h2>", unsafe_allow_html=True)
st.write(
    """
        Idealmente, a análise seria repetida para diferentes épocas e ligas, mantendo o processo metodológico constante. Isto ajudaria a 
        perceber se os perfis encontrados são robustos ou se variam consoante o contexto competitivo e temporal.
    """
)

st.markdown("<h2 style='text-align: center; color: white;'>Ponderações por Contexto (Treinador, Liga, Momento da Época)</h2>", unsafe_allow_html=True)
st.write(
    """
        Alguns autores argumentam que não se deve comparar jogadores sem ter em conta o contexto tático ou competitivo em que jogam. 
        Um possível trabalho futuro seria introduzir ponderadores que considerem variáveis como o treinador, a liga ou a fase da época, 
        de forma a refinar os perfis gerados e perceber como é que fatores 'externos' influenciam a performance e a forma de jogar dos jogadores
    """
)

st.markdown("<br><br><br><br>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: right; color: white;'>Dashboard realizada por:</h5>", unsafe_allow_html=True)
st.markdown("<p style='text-align: right; color: white;'>Filipa Alves<br>📞 961414482<br>📧 filipacarreira@gmail.com</p>", unsafe_allow_html=True)

