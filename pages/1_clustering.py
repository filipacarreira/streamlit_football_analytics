import streamlit as st
from PIL import Image
import pandas as pd
from sklearn.metrics import silhouette_score

from utils.clustering import (
    get_matches_df, 
    aggregate_player_metrics, 
    plot_correlation_heatmap, 
    rename_for_display, 
    plot_metric_histograms, 
    run_clustering_plotly, 
    plot_umap_interactive, 
    plot_radar_chart, 
    plot_size,
    plot_gender_distribution
)

st.markdown("<h1 style='text-align: center; color: white;'>Perfis de Jogadores</h1>", unsafe_allow_html=True)

st.write(
    """
        A segunda parte deste exercício consistia na criação de perfis de jogadores, recorri às métricas criadas anteriormente, 
        como solicitado.

        Mas vamos por partes... A primeira coisa a decidir foi que jogos usar para fazer esta análise. Como estamos sempre a ouvir que 
        futebol feminino e masculino são desportos completamente diferentes, para esta segmentação de jogadores decidi usar dados 
        de ligas femininas e masculinas. Pode parecer uma ideia estranha, mas a ideia era ver se realmente estas diferenças eram tão 
        carregadas que levariam à criação natural de clusters de homens e mulheres, separados, ou se estas diferenças, quando vistas 
        estatísticamente acabavam por não fazer muita diferença, o que levaria à criação de clusters mistos.

        Tendo tomado esta decisão, restava decidir que jogos analisar. Para isso, fiz uma análise aos jogos e ligas que tinham eventos disponíveis
        e constatei que, infelizmente, não ia ser possível usar jogos do mesmo ano.As únicas ligas onde isso era possível fazer não eram do mesmo 
        país (Espanha e Inglaterra), o que retira sentido aos resultados porque, normalmente, países diferentes têm
        formas de jogar futebol diferentes.

        Tendo isto em conta, decidi usar:

        - Todos os jogos da FA Women's Super League, época 2020/2021
        - Todos os jogos da Premier League Masculina, época 2015/2016

        Escolhidos os jogos, faltava decidir também que perfis analisar. Para este estudo decidi considerar 2 perfis principais de jogadores:
        defesas e avançados. Esta escolha prendeu-se com o facto de considerar que estes dois tipos de jogadores tèm características
        muito diferentes, e por isso não faz sentido agrupá-los da mesma forma.

        Tendo isto em conta, na imagem abaixo é possível ver os perfis de jogadores que considerei.
    """         
)

slide1 = Image.open("images/posiçoes.png")
st.image(slide1, use_container_width=True)

st.markdown("<h2 style='text-align: center; color: white;'>Criação de Métricas para Clustering</h2>", unsafe_allow_html=True)

st.write(
    """
        Tendo os jogadores e jogos escolhidos, faltava decidir quais iam ser as métricas a considerar para o clustering. Tendo isso em conta,
        abaixo fica uma tabela com todas as variáveis consideradas para cada um dos perfis. Escolhi estas métricas porque capturam aspetos essenciais
        do contributo ofensivo e defensivo dos jogadores. Estas features permitem uma análise mais contextualizada do que se tivessemos só usado as duas
        features construídas anteriormente e destacam impacto tático dos avançados e defesas no jogo.
    """
)

st.markdown("""
| Variável                        | Perfil                   | Descrição                                                                                              |
|-------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------|
| `xg`                          | Avançados                  | Média do xG (expected goals) por jogador e jogo. Calculado a partir de eventos de remate.              |
| `shots`                       | Avançados                  | Média de remates feitos por um jogador em cada jogo.                                                   |
| `key_passes`                  | Avançados                  | Média por jogo de passes que resultaram diretamente em finalizações (ou remates ou golos), por jogador.|
| `progressive_passes_received` | Avançados                  | Média de passes recebidos com ganho de mais de 30 metros no campo por jogador e jogo.                  |
| `touches_in_box`              | Avançados                  | Média de quantas vezes cada jogador toca na bola dentro da grande área adversária, através de passes.  |
| `clearances`                  | Defesas                    | Média de afastamentos da bola da zona de perigo por jogo e por jogador.                                |
| `interceptions`               | Defesas                    | Média de interceções realizadas por jogo e por jogador.                                                |
| `tackles_won`                 | Defesas                    | Média de desarmes feitos por jogo e por jogador.                                                       |
| `aerial_duels_won`            | Defesas                    | Média de passes aéreos recebidos com sucesso, por jogo e por jogador.                                  |
| `pass_completion_pct`         | Defesas                    | Percentagem média de passes feitos com sucesso, por jogo e por jogador.                                |
| `long_passes_completed`       | Defesas                    | Média de passes com mais de 30m de distância feitos com sucesso, por jogo e por jogador.               |
| `fouls_committed`             | Defesas                    | Número médio de faltas cometidas, por jogo e por jogador.                                              |
| `pressures`                   | Defesas e Avançados        | Número médio de vezes em que o jogador pressionou o adversário, por jogo e por jogador.                |
| `recovery_time`               | Defesas e Avançados        | Tempo médio para recuperar a posse de bola após perdê-la (em segundos), por jogo e por jogador.        |
| `final_third_entries`         | Defesas e Avançados        | Número médio de vezes em que o jogador levou a bola até à zona de ataque, por jogo e por jogador       |
""")

defenders = pd.read_csv('data/defenders.csv')
attackers = pd.read_csv('data/attackers.csv')

competition_id_woman = 37
season_id_woman = 90
woman_matches = get_matches_df(competition_id_woman, season_id_woman)

competition_id_man = 2
season_id_man = 27
man_matches = get_matches_df(competition_id_man, season_id_man)

all_matches = pd.concat([woman_matches, man_matches])


defenders = defenders.merge(all_matches[['match_id', 'home_team.home_team_gender']], how='inner', on='match_id').rename(columns={'home_team.home_team_gender': 'gender'})
attackers = attackers.merge(all_matches[['match_id', 'home_team.home_team_gender']], how='inner', on='match_id').rename(columns={'home_team.home_team_gender': 'gender'})

avg_defenders = aggregate_player_metrics(defenders)
avg_attackers = aggregate_player_metrics(attackers)

st.markdown("<h2 style='text-align: center; color: white;'>Resumo de Jogadores por Perfil</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div style="background-color:#F0F2F6; padding:10px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.05);">
        <h3 style="color:#333;">🛡️ Defesas</h3>
        <p style="font-size:30px; margin:0; color:#0072C6;"><strong>{len(avg_defenders)}</strong></p>
        <p style="color:#555;">Jogadores considerados defensores</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background-color:#F0F2F6; padding: 10px; border-radius:10px; text-align:center; box-shadow:2px 2px 5px rgba(0,0,0,0.05);">
        <h3 style="color:#333;">⚔️ Avançados</h3>
        <p style="font-size:30px; margin:0; color:#D6336C;"><strong>{len(avg_attackers)}</strong></p>
        <p style="color:#555;">Jogadores considerados atacantes</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: white;'>Exploração de dados</h2>", unsafe_allow_html=True)
st.write(
    """
        Após selecionar estas métricas, criei uma matriz de correlação para compreender as relações entre as variáveis 
        e identificar possíveis padrões ou redundâncias. Além disso, utilizei histogramas para visualizar a distribuição de 
        cada variável, o que ajudou a ter uma ideia mais clara da dispersão e comportamento nos dados.
    """
)

st.markdown("<h3 style='text-align: center; color: white;'>Correlação entre variávies</h3>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🛡️ Defesas", "⚔️ Avançados"])

with tab1:
    st.plotly_chart(plot_correlation_heatmap(rename_for_display(avg_defenders), "Correlação Variáveis - Defesas"), use_container_width=True)

with tab2:
    st.plotly_chart(plot_correlation_heatmap(rename_for_display(avg_attackers), "Correlação Variáveis - Avançados"), use_container_width=True)

defender_histograms = plot_metric_histograms(avg_defenders)
attacker_histograms = plot_metric_histograms(avg_attackers)

st.markdown("<h3 style='text-align: center; color: white;'>Distribuição das variáveis</h3>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["🛡️ Defesas", "⚔️ Avançados"])
with tab1:
    st.markdown("<h4 style='text-align: center; color: white;'>Distribuições das Métricas - Defesas</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for i, (label, fig) in enumerate(defender_histograms):
        (col1 if i % 2 == 0 else col2).plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<h4 style='text-align: center; color: white;'>Distribuições das Métricas - Avançados</h4>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    for i, (label, fig) in enumerate(attacker_histograms):
        (col3 if i % 2 == 0 else col4).plotly_chart(fig, use_container_width=True)


st.markdown("<h2 style='text-align: center; color: white;'>Clustering</h2>", unsafe_allow_html=True)
st.write(
    """
    Criadas as variáveis e feita uma pequena análise exploratória dos dados que não mostrou nada fora do normal, chegou finalmente a hora
    de criar os perfis dos jogadores. 

    Optei por utilizar técnicas de clustering, com o objetivo de agrupar jogadores com características semelhantes com base nas métricas selecionadas. 
    Para isso, escolhi o algoritmo K-means maioritariamente devido à sua simplicidade e ampla utilização em análise exploratória de dados. 
    O K-means é especialmente eficaz quando queremos encontrar grupos esféricos em dados com características numéricas, como é o caso das variáveis
    criadas (variáveis de contagem e médias, especialmente quando padronizadas, como é o nosso caso, normalmente têm uma estrutura parecida 
    à esférica).

    Como métrica para avaliar a qualidade dos clusters, utilizei o silhouette score, que mede o quão semelhantes os objetos estão dentro do seu 
    cluster em comparação com os outros. Esta métrica oferece uma interpretação intuitiva da coesão e separação dos grupos. Optei por esta métrica
    em vez de outras de distância porque o silhouette score permite avaliar a estrutura global do cluster, sem depender exclusivamente de uma 
    métrica de distância específica, além de facilitar a comparação entre diferentes configurações.

    Para garantir que todas as variáveis contribuíam de forma equilibrada para o modelo, especialmente porque tinha percentagens no dataset, 
    utilizei o StandardScaler (que transforma cada variável numa distribuição com média 0 e desvio padrão 1) para standarizar os dados. Este passo
    é essencial no K Means, porque é um algoritmo que depdende exclusivamente de distâncias e, por isso, acaba por ser muito sensível a escalas.
    Para perceber qual o número ótimo de clusters, apliquei o Elbow Method, que avalia a inércia (soma das distâncias quadradas entre os pontos
    e o centroide do cluster). À medida que o número de clusters aumenta, a inércia diminui, mas tende a estabilizar a partir de certo ponto, 
    no "cotovelo" da curva. No nosso caso, esse valor foi 4 (o gráfico está disponível no notebook).

    Inicialmente, apliquei o K-means diretamente aos dados originais scaled, o que permitiu obter uma baseline para o processo. 
    Depois usei PCA (Principal Components Analysis) para reduzir a dimensionalidade dos dados antes de aplicar o K-means. Esta redução permitiu 
    eliminar redundâncias e ruído, o que melhorou a qualidade dos clusters. Este aumento de qualidade refletiu-se no silhouette score, que passou
    de 0,22 para 0,42. Apesar de o silhouette score não ser muito elevado, estes resultados são expectáveis, já que de qualquer das maneiras as 
    características dos jogadores acabam por se sobrepor, especialmente se tivermos em conta que estamos a considerar jogadores "parecidos" em termos
    de posição no campo.

    Ainda foram feitas umas experiências baseline com DBSCAN, que é bom para detetar outliers e clusters com formas não esféricas, e com o Gaussian
    Mixture, que permite fazer um clustering probabilístico, mas como os resultados foram piores, acabei por não seguir por aqui (experiências estão
    no notebook)

    Por fim, para uma visualização mais intuitiva e representativa da distribuição dos clusters no espaço multidimensional, utilizei UMAP 
    (Uniform Manifold Approximation and Projection). Este algoritmo faz uma redução dimensional não linear, mas é capaz de preservar tanto a estrutura
    local dos dados, quanto a global, e por isso é uma boa forma de ver a representação dos perfis dos jogadores.

    Abaixo, encontram-se divididos em dois tabs os resultados obtidos no clustering para o perfil de defesas e para o perfil de avançados.
    Estão também algumas visualizações das características dos clusters, assim como uma descrição final do que cada um representa.
    """
)

tab1, tab2 = st.tabs(["🛡️ Defesas", "⚔️ Avançados"])

with tab1:
    st.markdown("<h3 style='text-align: center; color: white;'>Perfis de Defesas</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>Representação Visual dos Perfis de Defesas</h4>", unsafe_allow_html=True)

    clustered_defenders, kmeans_defenders, X_pca, X_scaled = run_clustering_plotly(rename_for_display(avg_defenders), pca_comp=2, n_clusters=4, role_name="Defenders")
    fig = plot_umap_interactive(clustered_defenders, X_pca, title="UMAP de jogadores Defesas")

    st.plotly_chart(fig, use_container_width=True)
    st.write(
        """
            O UMAP mostra uma distribuição mais dispersa entre os diferentes clusters, o que sugere que os jogadores defensivos têm
            perfis estatísticos mais variados e acabam por se agrupar em diferentes especializações defensivas.
        """
    )

    st.markdown("<h4 style='text-align: center; color: white;'>Caracterização dos Clusters</h4>", unsafe_allow_html=True)

    if 'player_name' in clustered_defenders.columns:
        def_feat_cluster = clustered_defenders.merge(avg_defenders, on=['player_name', 'team', 'role', 'gender'], how='inner').drop(columns=['PCA1', 'PCA2'])
        features = def_feat_cluster.drop(columns=['player_name', 'team', 'role', 'gender', 'Cluster']).columns
    else:
        def_feat_cluster = clustered_defenders.merge(rename_for_display(avg_defenders), on=["Jogador", "Equipa", "Posição", "Género"], how='inner').drop(columns=['PCA1', 'PCA2'])
        features = def_feat_cluster.drop(columns=["Jogador", "Equipa", "Posição", "Género", "Cluster"]).columns

    fig = plot_radar_chart(def_feat_cluster, features)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Principais Variáveis</h5>", unsafe_allow_html=True)

    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df['cluster'] = def_feat_cluster['Cluster'].to_list()
    clusters = sorted(scaled_df['cluster'].unique())

    for i in range(0, len(clusters), 2):
        cols = st.columns(2)
        for j, cluster_id in enumerate(clusters[i:i+2]):
            cluster_data = scaled_df[scaled_df['cluster'] == cluster_id][features]
            mean_values = cluster_data.mean().sort_values(ascending=False)
            top_attrs = mean_values.head(3).index.tolist()
            bottom_attrs = mean_values.tail(3).index.tolist()

            with cols[j]:
                st.markdown(f"""
                <div style="
                    background-color:#F0F2F6; 
                    padding:15px; 
                    border-radius:10px; 
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
                    min-height:250px;
                ">
                    <h4 style="color:#333; text-align:center;">Cluster {cluster_id}</h4>
                    <h6 style="color:#0072C6;">⬆️ Métricas mais fortes</h6>
                    <ul style="color:#0072C6;">
                        {''.join([f'<li>{attr}</li>' for attr in top_attrs])}
                    </ul>
                    <h6 style="color:#D6336C;">⬇️ Métricas mais fracas</h6>
                    <ul style="color:#D6336C;">
                        {''.join([f'<li>{attr}</li>' for attr in bottom_attrs])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        if i + 2 < len(clusters):
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Tamanho dos Clusters</h5>", unsafe_allow_html=True)
    fig_bar = plot_size(def_feat_cluster)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Distribuição do Género dos Clusters</h5>", unsafe_allow_html=True)
    fig = plot_gender_distribution(def_feat_cluster, gender_col='Género')
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """
            A análise da distribuição de género por cluster mostra um padrão interessante que contraria a minha expectativa inicial de 
            que era possível que se criassem clusters completos com um só género. O cluster 1, por exemplo tem uma ligeira predominância feminina (55.3%),
            mas depois o cluster 3 já tem uma predominância masculina (72.6%), o que mostra que, ao contrário da nossa intuição, as características de
            jogo de homens e mulheres e os padrões de desempenho que têm no futebol são suficientemente complexos para fazer com que jogadores de ambos os
            géneros possam partilhar perfis estatísticos similares dentro dos mesmos clusters
        """
    )

    st.markdown("<h5 style='text-align: center; color: white;'>Notas e Interpretação dos Clusters criados</h5>", unsafe_allow_html=True)

    st.markdown(
        "Este tab mostra uma análise detalhada dos clusters táticos dos defesas, "
        "com interpretações para facilitar a compreensão dos perfis identificados."
    )

    # Cluster 0
    with st.expander("**Cluster 0: Defensores Modernos e Ativos (provavelmente laterais ou alas)**"):
        st.markdown("""
        | Métrica                     | Valor     | Comentário                                        |
        | --------------------------- | --------- | ------------------------------------------------- |
        | % de Passes Completos       | 73,5%     | Boa precisão de passe.                            |
        | Tempo de Recuperação        | 40s       | Ativos na recuperação da posse de bola.           |
        | **Pressões**                | **12,9**  | Muito alto - pressionam intensamente.             |
        | **Entradas na Zona de Ataque** | **9,2**| Avançam com frequência ao ataque.                 |
        | Passes Longos               | 3,8       | Participação razoável na construção ofensiva.     |
        | Cortes e Interceptações     | 3,1 / 1,6 | Ação defensiva moderada.                          |
        | **Desarmes Vencidos**       | 0,59      | Contribuem defensivamente                         |

        **Interpretação**:

        - Provavelmente são **laterais ou alas** em equipas com estilo de pressão alta.
        - Jogadores **agressivos**, que participam no ataque e ajudam na transição.
        - Exemplo de perfil: **João Cancelo, Kyle Walker, Alphonso Davies**.
        """)

    # Cluster 1
    with st.expander("**Cluster 1: Defensores Recuados e Passivos (provavelmente centrais ou laterais conservadores)**"):
        st.markdown("""
        | Métrica                     | Valor     | Comentário                     |
        | --------------------------- | --------- | ------------------------------ |
        | % de Passes Completos       | 70,8%     | Um pouco abaixo do Cluster 0.  |
        | Tempo de Recuperação        | 44,6s     | Transições mais lentas.        |
        | **Pressões**                | 6,9       | Baixo envolvimento em pressão. |
        | **Entradas na Zona de Ataque** | 4,1    | Pouco avançam no campo.        |
        | Ações defensivas            | Moderadas | Participação defensiva básica. |
        | Desarmes                    | 0,25      | Poucos duelos diretos.         |

        **Interpretação**:

        - Provavelmente são **centrais** ou **laterais mais recuados**, com pouca participação ativa.
        - Atuam em linhas mais baixas, raramente sobem ao ataque.
        - Exemplo de perfil: **Harry Maguire, Ben Mee, Yerry Mina**.
        """)

    # Cluster 2
    with st.expander("**Cluster 2: Defensores Periféricos ou Reservas (baixa participação)**"):
        st.markdown("""
        | Métrica                      | Valor    | Comentário                                |
        | ---------------------------- | -------- | ----------------------------------------- |
        | % de Passes Completos        | 60,1%    | Muito baixo - possível limitação técnica. |
        | Tempo de Recuperação         | **6,9s** | Muito curto - talvez de lances isolados.  |
        | **Todas as métricas baixas** |          | Participação mínima em qualquer área.     |

        **Interpretação**:

        - Provavelmente são **suplentes, reservas ou jogadores com poucos minutos de jogo**.
        - Podem também ser **jovens** ou jogadores de equipas com pouca posse de bola.
        - Os dados são insuficientes para definir um papel tático claro.
        """)

    # Cluster 3
    with st.expander("**Cluster 3: Defesas Centrais Construtores (centrais agressivos ou com boa saída de bola)**"):
        st.markdown("""
        | Métrica                   | Valor     | Comentário                                      |
        | ------------------------- | --------- | ------------------------------------------------|
        | **% de Passes Completos** | **78,4%** | Mais alto - seguros com a bola.                 |
        | Tempo de Recuperação      | 29,7s     | Recuperam rapidamente a posse de bola.          |
        | Pressões                  | 8,1       | Ativos, mas equilibrados.                       |
        | **Passes Longos**         | **6,6**   | Participam na inversão e construção de jogadas. |
        | **Cortes**                | **5,9**   | Muito presentes na defesa.                      |
        | **Interceptações**        | 1,35      | Atentos e bem posicionados.                     |
        | Duelos Aéreos             | 1,35      | Fortes no jogo aéreo.                           |

        **Interpretação**:

        - Provavelmente são **defesas centrais com papel de liderança**.
        - Bons com a bola, sólidos defensivamente e confiáveis na construção.
        - Exemplo de perfil: **Virgil van Dijk, Rúben Dias, Aymeric Laporte**.
        """)



with tab2:
    st.markdown("<h3 style='text-align: center; color: white;'>Perfis de Avançados</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>Representação Visual dos Perfis de Avançados</h4>", unsafe_allow_html=True)

    clustered_attackers, kmeans_attackers, X_pca, X_scaled = run_clustering_plotly(rename_for_display(avg_attackers), pca_comp=2, n_clusters=4, role_name="Defenders")
    fig = plot_umap_interactive(clustered_attackers, X_pca, title="UMAP de jogadores Avançados")

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """
            Este UMAP tem uma estrutura mais concentrada que o dos Defesas, o que indica que os jogadores ofensivos tendem a ter características 
            mais homogéneas e com apenas pequenas alterações entre os diferentes tipos de perfis de avançados.
        """
    )

    st.markdown("<h4 style='text-align: center; color: white;'>Caracterização dos Clusters</h4>", unsafe_allow_html=True)

    if 'player_name' in clustered_attackers.columns:
        def_feat_cluster = clustered_attackers.merge(avg_attackers, on=['player_name', 'team', 'role', 'gender'], how='inner').drop(columns=['PCA1', 'PCA2'])
        features = def_feat_cluster.drop(columns=['player_name', 'team', 'role', 'gender', 'Cluster']).columns
    else:
        def_feat_cluster = clustered_attackers.merge(rename_for_display(avg_attackers), on=["Jogador", "Equipa", "Posição", "Género"], how='inner').drop(columns=['PCA1', 'PCA2'])
        features = def_feat_cluster.drop(columns=["Jogador", "Equipa", "Posição", "Género", "Cluster"]).columns

    fig = plot_radar_chart(def_feat_cluster, features)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Principais Variáveis</h5>", unsafe_allow_html=True)

    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df['cluster'] = def_feat_cluster['Cluster'].to_list()
    clusters = sorted(scaled_df['cluster'].unique())

    for i in range(0, len(clusters), 2):
        cols = st.columns(2)
        for j, cluster_id in enumerate(clusters[i:i+2]):
            cluster_data = scaled_df[scaled_df['cluster'] == cluster_id][features]
            mean_values = cluster_data.mean().sort_values(ascending=False)
            top_attrs = mean_values.head(3).index.tolist()
            bottom_attrs = mean_values.tail(3).index.tolist()

            with cols[j]:
                st.markdown(f"""
                <div style="
                    background-color:#F0F2F6; 
                    padding:15px; 
                    border-radius:10px; 
                    box-shadow: 3px 3px 10px rgba(0,0,0,0.1);
                    min-height:250px;
                ">
                    <h4 style="color:#333; text-align:center;">Cluster {cluster_id}</h4>
                    <h6 style="color:#0072C6;">⬆️ Métricas mais fortes</h6>
                    <ul style="color:#0072C6;">
                        {''.join([f'<li>{attr}</li>' for attr in top_attrs])}
                    </ul>
                    <h6 style="color:#D6336C;">⬇️ Métricas mais fracas</h6>
                    <ul style="color:#D6336C;">
                        {''.join([f'<li>{attr}</li>' for attr in bottom_attrs])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        if i + 2 < len(clusters):
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Tamanho dos Clusters</h5>", unsafe_allow_html=True)
    fig_bar = plot_size(def_feat_cluster)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<h5 style='text-align: center; color: white;'>Distribuição do Género dos Clusters</h5>", unsafe_allow_html=True)
    fig = plot_gender_distribution(def_feat_cluster, gender_col='Género')
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        """
            Esta distribuição comprova o que já tínhamos visto também nos Defesas. A distribuição de género por cluster contraria a 
            expectativa de segregação completa, já que todos os clusters têm uma mistura de jogadores masculinos e femininos, 
            que vai desde a predominância masculina de 70% nos clusters 0 e 2 até à distribuição perfeita do cluster 3.
        """
    )

    st.markdown("<h5 style='text-align: center; color: white;'>Notas e Interpretação dos Clusters criados</h5>", unsafe_allow_html=True)

    st.markdown(
        "Este tab detalha os clusters dos avançados, e ajuda a interpretar os perfis ofensivos encontrados."
    )

    with st.expander('**Cluster 0: "Avançados de Apoio com Pressão Alta"**'):
        st.markdown("""
        | Métrica                           | Valor        | Comentário                                                     |
        | --------------------------------- | ------------ | -------------------------------------------------------------- |
        | **Pressões**                      | **12,6**     | Muito alto - pressionam bastante desde a frente.               |
        | **Tempo de Recuperação**          | 30,4s        | Equilibrado - costumam recuperar a posse.                      |
        | **Passes Progressivos Recebidos** | 5,47         | Participam razoavelmente na construção.                        |
        | **Toques na Área**                | 3,51         | Moderado - aparecem na área com alguma frequência.             |
        | **Remates / xG**                  | 1,53 / 0,18  | Contribuem no ataque, mas não são os principais finalizadores. |
        | **Passes para Finalização**       | 1,16         | Presença criativa.                                             |

        **Interpretação**:

        * Jogadores **criativos e trabalhadores**, provavelmente **extremos ou segundos avançados**.
        * Pressionam, combinam e apoiam, mas não são quem finaliza a maioria das jogadas.
        * Exemplo de perfil: **Bukayo Saka, Bernardo Silva, Ángel Di María**.
        """)

    with st.expander('**Cluster 1: "Avançados de Baixo Impacto" (baixa participação ou suplentes)**'):
        st.markdown("""
        | Métrica                                         | Valor       | Comentário |
        | ----------------------------------------------- | ----------- | ---------- |
        | **Todos os valores são baixos**, especialmente: |             |            |
        | Passes Progressivos Recebidos                   | 1,37        |            |
        | Toques na Área                                  | 0,75        |            |
        | Remates / xG                                    | 0,28 / 0,02 |            |
        | Passes para Finalização                         | 0,18        |            |

        **Interpretação**:

        * Provavelmente são **suplentes, jogadores rotativos ou com poucos minutos**.
        * Também pode incluir **extremos em equipas defensivas** (ex.: equipas que "estacionam o autocarro").
        * xG e passes para finalização baixos indicam pouca participação ofensiva.
        """)

    with st.expander('**Cluster 2: "Avançados Centrais Finalizadores"**'):
        st.markdown("""
        | Métrica                           | Valor    | Comentário                                    |
        | --------------------------------- | -------- | --------------------------------------------- |
        | **xG**                            | **0,36** | Maior valor - finalizam oportunidades claras. |
        | **Remates**                       | **2,47** | Principal ameaça ofensiva.                    |
        | **Toques na Área**                | **7,55** | Presença muito ativa dentro da área.          |
        | **Passes Progressivos Recebidos** | **13,4** | Muito ativos no ataque.                       |
        | Tempo de Recuperação              | 49,5s    | Jogam mais adiantados, demoram a recuperar.   |
        | Pressões                          | 12,1     | Ainda assim pressionam bastante.              |

        **Interpretação**:

        * Avançados-centro ou pontas **com papel principal na finalização**.
        * Estão frequentemente na área, recebem muitas bolas em zonas ofensivas.
        * Exemplo de perfil: **Erling Haaland, Robert Lewandowski, Darwin Núñez**.

        """)

    with st.expander('**Cluster 3: "Extremos Pressionantes ou Avançados Híbridos"**'):
        st.markdown("""
        | Métrica                       | Valor       | Comentário                                      |
        | ----------------------------- | ----------- | ----------------------------------------------- |
        | **Pressões**                  | 11,8        | Alto — 1ª linha de pressão.                     |
        | Passes Progressivos Recebidos | 4,26        | Participação razoável, mas abaixo do Cluster 2. |
        | Toques na Área                | 1,69        | Menor presença na área.                         |
        | Remates / xG                  | 0,67 / 0,05 | Ameaça reduzida em termos de finalização.       |
        | **Tempo de Recuperação**      | **54,3s**   | Alto — podem ter papel mais recuado.            |

        **Interpretação**:

        * Provavelmente são **extremos ou atacantes de apoio** que pressionam, mas não finalizam muito.
        * Jogam mais recuados ou servem para **dar apoio aos avançados centrais**.
        * Exemplo de perfil: **Riyad Mahrez, Ferran Torres, Cody Gakpo (quando joga nas alas)**
        """)

