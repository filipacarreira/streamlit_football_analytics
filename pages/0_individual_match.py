import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from utils.individual_match import get_recovery, get_danger_zones, get_two_metrics

st.markdown("<h1 style='text-align: center; color: white;'>Chelsea FCW - Reading WFC (09/05/2019), para a FA Women\'s Super League, época 2020/2021</h1>", unsafe_allow_html=True)

st.write(
    '''
        A presente análise usa dados de um jogo de uma liga de futebol feminina, concretamente o jogo da última 
        jornada do campeonato FA Women's Super League, na época 2020/2021.

        A escolha do jogo em concreto a usar foi feita tendo em conta os resultados finais dos jogos da última jornada.
        Havia um que se destacava claramente por ter um resultado mais "extremo": 5-0. Chelsea FCW - Reading WFC, no dia 9 de maio de 2019.
        Usei este jogo porque achei que talvez as diferenças entre estas duas equipas fossem mais evidentes.

        Vamos ver o que os dados nos dizem...
    '''
)

slide1 = Image.open("images/chelsea.webp")
st.image(slide1, use_container_width=True)

st.markdown("<h2 style='text-align: center; color: white;'>Métricas desenvolvidas</h2>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: white;'>Métrica 1: Tempo de Recuperação da Posse de Bola</h3>", unsafe_allow_html=True)

st.write(
    '''
        O objetivo do cálculo da variável era perceber como este tempo se alterava ao longo do jogo e se, no final do 
        jogo, este tempo aumentava devido ao cansaço das jogadoras, por exemplo.

        Para calcular o Tempo de Recuperação (em segundos) considerei todos os casos nos dados em que a possessão
        de bola mudava e calculei os segundos entre as posses de bola da mesma equipa. 

        Agregei os dados em intervalos de 5 minutos para ter valores mais consistentes e perceber como evoluiu o 
        tempo de recuperção ao longo do jogo.
    '''
)

match_id = 3775593
recovery_df = get_recovery(match_id)
team_avg = recovery_df.groupby('recovered_by')['recovery_time'].mean().to_dict()
bin_avg = recovery_df.groupby(['time_bin', 'recovered_by'])['recovery_time'].mean().reset_index()
bin_values_sorted = sorted(bin_avg['time_bin'].unique())
last_bin = max(bin_values_sorted)

fig = px.bar(
    bin_avg,
    x='time_bin',
    y='recovery_time',
    color='recovered_by',
    barmode='group',
    labels={'time_bin': 'Intervalo de 5 Minutos', 'recovery_time': 'Tempo Médio de Recuperação (s)', 'recovered_by': 'Equipa'},
    title='Tempo Médio de Recuperação a Cada 5 Minutos por Equipa'
)

# Add average lines per team
for team, avg in team_avg.items():
    fig.add_trace(go.Scatter(
        x=bin_avg['time_bin'].unique(),
        y=[avg] * len(bin_avg['time_bin'].unique()),
        mode='lines',
        name=f'{team} (Média)',
        line=dict(dash='dash'),
        hovertemplate=f'Recuperação média: {avg:.2f}s<br>Equipa: {team}',
        legendgroup=team,
        showlegend=True
    ))

    fig.add_annotation(
        xref='paper',
        x=1.01,
        y=avg,
        text=f"{team}: {avg:.2f}s",
        showarrow=False,
        font=dict(size=11),
        align='left',
        xanchor='left',
        yanchor='middle'
    )

# Update layout
fig.update_layout(
    xaxis_title='Intervalo de 5 Minutos',
    yaxis_title='Tempo Médio de Recuperação (s)',
    bargap=0.15,
    legend_title='Time',
    margin=dict(t=60, r=40, b=60, l=60),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white') 
)
fig.update_xaxes(showgrid=False, zeroline=False, showline=False)
fig.update_yaxes(showgrid=False, zeroline=False, showline=False)
st.plotly_chart(fig, use_container_width=True)

st.write(
    '''
        O gráfico mostra que o Chelsea FCW tem um tempo médio de recuperação da posse consistentemente mais baixo ao longo do
        jogo, com uma média geral de cerca de 36 segundos, o que pode indicar uma equipa mais agressiva na pressão pós-perda de bola.
        
        O Reading WFC, por sua vez, tem tempos de recuperação bem mais altos em vários momentos do jogo, com uma média geral de 54 segundos
        para recuperar a bola.

        Esta diferença era expectável, tendo em conta que o jogo teve um resultado extremamente desequilibrado.
    '''
)

st.markdown("<h3 style='text-align: center; color: white;'>Métrica 2: Conduções da bola para a zona de ataque</h3>", unsafe_allow_html=True)

st.write(
    '''
        O objetivo do cálculo desta métrica é perceber o quão eficientes as equipas são ao longo do jogo, além de perceber
        quantas oportunidades de perigo criam em cada minuto.

        Para perceber quantas vezes os jogadores levam a bola para a zona de ataque, considerei apenas um tipo de eventos: "carry" 
        (conduzir a bola). Depois usei a localização desses eventos para perceber em que zona do campo os jogadores estavam. 
        Criei dois eventos:

        - Entradas na zona de ataque (terceira "parte" do campo - zona de ataque: x > 80 na imagem abaixo)
        - Entradas na grande área (x >= 102 e 18 <= y <= 62)
    '''
)

slide1 = Image.open("images/coordinates.png")
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image(slide1, use_container_width=True)

events_danger, entry_counts = get_danger_zones(match_id)

entry_counts['hover_text'] = (
    'Minuto: ' + entry_counts['minute'].astype(str) + '<br>' +
    'Número de Entradas: ' + entry_counts['entries'].astype(str)
)

fig = px.line(
    entry_counts,
    x='minute',
    y='entries',
    color='zone_team',
    markers=True,
    hover_data='hover_text',
)

fig.update_layout(
    title='Conduções até o Terço Final e Grande Área por Minuto',
    xaxis_title='Minuto',
    yaxis_title='Número de Entradas',
    legend_title='Equipa e Zona',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

fig.update_traces(hovertemplate='%{customdata[0]}')

st.plotly_chart(fig, use_container_width=True)

st.write(
    '''
        O Chelsea FCW consegue levar a bola até à zona de ataque e grande área adversária consideravelmente
        mais vezes que o Reading WFC ao longo do jogo, sendo que há uma presença ofensiva constante da primeira equipa. 
        
        O Reading WFC, por sua vez, tem um número muito mais baixo de entradas em zonas de ataque, com picos esporádicos 
        e menor consistência. 
        
        A diferença entre as duas equipas evidencia que o Chelsea tem maior capacidade de construção de jogadas perigosas,
        o que se verificou no resultado do jogo: 5 golos de vantagem para o Chelsea
    '''
)

st.markdown("<h3 style='text-align: center; color: white;'>Comparação entre as duas métricas</h3>", unsafe_allow_html=True)

two_metrics = get_two_metrics(recovery_df,events_danger)

danger_min = -1
danger_max = two_metrics['dangerous_entries'].max() + 1
recovery_min = -1
recovery_max = two_metrics['recovery_time'].max() + 1

teams = two_metrics['team'].unique()
n_cols = 2
n_rows = -(-len(teams) // n_cols)

fig = sp.make_subplots(
    rows=n_rows,
    cols=n_cols,
    specs=[[{"secondary_y": True}] * n_cols] * n_rows,
    subplot_titles=teams
)

for idx, team in enumerate(teams):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    team_df = two_metrics[two_metrics['team'] == team]

    fig.add_trace(
        go.Scatter(x=team_df['minute_match'], y=team_df['dangerous_entries'],
                   name='Entradas perigosas', mode='lines+markers', line=dict(color='blue'),
                   customdata=team_df[['minute_match', 'team']],
            hovertemplate='Entradas perigosas = %{y}<br>Minutos de jogo = %{customdata[0]}<br>Equipa: %{customdata[1]}'),
        row=row, col=col, secondary_y=False
    )

    fig.add_trace(
        go.Scatter(x=team_df['minute_match'], y=team_df['recovery_time'],
                   name='Tempo Recuperação de Bola (s)', mode='lines+markers', line=dict(color='red'),
                   customdata=team_df[['minute_match', 'team']],
            hovertemplate='Tempo de Recuperação = %{y} segundos<br>Minutos de jogo = %{customdata[0]}<br>Equipa: %{customdata[1]}'),
        row=row, col=col, secondary_y=True
    )

    fig.update_yaxes(range=[danger_min, danger_max], row=row, col=col, secondary_y=False)
    fig.update_yaxes(range=[recovery_min, recovery_max], row=row, col=col, secondary_y=True)
    fig.update_xaxes(title_text="Minuto do jogo", row=row, col=col)


fig.update_layout(
    height=500 * n_rows,
    width=1450,          
    title_text="Entradas perigosas vs Tempo de Recuperação de Bola (s)",
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
)

for i in range(1, len(teams) + 1):
    fig.update_yaxes(title_text="Entradas perigosas", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1, secondary_y=False)
    fig.update_yaxes(title_text="Tempo de Recuperação (s)", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1, secondary_y=True)

st.plotly_chart(fig, use_container_width=True)

st.write(
    '''
        **Highlight 1:** O Chelsea FCW mantém uma pressão consistente na 2ª parte do jogo

        - Entre os minutos 40 e 80, o Chelsea tem vários picos de entradas perigosas, mantendo ao mesmo tempo um tempo de 
        recuperação de bola relativamente baixo.
        - O Chelsea tem uma estratégia eficaz de pressão pós-perda de bola (counter-pressing), o que também mantém a pressão ofensiva.
        - O baixo tempo de recuperação provavelmente é consequência de uma equipa eficiente na pressão alta — recuperam a bola rapidamente 
        e entram imediatamente na zona de ataque.

        **Highlight 2:** O Reading WFC tem recuperações de bola lentas, o que leva a oportunidades perdidas

        - O tempo de recuperação frequentemente ultrapassa os 100 segundos, especialmente entre os minutos 20 e 40, e as entradas perigosas 
        ficam baixas ou estáveis.
        - O Reading demora a recuperar a posse de bola e que, quando o faz, não consegue converter essa
        recuperação em ações ofensivas, já que não há um aumento relevante das entradas na zona de ataque.
        - O Reading parece ter dificuldades nas transições ou não ter uma estrutura suficiente para aproveitar a recuperação da posse de bola.
        Isso pode mostrar que existem problemas na construção ofensiva ou que a equipa toma uma postura defensiva mais recuada.
    '''
)

