import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap.umap_ as umap


from utils.individual_match import get_data

def get_matches_df(competition_id: int, season_id: int):
    url_matches = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{competition_id}/{season_id}.json"
    data = get_data(url_matches)
    matches = pd.json_normalize(data)
    return matches

def aggregate_player_metrics(df):
    id_cols = ['player_name', 'team', 'role', 'gender']

    df = df.drop('match_id', axis=1)

    grouped = df.groupby('player_name')
    meta = grouped[id_cols[1:]].agg(lambda x: x.mode()[0])
    metrics = grouped.mean(numeric_only=True)

    return meta.join(metrics, on='player_name').reset_index()

def plot_correlation_heatmap(df, title):
    corr_matrix = df.select_dtypes(include=np.number).corr()

    mask = np.tril(np.ones(corr_matrix.shape)).astype(bool)
    corr_matrix_masked = corr_matrix.where(mask)

    # Gera o heatmap
    fig = px.imshow(
        corr_matrix_masked,
        text_auto=".2f",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1,
        aspect="auto",
        labels=dict(color='Correlação')
    )
    fig.update_layout(title=title)
    return fig

column_labels_pt = {
    'xg': 'xG',
    'shots': 'Remates',
    'key_passes': 'Passes para Finalizações',
    'progressive_passes_received': 'Passes Progressivos Recebidos',
    'pressures': 'Pressões',
    'touches_in_box': 'Toques na Grande Área',
    
    'clearances': 'Afastamentos de zonas perigosas',
    'interceptions': 'Interceções',
    'tackles_won': 'Desarmes Vencidos',
    'aerial_duels_won': 'Duelos Aéreos Vencidos',
    'pass_completion_pct': 'Percentagem de Passes Bem Sucedidos',
    'long_passes_completed': 'Passes Longos Bem Sucedidos',
    'fouls_committed': 'Faltas Cometidas',
    'recovery_time': 'Tempo de Recuperação de Bola',
    'final_third_entries': 'Entradas na Zona de Ataque',
    'penalty_area_entries': 'Entradas na Grande Área',

    'match_id': 'Jogo',
    'player_name': 'Jogador',
    'team': 'Equipa',
    'role': 'Posição',
    'gender': 'Género'
}

def rename_for_display(df, col_map=column_labels_pt):
    return df.rename(columns=col_map)

def plot_metric_histograms(df, title=None, labels_map=column_labels_pt):
    exclude = {'player_name', 'team', 'role', 'gender'}
    numeric_cols = [col for col in df.columns if col not in exclude]
    plots = []
    
    for col in numeric_cols:
        label = labels_map.get(col, col) if labels_map else col
        fig = px.histogram(df, x=col, nbins=10, title=label)
        fig.update_layout(
            xaxis_title=label,
            yaxis_title='Percentagem' if label == 'Percentagem de Passes Bem Sucedidos' else 'Nº de vezes',
            margin=dict(l=10, r=10, t=30, b=30),
            height=300
        )
        plots.append((label, fig))
    
    return plots

def run_clustering_plotly(df, pca_comp=2, n_clusters=4, role_name="Attackers", labels_map=column_labels_pt):
    if labels_map:
        players = df['Jogador'].values
        teams = df['Equipa'].values if 'Equipa' in df.columns else ['Unknown'] * len(df)
        roles = df['Posição'].values if 'Posição' in df.columns else ['Unknown'] * len(df)
        genders = df['Género'].values if 'Género' in df.columns else ['Unknown'] * len(df)
        df_numeric = df.drop(columns=['Jogador', 'Equipa', 'Posição', "Género"], errors='ignore')

    else:
        players = df['player_name'].values
        teams = df['team'].values if 'team' in df.columns else ['Unknown'] * len(df)
        roles = df['role'].values if 'role' in df.columns else ['Unknown'] * len(df)
        genders = df['gender'].values if 'gender' in df.columns else ['Unknown'] * len(df)
        df_numeric = df.drop(columns=['player_name', 'team', 'role', "gender"], errors='ignore')

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_numeric)

    pca = PCA(n_components=pca_comp)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    if labels_map:
        cluster_df = pd.DataFrame({
            'Jogador': players,
            'Equipa': teams,
            'Posição': roles,
            'Género':genders,
            'Cluster': labels
        })
    else:
        cluster_df = pd.DataFrame({
            'player_name': players,
            'team': teams,
            'role': roles,
            'gender':genders,
            'Cluster': labels
        })
    for i in range(pca_comp):
        cluster_df[f'PCA{i+1}'] = X_pca[:, i]

    return cluster_df, kmeans, X_pca, X_scaled

def plot_umap_interactive(df, X_pca, title="UMAP"):
    for i in ["player_name", "team", "role", "gender", "Jogador", "Equipa", "Posição", "Género"]:
        if i in df.columns:
            features = df.drop(columns=i, errors='ignore')

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X_pca)

    if 'player_name' in df.columns:
        plot_df = df[["player_name", "team", "role", "gender"]].copy()
    else:
        plot_df = df[["Jogador", "Equipa", "Posição", "Género"]].copy()
    plot_df["UMAP1"] = embedding[:, 0]
    plot_df["UMAP2"] = embedding[:, 1]
    plot_df["cluster"] = df['Cluster']

    fig = px.scatter(
        plot_df, x="UMAP1", y="UMAP2", color=plot_df.cluster, labels={'color': 'cluster'},
        hover_data=["player_name", "team", "gender"] if 'player_name' in df.columns else ["Jogador", "Equipa", "Posição", "Género"],
        title=title
    )
    fig.update_layout(
        width=800, 
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white') 
    )
    return fig

def plot_radar_chart(df,features):
    minmax_scaler = MinMaxScaler()
    normalized_values = minmax_scaler.fit_transform(df[features])
    normalized_df = pd.DataFrame(normalized_values, columns=features)
    normalized_df['cluster'] = df['Cluster']

    mean_normalized = normalized_df.groupby('cluster')[features].mean()
    mean_raw = df.groupby('Cluster')[features].mean()


    feature_labels = features

    fig = go.Figure()

    for cluster_id in mean_normalized.index:
        norm_values = mean_normalized.loc[cluster_id].tolist()
        raw_values = mean_raw.loc[cluster_id].tolist()

        r_values = norm_values
        raw_vals = raw_values
        theta_vals = features

        hover_text = [f"{feature}: {value:.2f}" for feature, value in zip(features, raw_values)]
        hover_text += [hover_text[0]]  # close the loop

        summary_text = "<br>".join([f"{feat}: {val:.2f}" for feat, val in zip(features, raw_values)])

        fig.add_trace(go.Scatterpolar(
            theta=feature_labels,  # Repeat for closure
            r=r_values,
            fill='toself',
            name=f"Cluster {cluster_id}",
            customdata=np.array(raw_vals).reshape(-1, 1),
            hoverinfo="text",
            text=hover_text,
            hovertemplate="<b>%{theta}</b><br>Valor: %{text}<extra>" + summary_text + "</extra>"
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            bgcolor='rgba(0,0,0,0)'
        ),
        title="Perfis de Jogadores (Normalizados, valores reais na hover action)",
        showlegend=True,
        height=650,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
  )
    return fig

def plot_size(df):
    counts = df.groupby('Cluster')['Jogador'].count().reset_index()
    counts.columns = ['cluster', 'counts']

    fig_bar = px.bar(
        counts,
        x='cluster',
        y='counts',
        text='counts',
        title=f"Nº de jogadores em cada cluster",
        labels={'counts': 'Nº de jogadores', 'cluster': 'Cluster'}
    )

    fig_bar.update_traces(
        hovertemplate='Cluster: %{x}<br>Nº jogadores: %{y}<extra></extra>',
        textposition='outside'
    )

    fig_bar.update_layout(
        yaxis_title="Nº jogadores",
        xaxis_title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            dtick=1
        )
    )
    return fig_bar

def plot_gender_distribution(df, cluster_col='Cluster', gender_col='gender'):
    counts = df.groupby([cluster_col, gender_col]).size().reset_index(name='count')

    totals = counts.groupby(cluster_col)['count'].transform('sum')
    counts['Percentagem'] = counts['count'] / totals * 100

    fig = px.bar(
        counts,
        x=cluster_col,
        y='Percentagem',
        color=gender_col,
        text=counts['Percentagem'].apply(lambda x: f'{x:.1f}%'),
        title="Percentagem de cada Género por Cluster",
        labels={cluster_col: "Cluster", 'percentage': "Percentagem (%)", gender_col: "Género"}
    )
    fig.update_layout(barmode='stack', yaxis=dict(ticksuffix='%'))
    fig.update_traces(textposition='inside')
    return fig
