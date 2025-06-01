import requests
import pandas as pd

def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
    return data

def process_events(match_id: int):
    url_events = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{match_id}.json"
    data = get_data(url_events)
    events = pd.json_normalize(data)

    events['time_seconds'] = events.apply(
        lambda row: int(row['minute']) * 60 + int(row['second']), axis=1
    )

    events['time_bin'] = (events['time_seconds'] // 300).astype(int)
    events = events[events["time_seconds"] != 0][['index', 'possession', 'possession_team.name', 'team.name', 'type.name', 'carry.end_location', 'time_seconds', 'time_bin']]

    return events

def get_recovery(match_id: int):
    events = process_events(match_id)

    recovery_events = []

    last_team = events.iloc[0]['possession_team.name']
    last_time = events.iloc[0]['time_seconds']

    for i in range(1, len(events)):
        curr = events.iloc[i]
        curr_team = curr['possession_team.name']
        curr_time = curr['time_seconds']

        if curr_team != last_team:
            recovery_time = curr_time - last_time

            recovery_events.append({
                'lost_by': last_team,
                'recovered_by': curr_team,
                'recovery_time': recovery_time,
                'time_seconds': curr_time,
                'time_bin': curr['time_bin']
            })

            last_team = curr_team
            last_time = curr_time

    change_possession = pd.DataFrame(recovery_events)

    recovery_df = change_possession[change_possession['recovery_time'] > 0]
    recovery_df['minute'] = (recovery_df['time_seconds'] // 60).astype(int)

    return recovery_df

def get_danger_zones(match_id: int):
    events = process_events(match_id)

    events_dangerous = events[
        (events['type.name'] == 'Carry')
        & (events['carry.end_location'].notnull())
    ]
    events_dangerous[['x', 'y']] = events_dangerous['carry.end_location'].apply(pd.Series)

    events_dangerous['final_third_entry'] = events_dangerous['x'] >= 80

    events_dangerous['penalty_area_entry'] = (events_dangerous['x'] >= 102) & (events_dangerous['y'].between(18, 62))

    events_dangerous['zone_entry'] = events_dangerous['final_third_entry'] | events_dangerous['penalty_area_entry']

    entries = events_dangerous[events_dangerous['zone_entry']].copy()
    entries['minute'] = (entries['time_seconds'] // 60).astype(int)

    final_third = entries[entries['final_third_entry']].copy()
    final_third['zone'] = 'Final Third'

    penalty_area = entries[entries['penalty_area_entry']].copy()
    penalty_area['zone'] = 'Penalty Area'

    entries = pd.concat([final_third, penalty_area])
    entries['zone_pt'] = entries['zone'].replace({
        'Final Third': 'Zona de Ataque',
        'Penalty Area': 'Grande Área'
    })
    entries['zone_team'] = entries['team.name'] + ' - ' + entries['zone_pt']

    entry_counts = entries.groupby(['minute', 'zone_team', 'team.name']).size().reset_index(name='entries')

    return events_dangerous, entry_counts

def get_two_metrics(recovery_df: pd.DataFrame, events_danger: pd.DataFrame):
    # Recuperação de posse de bola
    recovery_by_minute = recovery_df.groupby(['recovered_by', 'minute'])['recovery_time'].mean().reset_index()
    recovery_by_minute.rename(columns={'recovered_by': 'team', 'minute': 'minute_match'}, inplace=True)

    # Condução da bola até à zona de ataque
    events_danger['dangerous_entries'] = events_danger['final_third_entry'] + events_danger['penalty_area_entry']
    events_danger['minute_match'] = (events_danger['time_seconds'] // 60).astype(int)
    danger_by_minute = events_danger.groupby(['team.name', 'minute_match'])['dangerous_entries'].sum().reset_index()
    danger_by_minute.rename(columns={'team.name': 'team', 'zone_entry': 'dangerous_entries'}, inplace=True)

    # Juntar as duas métricas
    combined_df = pd.merge(recovery_by_minute, danger_by_minute, on=['team', 'minute_match'], how='inner')

    return combined_df
