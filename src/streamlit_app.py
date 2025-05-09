import streamlit as st
import pandas as pd
from joblib import load
from nba_api.stats.endpoints import ScoreboardV2, LeagueDashTeamStats
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import LeagueGameLog



def get_game_logs(season, type, game_date, home_team, away_team):
    print(f"season: {season} type: {type} game_date: {game_date} home_team: {home_team} away_team: {away_team}")
    df_games = (
        LeagueGameLog(
            season = season,
            season_type_all_star=type
        ).get_data_frames()[0]
    )
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    for _, r in df_games.iterrows():
        matchup = r['MATCHUP']
        team_1, vs_or, team_2 = matchup.split()
        if team_1 == home_team and team_2 == away_team and r['GAME_DATE'] == game_date:
            df1 = pd.DataFrame({
                'TEAM_ABBR': [team_1],
                'OPP_ABBR': [team_2],
                'is_home': [1],
                'WL': [r['WL']]
            })
        elif team_1 == away_team and team_2 == home_team and r['GAME_DATE'] == game_date:
            df2 = pd.DataFrame({
                'TEAM_ABBR': [team_1],
                'OPP_ABBR': [team_2],
                'is_home': [0],
                'WL': [r['WL']]
            })
    
    if df1.empty or df2.empty:
        return None
    df_combined = pd.concat([df1, df2], ignore_index=True)
    return df_combined


def check_team_playoffs(team): 
    df_team = (
        LeagueDashTeamStats (
            season= '2024-25',
            season_type_all_star='Playoffs'
        )
        .get_data_frames()[0]
    )

    teams_list = nba_teams.get_teams()
    for i in teams_list:
        if i['abbreviation'] == team:
            full_name = i['full_name']
            break


    for _, r in df_team.iterrows():
        if r['TEAM_NAME'] == full_name:
            return True
    
    return False

# Load trained model and saved state
def load_model_and_state():
    model = load('moneyline_model.joblib')
    team_elos = load('team_elos.joblib')
    last_mom5 = load('last_mom5.joblib')
    model_spread = load('spread_model.joblib')
    return model, team_elos, last_mom5, model_spread

@st.cache_data
# Cache prior season stats to avoid repeated API calls
def load_prior_season_stats():
    df = (
        LeagueDashTeamStats(
            season='2024-25',
            season_type_all_star='Regular Season'
        )
        .get_data_frames()[0]
    )
    # Keep only required columns
    return df[['TEAM_ID','W_PCT','PTS_RANK','AST_RANK','REB_RANK']].set_index('TEAM_ID')

# Utility: map between team IDs and abbreviations
@st.cache_data
def get_team_maps():
    teams_list = nba_teams.get_teams()
    abbr_to_id = {t['abbreviation']: t['id'] for t in teams_list}
    id_to_abbr = {v: k for k, v in abbr_to_id.items()}
    return abbr_to_id, id_to_abbr

# Fetch upcoming games for a given date
@st.cache_data
def fetch_upcoming_games(home_team_abbreviation, away_team_abbreviation):
    
    team_name_to_id = {}
    for t in nba_teams.get_teams():
        team_name_to_id[t['abbreviation']] = t['id']

    # pre_id = df.loc[len(df) - 1]['GAME_ID']
    # pre_id = pre_id.astype(int)

    GAME_ID = 123456788
    HOME_TEAM_ID = team_name_to_id[home_team_abbreviation]
    VISITOR_TEAM_ID = team_name_to_id[away_team_abbreviation]
    
    df_upcoming = pd.DataFrame({
        'GAME_ID': GAME_ID,
        'HOME_TEAM_ID': HOME_TEAM_ID,
        'VISITOR_TEAM_ID': VISITOR_TEAM_ID
    }, index=[0])

    return df_upcoming

# Build one row per team-game with matchup info
def build_upcoming_df(df_games, id_to_abbr):
    records = []
    for _, row in df_games.iterrows():
        h_id = row['HOME_TEAM_ID']
        a_id = row['VISITOR_TEAM_ID']
        records.append({
            'TEAM_ID': h_id,
            'OPP_ID': a_id,
            'is_home': 1,
            'TEAM_ABBR': id_to_abbr[h_id],
            'OPP_ABBR': id_to_abbr[a_id]
        })
        records.append({
            'TEAM_ID': a_id,
            'OPP_ID': h_id,
            'is_home': 0,
            'TEAM_ABBR': id_to_abbr[a_id],
            'OPP_ABBR': id_to_abbr[h_id]
        })
    return pd.DataFrame(records)

# Main Streamlit app
def main():

    model, team_elos, last_mom5, model_spread = load_model_and_state()

    season_feats = load_prior_season_stats()

    abbr_to_id, id_to_abbr = get_team_maps()

    # Date selector

    played = st.sidebar.radio(
            "Has this game already happened?",
            options=[True, False],
            format_func=lambda x: "Yes, it's in the past" if x else "No, it's upcoming"
    )

    
    if played:
        st.title("NBA Past Games")
        selected_date = st.date_input("Select game date", value=pd.to_datetime("today").date())
        date_str = selected_date.strftime('%Y-%m-%d')
        season = st.sidebar.selectbox("Season", ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25'])
        mode = st.sidebar.selectbox("Game Type", ['Regular Season', 'Playoffs'])
        home_abbr = st.text_input("Home team abbreviation (e.g., LAL)", '')
        away_abbr = st.text_input("Away team abbreviation (e.g., BOS)", '')
    else:
        st.title("NBA Moneyline Model Predictions")
        selected_date = st.date_input("Select game date", value=pd.to_datetime("today").date())
        date_str = selected_date.strftime('%m/%d/%Y')
        home_abbr = st.text_input("Home team abbreviation (e.g., LAL)", "")
        away_abbr = st.text_input("Away team abbreviation (e.g., BOS)", "")
    


    

    if home_abbr and away_abbr:
        # Fetch and prepare upcoming games
        if home_abbr not in abbr_to_id or away_abbr not in abbr_to_id:
            st.error("Invalid team abbreviation(s). Please check and try again.")
            return
        
        if not played:
            if not check_team_playoffs(home_abbr) or not check_team_playoffs(away_abbr):
                 st.error("Invalid teams. Teams must be in the playoffs. Try again.")
                 return
        
            df_games = fetch_upcoming_games(home_abbr, away_abbr)
            df_up = build_upcoming_df(df_games, id_to_abbr)

            # Merge prior-season stats
            df_up = df_up.join(season_feats, on='TEAM_ID')
            df_up = df_up.join(season_feats, on='OPP_ID', rsuffix='_opp')

            # Compute Elo features
            df_up['elo'] = df_up['TEAM_ID'].map(team_elos)
            df_up['elo_opp'] = df_up['OPP_ID'].map(team_elos)
            df_up['elo_diff'] = df_up['elo'] - df_up['elo_opp']

            # Load momentum
            df_up['momentum_5'] = df_up['TEAM_ID'].map(last_mom5)

            # Compute diff features
            for stat in ['W_PCT','PTS_RANK','AST_RANK','REB_RANK']:
                df_up[f'{stat}_diff'] = df_up[stat] - df_up[f'{stat}_opp']

            # Select features for prediction
            feature_cols = [f'{s}_diff' for s in ['W_PCT','PTS_RANK','AST_RANK','REB_RANK']] + ['is_home','elo_diff','momentum_5']
            X_up = df_up[feature_cols]

            # Generate probabilities
            df_up['P_WIN'] = model.predict_proba(X_up)[:,1]
            df_up['P_WIN_%'] = (df_up['P_WIN'] * 100).round(1).astype(str) + '%'

            # Display results
            st.subheader(f"Predictions for {selected_date}")
            
            st.dataframe(df_up[['TEAM_ABBR','OPP_ABBR','is_home','P_WIN_%']].rename(
                columns={'is_home':'Home?'}
            ))


            # spread selection
            st.subheader("Predict Game Edge")
            user_spread = st.number_input("Enter Current Spread (home team): ", step=0.5, format="%.1f")
            X_sel = df_up[feature_cols]
            pred_spread = model_spread.predict(X_sel)[0]

            if user_spread:
                st.write(f"📈 **Model Predicted Spread**: {pred_spread:+.1f} points")

                if (user_spread > pred_spread):
                    st.success(f"✅ Take the **Home Team** spread! Model thinks they should be favored by {pred_spread:+.1f}, not {user_spread:+.1f}.")
                elif pred_spread > user_spread:
                    st.error(f"❌ Take the **Away Team** spread! Model thinks home team should only be favored by {pred_spread:+.1f}, not {user_spread:+.1f}.")

        else:
            season_split = season[: 4]
            season_split_2 = "20" + season[5 :]
            year = date_str[: 4]
            if year != season_split and year != season_split_2:
                st.error("Invalid date. Please check and try again.")
                return
            df_past = get_game_logs(season, mode, date_str, home_abbr, away_abbr)
            if df_past is None:
                st.error("Invalid teams or game type. Please check and try again.")
                return
            st.subheader(f"Outcome for {home_abbr} @ {away_abbr}")
            st.dataframe(df_past.rename(
                columns={'is_home':'Home?'}
            ))
        

    else:
        st.info("Enter both home and away team abbreviations to see predictions.")

if __name__ == "__main__":
    main()
