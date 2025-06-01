import streamlit as st
import pandas as pd
from joblib import load
from joblib import dump
from nba_api.stats.endpoints import PlayerGameLogs, LeagueDashTeamStats, CommonTeamRoster
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import LeagueGameLog
import os

from train_models import (
        retrain_moneyline_model,
        retrain_spread_model,
        retrain_player_props_model,
        load_team_elos,
        load_last_mom5
    )

MONEYLINE_PATH = os.path.join(os.path.dirname(__file__), 'moneyline_model.joblib')
MODEL_SPREAD   = os.path.join(os.path.dirname(__file__), 'spread_model.joblib')
PLAYER_PROPS   = os.path.join(os.path.dirname(__file__), 'player_props_model.joblib')
TEAM_ELOS_PATH = os.path.join(os.path.dirname(__file__), 'team_elos.joblib')
LAST_MOM5_PATH = os.path.join(os.path.dirname(__file__), 'last_mom5.joblib')



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


# Load trained model and saved state
def load_model_and_state():
    MODEL_DIR = os.path.join(os.path.dirname(__file__))
    MONEYLINE_PATH = os.path.join(MODEL_DIR, "moneyline_model.joblib")
    TEAM_ELOS_PATH = os.path.join(MODEL_DIR, "team_elos.joblib")
    LAST_MOM5 = os.path.join(MODEL_DIR, "last_mom5.joblib")
    MODEL_SPREAD = os.path.join(MODEL_DIR, "spread_model.joblib")
    PLAYER_PROPS = os.path.join(MODEL_DIR, "player_props_model.joblib")

    
    model = load(MONEYLINE_PATH)
    team_elos = load(TEAM_ELOS_PATH)
    last_mom5 = load(LAST_MOM5)
    model_spread = load(MODEL_SPREAD)
    player_props = load(PLAYER_PROPS)

    return model, team_elos, last_mom5, model_spread, player_props

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

    with st.sidebar.expander("⚙️ Retrain Models", expanded=False):
        # Moneyline
        if st.button("Retrain Moneyline"):
            with st.spinner("Updating moneyline model…"):
                model = retrain_moneyline_model()
                dump(model, MONEYLINE_PATH)
            st.success("✅ Moneyline model refreshed!")
            st.rerun()

        # Spread
        if st.button("Retrain Spread"):
            with st.spinner("Updating spread model…"):
                spread = retrain_spread_model()
                dump(spread, MODEL_SPREAD)
            st.success("✅ Spread model refreshed!")
            st.rerun()

        # Player Props
        if st.button("Retrain Player Props"):
            with st.spinner("Updating player props…"):
                props = retrain_player_props_model()
                dump(props, PLAYER_PROPS)
            st.success("✅ Player props model refreshed!")
            st.rerun()


    model, team_elos, last_mom5, model_spread, player_props_model = load_model_and_state()

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
            # if not check_team_playoffs(home_abbr) or not check_team_playoffs(away_abbr):
            #      st.error("Invalid teams. Teams must be in the playoffs. Try again.")
            #      return
        
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
            feature_cols = [f'{s}_diff' for s in ['W_PCT']] + ['elo_diff','momentum_5']
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
            feature_cols_2 = [f'{s}_diff' for s in ['W_PCT']] + ['elo_diff','is_home']
            X_sel = df_up[feature_cols_2]
            pred_spread = model_spread.predict(X_sel)[0]

            if user_spread:
                st.write(f"📈 **Model Predicted Spread**: {pred_spread:+.1f} points")

                if (user_spread > pred_spread):
                    st.success(f"✅ Take the **Home Team** spread! Model thinks they should be favored by {pred_spread:+.1f}, not {user_spread:+.1f}.")
                elif pred_spread > user_spread:
                    st.error(f"❌ Take the **Away Team** spread! Model thinks home team should only be favored by {pred_spread:+.1f}, not {user_spread:+.1f}.")
            
            


            # ─── 2) Fetch each roster via CommonTeamRoster ─────────────────────────────────
            # You can cache this if you like, e.g. @st.cache_data on a function.
            season_str = "2024-25"  # or make this dynamic if you want

            home_id = abbr_to_id[home_abbr]
            away_id = abbr_to_id[away_abbr]

            home_roster_df = CommonTeamRoster(team_id=home_id, season=season_str).get_data_frames()[0]
            away_roster_df = CommonTeamRoster(team_id=away_id, season=season_str).get_data_frames()[0]



            # ─── 4) Show a dropdown of all those names ────────────────────────────────────
            st.subheader("Player Props Predictor")
            side = st.selectbox("Choose roster to pick a player from:", ["Home", "Away"])

            if side == "Home":
                roster_df = home_roster_df
            else:
                roster_df = away_roster_df
            
            # Build a name→ID mapping for the selected side
            player_dict = dict(zip(roster_df["PLAYER"], roster_df["PLAYER_ID"]))
            player_names = sorted(player_dict.keys())

            # ─── 2) Let user pick a player from that roster ───────────────────────────────
            selected_player = st.selectbox("Select Player:", player_names)
            pid = player_dict[selected_player]
            
            # ─── 5) Let user enter an Over/Under line as before ───────────────────────────
            line = st.number_input("Over/Under line (pts):", step=0.5)

            # ─── 6) Only run prediction once we have pid & line ───────────────────────────
            if pid is not None and line:
                # (You can print(pid) for debugging if you want)
                logs = PlayerGameLogs(
                    player_id_nullable=pid,
                    season_nullable="2024-25",
                    season_type_nullable="Playoffs"
                ).get_data_frames()[0]

                if logs.empty:
                    st.error(f"No logs found for PLAYER_ID ({selected_player}). He did not play in the playoffs sorry!")
                else:
                    # ── Compute feature columns exactly as before ───────────────────────────
                    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
                    logs = logs.sort_values(["PLAYER_ID", "GAME_DATE"])
                    grp = logs.groupby("PLAYER_ID")
                    logs["PTS_PREV_GAME"] = grp["PTS"].shift(1)
                    logs["PTS_ROLL5"]     = grp["PTS"].shift(1).rolling(5).mean().fillna(method="ffill")
                    logs["MIN_ROLL5"]     = grp["MIN"].shift(1).rolling(5).mean().fillna(method="ffill")

                    # ── Get opponent ID from the “MATCHUP” column ──────────────────────────
                    team_name_to_id = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
                    opp_ids = []
                    for m in logs["MATCHUP"]:
                        # MATCHUP is like “LAL vs BOS” or “BOS @ LAL”
                        # We want the opponent’s abbreviation
                        parts = m.split()
                        if len(parts) == 3:
                            # parts = [team, “vs”/“@”, opp]
                            opp_abbr = parts[2]
                        else:
                            opp_abbr = None
                        opp_ids.append(team_name_to_id.get(opp_abbr, None))
                    logs["OPP_ID"] = opp_ids

                    # ── Build team‐level logs to compute DEF_LAG1 & DEF_ROLL5 ────────────
                    rs = LeagueGameLog(season="2024-25", season_type_all_star="Regular Season").get_data_frames()[0]
                    po = LeagueGameLog(season="2024-25", season_type_all_star="Playoffs").get_data_frames()[0]
                    team_logs = pd.concat([rs, po], ignore_index=True)
                    team_logs["GAME_DATE"] = pd.to_datetime(team_logs["GAME_DATE"])

                    # rename & merge so each row has TEAM_ID, PTS_SCORED, OPP_ID, PTS_ALLOWED
                    for_logs = team_logs.rename(columns={"TEAM_ID": "TEAM_ID", "PTS": "PTS_SCORED"})
                    opp_logs = (
                        team_logs.rename(columns={"TEAM_ID": "OPP_ID", "PTS": "PTS_ALLOWED"})[
                            ["GAME_ID", "OPP_ID", "PTS_ALLOWED", "GAME_DATE"]
                        ]
                    )
                    team_logs = (
                        for_logs
                        .merge(opp_logs, on=["GAME_ID", "GAME_DATE"])
                        .query("TEAM_ID != OPP_ID")
                        .sort_values(["TEAM_ID", "GAME_DATE"])
                    )
                    grp_team = team_logs.groupby("TEAM_ID")
                    team_logs["DEF_LAG1"] = grp_team["PTS_ALLOWED"].shift(1)
                    team_logs["DEF_ROLL5"] = grp_team["PTS_ALLOWED"].shift(1).rolling(5, min_periods=1).mean()

                    def_feats = (
                        team_logs[["TEAM_ID", "GAME_ID", "DEF_LAG1", "DEF_ROLL5"]]
                        .rename(columns={"TEAM_ID": "OPP_ID"})
                    )

                    logs = logs.merge(def_feats, on=["GAME_ID", "OPP_ID"], how="left")

                    last_season_df = (
                        LeagueDashTeamStats (
                            season= '2023-24',
                            season_type_all_star='Regular Season',
                        )
                        .get_data_frames()[0]
                    )

                    stats = last_season_df.set_index("TEAM_ID")
                    for col in ["W_PCT","PLUS_MINUS","DREB","STL","BLK"]:
                        logs[f"opp_{col.lower()}"] = logs["OPP_ID"].map(stats[col])

                    # ── Take the last row (most recent game) as feature input ───────────────
                    last = logs.iloc[-1]
                    feat = pd.DataFrame([{
                        "PTS_PREV_GAME": last["PTS_PREV_GAME"],
                        "PTS_ROLL5":      last["PTS_ROLL5"],
                        "MIN_ROLL5":      last["MIN_ROLL5"],
                        "DEF_LAG1":       last["DEF_LAG1"],
                        "DEF_ROLL5":      last["DEF_ROLL5"],
                        "opp_w_pct": last["opp_w_pct"],
                        "opp_plus_minus": last["opp_plus_minus"],
                        "opp_dreb": last["opp_dreb"],
                        "opp_stl": last["opp_stl"],
                        "opp_blk": last["opp_blk"]
                    }])

                    # ── Predict with your pre‐trained model ────────────────────────────────
                    pred = player_props_model.predict(feat)[0]

                    # ── Show a few recent log lines & the prediction ───────────────────────
                    st.write(logs[["GAME_DATE", "MATCHUP", "OPP_ID", "DEF_LAG1", "DEF_ROLL5", "PTS"]].tail())
                    st.write(f"**Selected Player:** {selected_player} (PLAYER_ID {pid})")
                    st.write(f"**Predicted PTS:** {pred:.1f}")

                    if pred >= line:
                        st.success("✅ Suggest you take the OVER")
                    else:
                        st.error("❌ Suggest you take the UNDER")
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
