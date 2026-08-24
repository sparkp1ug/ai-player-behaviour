import streamlit as st
import pandas as pd
from recommend import recommend

st.title("Player Behaviour Explorer")

df = pd.read_parquet("data/players_clustered.parquet")

player_id = st.number_input("Select Player ID", min_value=0, max_value=len(df)-1, value=0)

st.subheader("Player Profile")
st.write(df.iloc[player_id])

st.subheader("Recommended Experiences")
recs = recommend(player_id)
for name, score in recs:
    st.write(f"**{name}** — similarity score: {score:.3f}")
