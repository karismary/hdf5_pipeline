import streamlit as st
from hdf5_pipeline.core.config import load_config

st.set_page_config(page_title="Test", layout="wide")
config = load_config()
st.title("Test")
st.write(config["paths"])
