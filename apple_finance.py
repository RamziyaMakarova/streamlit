import yfinance as yf
import streamlit as st
import pandas as pd


st.write("""
# Simple stock Price App
""")

tickerSymbol = 'AAPL'
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

st.line_chart(tickerDf.Close, use_container_width=True)
st.line_chart(tickerDf.Volume, width=800, height=400)
