import numpy as np
import pandas as pd
import pyupbit  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from class_ying_yang_volatility import YingYangTradingBot
from dotenv import load_dotenv
from notion_client import Client
import os
import time

# access upbit with my access and secret key for retrieve balance and eventually make execution

def run_bot():
    # Instance creation with adjusted time interval and data count
    load_dotenv()

    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    upbit = pyupbit.Upbit(access_key, secret_key)
    balance = upbit.get_balance()
        
    ticker = 'KRW-BTC'
    interval = 'minute30'
    count = 300

    bot = YingYangTradingBot(ticker, interval, count, ema=True, window=20, span=10)
    
    # Download data
    bot.download_data()
    
    # Calculate volatility
    bot.calculate_volatility()
    
    # Calculate Pan Bands
    bot.calculate_pan_bands()
   
    # Generate trading signals
    bot.trading_signal()

    # Perform backtest
    backtest_results = bot.backtest()
    last_signal = bot.get_last_signal()
    bot.notion_update()
    
    # Plot signals
    bot.plot_results()

if __name__ == "__main__":
    print("Auto bot started")
    while True:
        try:
            run_bot()
        except Exception as e:
            print(f"An error occurred: {e}")
        # Wait for 30 minutes before next execution
        time.sleep(1800)