import requests
import json
from datetime import datetime
import os
from twilio.rest import Client
import time
import pickle
import numpy as np
import logging

logging.getLogger().setLevel(logging.INFO)

if __name__ == '__main__':

    while True:

        time.sleep(45)

        current_time = datetime.now()
        logging.info(f'Processed {current_time}')
        if (current_time.hour in [0,4,8,12,16,20]) and (current_time.minute == 0):

            logging.info('Checking prices...')

            end = int(time.mktime(datetime.now().timetuple()))

            start = end - 1000 * 4 * 60 * 60

            ethusd_url = f'https://futures.kraken.com/api/charts/v1/spot/PI_ETHUSD/4h?from={start}&to={end}'

            ethusd_candles = requests.get(ethusd_url).json()['candles']

            ethusd_close_prices = []
            ethusd_open_prices = []
            timestamps = []
            for item in ethusd_candles:
                timestamps.append(item['time'])
                ethusd_close_prices.append(float(item['close']))
                ethusd_open_prices.append(float(item['open']))

            btcusd_url = f'https://futures.kraken.com/api/charts/v1/spot/PI_XBTUSD/4h?from={start}&to={end}'

            btcusd_candles = requests.get(btcusd_url).json()['candles']

            btcusd_close_prices = []
            btcusd_open_prices = []
            timestamps = []
            for item in btcusd_candles:
                timestamps.append(item['time'])
                btcusd_close_prices.append(float(item['close']))
                btcusd_open_prices.append(float(item['open']))

            close_prices = np.divide(np.array(ethusd_close_prices), np.array(btcusd_close_prices)).tolist()
            open_prices = np.divide(np.array(ethusd_open_prices), np.array(btcusd_open_prices)).tolist()

            with open('model.pickle', 'rb') as f:
                model = pickle.load(f)

            close_prices = np.array(close_prices)
            open_prices = np.array(open_prices)

            input_features = np.concatenate((close_prices[-10:], close_prices[-10:]/min(close_prices[-10:]), [min(close_prices[-10:])], 
                                             open_prices[-10:], open_prices[-10:]/min(open_prices[-10:]), [min(open_prices[-10:])]), axis=0)

            pred = model.predict([input_features])[0]

            if pred == 'True':

                api_key = os.environ['API_KEY']
                api_secret = os.environ['API_SECRET']
                account_sid = os.environ['ACCOUNT_SID']
                client = Client(api_key, api_secret, account_sid)

                message = client.messages.create(
                    from_= os.environ['TWILIO_PHONE_NUMBER'],
                    body = 'ETH should spike against BTC',
                    to = os.environ['RECEPIENT_PHONE_NUMBER']
                )