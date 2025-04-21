# backend.py
from flask import Flask, jsonify
import boto3
import pandas as pd
from io import StringIO

app = Flask(__name__)

@app.route('/api/emg')
def get_emg_data():
    s3 = boto3.client('s3')
    bucket = 'emg-signal-storage'
    key = 'emg_data/emg_data.csv'

    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'])  # 读取CSV内容
    return df.to_json(orient='records')  # 返回前端 JSON 格式

if __name__ == '__main__':
    app.run(debug=True)

