from flask import Flask, request, render_template, redirect, Response, jsonify
import os
from datetime import datetime
import pandas as pd
import json
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import numpy as np
import plotly.express as px
import csv
from graph_utils import convert_log_to_csv

ACCESS_TOKEN = "822ecd23815743f4ad241eda2f60b18b-bf960bf2301bf7157a7694874c2841c7"  # ← 実際のトークンに置き換えてください
client = API(access_token=ACCESS_TOKEN)
SECRET_TOKEN = "your_secret_token"

def generate_trade_signals(df, use_rsi=True, use_macd=True):
    buy_points = []
    sell_points = []

    for i in range(1, len(df)):
        # RSIシグナル
        if use_rsi and "RSI" in df.columns:
            if df["RSI"].iloc[i] < 30 and df["RSI"].iloc[i - 1] >= 30:
                buy_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))
            if df["RSI"].iloc[i] > 70 and df["RSI"].iloc[i - 1] <= 70:
                sell_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))

        # MACDシグナル
        if use_macd and "MACD" in df.columns and "Signal" in df.columns:
            if df["MACD"].iloc[i] > df["Signal"].iloc[i] and df["MACD"].iloc[i - 1] <= df["Signal"].iloc[i - 1]:
                buy_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))
            if df["MACD"].iloc[i] < df["Signal"].iloc[i] and df["MACD"].iloc[i - 1] >= df["Signal"].iloc[i - 1]:
                sell_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))

    return buy_points, sell_points

def generate_stats_html(df, buy_points, sell_points):
    buy_prices = [price for _, price in buy_points]
    sell_prices = [price for _, price in sell_points]
    stats = {
        '買い回数': len(buy_prices),
        '売り回数': len(sell_prices),
        '平均買い価格': round(np.mean(buy_prices), 2) if buy_prices else '-',
        '平均売り価格': round(np.mean(sell_prices), 2) if sell_prices else '-',
    }
    html = "<table><tr><th>項目</th><th>値</th></tr>"
    for k, v in stats.items():
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"
    html += "</table>"
    return html

def generate_main_chart(df, chart_type="line", buy_points=None, sell_points=None):
    fig = go.Figure()

    # メインチャート（価格）
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        mode="lines",
        name="価格",
        line=dict(color="blue")
    ))

    # ✅ 買いシグナルマーカー
    if buy_points:
        buy_df = pd.DataFrame(buy_points, columns=["timestamp", "price"])
        fig.add_trace(go.Scatter(
            x=buy_df["timestamp"],
            y=buy_df["price"],
            mode="markers",
            name="買いシグナル",
            marker=dict(color="green", size=10, symbol="triangle-up")
        ))

    # ✅ 売りシグナルマーカー
    if sell_points:
        sell_df = pd.DataFrame(sell_points, columns=["timestamp", "price"])
        fig.add_trace(go.Scatter(
            x=sell_df["timestamp"],
            y=sell_df["price"],
            mode="markers",
            name="売りシグナル",
            marker=dict(color="red", size=10, symbol="triangle-down")
        ))

    fig.update_layout(
        title="価格チャート（売買シグナル付き）",
        xaxis_title="日時",
        yaxis_title="価格",
        template="plotly_white"
    )

    return fig.to_html(full_html=False)

def calculate_signal_stats(buy_points, sell_points):
    stats = {}
    buy_prices = [p for _, p in buy_points]
    sell_prices = [p for _, p in sell_points]

    stats["買い回数"] = len(buy_prices)
    stats["売り回数"] = len(sell_prices)
    stats["買い平均価格"] = round(np.mean(buy_prices), 2) if buy_prices else None
    stats["売り平均価格"] = round(np.mean(sell_prices), 2) if sell_prices else None

    wins = 0
    for i in range(min(len(buy_points), len(sell_points))):
        if sell_points[i][1] > buy_points[i][1]:
            wins += 1
    stats["勝率"] = round(wins / min(len(buy_points), len(sell_points)) * 100, 1) if wins else 0

    return stats

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df):
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def fetch_candles():
    params = {
        "granularity": "H1",
        "count": 100,
        "price": "M"
    }
    r = instruments.InstrumentsCandles(instrument="USD_JPY", params=params)
    client.request(r)
    data = r.response["candles"]
    df = pd.DataFrame([{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"])
    } for c in data])
    return df

def generate_ma_chart(df, cross_signals):
    fig = go.Figure()

    # 移動平均線
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA_short"], mode="lines", name="短期MA"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MA_long"], mode="lines", name="長期MA"))

    # クロスマーカー
    for _, row in cross_signals.iterrows():
        color = "green" if row["signal"] == "Golden Cross" else "red"
        fig.add_trace(go.Scatter(
            x=[row["timestamp"]],
            y=[df.loc[df["timestamp"] == row["timestamp"], "MA_short"].values[0]],
            mode="markers",
            marker=dict(color=color, size=10, symbol="x"),
            name=row["signal"]
        ))

    fig.update_layout(title="移動平均クロス検出", xaxis_title="日時", yaxis_title="価格")
    return fig.to_html(full_html=False)

def calculate_moving_averages(df, short_window=5, long_window=25):
    df = df.sort_values("timestamp").copy()
    df["MA_short"] = df["price"].rolling(window=short_window).mean()
    df["MA_long"] = df["price"].rolling(window=long_window).mean()
    return df

def extract_ma_cross_signals(df):
    signals = []
    for i in range(1, len(df)):
        prev_short = df["MA_short"].iloc[i - 1]
        prev_long = df["MA_long"].iloc[i - 1]
        curr_short = df["MA_short"].iloc[i]
        curr_long = df["MA_long"].iloc[i]

        if pd.notna(prev_short) and pd.notna(prev_long) and pd.notna(curr_short) and pd.notna(curr_long):
            if prev_short < prev_long and curr_short > curr_long:
                signals.append({"timestamp": df["timestamp"].iloc[i], "signal": "Golden Cross"})
            elif prev_short > prev_long and curr_short < curr_long:
                signals.append({"timestamp": df["timestamp"].iloc[i], "signal": "Dead Cross"})
    return pd.DataFrame(signals)

def extract_rsi_macd_signals(df):
    signals = []

    for i in range(1, len(df)):
        if df['RSI'][i] <= 30 and df['RSI'][i-1] > 30:
            signals.append({'date': df['Date'][i], 'type': 'Buy', 'reason': 'RSI <= 30'})
        elif df['RSI'][i] >= 70 and df['RSI'][i-1] < 70:
            signals.append({'date': df['Date'][i], 'type': 'Sell', 'reason': 'RSI >= 70'})
        elif df['MACD'][i] > df['Signal'][i] and df['MACD'][i-1] <= df['Signal'][i-1]:
            signals.append({'date': df['Date'][i], 'type': 'Buy', 'reason': 'MACD cross up'})
        elif df['MACD'][i] < df['Signal'][i] and df['MACD'][i-1] >= df['Signal'][i-1]:
            signals.append({'date': df['Date'][i], 'type': 'Sell', 'reason': 'MACD cross down'})
    return pd.DataFrame(signals)

def save_signals_to_csv(signal_df, filename="static/rsi_macd_signals.csv"):
    signal_df.to_csv(filename, index=False)

from graph_utils import (
    load_csv, preprocess, generate_pie_chart,
    daily_summary, symbol_statistics, trend_comparison, monthly_summary,
    volatility_ranking, condition_filter, moving_average_chart, rsi_macd_chart,
    save_chart_as_png
)

def calculate_rsi_macd(df):
    df = df.sort_values("timestamp").copy()
    close = df["price"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["Date"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

app = Flask(__name__, template_folder="templates")
FILTER_FILE = "saved_filter.json"
UPLOAD_FOLDER = "uploads"
convert_log_to_csv()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

latest_html = ""
filtered_df = pd.DataFrame()

# ここに generate_trade_stats() を定義
def generate_trade_stats(df):
    stats = {
        "シグナル回数": len(df),
        "平均価格": round(df["price"].mean(), 2),
        "勝率": f"{round((df['result'] == 'win').mean() * 100, 1)}%",
        "平均利益": round(df["profit"].mean(), 2),
        "最大利益": df["profit"].max(),
        "最大損失": df["profit"].min(),
    }

    stats_html = "<ul>"
    for key, value in stats.items():
        stats_html += f"<li><strong>{key}:</strong> {value}</li>"
    stats_html += "</ul>"

    return stats_html

from flask import Flask, request, jsonify
from datetime import datetime
import os
import csv

app = Flask(__name__)
SECRET_TOKEN = "your_secret_token"  # ← 必要に応じて設定

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()

    # 🔐 トークン認証
    if not data or data.get("token") != SECRET_TOKEN:
     return jsonify({"status": "unauthorized"}), 403


    # 🔽 データ取得
    symbol = data.get("symbol", "unknown")
    price = data.get("price", 0.0)
    time = data.get("time", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    action = "alert"
    extra_info = ""

    print(f"✅ Alert received: {symbol} at {price} on {time}")

    # 🔽 個別ファイル保存（static/log_YYYYMMDD_HHMMSS.csv）
    filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join("static", filename)
    print(f"📁 Saving to: {filepath}")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "action", "symbol", "price", "extra_info"])
        writer.writerow([timestamp, action, symbol, price, extra_info])

    # 🔽 統一ログファイルにも追記（static/webhook_log.csv）
    log_path = os.path.join("static", "webhook_log.csv")
    write_header = not os.path.exists(log_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "action", "symbol", "price", "extra_info"])
        writer.writerow([timestamp, action, symbol, price, extra_info])

    return jsonify({"status": "success", "saved": filename}), 200

# ここにルート処理を追加
@app.route("/graph_upload", methods=["POST"])
def show_graph():
    file = request.files["logfile"]
    df = pd.read_csv(file)

    # 例：価格の折れ線グラフ
    fig_latest = px.line(df, x="timestamp", y="price", title="価格推移")
    latest_html = fig_latest.to_html(full_html=False)

    # 例：勝敗の円グラフ
    win_counts = df["result"].value_counts()
    fig_pie = px.pie(names=win_counts.index, values=win_counts.values, title="勝敗比率")
    pie_html = fig_pie.to_html(full_html=False)

    # データテーブル（HTML化）
    df_table = df.to_html(classes="table table-striped", index=False)

    # 統計HTML
    stats_html = generate_trade_stats(df)

    return render_template(
        "graph.html",
        latest_html=latest_html,
        pie_html=pie_html,
        df_table=df_table,
        stats_html=stats_html,
        daily_html="（未実装）",
        trend_html="（未実装）",
        monthly_html="（未実装）",
        volatility_html="（未実装）",
        condition_html="（未実装）",
        ma_html="（未実装）",
        rsi_macd_html="（未実装）"
    )

@app.route("/save_signals")
def save_signals():
    global filtered_df
    if filtered_df.empty:
        return "データがありません"

    # RSI・MACDを計算
    df = calculate_rsi_macd(filtered_df)

    # 移動平均（短期5・長期25）を計算
    df = calculate_moving_averages(df, short_window=5, long_window=25)

    # シグナル抽出（RSI・MACD）
    rsi_macd_signals = extract_rsi_macd_signals(df)

    # シグナル抽出（移動平均クロス）
    ma_cross_signals = extract_ma_cross_signals(df)

    # 両方のシグナルを結合
    all_signals = pd.concat([rsi_macd_signals, ma_cross_signals], ignore_index=True)

    # CSV保存
    save_signals_to_csv(all_signals)

    return redirect("/static/rsi_macd_signals.csv")

FILTER_FILE = "static/filter.json"

@app.route("/save_filter")
def save_filter():
    filter_data = {
    "symbol": request.args.getlist("symbol"),
    "start": request.args.get("start"),
    "end": request.args.get("end"),
    "price_min": request.args.get("price_min"),
    "price_max": request.args.get("price_max"),
    "action": request.args.get("action")
}
    with open(FILTER_FILE, "w", encoding="utf-8") as f:
        json.dump(filter_data, f, ensure_ascii=False, indent=2)
    return "フィルター条件を保存しました"

@app.route("/load_filter")
def load_filter():
    try:
        with open(FILTER_FILE, "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        return jsonify(filter_data)
    except Exception as e:
        return jsonify({"error": f"読み込み失敗: {str(e)}"})

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename.endswith(".csv"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(UPLOAD_FOLDER, f"log_{timestamp}.csv")
                file.save(save_path)
        elif "delete" in request.form:
            delete_file = request.form["delete"]
            try:
                os.remove(os.path.join(UPLOAD_FOLDER, delete_file))
            except Exception as e:
                print(f"削除失敗: {e}")
        return redirect("/")

    files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    latest_file = os.path.join(UPLOAD_FOLDER, files[0]) if files else None
    symbols = []
    if latest_file:
        try:
            df = pd.read_csv(latest_file, encoding="utf-8")
            print("列名一覧:", df.columns)
            symbols = df["symbol"].unique().tolist()
        except Exception as e:
            print(f"CSV読み込みエラー: {e}")

    return render_template("index.html", files=files, symbols=symbols)

@app.route("/graph")
def graph():

    filename = request.args.get("file")
    print("受け取ったファイル名:", filename)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    print("ファイルパス:", filepath)

    if not filename or not os.path.exists(filepath):
        print("ファイルが存在しないか、ファイル名が空です")
        abort(404)

    try:
        df = load_csv(filepath)
    except Exception as e:
        print("CSV読み込みエラー:", e)
        abort(404)
    global latest_html, filtered_df

    use_rsi = request.args.get("use_rsi") == "on"
    use_macd = request.args.get("use_macd") == "on"
    filename = request.args.get("file")
    symbols = request.args.getlist("symbol")
    start = request.args.get("start")
    end = request.args.get("end")
    chart_type = request.args.get("chart", "line")
    price_min = request.args.get("price_min")
    price_max = request.args.get("price_max")
    action = request.args.get("action")

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = load_csv(filepath)
    df = preprocess(df, symbols, start, end, price_min, price_max, action)
    df = calculate_rsi_macd(df)
    filtered_df = df.copy()

    # 売買シグナル抽出（重複なし）
    buy_points, sell_points = generate_trade_signals(df, use_rsi, use_macd)

    # 統計集計
    signal_stats = calculate_signal_stats(buy_points, sell_points)
    stats_html = generate_stats_html(df, buy_points, sell_points)

    # チャート生成（売買マーカー付き）
    latest_html = generate_main_chart(df, chart_type, buy_points, sell_points)
    pie_html = generate_pie_chart(df)
    daily_html = daily_summary(df)
    trend_html = trend_comparison(df)
    monthly_html = monthly_summary(df)
    volatility_html = volatility_ranking(df)
    condition_html = condition_filter(df)
    ma_html = moving_average_chart(df)
    rsi_macd_html = rsi_macd_chart(df)

    # 売買シグナルをCSVに保存
    buy_df = pd.DataFrame(buy_points, columns=["timestamp", "price"])
    sell_df = pd.DataFrame(sell_points, columns=["timestamp", "price"])
    buy_df.to_csv(os.path.join("static", "buy_signals.csv"), index=False)
    sell_df.to_csv(os.path.join("static", "sell_signals.csv"), index=False)

    # HTMLテーブルに変換
    buy_table_html = buy_df.to_html(classes='table table-bordered table-sm', index=False)
    sell_table_html = sell_df.to_html(classes='table table-bordered table-sm', index=False)

    return render_template("graph.html",
        latest_html=latest_html,
        pie_html=pie_html,
        df_table=df.to_html(index=False, classes="data-table", border=1),
        daily_html=daily_html,
        stats_html=stats_html,
        trend_html=trend_html,
        monthly_html=monthly_html,
        volatility_html=volatility_html,
        condition_html=condition_html,
        ma_html=ma_html,
        rsi_macd_html=rsi_macd_html,
        signal_stats=signal_stats,
        filtered_df=filtered_df,
        buy_table_html=buy_table_html,
        sell_table_html=sell_table_html
    )

@app.route("/download")
def download():
    global latest_html
    return Response(
        latest_html,
        mimetype="text/html",
        headers={"Content-Disposition": "attachment;filename=graph.html"}
    )

@app.route("/save_png")
def save_png_route():  # ← 関数名を変える
    kind = request.args.get("kind", "ma")
    fig = generate_graph(kind)
    save_path = os.path.join("static", "save_png.png")
    fig.write_image(save_path)
    return redirect(url_for("static", filename="save_png.png"))

@app.route("/export")
def export():
    global filtered_df
    csv_data = filtered_df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=filtered_data.csv"}
    )

# ✅ PNG保存エンドポイント
@app.route("/save_png")
def save_png():
    global filtered_df
    chart_type = request.args.get("chart", "line")
    filename = "static/chart.png"
    save_chart_as_png(filtered_df, chart_type, filename)
    return redirect("/" + filename)

@app.route("/oanda_chart")
def oanda_chart():
    df = fetch_candles()  # 先ほどのOANDAデータ取得関数
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )])
    fig.update_layout(title="USD/JPY 1時間足", xaxis_rangeslider_visible=False)
    graph_html = fig.to_html(full_html=False)
    return render_template("graph.html", latest_html=graph_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)