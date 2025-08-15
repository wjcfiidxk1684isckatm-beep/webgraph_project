import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import json
import os
import csv


def load_csv(filepath):
    return pd.read_csv(filepath, encoding="utf-8")

def preprocess(df, symbols=None, start=None, end=None, price_min=None, price_max=None, action=None):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if symbols:
        df = df[df["symbol"].isin(symbols)]
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]
    if price_min:
        df = df[df["price"] >= float(price_min)]
    if price_max:
        df = df[df["price"] <= float(price_max)]
    if action:
        df = df[df["action"] == action]
    return df

def generate_main_chart(df, chart_type="line", return_fig=False):
    fig = go.Figure()
    for symbol in df["symbol"].unique():
        for action in df["action"].unique():
            subset = df[(df["symbol"] == symbol) & (df["action"] == action)]
            marker = dict(symbol="circle" if action == "buy" else "x", size=8)

            if chart_type == "line":
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["price"],
                                         mode="lines+markers", name=f"{symbol} ({action})",
                                         marker=marker))
            elif chart_type == "bar":
                fig.add_trace(go.Bar(x=subset["timestamp"], y=subset["price"],
                                     name=f"{symbol} ({action})"))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["price"],
                                         mode="markers", name=f"{symbol} ({action})",
                                         marker=marker))

            if len(subset) >= 2:
                X = np.array((subset["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()).reshape(-1, 1)
                y = subset["price"].values
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=y_pred,
                                         mode="lines", name=f"{symbol} ({action}) 回帰線",
                                         line=dict(dash="dot", color="gray")))
    fig.update_layout(title="価格推移＋回帰予測", xaxis_title="Timestamp", yaxis_title="Price")
    
    if return_fig:
        return fig
    else:
        return fig.to_html(full_html=False)
    
def generate_pie_chart(df, return_fig=False):
    if "action" not in df.columns:
        return "<p>⚠️ このファイルには 'action' 列がないため、円グラフは表示できません。</p>"
    pie_fig = go.Figure()
    action_counts = df["action"].value_counts()
    pie_fig.add_trace(go.Pie(labels=action_counts.index, values=action_counts.values, hole=0.4))
    pie_fig.update_layout(title="📊 アクション比率（全体）")

    if return_fig:
        return pie_fig
    else:
        return pie_fig.to_html(full_html=False)

def daily_summary(df):
    required_cols = {"timestamp", "symbol", "action"}
    missing = required_cols - set(df.columns)
    if missing:
        return f"<p>⚠️ このファイルには {', '.join(missing)} 列がないため、日次サマリーは表示できません。</p>"

    summary = df.groupby([df["timestamp"].dt.date, "symbol", "action"]).agg(
        count=("action", "count"),
        avg_price=("price", "mean")
    ).reset_index()

    return summary.to_html(classes="table table-bordered", index=False)

def symbol_statistics(df):
    stats = df.groupby(["symbol", "action"]).agg(
        count=("price", "count"),
        avg_price=("price", "mean"),
        max_price=("price", "max"),
        min_price=("price", "min")
    ).reset_index()
    return stats.to_html(index=False, classes="data-table", border=1)

def trend_comparison(df):
    trend_df = df.groupby([df["timestamp"].dt.date, "symbol"]).agg(avg_price=("price", "mean")).reset_index()
    fig = go.Figure()
    for symbol in trend_df["symbol"].unique():
        subset = trend_df[trend_df["symbol"] == symbol]
        fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["avg_price"],
                                 mode="lines+markers", name=f"{symbol} 平均価格"))
    fig.update_layout(title="📈 銘柄別トレンド比較（日別平均）", xaxis_title="日付", yaxis_title="平均価格")
    return fig.to_html(full_html=False)

def monthly_summary(df):
    df["month"] = df["timestamp"].dt.to_period("M")
    summary = df.groupby(["month", "symbol", "action"]).agg(
        count=("price", "count"),
        avg_price=("price", "mean")
    ).reset_index()
    return summary.to_html(index=False, classes="data-table", border=1)

def volatility_ranking(df):
    vol_df = df.groupby("symbol").agg(
        max_price=("price", "max"),
        min_price=("price", "min")
    ).reset_index()
    vol_df["volatility"] = vol_df["max_price"] - vol_df["min_price"]
    vol_df = vol_df.sort_values(by="volatility", ascending=False)
    return vol_df.to_html(index=False, classes="data-table", border=1)

def condition_filter(df):
    filtered = df[(df["price"] >= 1000) & (df["action"] == "buy")]
    return filtered.to_html(index=False, classes="data-table", border=1)

def moving_average_chart(df, return_fig=False):
    fig = go.Figure()
    for symbol in df["symbol"].unique():
        subset = df[df["symbol"] == symbol].sort_values("timestamp")
        subset["MA5"] = subset["price"].rolling(window=5).mean()
        subset["MA20"] = subset["price"].rolling(window=20).mean()

        fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["price"],
                                 mode="lines", name=f"{symbol} 実価格"))
        fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["MA5"],
                                 mode="lines", name=f"{symbol} MA5", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["MA20"],
                                 mode="lines", name=f"{symbol} MA20", line=dict(dash="dash")))

        crossover = (subset["MA5"] > subset["MA20"]) & (subset["MA5"].shift(1) <= subset["MA20"].shift(1))
        cross_points = subset[crossover]
        fig.add_trace(go.Scatter(x=cross_points["timestamp"], y=cross_points["price"],
                                 mode="markers", name=f"{symbol} 転換点",
                                 marker=dict(symbol="star", size=10, color="red")))
    fig.update_layout(title="📈 移動平均線＋トレンド転換検出", xaxis_title="日時", yaxis_title="価格")

    if return_fig:
        return fig
    else:
        return fig.to_html(full_html=False)

def rsi_macd_chart(df, return_fig=False):
    df = df.sort_values("timestamp").copy()
    close = df["price"]

    # RSI計算
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD計算
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # シグナル抽出
    buy_points = []
    sell_points = []
    for i in range(1, len(df)):
        if df["RSI"].iloc[i] < 30 and df["RSI"].iloc[i-1] >= 30:
            buy_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))
        if df["RSI"].iloc[i] > 70 and df["RSI"].iloc[i-1] <= 70:
            sell_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))
        if df["MACD"].iloc[i] > df["Signal"].iloc[i] and df["MACD"].iloc[i-1] <= df["Signal"].iloc[i-1]:
            buy_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))
        if df["MACD"].iloc[i] < df["Signal"].iloc[i] and df["MACD"].iloc[i-1] >= df["Signal"].iloc[i-1]:
            sell_points.append((df["timestamp"].iloc[i], df["price"].iloc[i]))

    # サブプロット構成
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.25, 0.25],
                        vertical_spacing=0.05,
                        subplot_titles=("価格チャート", "RSI", "MACD"))

    # 価格チャート
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], name='価格'), row=1, col=1)

    # シグナルマーカー（価格チャート上に追加）
    for t, p in buy_points:
        fig.add_trace(go.Scatter(x=[t], y=[p], mode="markers",
                                 marker=dict(color="green", size=10, symbol="triangle-up"),
                                 name="買いシグナル"), row=1, col=1)
    for t, p in sell_points:
        fig.add_trace(go.Scatter(x=[t], y=[p], mode="markers",
                                 marker=dict(color="red", size=10, symbol="triangle-down"),
                                 name="売りシグナル"), row=1, col=1)

    # RSIチャート
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], name='RSI'), row=2, col=1)
    fig.add_shape(type="line", x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1], y0=70, y1=70,
                  line=dict(color="red", dash="dash"), row=2, col=1)
    fig.add_shape(type="line", x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1], y0=30, y1=30,
                  line=dict(color="green", dash="dash"), row=2, col=1)

    # MACDチャート
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"], name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["Signal"], name='Signal'), row=3, col=1)

    fig.update_layout(height=800, title="📍 RSI・MACD付きチャート", xaxis_title="日時", yaxis_title="値")

    if return_fig:
        return fig
    else:
        return fig.to_html(full_html=False)

# ✅ PNG保存関数
def save_chart_as_png(df, chart_type="line", filename="static/chart.png"):
    fig = go.Figure()
    for symbol in df["symbol"].unique():
        for action in df["action"].unique():
            subset = df[(df["symbol"] == symbol) & (df["action"] == action)]
            marker = dict(symbol="circle" if action == "buy" else "x", size=8)

            if chart_type == "line":
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["price"],
                                         mode="lines+markers", name=f"{symbol} ({action})",
                                         marker=marker))
            elif chart_type == "bar":
                fig.add_trace(go.Bar(x=subset["timestamp"], y=subset["price"],
                                     name=f"{symbol} ({action})"))
            elif chart_type == "scatter":
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=subset["price"],
                                         mode="markers", name=f"{symbol} ({action})",
                                         marker=marker))

            if len(subset) >= 2:
                X = np.array((subset["timestamp"] - pd.Timestamp("1970-01-01")).dt.total_seconds()).reshape(-1, 1)
                y = subset["price"].values
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                fig.add_trace(go.Scatter(x=subset["timestamp"], y=y_pred,
                                         mode="lines", name=f"{symbol} ({action}) 回帰線",
                                         line=dict(dash="dot", color="gray")))

    fig.update_layout(title="📥 保存用チャート", xaxis_title="日時", yaxis_title="価格")
    pio.write_image(fig, filename, format="png")

    return filename
    fig.update_layout(title="📍 RSI・MACD 指標", xaxis_title="日時", yaxis_title="値")
    return fig.to_html(full_html=False)
def save_chart_as_png_by_type(df, chart_type="main", filename="static/chart.png"):
    if chart_type == "main":
        fig = generate_main_chart(df, chart_type="line", return_fig=True)
    elif chart_type == "pie":
        fig = generate_pie_chart(df, return_fig=True)
    elif chart_type == "ma":
        fig = moving_average_chart(df, return_fig=True)
    else:
        return None
    fig.write_image(filename)
    return filename

def convert_log_to_csv(log_path="uploads/webhook.log", csv_path="static/webhook.csv"):
    if not os.path.exists(log_path):
        print("ログファイルが存在しません")
        return

    with open(log_path, "r", encoding="utf-8") as log_file, open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "symbol", "price", "time", "action"])

        for line in log_file:
            try:
                entry = json.loads(line)
                data = entry.get("data", {})
                writer.writerow([
                    entry.get("timestamp"),
                    data.get("symbol"),
                    data.get("price"),
                    data.get("time"),
                    data.get("action", "")
                ])
            except json.JSONDecodeError:
                continue
