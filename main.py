from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import logging 
from datetime import datetime
from LiveStockPricePipeline import FinnhubWebSocket
from TextFetchPipeline import TextFetchPipeline
from SignalGenerator import SignalGeneration
import threading
from html import escape
import time 

app = FastAPI()

# Example keys if needed
reddit_client_secret = 'sfFdYwuZWqofiqro51zGlKcJiC2YiQ'
reddit_client_id = 'mNEnO4swPUjezEf92dxgxg'
reddit_user_agent = 'LLM class project'
news_api_key = '4703e4dcf50c47e1b390c81a6d0f0080'
cohere_key = 'g1jNECQNHhEnlRhvMjba89qnPdeEPch9SvhmFMiN'
finnhub_token = 'ctahnvpr01qrt5hhnbg0ctahnvpr01qrt5hhnbgg'
tickers = ['AAPL','AMZN','TSLA']

text_pipeline = TextFetchPipeline(
    news_api_key,
    reddit_client_id,
    reddit_client_secret,
    reddit_user_agent,
    cohere_key
)

stock_pipeline = FinnhubWebSocket(finnhub_token,tickers)

signal_generator = SignalGeneration(buffer_size=30)


# Start the WebSocket in a separate thread when the application starts
@app.on_event("startup")
def start_websocket():
    threading.Thread(target=stock_pipeline.start, daemon=True).start()


@app.on_event("startup")
def start_vwap_collection():
    data_thread = threading.Thread(
        target=signal_generator.collect_vwap,
        args=(stock_pipeline,),
        daemon=True
    )
    data_thread.start()

@app.on_event("startup")
def start_text_aggregation():
    aggregation_thread = threading.Thread(
        target=text_pipeline.run_periodically,
        daemon=True
    )
    aggregation_thread.start()

class CombinedData(BaseModel):
    ticker: str
    VWAP: float
    time: str
    text: str
    sentiment: int
    probability: float
    MA_Crossover: int
    RSI: int
    Breakout: int
    Oscillator: int
    Signal: int


@app.get("/mock_data", response_model=List[CombinedData])
def get_mock_data():
    """
    Fetch the latest VWAP values and trading signals dynamically.
    """
    # Fetch the latest VWAP values from the stock pipeline
    latest_vwap = stock_pipeline.latest_vwap  # Dynamically fetch VWAP
    # Fetch the latest signals from the signal generator
    latest_signals = signal_generator.get_signals()  # Fetch generated signals
    # Fetch the latest news and reddit from the text pipeline 
    latest_news = text_pipeline.agg_text
    # Current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate mock data using both VWAP and signals
    mock_data = [
        CombinedData(
            ticker="AAPL",
            VWAP=latest_vwap.get("AAPL", 175.23),
            time=current_time,
            text=latest_news.get("AAPL","No data available."),
            sentiment=1,
            probability=0.89,
            MA_Crossover=latest_signals.get("AAPL", {}).get("SMA", 0),
            RSI=latest_signals.get("AAPL", {}).get("RSI", 0),
            Breakout=0,  # Placeholder if there's no Breakout signal
            Oscillator=latest_signals.get("AAPL", {}).get("Stochastic", 0),
            Signal=1
        ),
        CombinedData(
            ticker="AMZN",
            VWAP=latest_vwap.get("AMZN", 224.55),
            time=current_time,
            text=latest_news.get("AMZN","No data available."),
            sentiment=1,
            probability=0.85,
            MA_Crossover=latest_signals.get("AMZN", {}).get("SMA", 0),
            RSI=latest_signals.get("AMZN", {}).get("RSI", 0),
            Breakout=0,
            Oscillator=latest_signals.get("AMZN", {}).get("Stochastic", 0),
            Signal=1
        ),
        CombinedData(
            ticker="TSLA",
            VWAP=latest_vwap.get("TSLA", 400.45),
            time=current_time,
            text=latest_news.get("TSLA","No data available."),
            sentiment=1,
            probability=0.92,
            MA_Crossover=latest_signals.get("TSLA", {}).get("SMA", 0),
            RSI=latest_signals.get("TSLA", {}).get("RSI", 0),
            Breakout=0,
            Oscillator=latest_signals.get("TSLA", {}).get("Stochastic", 0),
            Signal=1
        ),
    ]

    return mock_data


# A list to store the trade logs
trade_log = []

def signal_icon(value: int) -> str:
    """Returns a checkmark if value=1, or a cross if value=0."""
    return "✔" if value == 1 else "✘"

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Dashboard</title>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                text-align: center;
                border: 1px solid black;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
        <script>
            // Function to fetch and update the table
            async function fetchData() {
                const response = await fetch('/mock_data');
                const data = await response.json();

                // Update the table body
                const tableBody = document.getElementById('table-body');
                tableBody.innerHTML = '';  // Clear existing rows

                data.forEach(item => {
                    const sentimentLabel = item.sentiment === 1 ? 'Positive' : 'Negative';
                    const row = `
                        <tr>
                            <td>${item.ticker}</td>
                            <td>${item.VWAP}</td>
                            <td>${item.time}</td>
                            <td>${item.text}</td>
                            <td>${sentimentLabel}</td>
                            <td>${item.probability}</td>
                            <td>${item.MA_Crossover === 1 ? '✅' : (item.MA_Crossover === -1 ? '❌' : 'Hold')}</td>
                            <td>${item.RSI === 1 ? '✅' : (item.RSI === -1 ? '❌' : 'Hold')}</td>
                            <td>${item.Breakout === 1 ? '✅' : (item.Breakout === -1 ? '❌' : 'Hold')}</td>
                            <td>${item.Oscillator === 1 ? '✅' : (item.Oscillator === -1 ? '❌' : 'Hold')}</td>
                            <td>${item.Signal === 1 ? '✅' : (item.Signal === -1 ? '❌' : 'Hold')}</td>
                            <td>
                                <form action="/buy" method="post">
                                    <input type="hidden" name="ticker" value="${item.ticker}">
                                    <input type="hidden" name="price" value="${item.VWAP}">
                                    <input type="number" name="amount" min="1" required>
                                    <button type="submit">Buy</button>
                                </form>
                            </td>
                            <td>
                                <form action="/sell" method="post">
                                    <input type="hidden" name="ticker" value="${item.ticker}">
                                    <input type="hidden" name="price" value="${item.VWAP}">
                                    <input type="number" name="amount" min="1" required>
                                    <button type="submit">Sell</button>
                                </form>
                            </td>
                        </tr>
                    `;
                    tableBody.insertAdjacentHTML('beforeend', row);
                });
            }

            // Refresh data every 5 seconds
            setInterval(fetchData, 5000);
            window.onload = fetchData;  // Fetch data when the page loads
        </script>
    </head>
    <body>
        <h1>Stock Dashboard</h1>
        <table border="1">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>VWAP</th>
                    <th>Time</th>
                    <th>Text</th>
                    <th>Sentiment</th>
                    <th>Probability</th>
                    <th>MA Crossover</th>
                    <th>RSI</th>
                    <th>Breakout</th>
                    <th>Oscillator</th>
                    <th>Signal</th>
                    <th>Buy</th>
                    <th>Sell</th>
                </tr>
            </thead>
            <tbody id="table-body">
                <!-- Rows will be dynamically populated here -->
            </tbody>
        </table>

        <h2>Trade Log</h2>
        <div id="trade-log">
            <ul>
    """
    if not trade_log:
        html_content += "<p>No trades have been made yet.</p>"
    else:
        for entry in trade_log:
            html_content += f"<li>{escape(entry)}</li>"
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/buy")
async def buy(ticker: str = Form(...), amount: int = Form(...), price: float = Form(...)):
    trade_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bought {amount} shares of {ticker} @ {price}"
    trade_log.append(trade_entry)
    return RedirectResponse(url="/", status_code=303)

@app.post("/sell")
async def sell(ticker: str = Form(...), amount: int = Form(...), price: float = Form(...)):
    trade_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sold {amount} shares of {ticker} @ {price}"
    trade_log.append(trade_entry)
    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
