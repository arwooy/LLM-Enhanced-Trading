# LLM-Enhanced-Trading
A live sentiment analysis system leveraging LLMs like FinGPT for real-time financial news and social media sentiment extraction to optimize trading strategies.

# Project Overview
Financial markets are unpredictable, and no formula reliably forecasts stock prices amid constant change. A compelling approach is to understand the market’s consensus—a collective sentiment shaped by participants' moods, expectations, and reactions. Capturing this sentiment in real-time from diverse sources, such as financial news and social media, is crucial. However, existing sentiment analysis tools often fall short, failing to provide traders with actionable insights.

This project addresses this gap using Large Language Models (LLMs), which excel in synthesizing complex language and sentiment patterns from multiple data sources. While sentiment prediction in financial texts is a well-explored problem, traditional models have met with mixed success. By deploying an LLM-driven system, we capture nuanced sentiment in real-time, designed to support trading strategies with higher predictive accuracy.

Our end-to-end system leverages LLMs for precise sentiment extraction and tracks dynamic sentiment shifts essential for financial markets. It integrates sentiment analysis into trading strategies, allowing for practical validation through performance metrics. This comprehensive approach combines LLM-based accuracy, live multi-source data, and real-world strategy testing to offer a robust and reliable tool for informed trading decisions.

# Benchmarking

The benchmarking of FinGPT, IBM Granite 3.0, and Meta LLaMA 3.1 was conducted using four key metrics: accuracy, precision, recall, and F1-score. These metrics comprehensively evaluate model performance by considering both the proportion of correct predictions and the balance between false positives and false negatives.

We used the Kaggle Financial Sentiment Analysis dataset for this exercise, as it features financial news headlines and their corresponding sentiment labels. This dataset is well-suited for assessing models in a trading context due to its domain-specific vocabulary and diverse sentiment expressions, capturing the complexities of financial language.

| Model             | Accuracy | Precision | Recall  | F1-Score |
|--------------------|----------|-----------|---------|----------|
| **FinGPT**         | 0.7462   | 0.7675    | 0.7642  | 0.7488   |
| **IBM Granite 3.0**| 0.5861   | 0.6942    | 0.5861  | 0.6207   |
| **Meta LLaMA 3.1** | 0.6565   | 0.6657    | 0.6565  | 0.6440   |


FinGPT demonstrated superior performance across all metrics, driven by its domain-specific fine-tuning using Low-Rank Adaptation (LoRA). LoRA specializes in optimizing only a subset of model parameters, introducing low-rank matrices to capture task-specific features. This technique ensures computational efficiency while enhancing the model's adaptability to financial jargon and sentiment patterns.

In comparison, IBM Granite 3.0 and Meta LLaMA 3.1 struggled due to their lack of targeted fine-tuning for financial contexts. While robust in general NLP tasks, these models failed to capture subtle sentiment shifts and financial nuances effectively.

FinGPT’s specialization and adaptability make it the most promising candidate for real-time decision-making in sentiment-driven trading strategies.

# Backtesting Sentiment Integration

Backtesting of the trading system was conducted using historical stock prices and sentiment data from Reddit (2022–2023). Two frameworks were evaluated:

1. **Baseline Framework**: Utilized traditional technical indicators without sentiment integration.
2. **Sentiment-Enhanced Framework**: Incorporated sentiment signals with technical indicators, dynamically adjusting trade sizes based on sentiment strength.
Key performance metrics such as Sharpe Ratio and Win Ratio were used to evaluate the effectiveness of sentiment integration.

### Backtesting Metrics

**Sharpe Ratio**: The **Sharpe Ratio** measures risk-adjusted returns by evaluating the excess return per unit of volatility. A higher Sharpe Ratio indicates better risk-adjusted performance. 

**Win Ratio**: The **Win Ratio** represents the percentage of profitable trades executed over the total number of trades. A higher Win Ratio indicates greater consistency in achieving profitable trades.  

### Backtesting Results and Comparison Table

Sentiment integration demonstrated significant improvements in trading performance across both the evaluated metrics:

**Sharpe Ratio**: Sentiment integration significantly boosted risk-adjusted returns across all tickers.

- TSLA's SMA strategy improved from **0.34 to 3.47**.
- AAPL and AMZN also exhibited substantial positive shifts.

**Win Ratio**: The proportion of profitable trades increased notably with sentiment integration.
  
- TSLA’s SMA strategy win ratio rose from **32.2% to 57.0%**.
- Similar improvements were observed in AAPL and AMZN for sentiment-driven strategies.


| **Tickers** | **Strat1 Sharpe** | **Strat1 Win Ratio** | **Strat2 Sharpe** | **Strat2 Win Ratio** | **Strat3 Sharpe** | **Strat3 Win Ratio** |
|-------------|--------------------|-----------------------|--------------------|-----------------------|--------------------|-----------------------|
| **Baseline**            |                    |                       |                    |                       |                    |                       |
| TSLA        | 0.34               | 32.2%                 | 0.15               | 52.3%                 | -1.58              | 70.7%                 |
| AAPL        | -4.03              | 29.9%                 | -0.97              | 49.5%                 | 1.20               | 78.0%                 |
| AMZN        | -2.75              | 30.1%                 | -0.85              | 49.2%                 | -1.05              | 78.5%                 |
| **With Sentiment**       |                    |                       |                    |                       |                    |                       |
| TSLA        | 3.47               | 57.0%                 | 2.37               | 51.4%                 | 1.79               | 64.3%                 |
| AAPL        | 2.13               | 54.9%                 | 1.58               | 50.9%                 | 1.61               | 72.1%                 |
| AMZN        | 3.14               | 64.3%                 | 2.32               | 52.0%                 | -1.03              | 65.1%                 |

### Key Insights 

1. **Enhanced Predictive Power**: Sentiment signals capture market dynamics and investor sentiment that traditional indicators often miss, particularly for volatile stocks like TSLA.
2. **Strategy Versatility**: Sentiment integration complements various trading strategies, significantly improving their robustness and effectiveness.
3. **Stock-Specific Sensitivity**: Sentiment-driven strategies work exceptionally well for sentiment-sensitive stocks like TSLA and AMZN, showcasing the model’s adaptability to specific market conditions.
4. **Practical Utility for Traders**: Sentiment-integrated strategies offer actionable insights, enhancing decision-making even when technical indicators produce ambiguous signals.

These findings demonstrate that integrating sentiment with technical indicators improves both risk-adjusted performance (Sharpe Ratio) and trade profitability consistency (Win Ratio). This approach reduces volatility and enables more reliable trading outcomes.



# Tutorial
[![User Tutorial](https://img.youtube.com/vi/6WdB-Rn9ieA/0.jpg)](https://youtu.be/6WdB-Rn9ieA)

