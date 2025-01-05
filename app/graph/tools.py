from typing import List
from typing import Literal, Dict
import yfinance as yf
import pandas as pd
import logging
from langchain_core.tools import tool
from utils.vector_store import get_vector_store
from langchain_core.documents import Document


@tool(parse_docstring=True)
def retrieve_stocks_data(
        stock_symbols: List[str],
        period: Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'] = "1mo") -> Dict:
    
    """Retrieve essential stock data for given stock symbols.

    Args:
        stock_symbols (List[str]): List of stock symbols to retrieve data for
        period (Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], optional):
            Time period for historical data. Defaults to '1mo'

    Returns:
        Dict[str, Dict[str, dict]]: Dictionary with essential stock metrics
    """
    results = {}
    if not stock_symbols:
        raise ValueError("No stock symbols were found")
    for symbol in stock_symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(period)
        if hist.empty:
            raise ValueError("Invalid stock symbol")

        # Get only essential stock info
        info = stock.info
        essential_info = {
            'currentPrice': info.get('currentPrice'),
            'marketCap': info.get('marketCap'),
            'trailingPE': info.get('trailingPE'),
            'forwardPE': info.get('forwardPE'),
            'beta': info.get('beta'),
            'dividendYield': info.get('dividendYield'),
            'profitMargins': info.get('profitMargins'),
            'revenueGrowth': info.get('revenueGrowth'),
            'recommendationKey': info.get('recommendationKey'),
            'targetMeanPrice': info.get('targetMeanPrice')
        }

        results[symbol] = {
            "historical_data": summarize_stock_data(hist),
            "stock_info": essential_info
        }
    return results


@tool(parse_docstring=True)
def retreive_stock_indicators_for_single_stock(
    stock_symbol: str,
    period: Literal['1d', '5d', '1mo', '3mo', '6mo',
                    '1y', '2y', '5y', '10y', 'ytd', 'max'] = "1mo"
) -> dict:
    """Calculate key stock performance indicators for a given stock symbol.

    Args:
        stock_symbol (str): The stock symbol to analyze (e.g., "AAPL")
        period (Literal['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], optional):
            Time period for historical data. Defaults to "1mo"

    Returns:
        Dict[str, float | str]: Dictionary containing calculated stock performance indicators:
            - trend (str): "bullish" or "bearish" based on Simple Moving Averages (SMA)
            - rsi (float): Relative Strength Index value
            - support (float): Lowest historical price in the given period
            - resistance (float): Highest historical price in the given period
            - volume_trend (str): "increasing" or "decreasing" based on average volume change
            - momentum (str): "positive" or "negative"

    Raises:
        ValueError: If stock_symbol is empty or invalid
    """
   

    logging.info("---Calculating stock performance indicators---")

    # Validate stock symbol
    if not stock_symbol:
        raise ValueError("No stock symbol provided.")

    # Fetch historical data
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(period)
    if hist.empty:
        raise ValueError(f"Invalid stock symbol: {stock_symbol}")

    # Calculate technical indicators
    df = pd.DataFrame(hist)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])

    # Determine trend, support, resistance, and volume trend
    trend = "bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "bearish"
    support = df['Low'].min()
    resistance = df['High'].max()
    volume_trend = "increasing" if df['Volume'].pct_change(
    ).mean() > 0 else "decreasing"
    momentum = "positive" if df['Close'].pct_change(
        5).mean() > 0 else "negative"

    logging.info("---Stock performance indicators calculated successfully---")

    return {
        "trend": trend,
        "rsi": df['RSI'].iloc[-1],
        "support": support,
        "resistance": resistance,
        "volume_trend": volume_trend,
        "momentum": momentum
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)
    
    Args:
        prices (pd.Series): Series of prices
        period (int): The number of periods to use for RSI calculation (default is 14)
    
    Returns:
        pd.Series: RSI values for the given price series
    """
    # Calculate price differences
    delta = prices.diff()

    # Separate gains (up) and losses (down)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi



def summarize_stock_data(hist):
    """
    Generate essential summary of stock historical data
    
    Parameters:
    hist (pd.DataFrame): Historical stock data from yfinance
    
    Returns:
    dict: Key stock metrics
    """
    try:
        price_change = (hist['Close'].iloc[-1] -
                        hist['Close'].iloc[0]).round(2)
        price_change_pct = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) /
                            hist['Close'].iloc[0] * 100).round(2)

        summary = {
            "price_metrics": {
                "current_price": round(hist['Close'].iloc[-1], 2),
                "price_change": price_change,
                "price_change_percent": price_change_pct,
                "high": round(hist['High'].max(), 2),
                "low": round(hist['Low'].min(), 2)
            },
            "volume": int(hist['Volume'].mean()),
            "volatility": round(hist['Close'].pct_change().std() * 100, 2),
            "date_range": {
                "start": hist.index[0].strftime('%Y-%m-%d'),
                "end": hist.index[-1].strftime('%Y-%m-%d')
            }
        }
        return summary

    except Exception as e:
        raise ValueError(f"Error generating stock summary: {str(e)}")

@tool(parse_docstring=True)
def retrieve_news_data(news_data_request: str) -> List[Document]:
    """
    Retrieve relevant news documents based on a given query.

    Args:
        news_data_request (str): Query string to search for relevant news data

    Returns:
        List[Document]: List of Document objects containing:
            - page_content (str): Text content of the retrieved document
            - metadata (dict): Document metadata excluding 'embedding' field

    Note:
        - Uses MMR (Maximal Marginal Relevance) search with k=4 and fetch_k=10
        - Logs retrieval information and errors for debugging
    """
  
    try:
        # Get the vector store (ensure this is implemented properly elsewhere)
        vectorstore = get_vector_store()

        # Retrieve documents using the specified search parameters
        documents = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 4, 'fetch_k': 10}
        ).invoke(news_data_request)

        # Filter out the 'embedding' metadata and collect results
        vector_store_documents = []
        for doc in documents:
            page_content = doc.page_content
            metadata = doc.metadata
            # Remove 'embedding' from metadata to avoid unnecessary data
            new_metadata = {key: value for key,
                            value in metadata.items() if key != "embedding"}

            # Append the processed document to the result list
            vector_store_documents.append(
                Document(page_content=page_content, metadata=new_metadata))

        logging.info(
            f"{len(vector_store_documents)} documents retrieved for query: '{news_data_request}'")

        return vector_store_documents

    except Exception as e:
        logging.error(
            f"Error retrieving news data for query '{news_data_request}': {e}")
        return []


