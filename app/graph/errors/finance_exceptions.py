class FinanceError(Exception):
    """Base class for all finance-related exceptions."""

    def chat_message(self):
        """Override in child classes to provide user-friendly messages."""
        return "An error occurred in the finance domain."


class MissingStockSymbolError(FinanceError):
    def __init__(self, message="No stock symbol provided. Please specify a valid stock symbol (e.g., AAPL for Apple)."):
        super().__init__(message)

    def chat_message(self):
        return (
            "It looks like you haven't provided any stock symbols. "
            "To assist you, could you please specify a stock symbol, such as **AAPL** for Apple or **TSLA** for Tesla? "
            "I'm here to help with your financial queries!"
        )


class InsufficientStockSymbolsError(FinanceError):
    def __init__(self, message="At least two stock symbols are required for comparison. Please provide more symbols."):
        super().__init__(message)

    def chat_message(self):
        return (
            "To compare stocks, I need at least two stock symbols. "
            "For example, you could compare **AAPL** (Apple) with **GOOG** (Google). "
            "Could you please provide at least two symbols for me to analyze and compare?"
        )


class InvalidStockSymbolError(FinanceError):
    def __init__(self, symbol=None, message=None):
        if message is None:
            if symbol:
                message = f"Invalid stock symbol: {symbol}"
            else:
                message = "Invalid stock symbol provided."
        super().__init__(message)
        self.symbol = symbol

    def chat_message(self):
        if self.symbol:
            return (
                f"The stock symbol **{self.symbol}** seems to be invalid. "
                "Please double-check the symbol and try again. "
                "For instance, **AAPL** represents Apple, and **MSFT** represents Microsoft. "
                "Let me know the correct symbol, and I'll assist you further!"
            )
        return (
            "The stock symbol provided seems to be invalid. "
            "Please double-check the symbol and try again. "
            "For instance, **AAPL** represents Apple, and **MSFT** represents Microsoft."
        )
