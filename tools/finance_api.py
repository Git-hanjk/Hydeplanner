from typing import Dict, List

import yfinance as yf
from langchain.tools import Tool


class FinanceDataTool:
    def __init__(self):
        self.company_tickers = {
            "lg_energy": "373220.KS",  # LG Energy Solution
            "samsung_sdi": "006400.KS",  # Samsung SDI
            "sk_innovation": "096770.KS",  # SK Innovation (SK On)
            "catl": "300750.SZ",  # CATL (China)
            "byd": "1211.HK",  # BYD (Hong Kong)
            "panasonic": "6752.T",  # Panasonic (Tokyo)
        }

    def get_company_info(self, company_name: str) -> Dict:
        ticker_symbol = self.company_tickers.get(company_name.lower().replace(" ", "_"))

        if not ticker_symbol:
            return {"error": f"Company {company_name} not found in database"}

        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            return {
                "name": info.get("longName", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "revenue": info.get("totalRevenue", "N/A"),
                "profit_margin": info.get("profitMargins", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "employees": info.get("fullTimeEmployees", "N/A"),
                "website": info.get("website", "N/A"),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_financial_metrics(self, company_name: str) -> Dict:
        ticker_symbol = self.company_tickers.get(company_name.lower().replace(" ", "_"))

        if not ticker_symbol:
            return {"error": f"Company {company_name} not found"}

        try:
            ticker = yf.Ticker(ticker_symbol)

            financials = ticker.financials

            if financials.empty:
                return {"error": "No financial data available"}

            latest = financials.iloc[:, 0]

            return {
                "total_revenue": latest.get("Total Revenue", "N/A"),
                "gross_profit": latest.get("Gross Profit", "N/A"),
                "operating_income": latest.get("Operating Income", "N/A"),
                "net_income": latest.get("Net Income", "N/A"),
                "period": str(financials.columns[0])[:10],
            }
        except Exception as e:
            return {"error": str(e)}

    def compare_companies(self, companies: List[str]) -> Dict:
        comparison = {}

        for company in companies:
            info = self.get_company_info(company)
            comparison[company] = {
                "market_cap": info.get("market_cap", "N/A"),
                "revenue": info.get("revenue", "N/A"),
                "profit_margin": info.get("profit_margin", "N/A"),
            }

        return comparison

    def as_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="yahoo_finance",
            func=lambda q: self.get_company_info(q),
            description=(
                "Get financial data for battery companies. "
                "Available companies: LG Energy, Samsung SDI, CATL, BYD, Panasonic. "
                "Returns: market cap, revenue, margins, P/E ratio, etc."
            ),
        )
