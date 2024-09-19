from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncChromiumLoader

# Supresses warning
import os
os.environ['USER_AGENT'] = 'myagent'

def extract_webpage_data(urls, tags=["h3", "p"]):
    
    # Load HTML content using AsyncChromiumLoader
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()

    # Transform the loaded HTML using BeautifulSoupTransformer
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=tags
    )

    return [doc.page_content for doc in docs_transformed]

urls = ['https://zerodha.com/varsity/chapter/supplementary-note-the-20-market-depth/',
        'https://zerodha.com/varsity/chapter/key-events-and-their-impact-on-markets/',
        'https://zerodha.com/varsity/chapter/momentum-portfolios/',
#        'https://zerodha.com/varsity/chapter/sector-analysis-overview/',
#        'https://zerodha.com/varsity/chapter/cement/',
#        'https://zerodha.com/varsity/chapter/episode-1-ideas-by-the-lake/',
#        'https://zerodha.com/varsity/chapter/why-do-stock-prices-fluctuate/',
#        'https://zerodha.com/varsity/chapter/why-should-you-invest/',
#        'https://zerodha.com/varsity/chapter/who-are-the-different-actors-in-market/',
#        'https://zerodha.com/varsity/chapter/why-and-how-do-companies-list-and-what-is-an-ipo/',
#        'https://zerodha.com/varsity/chapter/understanding-corporate-actions-like-dividends-bonuses-and-buybacks/',
#        'https://zerodha.com/varsity/chapter/understanding-the-various-order-types/',
#        'https://zerodha.com/varsity/chapter/getting-started-2/',
#        'https://zerodha.com/varsity/chapter/what-is-a-stock-market-index/',
#        'https://zerodha.com/varsity/chapter/how-does-a-trading-platform-work/',
#        'https://zerodha.com/varsity/chapter/fundamental-analysis-vs-technical-analysis/',
#        'https://zerodha.com/varsity/chapter/setting-realistic-expectations/',
#        'https://zerodha.com/varsity/chapter/types-of-charts/',
#        'https://zerodha.com/varsity/chapter/timeframes/',
#        'https://zerodha.com/varsity/chapter/key-assumption-of-technical-analysis/',
#        'https://zerodha.com/varsity/chapter/understanding-candlestick-patterns/',
#        'https://zerodha.com/varsity/chapter/single-candlestick-patterns/',
#        'https://zerodha.com/varsity/chapter/multiple-candlestick-patterns/',
#        'https://zerodha.com/varsity/chapter/support-and-resistance/',
#        'https://zerodha.com/varsity/chapter/technical-indicators/',
#        'https://zerodha.com/varsity/chapter/your-trading-checklist/',
#        'https://zerodha.com/varsity/chapter/moving-averages-2/',
#        'https://zerodha.com/varsity/chapter/introduction-to-fundamental-analysis/',
#        'https://zerodha.com/varsity/chapter/mindset-of-an-investor/',
#        'https://zerodha.com/varsity/chapter/how-to-read-the-annual-report-of-a-company/',
#        'https://zerodha.com/varsity/chapter/understanding-the-pl-statement/',
#        'https://zerodha.com/varsity/chapter/understanding-the-balance-sheet-statement/',
#        'https://zerodha.com/varsity/chapter/understanding-the-cash-flow-statement/',
#        'https://zerodha.com/varsity/chapter/the-connection-between-balance-sheet-pl-statement-and-cash-flow-statement/',
#        'https://zerodha.com/varsity/chapter/the-financial-ratio-analysis/',
#        'https://zerodha.com/varsity/chapter/quick-note-on-relative-valuation/',
#        'https://zerodha.com/varsity/chapter/fundamental-investment-checklist/',
#        'https://zerodha.com/varsity/chapter/introduction-to-forwards-market/',
#        'https://zerodha.com/varsity/chapter/introducing-the-futures-contract/',
#        'https://zerodha.com/varsity/chapter/margins/',
#        'https://zerodha.com/varsity/chapter/leverage-and-payoff/',
#        'https://zerodha.com/varsity/chapter/futures-trade-2/',
#        'https://zerodha.com/varsity/chapter/settlement/',
#        'https://zerodha.com/varsity/chapter/open-interest-2/',
#        'https://zerodha.com/varsity/chapter/shorting-futures/',
#        'https://zerodha.com/varsity/chapter/overview-of-contracts/',
#        'https://zerodha.com/varsity/chapter/introduction-to-options/',
#        'https://zerodha.com/varsity/chapter/option-jargons/',
#        'https://zerodha.com/varsity/chapter/long-call-payoff-and-short-call-trade/',
#        'https://zerodha.com/varsity/chapter/put-buy-and-put-sell/',
#        'https://zerodha.com/varsity/chapter/summarizing-call-put-options-2/',
#        'https://zerodha.com/varsity/chapter/moneyness-of-option/',
#        'https://zerodha.com/varsity/chapter/the-option-greeks-delta/',
#        'https://zerodha.com/varsity/chapter/gamma/',
#        'https://zerodha.com/varsity/chapter/theta-2/',
#        'https://zerodha.com/varsity/chapter/vega-2/',
#        'https://zerodha.com/varsity/chapter/options-m2m-and-pl-calculation/',
#        'https://zerodha.com/varsity/chapter/physical-settlement-of-futures-and-options/',
#        'https://zerodha.com/varsity/chapter/bull-call-spread-2/',
#        'https://zerodha.com/varsity/chapter/the-straddle/',
#        'https://zerodha.com/varsity/chapter/the-ipo-markets-part-1/',
#        'https://zerodha.com/varsity/chapter/the-vegetable-list/',
#        'https://zerodha.com/varsity/chapter/exchange-traded-funds-etf/',
#        'https://zerodha.com/varsity/chapter/clearing-and-settlement-process/',
#        'https://zerodha.com/varsity/chapter/impact-cost-and-how-it-can-ruin-a-trade/',
#        'https://zerodha.com/varsity/chapter/5-types-of-share-capital/',
#        'https://zerodha.com/varsity/chapter/how-ofs-allotment-is-done/',
#        'https://zerodha.com/varsity/chapter/building-a-mutual-fund-portfolio/',
#        'https://zerodha.com/varsity/chapter/clearing-and-settlement-process-2/',
#        'https://zerodha.com/varsity/chapter/five-corporate-actions-and-its-impact-on-stock-prices/',
#        'https://zerodha.com/varsity/chapter/iron-condor/',
#        'https://zerodha.com/varsity/chapter/who-can-raise-funds-on-sse/',
#        'https://zerodha.com/varsity/chapter/hotels-part-1/',
#        'https://zerodha.com/varsity/chapter/hotels-part-2/',
#        'https://zerodha.com/varsity/chapter/modes-of-raising-funds-part-1-zczp-and-other-instruments/',
#        'https://zerodha.com/varsity/chapter/modes-of-raising-funds-part-2/',
#        'https://zerodha.com/varsity/chapter/margin-m2m/',
#        'https://zerodha.com/varsity/chapter/foreign-stocks-and-taxation/',
#        'https://zerodha.com/varsity/chapter/the-haircut-affair/',
#        'https://zerodha.com/varsity/chapter/steel-part-2/',
#        'https://zerodha.com/varsity/chapter/steel-part-1/',
#        'https://zerodha.com/varsity/chapter/banking-part-2/',
#        'https://zerodha.com/varsity/chapter/banking-part-1/',
#        'https://zerodha.com/varsity/chapter/automobiles-part-1/',
#        'https://zerodha.com/varsity/chapter/automobiles-part-2/',
#        'https://zerodha.com/varsity/chapter/information-technology/',
#        'https://zerodha.com/varsity/chapter/insurance-part-1/',
#        'https://zerodha.com/varsity/chapter/understanding-insurance-sector-part-2/',
#        'https://zerodha.com/varsity/chapter/government-securities/',
#        'https://zerodha.com/varsity/chapter/retail-part-1/',
#        'https://zerodha.com/varsity/chapter/retail-part-2/',
#        'https://zerodha.com/varsity/chapter/social-stock-exchanges-an-introduction/',
#        'https://zerodha.com/varsity/chapter/basics/',
#        'https://zerodha.com/varsity/chapter/classifying-your-market-activity/',
#        'https://zerodha.com/varsity/chapter/taxation-for-traders/',
#        'https://zerodha.com/varsity/chapter/taxation-for-investors/',
#        'https://zerodha.com/varsity/chapter/turnover-balance-sheet-and-pl/',
#        'https://zerodha.com/varsity/chapter/itr-forms/'
]
    
if __name__ == "__main__":
    data = ''.join(str(x+'\n\n') for x in extract_webpage_data(urls))

    with open('data.txt', 'w', encoding="utf-8") as file:
        file.write(data)
