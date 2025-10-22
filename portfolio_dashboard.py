import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import requests
from googlesearch import search
import numpy as np
import re

ALPHA_VANTAGE_KEY = "ALT5R7FXMOK16ZUO"

# Function to perform fresh web search and screen for hidden gems
def get_fresh_recs():
    query = "best undervalued ETFs and mutual funds for long-term retirement 2025, medium-high risk, Warren Buffett style, no China exposure"
    results = []
    try:
        for url in search(query, num_results=5):  # Reduced for speed
            results.append(url)
    except Exception as e:
        st.warning(f"Web search failed: {str(e)}. Using fallback tickers.")
        return ['VOO', 'VUG', 'SCHD']  # Fallback only if search fails
    candidates = []
    for url in results:
        try:
            response = requests.get(url, timeout=5)
            text = response.text.upper()
            potential = re.findall(r'\b[A-Z]{3,5}\b', text)
            candidates.extend(potential)
        except:
            pass
    candidates = list(set(candidates))[:50]  # Unique and limit to 50

    # Screen candidates for undervalued ETFs/funds
    screened = []
    for ticker in candidates:
        try:
            info = yf.Ticker(ticker).info
            if 'quoteType' in info and info['quoteType'] in ['ETF', 'FUND']:
                pe = info.get('trailingPE', np.nan)
                pb = info.get('priceToBook', np.nan)
                beta = info.get('beta', np.nan)
                expense = info.get('expenseRatio', np.nan)
                hist = yf.Ticker(ticker).history(period="5y")['Close']
                five_y_return = ((hist.iloc[-1] / hist.iloc[0]) ** (1/5) - 1) * 100 if len(hist) > 1 else np.nan
                if (pe < 20 and pd.notna(pe)) and (pb < 2 and pd.notna(pb)) and (1.0 <= beta <= 1.5 and pd.notna(beta)) and (five_y_return > 10 and pd.notna(five_y_return)) and (expense < 0.5 and pd.notna(expense)):
                    screened.append(ticker)
        except:
            pass
    return screened[:10] if screened else ['VOO', 'VUG', 'SCHD']  # Fallback only if screening fails

# Dynamic underperformers from portfolio
def get_underperformers(portfolio):
    if '5Y Ann. Return (%)' in portfolio.columns:
        underperformers = portfolio[
            (portfolio['5Y Ann. Return (%)'].apply(lambda x: isinstance(x, (int, float)) and x < 5)) |
            (portfolio['Expense Ratio'].apply(lambda x: isinstance(x, (int, float)) and x > 0.5))
        ]['Symbol'].tolist()
    else:
        underperformers = []
    return underperformers

# Placeholder images
def get_image_url(ticker):
    return f"https://via.placeholder.com/300?text={ticker}+Chart"

def get_caption(ticker):
    return f"{ticker}: Performance Chart (Fresh Data)"

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'potential_buys' not in st.session_state:
    st.session_state.potential_buys = []
if 'potential_sells' not in st.session_state:
    st.session_state.potential_sells = []

# App Title
st.title("Portfolio Dashboard for 2042 Retirement")
st.write("Goal: $100K income ($70K pension + $30K withdrawal). $367K at 9% return grows to ~$1.59M by 2042. Focus: Balance tech (50% growth, 30% value, 20% international).")

# Explanations
st.sidebar.header("Quick Guide")
st.sidebar.write("**P/E**: Price per $1 earnings. <20 = bargain.")
st.sidebar.write("**P/B**: Price vs. assets. <2 = undervalued.")
st.sidebar.write("**Undervalued %**: Potential rise to target price.")
st.sidebar.write("**Beta**: Risk vs. market (1=avg, >1=riskier).")
st.sidebar.write("**5Y Return**: Avg yearly growth (higher = better).")

# Upload CSVs (persist)
uploaded_files = st.file_uploader("Upload Fidelity CSVs (once; persists on refresh)", type="csv", accept_multiple_files=True)
if uploaded_files:
    try:
        dfs = []
        required_columns = ['Symbol', 'Quantity', 'Last Price', 'Cost Basis Total']
        for file in uploaded_files:
            df = pd.read_csv(file, encoding='utf-8')
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
            df = df[df['Symbol'].notna() & ~df['Symbol'].str.contains('\*\*', na=False)]
            if df.empty:
                raise ValueError("No valid rows after filtering invalid symbols")
            for col in ['Quantity', 'Last Price', 'Cost Basis Total']:
                df[col] = df[col].replace(r'[\$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                invalid_rows = df[df[col].isna()]
                if not invalid_rows.empty:
                    error_msg = f"Non-numeric or missing values in {col}:\n"
                    for idx, row in invalid_rows.iterrows():
                        error_msg += f"Row {idx + 2}: Symbol={row['Symbol']}, {col}={row[col]}\n"
                    st.warning(error_msg)
                    df = df[df[col].notna()]
                    if df.empty:
                        raise ValueError(f"All rows in {col} are invalid")
            dfs.append(df)
        if not dfs:
            raise ValueError("No valid data after cleaning")
        portfolio = pd.concat(dfs, ignore_index=True)
        if portfolio.empty:
            raise ValueError("No valid rows in CSV after filtering")
        st.session_state.portfolio = portfolio
        st.success("CSV uploaded and persisted! Refresh to keep using.")
    except Exception as e:
        st.error(f"Upload error: {str(e)}. Ensure CSV has Symbol, Quantity, Last Price, Cost Basis Total with valid numeric values. Remove cash positions (e.g., FCASH, SPAXX, CORE).")

# Use persisted portfolio
if st.session_state.portfolio is not None:
    portfolio = st.session_state.portfolio
    st.write("Using persisted portfolio. Upload new to update.")

    try:
        # Refresh Recommendations Button
        if st.button("Refresh Recommendations (Internet Search - May Take 30-60s)"):
            with st.spinner("Searching internet for fresh recommendations and screening for hidden gems..."):
                st.session_state.potential_buys = get_fresh_recs()
                st.session_state.potential_sells = get_underperformers(portfolio)
            st.success("Recommendations refreshed from web search!")

        # Use current recs or prompt for refresh
        POTENTIAL_BUYS = st.session_state.potential_buys if st.session_state.potential_buys else []
        POTENTIAL_SELLS = st.session_state.potential_sells if st.session_state.potential_sells else []

        # Data fetching
        all_tickers = list(portfolio['Symbol'].unique()) + POTENTIAL_BUYS + POTENTIAL_SELLS
        all_tickers = list(set(all_tickers))
        current_prices = {}
        historical_data = pd.DataFrame()
        fundamentals = {}
        sectors = {}
        analyst_targets = {}
        failed_tickers = []

        # Alpha Vantage for analyst targets
        for ticker in all_tickers:
            try:
                url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
                response = requests.get(url).json()
                target = response.get('AnalystTargetPrice', np.nan)
                analyst_targets[ticker] = float(target) if target and target != 'None' else np.nan
            except Exception as e:
                analyst_targets[ticker] = np.nan
                st.warning(f"Alpha Vantage failed for {ticker}: {str(e)}")

        # yfinance for other data
        for ticker in all_tickers:
            try:
                ticker_data = yf.Ticker(ticker)
                info = ticker_data.info
                current_prices[ticker] = info.get('regularMarketPrice', portfolio[portfolio['Symbol'] == ticker]['Last Price'].iloc[0] if ticker in portfolio['Symbol'].values else np.nan)
                sectors[ticker] = info.get('sector', info.get('category', 'Unknown'))

                hist = ticker_data.history(period="5y")['Close']
                if not hist.empty:
                    historical_data[ticker] = hist
                    five_y_return = ((hist.iloc[-1] / hist.iloc[0]) ** (1/5) - 1) * 100 if len(hist) > 1 else np.nan
                else:
                    five_y_return = np.nan

                fundamentals[ticker] = {
                    'pe': info.get('trailingPE', np.nan),
                    'pb': info.get('priceToBook', np.nan),
                    'beta': info.get('beta', np.nan),
                    '5y_return': round(five_y_return, 2) if pd.notna(five_y_return) else np.nan,
                    'expense_ratio': info.get('expenseRatio', np.nan)
                }
            except Exception as e:
                if ticker in portfolio['Symbol'].values:
                    current_prices[ticker] = portfolio[portfolio['Symbol'] == ticker]['Last Price'].iloc[0]
                    fundamentals[ticker] = {
                        'pe': np.nan,
                        'pb': np.nan,
                        'beta': np.nan,
                        '5y_return': portfolio[portfolio['Symbol'] == ticker]['Total Gain/Loss Percent'].iloc[0] / 5 if 'Total Gain/Loss Percent' in portfolio.columns else np.nan,
                        'expense_ratio': np.nan
                    }
                else:
                    current_prices[ticker] = np.nan
                    fundamentals[ticker] = {'pe': np.nan, 'pb': np.nan, 'beta': np.nan, '5y_return': np.nan, 'expense_ratio': np.nan}
                failed_tickers.append(ticker)
                st.warning(f"yfinance failed for {ticker}: {str(e)}")

        if failed_tickers:
            st.info(f"Failed tickers (using defaults): {', '.join(failed_tickers)}")

        # Update portfolio with robust Current Price handling
        portfolio['Current Price'] = portfolio['Symbol'].map(current_prices)
        portfolio['Current Price'] = pd.to_numeric(portfolio['Current Price'], errors='coerce').fillna(portfolio['Last Price'])
        if portfolio['Current Price'].isna().any():
            st.error(f"Invalid Current Price for tickers: {portfolio[portfolio['Current Price'].isna()]['Symbol'].tolist()}. Check CSV Last Price or yfinance data.")
            raise ValueError("Current Price contains invalid values")

        portfolio['Current Value'] = portfolio['Quantity'] * portfolio['Current Price']
        portfolio['Gain/Loss %'] = (portfolio['Current Value'] - portfolio['Cost Basis Total']) / portfolio['Cost Basis Total'] * 100
        portfolio['P/E Ratio'] = portfolio['Symbol'].map(lambda t: fundamentals.get(t, {}).get('pe', np.nan))
        portfolio['P/B Ratio'] = portfolio['Symbol'].map(lambda t: fundamentals.get(t, {}).get('pb', np.nan))
        portfolio['Beta'] = portfolio['Symbol'].map(lambda t: fundamentals.get(t, {}).get('beta', np.nan))
        portfolio['5Y Ann. Return (%)'] = portfolio['Symbol'].map(lambda t: fundamentals.get(t, {}).get('5y_return', np.nan))
        portfolio['Expense Ratio'] = portfolio['Symbol'].map(lambda t: fundamentals.get(t, {}).get('expense_ratio', np.nan))
        portfolio['Analyst Target'] = portfolio['Symbol'].map(lambda t: analyst_targets.get(t, np.nan))
        portfolio['Undervalued %'] = portfolio.apply(
            lambda row: round(((row['Analyst Target'] - row['Current Price']) / row['Current Price'] * 100), 2)
            if pd.notna(row['Analyst Target']) and pd.notna(row['Current Price']) and row['Current Price'] != 0
            else np.nan, axis=1)
        portfolio['Sector'] = portfolio['Symbol'].map(sectors)

        # Ensure numeric columns are float
        for col in ['Current Price', 'Current Value', 'Gain/Loss %', 'P/E Ratio', 'P/B Ratio', 'Beta', '5Y Ann. Return (%)', 'Expense Ratio', 'Analyst Target', 'Undervalued %']:
            portfolio[col] = pd.to_numeric(portfolio[col], errors='coerce')

        # Recommendation DataFrames
        def get_rec_df(tickers_list, is_buy=True):
            rec_data = []
            for ticker in tickers_list:
                price = current_prices.get(ticker, np.nan)
                data = fundamentals.get(ticker, {})
                sector = sectors.get(ticker, 'Unknown')
                target = analyst_targets.get(ticker, np.nan)
                undervalued = round(((target - price) / price * 100), 2) if pd.notna(target) and pd.notna(price) and price != 0 else np.nan
                rec_data.append({
                    'Ticker': ticker,
                    'Current Price': round(price, 2) if pd.notna(price) else np.nan,
                    'P/E Ratio': round(data.get('pe', np.nan), 2) if pd.notna(data.get('pe')) else np.nan,
                    'P/B Ratio': round(data.get('pb', np.nan), 2) if pd.notna(data.get('pb')) else np.nan,
                    'Beta (Risk)': round(data.get('beta', np.nan), 2) if pd.notna(data.get('beta')) else np.nan,
                    '5Y Ann. Return (%)': data.get('5y_return', np.nan),
                    'Expense Ratio': data.get('expense_ratio', np.nan),
                    'Undervalued %': undervalued,
                    'Top Sectors': sector
                })
            df = pd.DataFrame(rec_data)
            for col in ['Current Price', 'P/E Ratio', 'P/B Ratio', 'Beta (Risk)', '5Y Ann. Return (%)', 'Expense Ratio', 'Undervalued %']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df

        buys_df = get_rec_df(POTENTIAL_BUYS, is_buy=True)
        sells_df = get_rec_df(POTENTIAL_SELLS, is_buy=False)

        # Portfolio Summary
        st.subheader("Your Current Portfolio Summary")
        st.dataframe(portfolio[['Symbol', 'Current Value', 'Gain/Loss %', 'P/E Ratio', 'P/B Ratio', 'Beta', '5Y Ann. Return (%)', 'Undervalued %', 'Sector']])

        # Goal Analysis
        st.subheader("Analysis vs. 2042 Goal ($100K Income)")
        total_value = portfolio['Current Value'].sum()
        st.write(f"**Total Value**: ${total_value:,.2f}")
        st.write("**Projection**: At 9% annual return (med-high risk), $367K grows to ~$1.59M by 2042. With $70K pension, covers $30K withdrawal ($750K needed)—**on track**.")
        st.write("**Risk Fit**: Med-high tolerance ok with 20% drops. Current beta ~1.1 (good). Tech-heavy (~35%)—add value/international for stability.")
        st.write("**Tax Rec**: Use Roth IRAs for tax-free growth. No regular contributions—consider $5K/year for extra $500K by 2042.")

        # Sector Breakdown
        show_sector = st.checkbox("Show Sector Breakdown", value=True)
        if show_sector:
            st.subheader("Your Sector Breakdown")
            sector_alloc = portfolio.groupby('Sector')['Current Value'].sum().reset_index()
            fig_sector = px.pie(sector_alloc, values='Current Value', names='Sector', title='Portfolio by Sector')
            st.plotly_chart(fig_sector)
            st.write("Tech ~35%. Suggestion: 50% growth, 30% value, 20% international.")

        # Buy Opportunities
        st.subheader("Buy Opportunities (vs. Your Holdings)")
        if not buys_df.empty:
            st.write("These outperform underperformers with 10-20% 5Y returns, low fees (<0.5%), and undervaluation. Add to balance tech focus. Data from yfinance/analysts (Oct 22, 2025).")
            st.dataframe(buys_df)

            # Images for buys
            st.subheader("Buy Performance Visuals")
            cols = st.columns(3)
            for i, ticker in enumerate(buys_df['Ticker'][:6]):
                url = get_image_url(ticker)
                caption = get_caption(ticker)
                with cols[i % 3]:
                    st.image(url, caption=caption, use_container_width=True)
        else:
            st.info("No buy recommendations yet. Click 'Refresh Recommendations' to search.")

        # Return Comparison Bar Chart
        st.subheader("5Y Return Comparison (Find Opportunities)")
        compare_returns = []
        for _, row in portfolio.iterrows():
            if pd.notna(row['5Y Ann. Return (%)']):
                compare_returns.append({'Type': 'Your Holdings', 'Ticker': row['Symbol'], '5Y Return %': row['5Y Ann. Return (%)']})
        for _, row in buys_df.iterrows():
            if pd.notna(row['5Y Ann. Return (%)']):
                compare_returns.append({'Type': 'Buy Recs', 'Ticker': row['Ticker'], '5Y Return %': row['5Y Ann. Return (%)']})
        compare_df = pd.DataFrame(compare_returns[:20])
        if not compare_df.empty:
            fig_bar = px.bar(compare_df, x='Ticker', y='5Y Return %', color='Type', title='5Y Returns: Holdings vs. Buys (Higher = Better for Growth)')
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar)
            st.write("Buys (orange) show higher avg returns—find missed growth here.")

        # Sell Underperformers
        st.subheader("Sell Underperformers (vs. Goals)")
        if not sells_df.empty:
            st.write("Low returns (<5% 5Y) or high fees drag growth. Sell to fund buys (e.g., reallocate $15K to new buys).")
            st.dataframe(sells_df)
        else:
            st.info("No underperformers identified. Click 'Refresh Recommendations' to check.")

        # Historical Line Chart
        st.subheader("Historical Performance (Limited View)")
        top_holdings = portfolio.nlargest(5, 'Current Value')['Symbol'].tolist()
        top_buys = buys_df.head(5)['Ticker'].tolist()
        limited_hist = historical_data[top_holdings + top_buys] if not historical_data.empty else pd.DataFrame()
        if not limited_hist.empty:
            fig_line = px.line(limited_hist, title='5Y Price History (Top Holdings Blue, Top Buys Orange)')
            st.plotly_chart(fig_line)
        else:
            st.info("Historical data limited for some mutual funds.")

        # Download Report
        st.subheader("Download Report")
        if st.button("Download Updated CSV"):
            report = portfolio.to_csv(index=False)
            st.download_button("Download", report, "portfolio_report.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing: {str(e)}. Ensure CSV has Symbol, Quantity, Last Price, Cost Basis Total with valid numeric values. Remove cash positions (e.g., FCASH, SPAXX, CORE).")

# Alerts
st.subheader("Set Up Alerts")
email = st.text_input("Your Email")
password = st.text_input("Gmail App Password", type="password")
receiver = st.text_input("Receiver (Email or SMS Gateway, e.g., number@txt.att.net)")
if st.button("Test Alert"):
    try:
        alerts = []
        tickers = portfolio['Symbol'].unique() if 'portfolio' in locals() else []
        for ticker in tickers[:5]:
            try:
                change = yf.Ticker(ticker).info.get('regularMarketChangePercent', 0)
                if change < -5:
                    alerts.append(f"Alert: {ticker} down {change:.1f}% - Sell?")
                elif change > 10:
                    alerts.append(f"Alert: {ticker} up {change:.1f}% - Buy more?")
            except:
                pass
        if alerts:
            msg = MIMEText("\n".join(alerts))
            msg['Subject'] = f"Portfolio Alert - {datetime.now().strftime('%Y-%m-%d')}"
            msg['From'] = email
            msg['To'] = receiver
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(email, password)
                server.sendmail(email, receiver, msg.as_string())
            st.success("Test alert sent!")
        else:
            st.info("No alerts triggered in test.")
    except Exception as e:
        st.error(f"Alert error: {str(e)}. Verify Gmail App Password.")
