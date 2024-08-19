from flask import Flask, jsonify, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import minimize

app = Flask(__name__)

# Function to calculate annual return
def calculate_annual_return(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return "Error: No data found"
    start_value = data['Close'].iloc[0]
    end_value = data['Close'].iloc[-1]
    annual_return = (end_value - start_value) / start_value * 100
    return annual_return

# Define the asset classes and their respective tickers
assets = {
    'Equities': '^ISEQ',          # ISEQ Overall Index
    'Bonds': 'IBGL.L',            # iShares Euro Government Bond 7-10yr UCITS ETF
    'Real Estate': 'IYR',         # iShares U.S. Real Estate ETF
    'Cash': 'BIL',                # SPDR Bloomberg Barclays 1-3 Month T-Bill ETF
}

# Define the time period
start_date = '2023-01-01'
end_date = '2023-12-31'

# Fetch data and calculate returns
annual_returns = {}
for asset, ticker in assets.items():
    try:
        annual_return = calculate_annual_return(ticker, start_date, end_date)
        annual_returns[asset] = annual_return
    except Exception as e:
        annual_returns[asset] = f"Error fetching data: {e}"

# Function to predict annual coffee spending using machine learning
def predict_annual_spending(data):
    X = data[['Drinks Coffee', 'Daily Cups', 'Reusable Cup', 'Importance Morning']]
    X = pd.get_dummies(X, drop_first=True)
    y = data['Annual Spend']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Pooled investment strategy
def pooled_investment_strategy(annual_cost, allocation):
    investment = {asset: annual_cost * proportion for asset, proportion in allocation.items()}
    return investment

# Monte Carlo simulation
def monte_carlo_simulation(investment, years, num_simulations):
    np.random.seed(42)
    simulation_results = {asset: [] for asset in investment}
    
    for _ in range(num_simulations):
        for asset, amount in investment.items():
            annual_return = annual_returns.get(asset, 0) / 100
            future_value = amount * (1 + np.random.normal(annual_return, 0.1, years)).prod()
            simulation_results[asset].append(future_value)
    
    return simulation_results

# Portfolio optimization
def portfolio_optimization(mean_returns, cov_matrix, risk_free_rate=0.01):
    num_assets = len(mean_returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets,]
    
    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Individual returns calculation
def individual_returns(data, allocation, annual_returns, years=10):
    individual_future_returns = []

    for i in range(len(data)):
        annual_investment = data.iloc[i]['Annual Spend'] * data.iloc[i]['Total Investment Proportion']
        individual_investment = {asset: annual_investment * proportion for asset, proportion in allocation.items()}
        future_returns = monte_carlo_simulation(individual_investment, years, num_simulations=1)
        mean_return = sum(np.mean(values) for values in future_returns.values())
        individual_future_returns.append(mean_return)

    return individual_future_returns

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    age_range = data.get('ageRange')
    gender = data.get('gender')
    drinks_coffee = data.get('drinksCoffee') == 'true'
    daily_cups = int(data.get('dailyCups'))
    libido_effect = data.get('libidoEffect')
    reusable_cups = data.get('reusableCups')
    importance_morning = data.get('importanceMorning')

    coffee_price = 3.5
    annual_spend = daily_cups * coffee_price * 365

    user_data = pd.DataFrame({
        'Age': [age_range],
        'Gender': [gender],
        'Drinks Coffee': [drinks_coffee],
        'Daily Cups': [daily_cups],
        'Libido Effect': [libido_effect],
        'Reusable Cup': [reusable_cups],
        'Annual Spend': [annual_spend],
        'Importance Morning': [importance_morning]
    })

    user_data['Total Investment Proportion'] = np.random.uniform(0.05, 0.2)
    allocation = {'Equities': 0.5, 'Bonds': 0.3, 'Real Estate': 0.15, 'Cash': 0.05}
    user_data['Individual Future Return'] = individual_returns(user_data, allocation, annual_returns)

    roi = user_data['Individual Future Return'][0]
    roi_breakdown = {asset: roi * proportion for asset, proportion in allocation.items()}

    result = {
        'roi': roi,
        'roi_breakdown': roi_breakdown,
        'assets': [
            {'name': asset, 'return': annual_returns[asset]} for asset in allocation
        ]
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
