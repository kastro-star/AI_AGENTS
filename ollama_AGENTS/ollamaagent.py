from ollama import chat
from pydantic import BaseModel
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.tools import BaseTool
import yfinance as yf


class Identifier(BaseModel):
  is_search: bool
  is_weather: bool
  is_yfinance: bool

class Weather(BaseModel):
  place: str

YFinanceToolsPrompt = """
  your are a contains all the information about the stock market and stock prices.
  You are a agent good in finding stock prices from the user's query.
  tell the user the stock price of the company.
  further tell the user the analyst recommendations for the company.
  User's query: {user_input}
  Stock Price:
"""

WEATHER_PROMPT = """
  You are a agent good in finding places from the user's query.
  User's query: {user_input}
  Place:
"""

IDENTIFIER_PROMPT = """
  User's query: {user_input}

  You are an agent specialized in query classification.
  Your task is to determine if the user's query requires an internet search, weather information, or stock market data.
  
  Guidelines:
  - Classify as search true if the query is general knowledge (e.g. "Who is...", "What is...", "When did...")
  - Classify as search if the query asks about facts, history, or current events
  - Do not classify as search if the query is about weather or local conditions
  - Classify as weather if the query asks about temperature, precipitation, or weather conditions
  - Classify as weather if the query mentions forecast, weather, rain, sun, etc.
  - Classify as weather if the query asks about current or future weather conditions for a location
  - Classify as yfinance if the query asks about stocks, stock prices, or company market data
  - Classify as yfinance if the query mentions stock symbol, share price, or market value

  Analyze the query and determine the appropriate classification.
"""

#--------------------------------------Functions--------------------------------------
def search(user_input: str):
  search = DuckDuckGoSearchRun()

  search = search.invoke(user_input)
  return search

def yfinance_result(user_input: str):
    try:
        # Look for stock symbols or company names
        words = user_input.upper().split()
        common_words = {'STOCK', 'PRICE', 'OF', 'THE', 'AND', 'FOR', 'IN', 'IS'}
        potential_symbols = [word for word in words if word not in common_words]
        
        for symbol in potential_symbols:
            ticker = yf.Ticker(symbol)
            try:
                info = ticker.info
                if info and 'longName' in info:
                    return {
                        'name': info.get('longName'),
                        'symbol': symbol,
                        'price': info.get('currentPrice'),
                        'currency': info.get('currency', 'USD'),
                        'recommendation': info.get('recommendationKey', 'N/A')
                    }
            except:
                continue
        return "Could not find valid stock information. Please provide a valid stock symbol."
    except Exception as e:
        return f"Error: {str(e)}"


def weather_call(user_input: str):
  # Importing necessary libraries

  weather = OpenWeatherMapAPIWrapper(openweathermap_api_key="f321902e1472611b6574e8b293b256ad")  

  weather_call = weather.run(user_input)
  return weather_call

#--------------------------------------Completion--------------------------------------
def struct_completion(AgentClass: BaseModel, user_input: str):
  response = chat(
      messages=[
        {
        'role': 'user',
        'content': user_input,
      }
    ],
    model='llama3.2',
    format=AgentClass.model_json_schema(),
  )

  filtered_response = AgentClass.model_validate_json(response.message.content)
  return filtered_response
  
def completion(user_input: str):
  response = chat(
      messages=[
        {
        'role': 'user',
        'content': user_input,
      }
    ],
    model='llama3.2',
  )

  return response.message.content

#--------------------------------------Main--------------------------------------

user_input = input("Enter your query: ")

identifier = struct_completion(Identifier, IDENTIFIER_PROMPT.format(user_input=user_input))
print(identifier)

if identifier.is_yfinance:
  print("YFinance Agent is working...")
  yfinance_data = yfinance_result(user_input)
  if isinstance(yfinance_data, dict):
    final_result = f"""
Stock Information:
- Company: {yfinance_data['name']}
- Symbol: {yfinance_data['symbol']}
- Current Price: {yfinance_data['price']} {yfinance_data['currency']}
- Analyst Recommendation: {yfinance_data['recommendation']}
"""
  else:
    final_result = yfinance_data
  print(final_result)

if identifier.is_search:
  print("Search Agent is working...")
  search_result = search(user_input)
  final_result = completion(f"from the given search result {search_result}, summarize the search result.tell as an local madurai person as all details.")
  print(final_result)

if identifier.is_weather:
  print("Weather Agent is working...")
  weather_place = struct_completion(Weather, WEATHER_PROMPT.format(user_input=user_input))
  weather_result = weather_call(weather_place.place)
  print(weather_result)
  final_result = completion(f"The weather in {weather_place.place} is {weather_result}. from the given weather result, summarize the weather in {weather_place.place}. do not add any other text except the weather summary.")

  print(final_result)