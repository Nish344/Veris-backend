# tools/tavily_tool.py
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# This tool will automatically use the TAVILY_API_KEY from your .env file
tavily_search = TavilySearchResults(max_results=5)

# You can test this tool directly by running this file
if __name__ == '__main__':
    print("--- Testing Tavily Search Tool ---")
    
    # Ensure your TAVILY_API_KEY is set in your .env file
    if not os.getenv("TAVILY_API_KEY"):
        print("ERROR: TAVILY_API_KEY is not set in the .env file.")
    else:
        query = "What is the latest news on the VerisProject?"
        results = tavily_search.invoke({"query": query})
        
        print(f"Query: {query}")
        print("Results:")
        for result in results:
            print(f"- URL: {result['url']}")
            print(f"  Content: {result['content'][:100]}...")
            print("-" * 20)
