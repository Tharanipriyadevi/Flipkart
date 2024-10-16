import streamlit as st
import pandas as pd
from langchain_community.llms import Cohere  # Ensure you have this package installed
import cohere

# Load sentiment data
sentiment_data = pd.read_csv("product_sentiment.csv")

# Initialize Cohere (Replace with your Cohere API Key)
cohere_api_key = 'tfKQMxhyKWSqauEbXbWih33cDMSf4YzkZevV2lr9'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

# Function to recommend products based on user query
def recommend_products(query, sentiment_data):
    # Analyze the user's input using Cohere
    response = co.generate(prompt=query, model='command-xlarge-nightly')

    # Process response (you can also use a custom NLP method if needed)
    search_keywords = response.generations[0].text.strip().split()

    # Filter sentiment data based on query analysis (e.g., find related product names or positive reviews)
    recommendations = sentiment_data[sentiment_data['Review'].str.contains('|'.join(search_keywords), case=False)]

    # Sort products by sentiment score
    recommendations = recommendations.sort_values(by='Sentiment_Score', ascending=False)

    # Return top 5 products
    return recommendations[['Product_Name']].head(5)

# Streamlit App Interface
st.set_page_config(page_title="Product Recommendation Engine", layout="wide")
st.title("🌟 Product Recommendation Engine 🌟")
st.markdown("### Find the best products based on your preferences!")
st.markdown("Enter your product preferences or query below, then click **Search**.")

# Input for user query
user_query = st.text_input("What are you looking for? (e.g., best camera phone, lightweight, etc.)", "")

# Search button
if st.button("Search"):
    if user_query:  # Check if the user has entered a query
        # Get top 5 product recommendations
        recommended_products = recommend_products(user_query, sentiment_data)

        if not recommended_products.empty:
            st.markdown("### Recommended Products:")
            # Display recommended products in a well-formatted manner
            for index, row in recommended_products.iterrows():
                st.markdown(f"- **{row['Product_Name']}**")  # Display each product as a bullet point
        else:
            st.warning("No recommendations found based on your query. Please try a different query.")
    else:
        st.warning("Please enter a product preference or query before clicking the search button.")

# Footer
st.markdown("---")
st.markdown("### Need Help?")
st.markdown("Feel free to reach out if you have any questions or need assistance!")

# To run the app, use the command below in your terminal:
# streamlit run "D:\vscode\apps.py"
