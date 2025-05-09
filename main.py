import streamlit as st
from utils import *

# Page configuration
st.set_page_config(page_title="Car Recommendation System", layout="wide")

# App title and introduction
st.title("Smart Car Recommendation System")
st.markdown(
    """
This application helps you find the perfect car based on your preferences. Simply describe what 
you're looking for in natural language, and our AI will recommend suitable second hand cars from our database.
"""
)

# Main content area
query_container = st.container()
results_container = st.container()

with query_container:
    st.subheader("What kind of car are you looking for?")
    query_example = "I need a 7 seater family SUV with low price and great features. I'll prefer petrol engine with low power"
    user_query = st.text_area(
        "Describe your ideal car", placeholder=query_example, height=100
    )

    col = st.container()
    with col:
        search_button = st.button("Find Cars", type="primary")

# Process and display results
if search_button and user_query:
    with st.spinner("🔍 Searching for the perfect car match..."):
        try:
            # Get vector embeddings
            sparse_vectors = get_query_sparse_embeddings(user_query)
            query_vectors = get_query_dense_embeddings(user_query)

            # Connect to Pinecone and query
            index = connect_to_pinecone()
            search_results = index.query(
                vector=query_vectors,
                top_k=10,
                include_metadata=True,
                sparse_vector={
                    "indices": sparse_vectors["indices"],
                    "values": sparse_vectors["values"],
                },
            )

            # Process with Gemini
            client = initialize_gemini()
            prompt = f"""Analyze the User's Query:
                Parse the query to extract all essential details such as:
                    Car Make/Model/Year (e.g., "Honda Civic 2012")
                    Location (e.g., "Lahore, Punjab")
                    Color Requirements (e.g., "white")
                    Engine Type (e.g., "Petrol")
                    Transmission (e.g., "Manual" or "Automatic")
                    Mileage (e.g., "around 33,000 km")
                    Body Type (e.g., "Sedan")
                    Features (e.g., "AM/FM Radio, Alloy Rims, Air Bags, Power Steering")
                    Price Range/Conditions (e.g., "Negotiable" or specific amount)

            Evaluate the Provided Content:
                Review each car record from the provided content. Each record contains detailed information such as car model, year, location, engine type, mileage, features, price, and more.
                Compare the details in each record with the user's query:
                    Exact and Partial Matches: Identify which records meet or closely align with the specifications.
                    Relevance Ranking: Rank the records based on how well they match the user's criteria.
                    Top 5 Selection: Choose the top 5 records that best satisfy the user's requirements.

            Generate the Final Response:
                If at least one matching record is found, return a formatted list of the top 5 best car recommendations.
                Each recommendation should include all relevant details: Car Make/Model, Year, Location, Color, Engine Type, Transmission, Mileage, Body Type, Features, and Price.
                Use clear bullet points or numbered lists for readability.
                Include a brief summary of why each car is recommended relative to the query.
                Important: If no records in the provided content match the user's query, then return only the message:
                "NOTHING"

            Formatting Requirements:
                Return the answer as a very well-structured list.
                Ensure that the response is clear, structured, and professional.
                Use markdown formatting including headers, bold text, and lists.
                
            RETURN ONLY RECOMMENDED CARS WITH PROPER FORMATTING, NOTHING ELSE

            User's Query:
            {user_query}

            Content:
            {search_results}
            """

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )

            with results_container:
                st.subheader("Best Car Matches For You")
                if "NOTHING" in response.text:
                    st.error(
                        "Sorry, we couldn't find any cars matching your criteria. Please try a different search query or adjust your filters."
                    )
                else:
                    st.markdown(response.text)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Additional sections below results
if not search_button or not user_query:
    with results_container:
        # Show sample queries to help users
        st.subheader("Example Queries")
        example_queries = [
            "I need a family SUV with 7 seats, automatic transmission, and good fuel economy under 5 million",
            "Looking for a luxury sedan with leather seats and sunroof in Islamabad",
            "I want a small hatchback for city driving with low maintenance cost and good mileage",
            "Need a 4x4 vehicle for off-road use with diesel engine and under 10 million budget",
        ]

        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                st.info(example)
