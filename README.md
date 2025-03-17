# Car Recommendation System

![car-recommendation-system](https://img.shields.io/badge/Project-Car%20Recommendation%20System-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-yellow)
![Gemini](https://img.shields.io/badge/Gemini%20AI-2.0-purple)

A smart car recommendation system that uses natural language processing, vector search, and AI to help users find their ideal vehicle based on specific preferences and requirements.

## 🚗 Features

- **Natural Language Search**: Simply describe the car you're looking for in plain language
- **AI-Powered Recommendations**: Get personalized car suggestions based on your specific needs
- **Hybrid Search**: Combines dense and sparse vector embeddings for more accurate results
- **User-Friendly Interface**: Clean, responsive UI built with Streamlit
- **Detailed Recommendations**: View comprehensive information about each recommended car

## 📋 Requirements

- Python 3.8+
- Poetry (dependency management)
- Streamlit
- Pinecone
- Sentence Transformers
- Google Generative AI (Gemini)
- Pinecone Text
- dotenv

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/car-recommendation-system.git
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root and add your API keys:
```
PINECON_API=your_pinecone_api_key
GEMINI_API=your_gemini_api_key
```

## 🚀 Usage

1. Start the application using Poetry:
```bash
poetry run main.py
```

2. Open your browser and navigate to the URL shown in the terminal

3. Enter your car preferences in natural language (e.g., "I need a 7 seater family SUV with low price and great features")

4. View the AI-generated recommendations based on your query

## 🏗️ System Architecture


The system uses a multi-step process to provide accurate recommendations:

1. **Natural Language Processing**: User queries are processed using Sentence Transformers
2. **Vector Embeddings**: Creates both dense and sparse embeddings for semantic and keyword-based search
3. **Vector Database**: Pinecone vector database stores and retrieves car data based on similarity
4. **AI Processing**: Gemini AI model analyzes search results and generates personalized recommendations
5. **User Interface**: Streamlit provides an interactive frontend for user input and displaying results

## 📊 Data Structure

The system stores car data with the following attributes:
- Make/Model/Year
- Location
- Color
- Engine Type
- Transmission
- Mileage
- Body Type
- Features
- Price
- And more...

## 🧠 AI Recommendation Process

1. **Query Analysis**: Extracts essential details from the user's query
2. **Content Evaluation**: Reviews matching car records from the vector database
3. **Relevance Ranking**: Ranks records based on how well they match user criteria
4. **Response Generation**: Creates a formatted list of the top recommendations

## 🔍 Example Queries

- "I need a family SUV with 7 seats, automatic transmission, and good fuel economy under 5 million"
- "Looking for a luxury sedan with leather seats and sunroof in Islamabad"
- "I want a small hatchback for city driving with low maintenance cost and good mileage"
- "Need a 4x4 vehicle for off-road use with diesel engine and under 10 million budget"

## 🛠️ Future Improvements

- [ ] Add user accounts and saved searches
- [ ] Implement image gallery for car listings
- [ ] Add comparative analysis between recommended cars
- [ ] Integrate with external car listings APIs
- [ ] Add more advanced filtering options
- [ ] Implement sentiment analysis for user reviews

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence Transformers for providing the embedding models
- Pinecone for vector database services
- Google for Gemini AI capabilities
- Streamlit for the easy-to-use web application framework

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

Email - hkashan.dev@gmail.com
