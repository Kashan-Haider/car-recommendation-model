from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
from google import genai

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECON_API")
GEMINI_API_KEY = os.getenv("GEMINI_API")


def load_models():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_models()


def get_query_dense_embeddings(text):
    embeddings = model.encode(text)
    return embeddings


def get_query_sparse_embeddings(query):
    encoder = BM25Encoder()
    texts = [
        "Looking for a Toyota Fortuner Legender 2022 in I-8, Islamabad Islamabad; mileage 5 km; Diesel engine, Automatic transmission, white color, local status, 2800 cc engine capacity, SUV body type; features include ABS, AM/FM Radio, Air Bags, Air Conditioning, Alloy Rims, Cassette Player, Cruise Control, Immobilizer Key, Keyless Entry, Navigation System, Power Locks, Power Mirrors, Power Steering, Power Windows; price: 15000000",
        "Need a Toyota Premio X EX Package 1.8 2018 at Askari 6, Peshawar KPK with 17,000 km mileage; Petrol engine, Automatic transmission, pearl white color, imported status, 1800 cc engine, Sedan body type; features: ABS, AM/FM Radio, Air Bags, Air Conditioning, Alloy Rims, Cruise Control, DVD Player, Immobilizer Key, Keyless Entry, Navigation System, Power Locks, Power Mirrors, Power Steering, Power Windows; price: 8500000.0.",
    ]
    encoder.fit(texts)
    embeddings = encoder.encode_documents(query)
    return embeddings


def connect_to_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "cars-recommendation-system-index"
    index = pc.Index(index_name)
    return index


def initialize_gemini():
    return genai.Client(api_key=GEMINI_API_KEY)
