import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

class FoodDigestionDataCollector:
    def __init__(self):
        self.raw_data = []
        self.processed_data = []
        
    def create_initial_dataset(self):
        """Create initial dataset with known digestion times"""
        # Based on nutritional science research
        food_data = [
            # Liquids (fastest digestion)
            {"food_name": "water", "digestion_time_minutes": 0, "category": "liquid", "fiber": 0, "fat": 0, "protein": 0, "carbs": 0},
            {"food_name": "fruit juice", "digestion_time_minutes": 15, "category": "liquid", "fiber": 0.5, "fat": 0, "protein": 1, "carbs": 24},
            {"food_name": "sports drink", "digestion_time_minutes": 10, "category": "liquid", "fiber": 0, "fat": 0, "protein": 0, "carbs": 14},
            {"food_name": "milk", "digestion_time_minutes": 30, "category": "liquid", "fiber": 0, "fat": 3.5, "protein": 8, "carbs": 12},
            {"food_name": "smoothie", "digestion_time_minutes": 45, "category": "liquid", "fiber": 3, "fat": 2, "protein": 5, "carbs": 25},
            
            # Fruits (30-60 minutes)
            {"food_name": "watermelon", "digestion_time_minutes": 20, "category": "fruit", "fiber": 0.4, "fat": 0.2, "protein": 0.6, "carbs": 8},
            {"food_name": "grapes", "digestion_time_minutes": 30, "category": "fruit", "fiber": 0.9, "fat": 0.2, "protein": 0.7, "carbs": 16},
            {"food_name": "orange", "digestion_time_minutes": 30, "category": "fruit", "fiber": 3.1, "fat": 0.1, "protein": 1.2, "carbs": 12},
            {"food_name": "apple", "digestion_time_minutes": 40, "category": "fruit", "fiber": 2.4, "fat": 0.3, "protein": 0.3, "carbs": 14},
            {"food_name": "banana", "digestion_time_minutes": 30, "category": "fruit", "fiber": 2.6, "fat": 0.3, "protein": 1.1, "carbs": 23},
            {"food_name": "berries", "digestion_time_minutes": 40, "category": "fruit", "fiber": 8, "fat": 0.3, "protein": 1.4, "carbs": 12},
            
            # Vegetables (30-90 minutes)
            {"food_name": "lettuce", "digestion_time_minutes": 30, "category": "vegetable", "fiber": 1.3, "fat": 0.2, "protein": 1.4, "carbs": 2.9},
            {"food_name": "cucumber", "digestion_time_minutes": 30, "category": "vegetable", "fiber": 0.5, "fat": 0.1, "protein": 0.7, "carbs": 4},
            {"food_name": "tomato", "digestion_time_minutes": 30, "category": "vegetable", "fiber": 1.2, "fat": 0.2, "protein": 0.9, "carbs": 3.9},
            {"food_name": "spinach", "digestion_time_minutes": 40, "category": "vegetable", "fiber": 2.2, "fat": 0.4, "protein": 2.9, "carbs": 3.6},
            {"food_name": "broccoli", "digestion_time_minutes": 45, "category": "vegetable", "fiber": 2.6, "fat": 0.4, "protein": 2.8, "carbs": 6},
            {"food_name": "carrots", "digestion_time_minutes": 50, "category": "vegetable", "fiber": 2.8, "fat": 0.2, "protein": 0.9, "carbs": 10},
            {"food_name": "sweet potato", "digestion_time_minutes": 60, "category": "vegetable", "fiber": 3, "fat": 0.1, "protein": 2, "carbs": 20},
            
            # Grains and Starches (1-3 hours)
            {"food_name": "white rice", "digestion_time_minutes": 60, "category": "grain", "fiber": 0.4, "fat": 0.3, "protein": 2.7, "carbs": 28},
            {"food_name": "brown rice", "digestion_time_minutes": 90, "category": "grain", "fiber": 1.8, "fat": 0.9, "protein": 2.6, "carbs": 23},
            {"food_name": "white bread", "digestion_time_minutes": 60, "category": "grain", "fiber": 2.7, "fat": 3.2, "protein": 9, "carbs": 49},
            {"food_name": "whole grain bread", "digestion_time_minutes": 90, "category": "grain", "fiber": 6, "fat": 4, "protein": 13, "carbs": 41},
            {"food_name": "pasta", "digestion_time_minutes": 90, "category": "grain", "fiber": 2.5, "fat": 1.1, "protein": 5, "carbs": 25},
            {"food_name": "quinoa", "digestion_time_minutes": 120, "category": "grain", "fiber": 2.8, "fat": 1.9, "protein": 4.4, "carbs": 22},
            {"food_name": "oatmeal", "digestion_time_minutes": 90, "category": "grain", "fiber": 4, "fat": 2.5, "protein": 5.3, "carbs": 28},
            
            # Proteins (2-4 hours)
            {"food_name": "egg", "digestion_time_minutes": 45, "category": "protein", "fiber": 0, "fat": 5, "protein": 6, "carbs": 0.6},
            {"food_name": "greek yogurt", "digestion_time_minutes": 60, "category": "protein", "fiber": 0, "fat": 0.4, "protein": 10, "carbs": 3.6},
            {"food_name": "cottage cheese", "digestion_time_minutes": 90, "category": "protein", "fiber": 0, "fat": 4.3, "protein": 11, "carbs": 3.4},
            {"food_name": "chicken breast", "digestion_time_minutes": 150, "category": "protein", "fiber": 0, "fat": 3.6, "protein": 31, "carbs": 0},
            {"food_name": "fish", "digestion_time_minutes": 120, "category": "protein", "fiber": 0, "fat": 6, "protein": 22, "carbs": 0},
            {"food_name": "tofu", "digestion_time_minutes": 120, "category": "protein", "fiber": 0.3, "fat": 4.8, "protein": 8, "carbs": 1.9},
            {"food_name": "beans", "digestion_time_minutes": 180, "category": "protein", "fiber": 7, "fat": 0.5, "protein": 8, "carbs": 22},
            {"food_name": "lentils", "digestion_time_minutes": 150, "category": "protein", "fiber": 7.9, "fat": 0.4, "protein": 9, "carbs": 20},
            
            # Nuts and Seeds (3-4 hours)
            {"food_name": "almonds", "digestion_time_minutes": 180, "category": "nuts", "fiber": 3.5, "fat": 14, "protein": 6, "carbs": 6},
            {"food_name": "walnuts", "digestion_time_minutes": 200, "category": "nuts", "fiber": 1.9, "fat": 18, "protein": 4.3, "carbs": 3.9},
            {"food_name": "peanut butter", "digestion_time_minutes": 180, "category": "nuts", "fiber": 1.6, "fat": 16, "protein": 7.7, "carbs": 6.3},
            {"food_name": "chia seeds", "digestion_time_minutes": 240, "category": "seeds", "fiber": 10, "fat": 8.6, "protein": 4.7, "carbs": 12},
            
            # Mixed/Complex meals (2-5 hours)
            {"food_name": "salad with chicken", "digestion_time_minutes": 135, "category": "mixed", "fiber": 4, "fat": 8, "protein": 25, "carbs": 10},
            {"food_name": "sandwich", "digestion_time_minutes": 120, "category": "mixed", "fiber": 4, "fat": 12, "protein": 20, "carbs": 35},
            {"food_name": "pizza", "digestion_time_minutes": 180, "category": "mixed", "fiber": 2.5, "fat": 10, "protein": 12, "carbs": 36},
            {"food_name": "burger", "digestion_time_minutes": 240, "category": "mixed", "fiber": 3, "fat": 17, "protein": 25, "carbs": 31},
            {"food_name": "steak", "digestion_time_minutes": 180, "category": "protein", "fiber": 0, "fat": 20, "protein": 26, "carbs": 0},
            {"food_name": "avocado", "digestion_time_minutes": 180, "category": "fat", "fiber": 10, "fat": 15, "protein": 2, "carbs": 9},
            
            # Dairy (varies)
            {"food_name": "cheese", "digestion_time_minutes": 120, "category": "dairy", "fiber": 0, "fat": 9, "protein": 7, "carbs": 1},
            {"food_name": "ice cream", "digestion_time_minutes": 120, "category": "dairy", "fiber": 0.7, "fat": 11, "protein": 3.5, "carbs": 23},
        ]
        
        self.raw_data = food_data
        return pd.DataFrame(food_data)
    
    def add_food_variations(self, df):
        """Add variations and combinations of existing foods"""
        variations = []
        
        # Add cooking method variations
        cooking_methods = ["grilled", "baked", "fried", "steamed", "raw", "boiled"]
        base_proteins = ["chicken", "fish", "salmon", "turkey", "beef"]
        
        for protein in base_proteins:
            base_time = df[df['food_name'].str.contains(protein.split()[0], case=False, na=False)]
            if not base_time.empty:
                base_minutes = base_time.iloc[0]['digestion_time_minutes']
                for method in cooking_methods:
                    # Fried foods take longer, steamed/boiled slightly faster
                    time_modifier = 1.3 if method == "fried" else (0.9 if method in ["steamed", "boiled"] else 1.0)
                    fat_modifier = 1.5 if method == "fried" else (0.8 if method in ["steamed", "boiled"] else 1.0)
                    
                    variations.append({
                        "food_name": f"{method} {protein}",
                        "digestion_time_minutes": int(base_minutes * time_modifier),
                        "category": "protein",
                        "fiber": base_time.iloc[0]['fiber'],
                        "fat": base_time.iloc[0]['fat'] * fat_modifier,
                        "protein": base_time.iloc[0]['protein'],
                        "carbs": base_time.iloc[0]['carbs']
                    })
        
        # Add combination meals
        combinations = [
            {"name": "greek yogurt with berries", "base1": "greek yogurt", "base2": "berries", "ratio": 0.7},
            {"name": "oat latte", "base1": "oatmeal", "base2": "milk", "ratio": 0.6},
            {"name": "chicken salad with quinoa", "base1": "chicken breast", "base2": "quinoa", "ratio": 0.8},
            {"name": "apple with peanut butter", "base1": "apple", "base2": "peanut butter", "ratio": 0.6},
            {"name": "avocado toast", "base1": "avocado", "base2": "whole grain bread", "ratio": 0.75},
            {"name": "smoothie bowl", "base1": "smoothie", "base2": "berries", "ratio": 0.8},
        ]
        
        for combo in combinations:
            base1_data = df[df['food_name'].str.contains(combo["base1"], case=False, na=False)]
            base2_data = df[df['food_name'].str.contains(combo["base2"], case=False, na=False)]
            
            if not base1_data.empty and not base2_data.empty:
                b1, b2 = base1_data.iloc[0], base2_data.iloc[0]
                ratio = combo["ratio"]
                
                variations.append({
                    "food_name": combo["name"],
                    "digestion_time_minutes": int(b1['digestion_time_minutes'] * ratio + b2['digestion_time_minutes'] * (1-ratio)),
                    "category": "mixed",
                    "fiber": b1['fiber'] * ratio + b2['fiber'] * (1-ratio),
                    "fat": b1['fat'] * ratio + b2['fat'] * (1-ratio),
                    "protein": b1['protein'] * ratio + b2['protein'] * (1-ratio),
                    "carbs": b1['carbs'] * ratio + b2['carbs'] * (1-ratio)
                })
        
        return pd.concat([df, pd.DataFrame(variations)], ignore_index=True)
    
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Convert digestion time to hours for easier interpretation
        df['digestion_time_hours'] = df['digestion_time_minutes'] / 60
        
        # Create additional features
        df['total_macros'] = df['protein'] + df['carbs'] + df['fat']
        df['protein_ratio'] = df['protein'] / (df['total_macros'] + 1e-6)
        df['fat_ratio'] = df['fat'] / (df['total_macros'] + 1e-6)
        df['carb_ratio'] = df['carbs'] / (df['total_macros'] + 1e-6)
        df['fiber_ratio'] = df['fiber'] / (df['total_macros'] + 1e-6)
        
        # Clean food names
        df['food_name_clean'] = df['food_name'].str.lower().str.strip()
        
        # Create word features for NLP model
        df['word_count'] = df['food_name_clean'].str.split().str.len()
        df['name_length'] = df['food_name_clean'].str.len()
        
        # Encode categories
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        
        return df, le
    
    def save_data(self, df, filename="food_digestion_dataset.csv"):
        """Save processed dataset"""
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(df)}")
        print(f"Categories: {df['category'].value_counts().to_dict()}")
        return df

# Usage example
if __name__ == "__main__":
    collector = FoodDigestionDataCollector()
    
    # Create initial dataset
    print("Creating initial dataset...")
    df = collector.create_initial_dataset()
    
    # Add variations
    print("Adding food variations...")
    df = collector.add_food_variations(df)
    
    # Preprocess
    print("Preprocessing data...")
    df, label_encoder = collector.preprocess_data(df)
    
    # Save everything
    final_df = collector.save_data(df)
    
    # Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Display sample
    print("\nSample data:")
    print(df[['food_name', 'digestion_time_minutes', 'digestion_time_hours', 'category', 'fiber', 'fat', 'protein', 'carbs']].head(10))