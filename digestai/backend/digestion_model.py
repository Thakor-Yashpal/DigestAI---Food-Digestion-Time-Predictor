# digestion_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class FoodDigestionDataset(Dataset):
    def __init__(self, text_features, numerical_features, targets):
        self.text_features = torch.FloatTensor(text_features)
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'text_features': self.text_features[idx],
            'numerical_features': self.numerical_features[idx],
            'target': self.targets[idx]
        }

class FoodDigestionPredictor(nn.Module):
    def __init__(self, text_input_dim, numerical_input_dim, hidden_dim=128, dropout=0.3):
        super(FoodDigestionPredictor, self).__init__()
        
        # Text processing branch
        self.text_branch = nn.Sequential(
            nn.Linear(text_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Numerical features branch
        self.numerical_branch = nn.Sequential(
            nn.Linear(numerical_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Combined prediction head
        combined_dim = (hidden_dim // 2) + (hidden_dim // 4)
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Single output: digestion time
        )
        
    def forward(self, text_features, numerical_features):
        text_out = self.text_branch(text_features)
        numerical_out = self.numerical_branch(numerical_features)
        
        # Combine features
        combined = torch.cat([text_out, numerical_out], dim=1)
        output = self.predictor(combined)
        
        return output.squeeze()

class DigestionModelTrainer:
    def __init__(self, data_path="food_digestion_dataset.csv"):
        self.data_path = data_path
        self.model = None
        self.text_vectorizer = None
        self.numerical_scaler = None
        self.target_scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        df = pd.read_csv(self.data_path)
        
        # Prepare text features using TF-IDF
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        text_features = self.text_vectorizer.fit_transform(df['food_name_clean']).toarray()
        
        # Prepare numerical features
        numerical_cols = ['fiber', 'fat', 'protein', 'carbs', 'total_macros', 
                         'protein_ratio', 'fat_ratio', 'carb_ratio', 'fiber_ratio',
                         'word_count', 'name_length', 'category_encoded']
        numerical_features = df[numerical_cols].values
        
        # Scale numerical features
        self.numerical_scaler = StandardScaler()
        numerical_features = self.numerical_scaler.fit_transform(numerical_features)
        
        # Prepare targets (digestion time in hours)
        targets = df['digestion_time_hours'].values
        self.target_scaler = StandardScaler()
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        return text_features, numerical_features, targets_scaled, df
    
    def create_data_loaders(self, text_features, numerical_features, targets, batch_size=32, test_size=0.2):
        """Create train and validation data loaders"""
        X_text_train, X_text_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
            text_features, numerical_features, targets, test_size=test_size, random_state=42
        )
        
        train_dataset = FoodDigestionDataset(X_text_train, X_num_train, y_train)
        val_dataset = FoodDigestionDataset(X_text_val, X_num_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def initialize_model(self, text_input_dim, numerical_input_dim):
        """Initialize the model"""
        self.model = FoodDigestionPredictor(
            text_input_dim=text_input_dim,
            numerical_input_dim=numerical_input_dim,
            hidden_dim=128,
            dropout=0.3
        ).to(self.device)
        
        return self.model
    
    def train_model(self, train_loader, val_loader, epochs=100, learning_rate=0.001):
        """Train the model"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                text_feat = batch['text_features'].to(self.device)
                num_feat = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(text_feat, num_feat)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    text_feat = batch['text_features'].to(self.device)
                    num_feat = batch['numerical_features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = self.model(text_feat, num_feat)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'text_vectorizer': self.text_vectorizer,
                    'numerical_scaler': self.numerical_scaler,
                    'target_scaler': self.target_scaler,
                    'text_input_dim': self.model.text_branch[0].in_features,
                    'numerical_input_dim': self.model.numerical_branch[0].in_features,
                }, 'best_digestion_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return train_losses, val_losses
    
    def evaluate_model(self, val_loader):
        """Evaluate model performance"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in val_loader:
                text_feat = batch['text_features'].to(self.device)
                num_feat = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(text_feat, num_feat)
                
                # Convert back to original scale
                pred_unscaled = self.target_scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()
                actual_unscaled = self.target_scaler.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).flatten()
                
                predictions.extend(pred_unscaled)
                actuals.extend(actual_unscaled)
        
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        print(f"Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f} hours")
        print(f"R² Score: {r2:.3f}")
        
        return predictions, actuals, mae, r2
    
    def predict_digestion_time(self, food_name: str) -> Dict:
        """Predict digestion time for a single food item"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        self.model.eval()
        
        # Prepare input
        text_features = self.text_vectorizer.transform([food_name.lower().strip()]).toarray()
        
        # Create dummy numerical features (you might want to implement food lookup)
        # For now, using average values
        numerical_features = np.array([[2.0, 5.0, 10.0, 15.0, 32.0, 0.3, 0.15, 0.47, 0.06, 2, 10, 2]])
        numerical_features = self.numerical_scaler.transform(numerical_features)
        
        # Convert to tensors
        text_tensor = torch.FloatTensor(text_features).to(self.device)
        num_tensor = torch.FloatTensor(numerical_features).to(self.device)
        
        with torch.no_grad():
            prediction_scaled = self.model(text_tensor, num_tensor)
            prediction = self.target_scaler.inverse_transform(prediction_scaled.cpu().numpy().reshape(-1, 1))[0, 0]
        
        # Convert to minutes and create response
        minutes = max(15, int(prediction * 60))  # Minimum 15 minutes
        hours = prediction
        
        # Determine confidence based on how common the food pattern is
        confidence = "Medium"  # You can implement more sophisticated confidence calculation
        
        return {
            "food_name": food_name,
            "digestion_time_hours": round(hours, 2),
            "digestion_time_minutes": minutes,
            "confidence": confidence
        }

def train_complete_model():
    """Complete training pipeline"""
    print("Starting model training pipeline...")
    
    # Initialize trainer
    trainer = DigestionModelTrainer()
    
    # Load and prepare data
    print("Loading and preparing data...")
    text_features, numerical_features, targets, df = trainer.load_and_prepare_data()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = trainer.create_data_loaders(
        text_features, numerical_features, targets, batch_size=16
    )
    
    # Initialize model
    print("Initializing model...")
    model = trainer.initialize_model(
        text_input_dim=text_features.shape[1],
        numerical_input_dim=numerical_features.shape[1]
    )
    
    print(f"Model architecture:")
    print(f"Text input dim: {text_features.shape[1]}")
    print(f"Numerical input dim: {numerical_features.shape[1]}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Training model...")
    train_losses, val_losses = trainer.train_model(train_loader, val_loader, epochs=150)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, actuals, mae, r2 = trainer.evaluate_model(val_loader)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', lw=2)
    plt.title(f'Predictions vs Actual (R²={r2:.3f})')
    plt.xlabel('Actual Digestion Time (hours)')
    plt.ylabel('Predicted Digestion Time (hours)')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test some predictions
    test_foods = [
        "greek yogurt with berries",
        "grilled chicken breast",
        "avocado toast",
        "apple with peanut butter",
        "quinoa salad"
    ]
    
    print("\nSample predictions:")
    for food in test_foods:
        result = trainer.predict_digestion_time(food)
        print(f"{food}: {result['digestion_time_hours']:.1f}h ({result['digestion_time_minutes']}min)")
    
    print("\nModel training completed!")
    return trainer

# Enhanced prediction class for deployment
class DigestAIPredictor:
    def __init__(self, model_path='best_digestion_model.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.text_vectorizer = None
        self.numerical_scaler = None
        self.target_scaler = None
        self.food_database = None
        self.load_model()
        self.load_food_database()
    
    def load_model(self):
        """Load trained model and preprocessors"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Load preprocessors
        self.text_vectorizer = checkpoint['text_vectorizer']
        self.numerical_scaler = checkpoint['numerical_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        # Initialize and load model
        text_dim = checkpoint['text_input_dim']
        num_dim = checkpoint['numerical_input_dim']
        
        self.model = FoodDigestionPredictor(text_dim, num_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def load_food_database(self):
        """Load food nutritional database for better predictions"""
        try:
            self.food_database = pd.read_csv('food_digestion_dataset.csv')
        except:
            print("Warning: Food database not found. Using default nutritional values.")
            self.food_database = None
    
    def get_food_nutrition(self, food_name: str) -> Dict:
        """Get nutritional information for a food item"""
        if self.food_database is None:
            # Default values if no database
            return {
                'fiber': 2.0, 'fat': 5.0, 'protein': 10.0, 'carbs': 15.0,
                'total_macros': 32.0, 'protein_ratio': 0.31, 'fat_ratio': 0.16,
                'carb_ratio': 0.47, 'fiber_ratio': 0.06, 'word_count': len(food_name.split()),
                'name_length': len(food_name), 'category_encoded': 2
            }
        
        # Search for similar foods in database
        food_lower = food_name.lower().strip()
        
        # Exact match
        exact_match = self.food_database[self.food_database['food_name_clean'] == food_lower]
        if not exact_match.empty:
            row = exact_match.iloc[0]
            return self._extract_nutrition(row, food_name)
        
        # Partial match
        partial_matches = self.food_database[
            self.food_database['food_name_clean'].str.contains('|'.join(food_lower.split()), case=False, na=False)
        ]
        
        if not partial_matches.empty:
            # Use the most similar match
            row = partial_matches.iloc[0]
            return self._extract_nutrition(row, food_name)
        
        # Keyword-based matching
        keywords = food_lower.split()
        for keyword in keywords:
            matches = self.food_database[
                self.food_database['food_name_clean'].str.contains(keyword, case=False, na=False)
            ]
            if not matches.empty:
                row = matches.iloc[0]
                return self._extract_nutrition(row, food_name)
        
        # Default fallback
        return self._extract_nutrition(self.food_database.iloc[0], food_name)
    
    def _extract_nutrition(self, row, food_name: str) -> Dict:
        """Extract nutrition data from database row"""
        return {
            'fiber': float(row.get('fiber', 2.0)),
            'fat': float(row.get('fat', 5.0)),
            'protein': float(row.get('protein', 10.0)),
            'carbs': float(row.get('carbs', 15.0)),
            'total_macros': float(row.get('total_macros', 32.0)),
            'protein_ratio': float(row.get('protein_ratio', 0.31)),
            'fat_ratio': float(row.get('fat_ratio', 0.16)),
            'carb_ratio': float(row.get('carb_ratio', 0.47)),
            'fiber_ratio': float(row.get('fiber_ratio', 0.06)),
            'word_count': len(food_name.split()),
            'name_length': len(food_name),
            'category_encoded': int(row.get('category_encoded', 2))
        }
    
    def predict(self, food_name: str) -> Dict:
        """Enhanced prediction with better error handling and confidence"""
        try:
            # Clean input
            food_name = food_name.strip()
            if not food_name:
                raise ValueError("Food name cannot be empty")
            
            # Get text features
            text_features = self.text_vectorizer.transform([food_name.lower()]).toarray()
            
            # Get nutritional features
            nutrition = self.get_food_nutrition(food_name)
            numerical_features = np.array([[
                nutrition['fiber'], nutrition['fat'], nutrition['protein'], nutrition['carbs'],
                nutrition['total_macros'], nutrition['protein_ratio'], nutrition['fat_ratio'],
                nutrition['carb_ratio'], nutrition['fiber_ratio'], nutrition['word_count'],
                nutrition['name_length'], nutrition['category_encoded']
            ]])
            
            # Scale features
            numerical_features = self.numerical_scaler.transform(numerical_features)
            
            # Convert to tensors
            text_tensor = torch.FloatTensor(text_features).to(self.device)
            num_tensor = torch.FloatTensor(numerical_features).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.model(text_tensor, num_tensor)
                prediction_hours = self.target_scaler.inverse_transform(
                    prediction_scaled.cpu().numpy().reshape(-1, 1)
                )[0, 0]
            
            # Post-process prediction
            prediction_hours = max(0.25, min(6.0, prediction_hours))  # Clamp between 15min and 6h
            prediction_minutes = int(prediction_hours * 60)
            
            # Calculate confidence based on data availability and prediction certainty
            confidence = self._calculate_confidence(food_name, nutrition, prediction_hours)
            
            # Generate description
            description = self._generate_description(nutrition, prediction_hours)
            
            return {
                "success": True,
                "food_name": food_name,
                "digestion_time_hours": round(prediction_hours, 2),
                "digestion_time_minutes": prediction_minutes,
                "confidence": confidence,
                "description": description,
                "nutritional_factors": {
                    "protein": round(nutrition['protein'], 1),
                    "carbs": round(nutrition['carbs'], 1),
                    "fat": round(nutrition['fat'], 1),
                    "fiber": round(nutrition['fiber'], 1)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "food_name": food_name
            }
    
    def _calculate_confidence(self, food_name: str, nutrition: Dict, prediction: float) -> str:
        """Calculate prediction confidence"""
        # Higher confidence for common foods and reasonable predictions
        common_foods = ['chicken', 'rice', 'apple', 'bread', 'milk', 'egg', 'fish', 'yogurt']
        is_common = any(food in food_name.lower() for food in common_foods)
        
        # Check if prediction is in reasonable range
        reasonable_range = 0.5 <= prediction <= 4.0
        
        if is_common and reasonable_range:
            return "High"
        elif reasonable_range:
            return "Medium"
        else:
            return "Low"
    
    def _generate_description(self, nutrition: Dict, prediction: float) -> str:
        """Generate description based on nutritional content"""
        descriptions = []
        
        if nutrition['protein'] > 15:
            descriptions.append("High protein content")
        if nutrition['fat'] > 10:
            descriptions.append("higher fat content")
        if nutrition['fiber'] > 5:
            descriptions.append("high fiber")
        if nutrition['carbs'] > 20:
            descriptions.append("carbohydrate-rich")
        
        if prediction < 1:
            speed = "quick digestion"
        elif prediction < 2:
            speed = "moderate digestion time"
        else:
            speed = "longer digestion time"
        
        if descriptions:
            return f"{', '.join(descriptions[:2])} leads to {speed}."
        else:
            return f"Balanced nutritional profile with {speed}."

if __name__ == "__main__":
    # Run complete training
    trainer = train_complete_model()