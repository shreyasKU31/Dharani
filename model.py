import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from joblib import dump, load
import time
warnings.filterwarnings('ignore')

class CropRecommendationSystem:
    def __init__(self, n_jobs=-1):
        """
        Initialize the crop recommendation system
        Predicts optimal crop based on soil and environmental conditions
        """
        self.model = None
        self.season_encoder = LabelEncoder()
        self.crop_encoder = LabelEncoder()  # For target (crop names)
        self.feature_names = None
        self.is_trained = False
        self.n_jobs = n_jobs
        self.crop_names = None
        
    def load_data(self, filename='dataset.csv'):
        """Load and validate the agricultural dataset"""
        try:
            df = pd.read_csv(filename, low_memory=False)
            return df
        except FileNotFoundError:
            return self._create_crop_dataset()
    
    def _create_crop_dataset(self, n_samples=2000):
        """Create a sample dataset for crop recommendation"""
        np.random.seed(42)
        
        # Define crop requirements (realistic agricultural data)
        crop_profiles = {
            'Rice': {'N': (120, 150), 'P': (60, 80), 'K': (40, 60), 'pH': (6.0, 7.0), 'moisture': (80, 95), 'temp': (22, 30)},
            'Wheat': {'N': (100, 130), 'P': (50, 70), 'K': (30, 50), 'pH': (6.5, 7.5), 'moisture': (50, 70), 'temp': (15, 25)},
            'Corn': {'N': (150, 200), 'P': (70, 90), 'K': (60, 80), 'pH': (6.0, 7.0), 'moisture': (60, 80), 'temp': (20, 30)},
            'Cotton': {'N': (120, 180), 'P': (50, 80), 'K': (80, 120), 'pH': (5.5, 8.0), 'moisture': (50, 80), 'temp': (25, 35)},
            'Soybean': {'N': (80, 120), 'P': (60, 90), 'K': (70, 100), 'pH': (6.0, 7.5), 'moisture': (60, 85), 'temp': (20, 30)},
            'Sugarcane': {'N': (200, 300), 'P': (80, 120), 'K': (150, 250), 'pH': (6.0, 8.0), 'moisture': (75, 95), 'temp': (25, 35)},
            'Potato': {'N': (150, 200), 'P': (80, 120), 'K': (200, 300), 'pH': (5.5, 6.5), 'moisture': (70, 90), 'temp': (15, 25)},
            'Tomato': {'N': (120, 180), 'P': (100, 150), 'K': (180, 250), 'pH': (6.0, 7.0), 'moisture': (70, 85), 'temp': (20, 30)}
        }
        
        seasons = ['Kharif', 'Rabi', 'Zaid', 'Annual']
        
        data = []
        samples_per_crop = n_samples // len(crop_profiles)
        
        for crop, requirements in crop_profiles.items():
            for _ in range(samples_per_crop):
                # Generate values around optimal ranges for each crop
                sample = {
                    'Season': np.random.choice(seasons),
                    'N (kg/ha)': np.random.normal(np.mean(requirements['N']), 20),
                    'P (kg/ha)': np.random.normal(np.mean(requirements['P']), 10),
                    'K (kg/ha)': np.random.normal(np.mean(requirements['K']), 15),
                    'pH': np.random.normal(np.mean(requirements['pH']), 0.3),
                    'Moisture (%)': np.random.normal(np.mean(requirements['moisture']), 8),
                    'Temp (°C)': np.random.normal(np.mean(requirements['temp']), 3),
                    'Crop Name': crop  # This is now our target variable
                }
                
                # Add some realistic bounds
                sample['N (kg/ha)'] = np.clip(sample['N (kg/ha)'], 50, 350)
                sample['P (kg/ha)'] = np.clip(sample['P (kg/ha)'], 20, 200)
                sample['K (kg/ha)'] = np.clip(sample['K (kg/ha)'], 20, 350)
                sample['pH'] = np.clip(sample['pH'], 4.5, 9.0)
                sample['Moisture (%)'] = np.clip(sample['Moisture (%)'], 20, 100)
                sample['Temp (°C)'] = np.clip(sample['Temp (°C)'], 10, 45)
                
                data.append(sample)
        
        df = pd.DataFrame(data)
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for crop recommendation"""
        start_time = time.time()
        
        df = df.copy()
        
        # Handle missing values
        numeric_columns = ['N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'pH', 'Moisture (%)', 'Temp (°C)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col].fillna(df[col].median(), inplace=True)
        
        # Handle season encoding
        if 'Season' in df.columns:
            df['Season'] = df['Season'].fillna('Annual')
            df['Season'] = self.season_encoder.fit_transform(df['Season'].astype(str))
        
        # Store crop names before encoding
        if 'Crop Name' in df.columns:
            self.crop_names = sorted(df['Crop Name'].unique())
        return df
    
    def train_model(self, df):
        """Train the crop recommendation model"""
        train_start = time.time()
        
        # Prepare features (everything except crop name) and target (crop name)
        feature_cols = ['Season', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'pH', 'Moisture (%)', 'Temp (°C)']
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features]
        y = self.crop_encoder.fit_transform(df['Crop Name'])
        
        self.feature_names = available_features
        
        # Check class distribution
        unique_crops, counts = np.unique(y, return_counts=True)
        crop_distribution = dict(zip([self.crop_encoder.classes_[i] for i in unique_crops], counts))
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        # Optimized hyperparameter search
        param_distributions = {
            'n_estimators': [100, 150, 200],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        n_iter = 15
        cv_folds = 5
        
        rf_base = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        
        random_search = RandomizedSearchCV(
            rf_base, param_distributions, 
            n_iter=n_iter, cv=cv_folds, 
            scoring='accuracy', random_state=42, n_jobs=self.n_jobs
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Training completed in {time.time() - train_start:.2f}s")
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        
        # Show classification report
        print(f"\n=== MODEL PERFORMANCE METRICS ===")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Training Samples: {len(X_train):,}")
        print(f"Test Samples: {len(X_test):,}")
        print(f"Number of Crops: {len(self.crop_encoder.classes_)}")
        print(f"Best CV Score: {random_search.best_score_:.4f}")
        
        # Calculate detailed metrics for visualizations
        from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
        
        # Store metrics for visualization
        self.test_metrics = {
            'y_test': y_test,
            'y_pred': y_pred,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'accuracy': accuracy,
            'crop_names': self.crop_encoder.classes_
        }
        
        self.is_trained = True
        return X_test, y_test, y_pred
    
    def show_comprehensive_visualizations(self):
        """Display all important graphs and heatmaps for model evaluation"""
        if not self.is_trained or not hasattr(self, 'test_metrics'):
            print("❌ Model not trained or metrics not available")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        metrics = self.test_metrics
        crop_names = metrics['crop_names']
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(metrics['y_test'], metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=crop_names, yticklabels=crop_names)
        plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Crop', fontsize=12)
        plt.ylabel('Actual Crop', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 2. Precision, Recall, F1-Score Heatmap
        plt.subplot(2, 3, 2)
        performance_matrix = np.array([
            metrics['precision'], 
            metrics['recall'], 
            metrics['f1']
        ])
        
        sns.heatmap(performance_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=crop_names, 
                   yticklabels=['Precision', 'Recall', 'F1-Score'],
                   cbar_kws={'label': 'Score'})
        plt.title('Performance Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Crops', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 3. Feature Importance Bar Chart
        plt.subplot(2, 3, 3)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
        bars = plt.bar(range(len(indices)), importances[indices], color=colors, alpha=0.8)
        
        plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        
        # Add percentage labels on bars
        total_importance = np.sum(importances)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percentage = 100 * height / total_importance
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, weight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 4. Model Accuracy Comparison
        plt.subplot(2, 3, 4)
        accuracy_metrics = ['Overall\nAccuracy', 'Avg\nPrecision', 'Avg\nRecall', 'Avg\nF1-Score']
        accuracy_values = [
            metrics['accuracy'],
            np.mean(metrics['precision']),
            np.mean(metrics['recall']),
            np.mean(metrics['f1'])
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = plt.bar(accuracy_metrics, accuracy_values, color=colors, alpha=0.8)
        plt.title('Overall Model Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, accuracy_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=11, weight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        
        # 5. Per-Crop Performance Radar Chart (converted to bar chart for simplicity)
        plt.subplot(2, 3, 5)
        crop_indices = np.arange(len(crop_names))
        width = 0.25
        
        bars1 = plt.bar(crop_indices - width, metrics['precision'], width, 
                       label='Precision', alpha=0.8, color='skyblue')
        bars2 = plt.bar(crop_indices, metrics['recall'], width,
                       label='Recall', alpha=0.8, color='lightcoral')  
        bars3 = plt.bar(crop_indices + width, metrics['f1'], width,
                       label='F1-Score', alpha=0.8, color='lightgreen')
        
        plt.title('Per-Crop Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Crops', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(crop_indices, crop_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 1.1)
        
        # 6. Support (Sample Count) Distribution
        plt.subplot(2, 3, 6)
        plt.pie(metrics['support'], labels=crop_names, autopct='%1.1f%%', startangle=90)
        plt.title('Test Set Distribution by Crop', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Additional Classification Report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        from sklearn.metrics import classification_report
        print(classification_report(metrics['y_test'], metrics['y_pred'], 
                                  target_names=crop_names, digits=4))
        
        # Summary Statistics Table
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY TABLE")
        print("="*60)
        print(f"{'Crop':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        for i, crop in enumerate(crop_names):
            if i < len(metrics['precision']):
                print(f"{crop:<12} {metrics['precision'][i]:<10.4f} "
                      f"{metrics['recall'][i]:<10.4f} {metrics['f1'][i]:<10.4f} "
                      f"{int(metrics['support'][i]):<10}")
        
        print("-" * 60)
        print(f"{'Average':<12} {np.mean(metrics['precision']):<10.4f} "
              f"{np.mean(metrics['recall']):<10.4f} {np.mean(metrics['f1']):<10.4f} "
              f"{int(np.sum(metrics['support'])):<10}")
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
    def show_feature_importance(self, show_plot=True, top_n=5):
        """Show which factors are most important for crop selection"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("❌ Model not trained")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if top_n:
            indices = indices[:top_n]
        
        if show_plot:
            # Create a focused feature importance plot
            plt.figure(figsize=(12, 6))
            colors = plt.cm.Set3(np.linspace(0, 1, len(indices)))
            bars = plt.bar(range(len(indices)), importances[indices], color=colors, alpha=0.8)
            
            plt.title('Top Feature Importance for Crop Selection', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Environmental & Soil Factors', fontsize=12)
            plt.ylabel('Importance Score', fontsize=12)
            
            labels = [self.feature_names[i] for i in indices]
            plt.xticks(range(len(indices)), labels, rotation=45, ha='right')
            
            # Add percentage labels
            total_importance = np.sum(importances)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                percentage = 100 * height / total_importance
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=11, weight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        print(f"\nTop {len(indices)} Important Features:")
        total_importance = np.sum(importances)
        for i, idx in enumerate(indices):
            importance = importances[idx]
            percentage = 100 * importance / total_importance
            print(f"{i+1}. {self.feature_names[idx]}: {percentage:.1f}%")
    
    def recommend_crop(self, conditions):
        """Recommend the best crop for given conditions"""
        if not self.is_trained:
            raise ValueError("❌ Model not trained! Call run_pipeline() first.")
        
        # Convert to DataFrame
        if isinstance(conditions, dict):
            input_df = pd.DataFrame([conditions])
        else:
            input_df = conditions.copy()
        
        # Handle season encoding
        if 'Season' in input_df.columns:
            # Handle unknown seasons
            season_val = input_df['Season'].iloc[0]
            if season_val not in self.season_encoder.classes_:
                input_df['Season'] = 'Annual'
            input_df['Season'] = self.season_encoder.transform(input_df['Season'])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                if feature == 'Season':
                    input_df[feature] = 0  # Default season
                else:
                    # Handle P2O5/K2O naming variations
                    alt_name = feature.replace('P (kg/ha)', 'P₂O₅ (kg/ha)').replace('K (kg/ha)', 'K₂O (kg/ha)')
                    if alt_name in input_df.columns:
                        input_df[feature] = input_df[alt_name]
                    else:
                        input_df[feature] = 0
                    
        # Select only the features used in training
        input_df = input_df[self.feature_names]
        
        # Get prediction and probabilities
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Get crop name and confidence
        recommended_crop = self.crop_encoder.classes_[prediction]
        confidence = probabilities[prediction]
        
        # Simple output - just the best crop
        print(f"Recommended Crop: {recommended_crop}")
        print(f"Confidence: {confidence:.3f}")
        
        return recommended_crop, confidence
    
    def analyze_conditions(self, conditions):
        """Provide detailed analysis of growing conditions"""
        print(f"\n🔬 DETAILED CONDITION ANALYSIS")
        print("=" * 45)
        
        # Define optimal ranges for major crops
        optimal_ranges = {
            'N (kg/ha)': (100, 200, "Nitrogen levels"),
            'P (kg/ha)': (50, 100, "Phosphorus levels"), 
            'K (kg/ha)': (50, 150, "Potassium levels"),
            'pH': (6.0, 7.5, "Soil pH"),
            'Moisture (%)': (60, 85, "Moisture content"),
            'Temp (°C)': (20, 30, "Temperature")
        }
        
        for param, value in conditions.items():
            if param in optimal_ranges:
                min_val, max_val, description = optimal_ranges[param]
                if min_val <= value <= max_val:
                    status = "✅ Optimal"
                elif value < min_val:
                    status = f"⬇️  Low (recommend: {min_val}+)"
                else:
                    status = f"⬆️  High (recommend: <{max_val})"
                    
                print(f"{description:20s}: {value:6.1f} - {status}")
    
    def save_model(self, filepath='crop_recommendation_model.joblib'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("❌ Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'season_encoder': self.season_encoder,
            'crop_encoder': self.crop_encoder,
            'feature_names': self.feature_names,
            'crop_names': self.crop_names,
            'is_trained': self.is_trained
        }
        dump(model_data, filepath, compress=3)
    
    def load_model(self, filepath='crop_recommendation_model.joblib'):
        """Load trained model"""
        try:
            model_data = load(filepath)
            self.model = model_data['model']
            self.season_encoder = model_data['season_encoder']
            self.crop_encoder = model_data['crop_encoder']
            self.feature_names = model_data['feature_names']
            self.crop_names = model_data['crop_names']
            self.is_trained = model_data['is_trained']
            print(f"📂 Model loaded from {filepath}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Model file {filepath} not found")
    
    def run_pipeline(self, filename='dataset.csv', save_model=True):
        """Run the complete crop recommendation pipeline"""
        
        pipeline_start = time.time()
        
        # Load data
        df = self.load_data(filename)
        
        # Auto-detect target column (should be crop name)
        if 'Crop Name' not in df.columns:
            possible_targets = [col for col in df.columns 
                             if any(keyword in col.lower() 
                                   for keyword in ['crop', 'plant', 'species', 'variety'])]
            if possible_targets:
                target_col = possible_targets[0]
                df.rename(columns={target_col: 'Crop Name'}, inplace=True)
            else:
                raise ValueError(f"❌ No crop name column found in {list(df.columns)}")
        
        # Preprocessing and training
        df = self.preprocess_data(df)
        X_test, y_test, y_pred = self.train_model(df)
        
        # Show comprehensive visualizations
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*50)
        self.show_comprehensive_visualizations()
        
        # Show feature importance with plot
        self.show_feature_importance(show_plot=True)
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        total_time = time.time() - pipeline_start
        
        return self.model


# Usage example
if __name__ == "__main__":
    # Initialize crop recommendation system
    crop_system = CropRecommendationSystem(n_jobs=-1)
    
    # Train the system with comprehensive visualizations
    trained_model = crop_system.run_pipeline('dataset.csv')
    
    # Simple crop recommendation test
    print("\n" + "="*40)
    print("CROP RECOMMENDATION TEST")
    print("="*40)
    
    test_conditions = {
        'Season': 'Kharif',
        'N (kg/ha)': 270,
        'P (kg/ha)': 82,
        'K (kg/ha)': 164,
        'pH': 7.0,
        'Moisture (%)': 80,
        'Temp (°C)': 31
    }
    
    print(f"Input conditions: {test_conditions}")
    recommended_crop, confidence = crop_system.recommend_crop(test_conditions)
    
    print("\n" + "="*50)
    print("To view all visualizations again, run:")
    print("crop_system.show_comprehensive_visualizations()")
    print("="*50)