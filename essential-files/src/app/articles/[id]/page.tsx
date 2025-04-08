'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowLeft, Calendar, Tag, User } from 'lucide-react'
import CodeSection from '@/components/code-section'
import CodeRepository from '@/components/code-repository'

// Mock data for articles - this would come from a database in a real application
const articlesData = {
  '1': {
    id: 1,
    title: 'Machine Learning Applications in Macroeconomic Forecasting',
    date: '2025-03-28',
    author: 'Dr. Jane Smith',
    category: 'Research',
    tags: ['Machine Learning', 'Forecasting', 'Policy'],
    content: `
# Machine Learning Applications in Macroeconomic Forecasting

## Introduction

Macroeconomic forecasting has traditionally relied on econometric models that often struggle to capture complex, non-linear relationships in economic data. With the advent of machine learning techniques, economists now have powerful new tools to improve forecast accuracy and gain deeper insights into economic dynamics.

## Traditional Approaches vs. Machine Learning

Traditional macroeconomic forecasting typically employs structural models based on economic theory or time series methods like ARIMA. While these approaches have served economists well, they come with limitations:

- They often assume linear relationships between variables
- They struggle with high-dimensional data
- They require strong assumptions about data distributions
- They have limited ability to incorporate diverse data sources

Machine learning approaches offer several advantages:

- They can capture complex non-linear relationships
- They excel at handling high-dimensional data
- They make fewer assumptions about underlying data distributions
- They can integrate diverse data sources, including unstructured data

## Key Machine Learning Techniques in Macroeconomic Forecasting

### Random Forests

Random forests have proven effective for macroeconomic forecasting due to their ability to handle non-linearities and variable interactions. Studies have shown that random forests can outperform traditional models in forecasting GDP growth, especially during periods of economic volatility.

### Recurrent Neural Networks (RNNs)

RNNs and their variants (LSTM, GRU) are particularly well-suited for time series forecasting. Their ability to capture temporal dependencies makes them valuable for predicting macroeconomic indicators like inflation rates and unemployment figures.

### Ensemble Methods

Combining multiple models often yields superior forecasting performance. Ensemble methods that integrate both traditional econometric models and machine learning algorithms have shown promising results in central bank applications.

## Challenges and Limitations

Despite their potential, machine learning approaches face several challenges in macroeconomic forecasting:

1. **Interpretability**: Many ML models function as "black boxes," making it difficult to understand the economic mechanisms driving predictions.

2. **Data limitations**: Economic data is often limited in frequency and sample size compared to other ML applications.

3. **Structural breaks**: Economic relationships can change fundamentally over time, particularly during crises.

4. **Overfitting**: Complex ML models risk capturing noise rather than signal in economic data.

## Future Directions

The integration of machine learning into macroeconomic forecasting continues to evolve. Promising areas include:

- **Explainable AI**: Developing techniques to make ML models more interpretable for policymakers
- **Hybrid models**: Combining economic theory with ML flexibility
- **Alternative data**: Incorporating high-frequency, non-traditional data sources
- **Causal ML**: Moving beyond correlation to better understand causal economic relationships

## Conclusion

Machine learning offers powerful tools to enhance macroeconomic forecasting, complementing rather than replacing traditional approaches. As these techniques continue to mature and address current limitations, they will likely play an increasingly important role in economic analysis and policy formulation.
    `,
    hasCode: true,
    codeFiles: [
      {
        name: 'random_forest_model.py',
        language: 'python',
        code: `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load economic data
data = pd.read_csv('economic_indicators.csv')

# Feature engineering
data['gdp_growth_lag1'] = data['gdp_growth'].shift(1)
data['inflation_lag1'] = data['inflation'].shift(1)
data['unemployment_lag1'] = data['unemployment'].shift(1)
data = data.dropna()

# Prepare features and target
X = data[['gdp_growth_lag1', 'inflation_lag1', 'unemployment_lag1', 
          'industrial_production', 'consumer_confidence']]
y = data['gdp_growth']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse:.4f}')

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)
        `
      },
      {
        name: 'visualization.py',
        language: 'python',
        code: `
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load predictions from model
def plot_predictions(actual, predicted, title='GDP Growth Predictions'):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.title(title)
    plt.xlabel('Time Period')
    plt.ylabel('GDP Growth Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance in GDP Forecasting')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    # This would use actual model results in a real application
    periods = np.arange(20)
    actual = np.sin(periods/3) + np.random.normal(0, 0.2, 20)
    predicted = np.sin(periods/3) + np.random.normal(0, 0.3, 20)
    
    plot_predictions(actual, predicted)
        `
      }
    ],
    dataFiles: [
      {
        name: 'economic_indicators.csv',
        description: 'Quarterly economic indicators for major economies (2000-2024)',
        size: '2.4 MB',
        url: '/data/economic_indicators.csv'
      },
      {
        name: 'model_results.csv',
        description: 'Prediction results and evaluation metrics',
        size: '1.1 MB',
        url: '/data/model_results.csv'
      }
    ],
    hasRepository: true
  },
  '2': {
    id: 2,
    title: 'Neural Networks for Time Series Analysis in Economics',
    date: '2025-03-15',
    author: 'Prof. Michael Chen',
    category: 'Methodology',
    tags: ['Neural Networks', 'Time Series', 'Deep Learning'],
    content: `
# Neural Networks for Time Series Analysis in Economics

## Introduction

Time series analysis is fundamental to economic research and forecasting. Traditional approaches like ARIMA and VAR models have been the workhorses of economic time series analysis for decades. However, neural networks are increasingly demonstrating their value in capturing complex patterns in economic data.

## Why Neural Networks for Economic Time Series?

Economic time series often exhibit complex characteristics:

- Non-linear relationships
- Regime changes
- Structural breaks
- Seasonal patterns
- Long-term dependencies

Neural networks, particularly recurrent architectures, can capture these complexities without requiring explicit specification of the underlying data generating process.

## Key Neural Network Architectures for Economic Time Series

### Recurrent Neural Networks (RNNs)

Basic RNNs maintain a hidden state that captures information from previous time steps, making them naturally suited for sequential data. However, they often struggle with long-term dependencies due to the vanishing gradient problem.

### Long Short-Term Memory (LSTM) Networks

LSTMs address the limitations of standard RNNs through a gating mechanism that allows them to capture both short and long-term dependencies. This makes them particularly valuable for economic time series that may contain both cyclical patterns and long-term trends.

### Temporal Convolutional Networks (TCNs)

TCNs apply convolutional operations with dilated filters to capture temporal patterns at different time scales. They offer advantages in terms of parallelization and can sometimes outperform recurrent architectures.

## Applications in Economics

### Forecasting Macroeconomic Indicators

Neural networks have shown promise in forecasting key indicators like GDP growth, inflation, and unemployment. Studies have demonstrated that deep learning approaches can outperform traditional models, especially during periods of economic volatility.

### Financial Market Prediction

In financial economics, neural networks are used to model asset returns, volatility, and risk. Their ability to capture non-linear relationships makes them valuable for understanding complex market dynamics.

### Early Warning Systems

Neural networks can be employed to develop early warning systems for economic crises by identifying patterns that precede economic downturns.

## Methodological Considerations

### Feature Engineering

While neural networks can learn representations automatically, thoughtful feature engineering remains important. Relevant transformations might include:

- Differencing to achieve stationarity
- Normalization of input variables
- Creation of lagged variables
- Inclusion of calendar effects

### Hyperparameter Tuning

Neural network performance is sensitive to hyperparameter choices. Cross-validation strategies adapted for time series (e.g., time series cross-validation) are essential for robust model selection.

### Interpretability

The "black box" nature of neural networks poses challenges for economic applications where interpretation is crucial. Techniques like partial dependence plots and SHAP values can help economists gain insights into model behavior.

## Conclusion

Neural networks offer powerful tools for economic time series analysis, complementing rather than replacing traditional econometric approaches. As these methods continue to evolve and address current limitations, they will likely become increasingly integrated into the economist's toolkit.
    `,
    hasCode: true,
    codeFiles: [
      {
        name: 'lstm_model.py',
        language: 'python',
        code: `
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load economic data
data = pd.read_csv('quarterly_economic_data.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# Prepare data for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['gdp_growth']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Select features and scale data
features = ['gdp_growth', 'inflation', 'unemployment', 'interest_rate']
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data[features]),
    columns=features,
    index=data.index
)

# Create sequences
seq_length = 8  # 2 years of quarterly data
X, y = create_sequences(scaled_data, seq_length)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {mae:.4f}')

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual')
plt.plot(data.index[-len(y_test):], y_pred, label='Predicted')
plt.title('GDP Growth Prediction')
plt.xlabel('Date')
plt.ylabel('GDP Growth Rate')
plt.legend()
plt.show()
        `
      }
    ],
    dataFiles: [
      {
        name: 'quarterly_economic_data.csv',
        description: 'Quarterly macroeconomic indicators (2000-2024)',
        size: '1.8 MB',
        url: '/data/quarterly_economic_data.csv'
      }
    ],
    hasRepository: false
  }
}

export default function ArticlePage() {
  const params = useParams()
  const articleId = params.id as string
  
  // Get article data
  const article = articlesData[articleId]
  
  // If article not found
  if (!article) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h1 className="text-4xl font-bold mb-4">Article Not Found</h1>
        <p className="text-muted-foreground mb-8">The article you're looking for doesn't exist or has been removed.</p>
        <Button asChild>
          <Link href="/articles">Back to Articles</Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-12">
      {/* Back button */}
      <div className="mb-8">
        <Button variant="outline" asChild>
          <Link href="/articles" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Articles
          </Link>
        </Button>
      </div>
      
      {/* Article header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4">{article.title}</h1>
        <div className="flex flex-wrap gap-4 text-muted-foreground">
          <div className="flex items-center">
            <Calendar className="mr-2 h-4 w-4" />
            {article.date}
          </div>
          <div className="flex items-center">
            <User className="mr-2 h-4 w-4" />
            {article.author}
          </div>
          <div className="flex items-center">
            <Tag className="mr-2 h-4 w-4" />
            {article.category}
          </div>
        </div>
      </div>
      
      {/* Article tags */}
      <div className="mb-8 flex flex-wrap gap-2">
        {article.tags.map((tag, index) => (
          <span key={index} className="bg-muted px-3 py-1 rounded-full text-sm">
            {tag}
          </span>
        ))}
      </div>
      
      {/* Article content */}
      <div className="prose prose-lg dark:prose-invert max-w-none mb-12">
        <div dangerouslySetInnerHTML={{ __html: article.content.replace(/\n/g, '<br>') }} />
      </div>
      
      {/* Code section */}
      {article.hasCode && (
        <div className="mb-12">
          <CodeSection 
            files={article.codeFiles || []}
            dataFiles={article.dataFiles || []}
          />
        </div>
      )}
      
      {/* Repository section */}
      {article.hasRepository && (
        <div className="mb-12">
          <CodeRepository />
        </div>
      )}
      
      {/* Related articles */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Related Articles</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.values(articlesData)
            .filter(a => a.id !== article.id)
            .slice(0, 2)
            .map(relatedArticle => (
              <div key={relatedArticle.id} className="border border-border rounded-lg p-6">
                <h3 className="text-xl font-semibold mb-2">{relatedArticle.title}</h3>
                <p className="text-muted-foreground mb-4">{relatedArticle.date} • {relatedArticle.category}</p>
                <Button variant="outline" asChild>
                  <Link href={`/articles/${relatedArticle.id}`}>Read Article</Link>
                </Button>
              </div>
            ))}
        </div>
      </div>
    </div>
  )
}
