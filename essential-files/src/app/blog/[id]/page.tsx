'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { ArrowLeft, Calendar, User, Clock, Tag } from 'lucide-react'
import CodeBlock from '@/components/code-block'

// Mock data for blog posts
const blogPostsData = {
  '1': {
    id: 1,
    title: 'Applying Transformer Models to Economic Policy Analysis',
    date: '2025-04-01',
    author: 'Your Name',
    category: 'Machine Learning',
    readTime: '8 min read',
    content: `
# Applying Transformer Models to Economic Policy Analysis

## Introduction

Economic policy documents—such as central bank minutes, regulatory frameworks, and fiscal policy statements—contain valuable information that shapes market expectations and economic outcomes. Traditionally, economists and analysts have manually reviewed these documents to extract insights, a process that is both time-consuming and potentially subject to human bias. With the advent of transformer-based language models, we now have powerful tools to analyze these documents at scale and with remarkable accuracy.

## The Challenge of Economic Policy Text Analysis

Economic policy documents present several unique challenges for text analysis:

1. **Specialized Language**: They contain domain-specific terminology and jargon that requires specialized understanding.

2. **Nuanced Communication**: Central banks and policy institutions often use deliberate, nuanced language where subtle changes in wording can signal significant policy shifts.

3. **Contextual Importance**: The meaning of statements often depends on broader economic contexts and previous communications.

4. **Structured Format**: Many policy documents follow specific structures and conventions that carry implicit information.

## Transformer Models: A Brief Overview

Transformer models, introduced in the landmark paper "Attention Is All You Need" (Vaswani et al., 2017), have revolutionized natural language processing. Unlike previous recurrent neural network architectures, transformers process entire sequences in parallel using a mechanism called self-attention, which allows them to capture long-range dependencies in text.

Large language models (LLMs) based on the transformer architecture, such as GPT, BERT, and their derivatives, have demonstrated remarkable capabilities in understanding and generating human language. These models are pre-trained on vast corpora of text and can be fine-tuned for specific tasks.

## Applications in Economic Policy Analysis

### Sentiment and Tone Analysis

Transformer models can be fine-tuned to detect the sentiment and tone of policy documents. For example, they can identify whether a central bank's statement is "hawkish" (suggesting tighter monetary policy) or "dovish" (suggesting accommodative policy). This analysis can be performed at both document and sentence levels.

### Policy Change Detection

By comparing current policy documents with previous versions, transformer models can highlight significant changes in language and emphasis. This can help identify shifts in policy stance that might not be explicitly stated.

### Topic Modeling and Theme Extraction

Transformers can identify the main topics and themes discussed in policy documents, allowing analysts to track how the focus of policymakers evolves over time. This can provide early signals of emerging economic concerns or priorities.

### Cross-document Analysis

Transformer models can analyze relationships between different policy documents, such as comparing central bank communications across different countries or institutions. This can reveal policy coordination or divergence.

## Implementation Approach

### Fine-tuning Strategy

While pre-trained LLMs have general language understanding capabilities, fine-tuning them on a corpus of economic policy documents significantly improves their performance for specialized tasks. The fine-tuning process involves:

1. **Data Collection**: Gathering a comprehensive dataset of policy documents with appropriate annotations (e.g., labeled for sentiment, policy decisions, etc.).

2. **Model Selection**: Choosing an appropriate base model. BERT-based models often work well for classification tasks, while GPT-based models excel at generation tasks.

3. **Training Procedure**: Fine-tuning the model using techniques like gradual unfreezing and discriminative learning rates to prevent catastrophic forgetting.

### Feature Engineering for Economic Context

To enhance model performance, we can incorporate economic context through feature engineering:

- Including economic indicators as additional inputs
- Providing historical policy decisions as context
- Adding document metadata (author institution, document type, etc.)

## Case Study: Analyzing Federal Reserve Communications

In a recent project, we fine-tuned a RoBERTa model on Federal Open Market Committee (FOMC) statements and minutes from 2000-2024. The model was trained to:

1. Classify the overall monetary policy stance
2. Identify sentences discussing inflation, employment, and financial stability
3. Detect significant changes in language compared to previous statements

The model achieved over 90% accuracy in classifying policy stance and could identify subtle shifts in language that correlated with subsequent policy actions.

## Challenges and Limitations

Despite their power, transformer models face several challenges in economic policy analysis:

1. **Interpretability**: The "black box" nature of these models can make it difficult to understand why they make certain predictions.

2. **Data Requirements**: Fine-tuning requires substantial amounts of labeled data, which can be costly to create for specialized economic applications.

3. **Context Window Limitations**: Many policy documents exceed the context window of current models, requiring chunking strategies that may lose important context.

4. **Evolving Language**: Economic language evolves over time, potentially reducing model performance on new documents.

## Future Directions

Several promising directions for future research include:

1. **Multimodal Models**: Incorporating economic data alongside text to provide richer context for language understanding.

2. **Causal Inference**: Moving beyond correlation to understand how policy language causally affects market outcomes.

3. **Cross-lingual Analysis**: Developing models that can analyze policy documents across different languages for global economic analysis.

4. **Explainable AI**: Creating more interpretable models that can justify their analyses in economic terms.

## Conclusion

Transformer models offer powerful tools for analyzing economic policy documents, enabling more comprehensive, consistent, and scalable analysis than traditional methods. While challenges remain, particularly around interpretability and data requirements, the potential benefits for economic research and policy analysis are substantial. As these models continue to evolve, they will likely become essential tools in the economist's toolkit.
    `,
    hasCode: true,
    codeSnippet: `
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Load pre-trained tokenizer and model
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Prepare dataset (example with FOMC statements)
df = pd.read_csv('fomc_statements.csv')
df['label'] = df['stance'].map({'dovish': 0, 'neutral': 1, 'hawkish': 2})

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save model
model.save_pretrained("./fomc_stance_classifier")
tokenizer.save_pretrained("./fomc_stance_classifier")

# Example inference
def predict_stance(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=1).item()
    stance_map = {0: 'dovish', 1: 'neutral', 2: 'hawkish'}
    return stance_map[predicted_class], predictions[0].tolist()

# Test on a new statement
new_statement = """
The Committee seeks to achieve maximum employment and inflation at the rate 
of 2 percent over the longer run. In support of these goals, the Committee 
decided to maintain the target range for the federal funds rate at 5 to 5-1/4 
percent. The Committee will continue to assess additional information and its 
implications for monetary policy.
"""

stance, confidence = predict_stance(new_statement)
print(f"Predicted stance: {stance}")
print(f"Confidence scores (dovish, neutral, hawkish): {confidence}")
    `
  },
  '2': {
    id: 2,
    title: 'Visualizing Economic Data with Python and D3.js',
    date: '2025-03-20',
    author: 'Your Name',
    category: 'Data Visualization',
    readTime: '6 min read',
    content: `
# Visualizing Economic Data with Python and D3.js

## Introduction

Effective data visualization is crucial for understanding complex economic relationships and communicating insights to both technical and non-technical audiences. While static visualizations have long been the standard in economic research, interactive visualizations offer significant advantages for exploring multidimensional economic data and uncovering hidden patterns.

In this blog post, I'll explore how to combine the data processing capabilities of Python with the interactive visualization power of D3.js to create compelling economic data visualizations.

## The Visualization Challenge in Economics

Economic data presents several unique visualization challenges:

1. **Multidimensional Relationships**: Economic variables often have complex interrelationships that are difficult to represent in two dimensions.

2. **Time Series Focus**: Much of economic analysis revolves around time series data, requiring specialized visualization approaches.

3. **Uncertainty Representation**: Economic forecasts and estimates come with uncertainty that should be visually represented.

4. **Scale Differences**: Economic indicators can vary dramatically in scale, making comparative visualization challenging.

5. **Structural Breaks**: Economic data often contains structural breaks (e.g., recessions, policy changes) that need visual emphasis.

## Python for Data Preparation

Python offers powerful libraries for economic data processing:

### Data Acquisition and Cleaning

Libraries like pandas, requests, and BeautifulSoup make it easy to acquire and clean economic data from various sources:

- **FRED API**: Access Federal Reserve Economic Data
- **World Bank API**: Global economic indicators
- **Web Scraping**: Extract data from statistical agencies and other sources

### Statistical Analysis

NumPy, SciPy, and statsmodels provide robust tools for economic analysis:

- Time series decomposition
- Seasonal adjustment
- Trend analysis
- Correlation and regression analysis

### Initial Visualization

Matplotlib and Seaborn are excellent for exploratory data analysis and creating static visualizations that can inform more complex interactive designs.

## D3.js for Interactive Visualization

D3.js (Data-Driven Documents) is a JavaScript library that binds data to DOM elements, enabling powerful interactive visualizations. It's particularly well-suited for economic data visualization because:

1. **Flexibility**: D3 can create virtually any visualization design, unlike template-based libraries.

2. **Interactivity**: It enables rich user interactions like brushing, zooming, and tooltips.

3. **Transitions**: Smooth transitions help users track changes in data.

4. **Scales and Axes**: D3 has sophisticated scaling functions for different data types.

## Bridging Python and D3.js

Several approaches can bridge the Python data analysis environment with D3.js visualizations:

### Approach 1: Python Web Frameworks

Using frameworks like Flask or Django, you can:
- Process data with Python
- Serve processed data as JSON
- Render D3.js visualizations in web templates

### Approach 2: Jupyter Notebooks with JavaScript

Libraries like ipywidgets and jupyter-d3 allow embedding D3.js visualizations directly in Jupyter notebooks.

### Approach 3: Observable and Python

Observable (observablehq.com) provides a reactive JavaScript notebook environment ideal for D3.js. You can:
- Process data in Python
- Export to CSV/JSON
- Import into Observable for visualization
- Embed the resulting visualizations in websites

## Case Study: Visualizing GDP Components Over Time

Let's walk through a simplified example of creating an interactive visualization of GDP components over time:

### Step 1: Data Preparation in Python

First, we'll use Python to acquire and process the data:

\`\`\`python
import pandas as pd
import requests
import json

# Fetch GDP components data from FRED API
api_key = "YOUR_FRED_API_KEY"
series_ids = {
    "Consumption": "PCECC96",
    "Investment": "GPDIC1",
    "Government": "GCEC1",
    "Exports": "EXPGSC1",
    "Imports": "IMPGSC1"
}

data = {}
for component, series_id in series_ids.items():
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    response = requests.get(url)
    series_data = response.json()["observations"]
    data[component] = {item["date"]: float(item["value"]) for item in series_data if item["value"] != "."}

# Combine into a single DataFrame
dates = sorted(set().union(*[set(component_data.keys()) for component_data in data.values()]))
df = pd.DataFrame(index=dates)

for component, component_data in data.items():
    df[component] = pd.Series(component_data)

# Fill missing values and calculate percentages
df = df.fillna(method="ffill").fillna(method="bfill")
total_gdp = df["Consumption"] + df["Investment"] + df["Government"] + df["Exports"] - df["Imports"]
for component in series_ids.keys():
    if component != "Imports":
        df[f"{component}_pct"] = (df[component] / total_gdp) * 100
    else:
        df[f"{component}_pct"] = (df[component] / total_gdp) * -100  # Negative for imports

# Export to JSON for D3.js
result = []
for date in df.index:
    entry = {"date": date}
    for component in series_ids.keys():
        entry[component] = float(df.loc[date, f"{component}_pct"])
    result.append(entry)

with open("gdp_components.json", "w") as f:
    json.dump(result, f)
\`\`\`

### Step 2: Creating the D3.js Visualization

Now, we'll create an interactive stacked area chart with D3.js:

\`\`\`javascript
// D3.js visualization code
const width = 900;
const height = 500;
const margin = {top: 20, right: 30, bottom: 30, left: 60};

const svg = d3.select("#visualization")
  .append("svg")
  .attr("width", width)
  .attr("height", height);

d3.json("gdp_components.json").then(data => {
  // Parse dates
  data.forEach(d => d.date = new Date(d.date));
  
  // Set up scales
  const x = d3.scaleTime()
    .domain(d3.extent(data, d => d.date))
    .range([margin.left, width - margin.right]);
    
  const y = d3.scaleLinear()
    .domain([0, 100])
    .range([height - margin.bottom, margin.top]);
    
  // Define components and colors
  const components = ["Consumption", "Investment", "Government", "Exports"];
  const colors = d3.scaleOrdinal()
    .domain(components)
    .range(d3.schemeCategory10);
    
  // Create stack generator
  const stack = d3.stack()
    .keys(components)
    .order(d3.stackOrderNone)
    .offset(d3.stackOffsetExpand);
    
  const series = stack(data);
  
  // Create area generator
  const area = d3.area()
    .x(d => x(d.data.date))
    .y0(d => y(d[0] * 100))
    .y1(d => y(d[1] * 100));
    
  // Add areas
  svg.append("g")
    .selectAll("path")
    .data(series)
    .join("path")
      .attr("fill", ({key}) => colors(key))
      .attr("d", area)
      .append("title")
      .text(({key}) => key);
      
  // Add axes
  svg.append("g")
    .attr("transform", \`translate(0,\${height - margin.bottom})\`)
    .call(d3.axisBottom(x).ticks(width / 80).tickSizeOuter(0));
    
  svg.append("g")
    .attr("transform", \`translate(\${margin.left},0)\`)
    .call(d3.axisLeft(y).ticks(10, "%"))
    .call(g => g.select(".domain").remove())
    .call(g => g.select(".tick:last-of-type text").clone()
      .attr("x", 3)
      .attr("text-anchor", "start")
      .attr("font-weight", "bold")
      .text("% of GDP"));
      
  // Add legend
  const legend = svg.append("g")
    .attr("font-family", "sans-serif")
    .attr("font-size", 10)
    .attr("text-anchor", "end")
    .selectAll("g")
    .data(components.slice().reverse())
    .join("g")
      .attr("transform", (d, i) => \`translate(0,\${i * 20})\`);
      
  legend.append("rect")
    .attr("x", width - 19)
    .attr("width", 19)
    .attr("height", 19)
    .attr("fill", colors);
    
  legend.append("text")
    .attr("x", width - 24)
    .attr("y", 9.5)
    .attr("dy", "0.35em")
    .text(d => d);
});
\`\`\`

## Advanced Techniques

For more sophisticated economic visualizations, consider these advanced techniques:

### 1. Linked Views

Create multiple coordinated visualizations where interactions in one view (e.g., selecting a time period) update other views (e.g., showing detailed breakdowns for that period).

### 2. Animated Transitions

Use animations to show how economic relationships evolve over time, particularly useful for visualizing business cycles or structural changes.

### 3. Uncertainty Visualization

Incorporate confidence intervals, prediction ranges, or fan charts to represent uncertainty in economic forecasts.

### 4. Small Multiples

Create arrays of similar charts to compare economic indicators across countries, regions, or time periods.

### 5. Interactive Storytelling

Combine visualizations with narrative elements to guide users through economic analyses and insights.

## Conclusion

The combination of Python's data processing capabilities and D3.js's visualization power offers a robust toolkit for creating compelling, interactive economic data visualizations. These visualizations can enhance both research and communication, making complex economic relationships more accessible to diverse audiences.

As economic data continues to grow in volume and complexity, interactive visualization will become increasingly important for extracting and communicating meaningful insights. By mastering these tools, economists and data scientists can significantly enhance their analytical capabilities and the impact of their work.
    `,
    hasCode: true,
    codeSnippet: `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Sample data generation for GDP components visualization
# In a real scenario, you would fetch this from an API or database

# Generate quarterly dates for 10 years
dates = pd.date_range(start='2015-01-01', end='2025-01-01', freq='Q')
quarters = [d.strftime('%Y-Q%q') for d in dates]

# Create synthetic GDP component data
np.random.seed(42)  # For reproducibility

# Base values for each component
consumption_base = 60  # ~60% of GDP
investment_base = 20   # ~20% of GDP
government_base = 15   # ~15% of GDP
exports_base = 15      # ~15% of GDP
imports_base = -10     # ~10% of GDP (negative as it's subtracted)

# Add some random variation and trends
n = len(quarters)
trend = np.linspace(0, 5, n)  # General upward trend

consumption = consumption_base + np.random.normal(0, 2, n) + trend * 0.2
investment = investment_base + np.random.normal(0, 3, n) + trend * 0.3
government = government_base + np.random.normal(0, 1, n) + trend * 0.1
exports = exports_base + np.random.normal(0, 2, n) + trend * 0.4
imports = imports_base - np.random.normal(0, 2, n) - trend * 0.3  # Negative values

# Create DataFrame
gdp_data = pd.DataFrame({
    'Quarter': quarters,
    'Consumption': consumption,
    'Investment': investment,
    'Government': government,
    'Exports': exports,
    'Imports': imports
})

# Calculate total GDP
gdp_data['GDP'] = (
    gdp_data['Consumption'] + 
    gdp_data['Investment'] + 
    gdp_data['Government'] + 
    gdp_data['Exports'] - 
    abs(gdp_data['Imports'])  # Use abs() since we stored imports as negative
)

# Calculate percentages
for component in ['Consumption', 'Investment', 'Government', 'Exports', 'Imports']:
    gdp_data[f'{component}_pct'] = (gdp_data[component] / gdp_data['GDP']) * 100

# Prepare data for D3.js visualization
d3_data = []
for _, row in gdp_data.iterrows():
    entry = {
        'date': row['Quarter'],
        'Consumption': float(row['Consumption_pct']),
        'Investment': float(row['Investment_pct']),
        'Government': float(row['Government_pct']),
        'Exports': float(row['Exports_pct']),
        'Imports': float(row['Imports_pct'])
    }
    d3_data.append(entry)

# Save to JSON file for D3.js
with open('gdp_components.json', 'w') as f:
    json.dump(d3_data, f)

# Create a simple visualization with matplotlib
plt.figure(figsize=(12, 8))

# Create stacked area chart
components = ['Consumption', 'Investment', 'Government', 'Exports']
values = [gdp_data[f'{c}_pct'] for c in components]

plt.stackplot(
    range(len(quarters)), 
    values,
    labels=components,
    alpha=0.8
)

# Add imports as a line (since they're negative)
plt.plot(
    range(len(quarters)), 
    -gdp_data['Imports_pct'],  # Negate to show as positive
    'r--', 
    label='Imports'
)

plt.title('GDP Components as Percentage of GDP', fontsize=16)
plt.xlabel('Quarter', fontsize=12)
plt.ylabel('Percentage of GDP', fontsize=12)
plt.xticks(
    range(0, len(quarters), 4),  # Show every 4th quarter
    [quarters[i] for i in range(0, len(quarters), 4)],
    rotation=45
)
plt.legend(loc='upper left')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('gdp_components_matplotlib.png', dpi=300)
plt.show()

print("Data and visualization prepared successfully!")
print(f"Generated {len(d3_data)} quarterly data points from {quarters[0]} to {quarters[-1]}")
print("Files created: gdp_components.json and gdp_components_matplotlib.png")
    `
  }
}

export default function BlogPostPage() {
  const params = useParams()
  const postId = params.id as string
  
  // Get blog post data
  const post = blogPostsData[postId]
  
  // If post not found
  if (!post) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <h1 className="text-4xl font-bold mb-4">Blog Post Not Found</h1>
        <p className="text-muted-foreground mb-8">The blog post you're looking for doesn't exist or has been removed.</p>
        <Button asChild>
          <Link href="/blog">Back to Blog</Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-12">
      {/* Back button */}
      <div className="mb-8">
        <Button variant="outline" asChild>
          <Link href="/blog" className="flex items-center">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Blog
          </Link>
        </Button>
      </div>
      
      <div className="max-w-3xl mx-auto">
        {/* Blog post header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4">{post.title}</h1>
          <div className="flex flex-wrap gap-4 text-muted-foreground">
            <div className="flex items-center">
              <Calendar className="mr-2 h-4 w-4" />
              {post.date}
            </div>
            <div className="flex items-center">
              <User className="mr-2 h-4 w-4" />
              {post.author}
            </div>
            <div className="flex items-center">
              <Clock className="mr-2 h-4 w-4" />
              {post.readTime}
            </div>
            <div className="flex items-center">
              <Tag className="mr-2 h-4 w-4" />
              {post.category}
            </div>
          </div>
        </div>
        
        {/* Blog post content */}
        <div className="prose prose-lg dark:prose-invert max-w-none mb-12">
          <div dangerouslySetInnerHTML={{ __html: post.content.replace(/\n/g, '<br>') }} />
        </div>
        
        {/* Code section */}
        {post.hasCode && (
          <div className="mb-12">
            <h2 className="text-2xl font-bold mb-4">Code Implementation</h2>
            <CodeBlock 
              code={post.codeSnippet}
              language="python"
              filename={`${post.title.toLowerCase().replace(/\s+/g, '_')}.py`}
            />
          </div>
        )}
        
        {/* Author section */}
        <div className="border-t border-border pt-8 mt-8">
          <h2 className="text-xl font-bold mb-4">About the Author</h2>
          <div className="flex items-center">
            <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mr-4">
              <User className="h-8 w-8 text-muted-foreground" />
            </div>
            <div>
              <h3 className="font-semibold">{post.author}</h3>
              <p className="text-muted-foreground">
                Researcher and writer focused on the intersection of machine learning and macroeconomics.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
