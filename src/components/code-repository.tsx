'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Download, FileDown, FileText, Archive } from 'lucide-react'
import CodeBlock from '@/components/code-block'

// Mock data for code repository
const mockRepositoryInfo = {
  name: 'ai-econ-forecasting',
  description: 'Machine learning models for macroeconomic forecasting',
  stars: 124,
  forks: 37,
  lastUpdated: '2025-03-15',
  url: 'https://github.com/example/ai-econ-forecasting'
}

interface CodeRepositoryProps {
  repositoryUrl?: string
  repositoryInfo?: {
    name: string
    description: string
    stars: number
    forks: number
    lastUpdated: string
    url: string
  }
}

export default function CodeRepository({ 
  repositoryUrl, 
  repositoryInfo = mockRepositoryInfo 
}: CodeRepositoryProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Code Repository</h2>
      
      <Card className="p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-2">
            <h3 className="text-xl font-semibold">{repositoryInfo.name}</h3>
            <p className="text-muted-foreground">{repositoryInfo.description}</p>
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span>⭐ {repositoryInfo.stars} stars</span>
              <span>🍴 {repositoryInfo.forks} forks</span>
              <span>Updated: {repositoryInfo.lastUpdated}</span>
            </div>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-2">
            <Button asChild>
              <a href={repositoryUrl || repositoryInfo.url} target="_blank" rel="noopener noreferrer">
                View Repository
              </a>
            </Button>
            <Button variant="outline">
              <Archive className="mr-2 h-4 w-4" />
              Download ZIP
            </Button>
          </div>
        </div>
      </Card>
      
      <div className="space-y-4">
        <h3 className="text-xl font-semibold">Getting Started</h3>
        <div className="space-y-4">
          <CodeBlock 
            code="git clone https://github.com/example/ai-econ-forecasting.git\ncd ai-econ-forecasting\npip install -r requirements.txt" 
            language="bash"
            filename="setup.sh"
          />
          
          <p className="text-muted-foreground">
            After cloning the repository and installing dependencies, you can run the example models:
          </p>
          
          <CodeBlock 
            code="python src/models/random_forest.py --data data/economic_indicators.csv --output results/" 
            language="bash"
            filename="run_model.sh"
          />
        </div>
      </div>
      
      <div className="space-y-4">
        <h3 className="text-xl font-semibold">Repository Structure</h3>
        <Card className="p-4">
          <pre className="text-sm font-mono">
{`ai-econ-forecasting/
├── data/                  # Dataset files
│   ├── economic_indicators.csv
│   └── README.md
├── notebooks/             # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model implementations
│   └── visualization/     # Visualization utilities
├── results/               # Output directory
├── requirements.txt       # Dependencies
└── README.md              # Documentation`}
          </pre>
        </Card>
      </div>
    </div>
  )
}
