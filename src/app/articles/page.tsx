'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import Link from 'next/link'

// Mock data for articles
const articles = [
  {
    id: 1,
    title: 'Machine Learning Applications in Macroeconomic Forecasting',
    excerpt: 'How AI models are revolutionizing economic predictions and policy planning.',
    date: '2025-03-28',
    category: 'Research',
    tags: ['Machine Learning', 'Forecasting', 'Policy']
  },
  {
    id: 2,
    title: 'Neural Networks for Time Series Analysis in Economics',
    excerpt: 'Exploring deep learning approaches to analyze economic time series data.',
    date: '2025-03-15',
    category: 'Methodology',
    tags: ['Neural Networks', 'Time Series', 'Deep Learning']
  },
  {
    id: 3,
    title: 'Reinforcement Learning in Monetary Policy Optimization',
    excerpt: 'How central banks can leverage AI for better policy decisions.',
    date: '2025-02-22',
    category: 'Policy',
    tags: ['Reinforcement Learning', 'Monetary Policy', 'Central Banking']
  },
  {
    id: 4,
    title: 'Natural Language Processing for Economic Sentiment Analysis',
    excerpt: 'Using NLP to analyze economic sentiment from news and social media.',
    date: '2025-02-10',
    category: 'Research',
    tags: ['NLP', 'Sentiment Analysis', 'Text Mining']
  },
  {
    id: 5,
    title: 'Predictive Analytics for GDP Growth Estimation',
    excerpt: 'Advanced techniques for predicting economic growth indicators.',
    date: '2025-01-30',
    category: 'Methodology',
    tags: ['Predictive Analytics', 'GDP', 'Economic Indicators']
  },
  {
    id: 6,
    title: 'Explainable AI in Economic Decision Making',
    excerpt: 'Making AI-driven economic models transparent and interpretable.',
    date: '2025-01-15',
    category: 'Research',
    tags: ['Explainable AI', 'Transparency', 'Decision Making']
  }
]

// Categories for filtering
const categories = ['All', 'Research', 'Methodology', 'Policy']

export default function ArticlesPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All')
  
  // Filter articles based on search query and category
  const filteredArticles = articles.filter(article => {
    const matchesSearch = article.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          article.excerpt.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          article.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    
    const matchesCategory = selectedCategory === 'All' || article.category === selectedCategory
    
    return matchesSearch && matchesCategory
  })

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8 text-center">Articles</h1>
      
      {/* Search and Filter */}
      <div className="mb-8 flex flex-col md:flex-row gap-4">
        <div className="flex-grow">
          <Input
            placeholder="Search articles..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full"
          />
        </div>
        <div className="w-full md:w-48">
          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger>
              <SelectValue placeholder="Category" />
            </SelectTrigger>
            <SelectContent>
              {categories.map((category) => (
                <SelectItem key={category} value={category}>
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      
      {/* Articles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {filteredArticles.map((article) => (
          <Card key={article.id} className="overflow-hidden flex flex-col h-full">
            <div className="p-6 flex flex-col h-full">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm text-primary">{article.category}</span>
                <span className="text-sm text-muted-foreground">{article.date}</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">{article.title}</h3>
              <p className="text-muted-foreground mb-4 flex-grow">{article.excerpt}</p>
              <div className="flex flex-wrap gap-2 mb-4">
                {article.tags.map((tag, index) => (
                  <span key={index} className="text-xs bg-muted px-2 py-1 rounded-full">
                    {tag}
                  </span>
                ))}
              </div>
              <Button variant="outline" asChild className="w-full">
                <Link href={`/articles/${article.id}`}>Read Article</Link>
              </Button>
            </div>
          </Card>
        ))}
      </div>
      
      {/* No Results */}
      {filteredArticles.length === 0 && (
        <div className="text-center py-12">
          <h3 className="text-xl font-semibold mb-2">No articles found</h3>
          <p className="text-muted-foreground">Try adjusting your search or filter criteria</p>
        </div>
      )}
    </div>
  )
}
