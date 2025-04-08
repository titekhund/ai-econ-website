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

// Mock data for blog posts
const blogPosts = [
  {
    id: 1,
    title: 'Applying Transformer Models to Economic Policy Analysis',
    excerpt: 'How large language models can help analyze and interpret economic policy documents.',
    date: '2025-04-01',
    category: 'Machine Learning',
    author: 'Your Name',
    readTime: '8 min read',
    imageUrl: '/placeholder-1.jpg'
  },
  {
    id: 2,
    title: 'Visualizing Economic Data with Python and D3.js',
    excerpt: 'Creating interactive visualizations to better understand macroeconomic trends.',
    date: '2025-03-20',
    category: 'Data Visualization',
    author: 'Your Name',
    readTime: '6 min read',
    imageUrl: '/placeholder-2.jpg'
  },
  {
    id: 3,
    title: 'Predicting Inflation Using Ensemble Methods',
    excerpt: 'Combining multiple machine learning models for more accurate inflation forecasts.',
    date: '2025-03-05',
    category: 'Forecasting',
    author: 'Your Name',
    readTime: '10 min read',
    imageUrl: '/placeholder-3.jpg'
  },
  {
    id: 4,
    title: 'Feature Engineering for Macroeconomic Time Series',
    excerpt: 'Techniques for creating effective features from economic time series data.',
    date: '2025-02-18',
    category: 'Methodology',
    author: 'Your Name',
    readTime: '7 min read',
    imageUrl: '/placeholder-1.jpg'
  },
  {
    id: 5,
    title: 'Explainable AI in Central Banking',
    excerpt: 'Making AI models interpretable for monetary policy applications.',
    date: '2025-02-01',
    category: 'Policy',
    author: 'Your Name',
    readTime: '9 min read',
    imageUrl: '/placeholder-2.jpg'
  }
]

// Categories for filtering
const categories = ['All', 'Machine Learning', 'Data Visualization', 'Forecasting', 'Methodology', 'Policy']

export default function BlogPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('All')
  
  // Filter blog posts based on search query and category
  const filteredPosts = blogPosts.filter(post => {
    const matchesSearch = post.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
                          post.excerpt.toLowerCase().includes(searchQuery.toLowerCase())
    
    const matchesCategory = selectedCategory === 'All' || post.category === selectedCategory
    
    return matchesSearch && matchesCategory
  })

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-3xl mx-auto text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">ML in Macroeconomics Blog</h1>
        <p className="text-xl text-muted-foreground">
          Exploring the intersection of machine learning and macroeconomics through practical examples and insights.
        </p>
      </div>
      
      {/* Search and Filter */}
      <div className="mb-12 flex flex-col md:flex-row gap-4 max-w-3xl mx-auto">
        <div className="flex-grow">
          <Input
            placeholder="Search blog posts..."
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
      
      {/* Blog Posts */}
      <div className="max-w-3xl mx-auto space-y-8">
        {filteredPosts.map((post) => (
          <Card key={post.id} className="overflow-hidden">
            <div className="md:flex">
              <div className="md:w-1/3 h-48 md:h-auto bg-muted flex items-center justify-center">
                <div className="text-muted-foreground">Blog Image</div>
              </div>
              <div className="p-6 md:w-2/3">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm text-primary">{post.category}</span>
                  <span className="text-sm text-muted-foreground">{post.date}</span>
                </div>
                <h2 className="text-xl font-semibold mb-2">{post.title}</h2>
                <p className="text-muted-foreground mb-4">{post.excerpt}</p>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">{post.readTime}</span>
                  <Button variant="outline" asChild>
                    <Link href={`/blog/${post.id}`}>Read Post</Link>
                  </Button>
                </div>
              </div>
            </div>
          </Card>
        ))}
        
        {/* No Results */}
        {filteredPosts.length === 0 && (
          <div className="text-center py-12">
            <h3 className="text-xl font-semibold mb-2">No blog posts found</h3>
            <p className="text-muted-foreground">Try adjusting your search or filter criteria</p>
          </div>
        )}
      </div>
    </div>
  )
}
