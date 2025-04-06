'use client'

import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

// Mock data for featured articles
const featuredArticles = [
  {
    id: 1,
    title: 'Machine Learning Applications in Macroeconomic Forecasting',
    excerpt: 'How AI models are revolutionizing economic predictions and policy planning.',
    date: '2025-03-28',
    category: 'Research',
    imageUrl: '/placeholder-1.jpg'
  },
  {
    id: 2,
    title: 'Neural Networks for Time Series Analysis in Economics',
    excerpt: 'Exploring deep learning approaches to analyze economic time series data.',
    date: '2025-03-15',
    category: 'Methodology',
    imageUrl: '/placeholder-2.jpg'
  },
  {
    id: 3,
    title: 'Reinforcement Learning in Monetary Policy Optimization',
    excerpt: 'How central banks can leverage AI for better policy decisions.',
    date: '2025-02-22',
    category: 'Policy',
    imageUrl: '/placeholder-3.jpg'
  }
]

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-background to-background/80 py-16 md:py-24">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold mb-6">
              AI in Economics
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-8">
              Exploring the intersection of artificial intelligence and macroeconomics through research, code, and analysis.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button size="lg" asChild>
                <Link href="/articles">Browse Articles</Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link href="/subscribe">Subscribe</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Articles Section */}
      <section className="py-16 bg-muted/30">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8 text-center">Featured Articles</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {featuredArticles.map((article) => (
              <Card key={article.id} className="overflow-hidden flex flex-col h-full">
                <div className="h-48 bg-muted flex items-center justify-center">
                  <div className="text-muted-foreground">Article Image</div>
                </div>
                <div className="p-6 flex flex-col flex-grow">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-primary">{article.category}</span>
                    <span className="text-sm text-muted-foreground">{article.date}</span>
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{article.title}</h3>
                  <p className="text-muted-foreground mb-4 flex-grow">{article.excerpt}</p>
                  <Button variant="outline" asChild className="w-full">
                    <Link href={`/articles/${article.id}`}>Read Article</Link>
                  </Button>
                </div>
              </Card>
            ))}
          </div>
          <div className="mt-10 text-center">
            <Button variant="outline" asChild>
              <Link href="/articles">View All Articles</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Topics Section */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold mb-8 text-center">Explore Topics</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {['Macroeconomic Modeling', 'Policy Analysis', 'Forecasting', 'Time Series Analysis'].map((topic, index) => (
              <Card key={index} className="p-6 text-center hover:border-primary transition-colors cursor-pointer">
                <h3 className="text-xl font-semibold mb-2">{topic}</h3>
                <p className="text-muted-foreground">Explore articles and research</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Newsletter Section */}
      <section className="py-16 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold mb-4">Stay Updated</h2>
            <p className="text-xl text-muted-foreground mb-8">
              Subscribe to receive the latest articles and research on AI in economics.
            </p>
            <Button size="lg" asChild>
              <Link href="/subscribe">Subscribe to Newsletter</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}
