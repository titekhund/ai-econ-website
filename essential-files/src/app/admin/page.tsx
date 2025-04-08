'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toaster } from '@/components/ui/toaster'
import { sendNotifications } from '../subscribe/actions'

// Mock data for admin dashboard
const subscriberStats = {
  total: 156,
  confirmed: 142,
  unconfirmed: 14,
  daily: 12,
  weekly: 98,
  monthly: 32,
  immediate: 14
}

const recentSubscribers = [
  { email: 'john.smith@example.com', date: '2025-04-05', status: 'confirmed' },
  { email: 'maria.johnson@example.com', date: '2025-04-05', status: 'confirmed' },
  { email: 'david.williams@example.com', date: '2025-04-04', status: 'unconfirmed' },
  { email: 'sarah.brown@example.com', date: '2025-04-04', status: 'confirmed' },
  { email: 'michael.jones@example.com', date: '2025-04-03', status: 'confirmed' }
]

const recentArticles = [
  { id: 1, title: 'Machine Learning Applications in Macroeconomic Forecasting', date: '2025-03-28', notified: true },
  { id: 2, title: 'Neural Networks for Time Series Analysis in Economics', date: '2025-03-15', notified: true },
  { id: 3, title: 'Reinforcement Learning in Monetary Policy Optimization', date: '2025-02-22', notified: false },
  { id: 4, title: 'Natural Language Processing for Economic Sentiment Analysis', date: '2025-02-10', notified: false },
  { id: 5, title: 'Predictive Analytics for GDP Growth Estimation', date: '2025-01-30', notified: true }
]

export default function AdminPage() {
  const [sendingNotification, setSendingNotification] = useState<number | null>(null)
  
  const handleSendNotification = async (articleId: number) => {
    setSendingNotification(articleId)
    
    try {
      // In a real application, this would call the server action to send notifications
      await sendNotifications(articleId)
      
      // Update the UI to show notification was sent
      // In a real application, we would refresh the data from the server
      setTimeout(() => {
        setSendingNotification(null)
      }, 1500)
    } catch (error) {
      console.error('Error sending notification:', error)
      setSendingNotification(null)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Subscription Management</h1>
      
      <Tabs defaultValue="overview">
        <TabsList className="mb-8">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="subscribers">Subscribers</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-2">Total Subscribers</h3>
              <p className="text-4xl font-bold">{subscriberStats.total}</p>
              <p className="text-muted-foreground mt-2">
                {subscriberStats.confirmed} confirmed, {subscriberStats.unconfirmed} pending
              </p>
            </Card>
            
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-2">Frequency Breakdown</h3>
              <div className="space-y-2 mt-4">
                <div className="flex justify-between">
                  <span>Daily</span>
                  <span className="font-semibold">{subscriberStats.daily}</span>
                </div>
                <div className="flex justify-between">
                  <span>Weekly</span>
                  <span className="font-semibold">{subscriberStats.weekly}</span>
                </div>
                <div className="flex justify-between">
                  <span>Monthly</span>
                  <span className="font-semibold">{subscriberStats.monthly}</span>
                </div>
                <div className="flex justify-between">
                  <span>Immediate</span>
                  <span className="font-semibold">{subscriberStats.immediate}</span>
                </div>
              </div>
            </Card>
            
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-2">Recent Activity</h3>
              <p className="text-muted-foreground">
                Last notification sent: <span className="font-semibold">April 2, 2025</span>
              </p>
              <p className="text-muted-foreground mt-2">
                New subscribers this week: <span className="font-semibold">14</span>
              </p>
            </Card>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Recent Subscribers</h3>
              <div className="space-y-4">
                {recentSubscribers.map((subscriber, index) => (
                  <div key={index} className="flex justify-between items-center border-b border-border pb-2 last:border-0">
                    <div>
                      <p className="font-medium">{subscriber.email}</p>
                      <p className="text-sm text-muted-foreground">{subscriber.date}</p>
                    </div>
                    <span className={`text-sm px-2 py-1 rounded-full ${
                      subscriber.status === 'confirmed' 
                        ? 'bg-green-500/20 text-green-500' 
                        : 'bg-amber-500/20 text-amber-500'
                    }`}>
                      {subscriber.status}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-4">
                <Button variant="outline" className="w-full">View All Subscribers</Button>
              </div>
            </Card>
            
            <Card className="p-6">
              <h3 className="text-xl font-semibold mb-4">Recent Articles</h3>
              <div className="space-y-4">
                {recentArticles.map((article) => (
                  <div key={article.id} className="flex justify-between items-center border-b border-border pb-2 last:border-0">
                    <div>
                      <p className="font-medium">{article.title}</p>
                      <p className="text-sm text-muted-foreground">{article.date}</p>
                    </div>
                    {article.notified ? (
                      <span className="text-sm px-2 py-1 rounded-full bg-green-500/20 text-green-500">
                        Notified
                      </span>
                    ) : (
                      <Button 
                        size="sm" 
                        variant="outline"
                        disabled={sendingNotification === article.id}
                        onClick={() => handleSendNotification(article.id)}
                      >
                        {sendingNotification === article.id ? 'Sending...' : 'Send Notification'}
                      </Button>
                    )}
                  </div>
                ))}
              </div>
              <div className="mt-4">
                <Button variant="outline" className="w-full">View All Articles</Button>
              </div>
            </Card>
          </div>
        </TabsContent>
        
        <TabsContent value="subscribers">
          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Subscriber Management</h3>
            <p className="text-muted-foreground mb-4">
              This section would contain a full list of subscribers with filtering, search, and management options.
            </p>
            <p className="text-muted-foreground">
              Features would include:
            </p>
            <ul className="list-disc list-inside space-y-1 mt-2 text-muted-foreground">
              <li>View all subscriber details</li>
              <li>Filter by status, frequency, and interests</li>
              <li>Export subscriber list</li>
              <li>Manually add or remove subscribers</li>
              <li>Send targeted notifications to specific subscriber segments</li>
            </ul>
          </Card>
        </TabsContent>
        
        <TabsContent value="notifications">
          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Notification History</h3>
            <p className="text-muted-foreground mb-4">
              This section would contain a history of all notifications sent, with details on delivery rates and engagement.
            </p>
            <p className="text-muted-foreground">
              Features would include:
            </p>
            <ul className="list-disc list-inside space-y-1 mt-2 text-muted-foreground">
              <li>View all past notifications</li>
              <li>See delivery statistics</li>
              <li>Track open and click rates</li>
              <li>Schedule future notifications</li>
              <li>Create notification templates</li>
            </ul>
          </Card>
        </TabsContent>
      </Tabs>
      
      <Toaster />
    </div>
  )
}
