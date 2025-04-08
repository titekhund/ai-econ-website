'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { toast } from '@/components/ui/use-toast'
import { Toaster } from '@/components/ui/use-toast'

export default function CommentSection() {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [comment, setComment] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // Mock comments data
  const [comments, setComments] = useState([
    {
      id: 1,
      name: 'John Smith',
      date: '2025-04-05',
      content: 'Great article! I especially liked the section on LSTM networks for time series forecasting.'
    },
    {
      id: 2,
      name: 'Maria Garcia',
      date: '2025-04-04',
      content: 'This was very helpful for my research. Would love to see more on explainable AI in economics.'
    }
  ])
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)
    
    // Simulate API call to post comment
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Add new comment to the list
      const newComment = {
        id: comments.length + 1,
        name,
        date: new Date().toISOString().split('T')[0],
        content: comment
      }
      
      setComments([newComment, ...comments])
      
      // Reset form
      setName('')
      setEmail('')
      setComment('')
      
      toast({
        title: "Comment posted",
        description: "Your comment has been posted successfully.",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: "There was an error posting your comment. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold">Comments</h2>
      
      {/* Comment form */}
      <Card className="p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <h3 className="text-lg font-semibold">Leave a Comment</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="email">Email (will not be published)</Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="comment">Comment</Label>
            <Textarea
              id="comment"
              rows={4}
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              required
            />
          </div>
          
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Posting..." : "Post Comment"}
          </Button>
        </form>
      </Card>
      
      {/* Comments list */}
      <div className="space-y-4">
        {comments.length === 0 ? (
          <p className="text-muted-foreground text-center py-4">
            No comments yet. Be the first to comment!
          </p>
        ) : (
          comments.map((comment) => (
            <Card key={comment.id} className="p-4">
              <div className="flex justify-between items-start mb-2">
                <h4 className="font-semibold">{comment.name}</h4>
                <span className="text-sm text-muted-foreground">{comment.date}</span>
              </div>
              <p>{comment.content}</p>
            </Card>
          ))
        )}
      </div>
      
      <Toaster />
    </div>
  )
}
