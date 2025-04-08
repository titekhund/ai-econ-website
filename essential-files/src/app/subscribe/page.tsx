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
import { toast } from '@/components/ui/use-toast'
import { Toaster } from '@/components/ui/use-toast'
import { subscribeUser } from './actions'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'

export default function SubscribePage() {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  
  const handleSubmit = async (formData: FormData) => {
    setIsSubmitting(true)
    
    try {
      const result = await subscribeUser(formData)
      
      if (result.success) {
        setIsSuccess(true)
        toast({
          title: "Subscription initiated",
          description: "Please check your email to confirm your subscription.",
        })
      } else {
        toast({
          title: "Subscription failed",
          description: "There was an error processing your subscription. Please try again.",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred. Please try again later.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-4xl font-bold mb-4">Subscribe to Updates</h1>
        <p className="text-xl text-muted-foreground mb-8">
          Stay informed about the latest research and articles on AI in economics.
        </p>
        
        {isSuccess ? (
          <Card className="p-6">
            <div className="text-center space-y-4">
              <h2 className="text-2xl font-semibold">Thank You for Subscribing!</h2>
              <p>
                We've sent a confirmation email to your inbox. Please click the link in the email to complete your subscription.
              </p>
              <p className="text-sm text-muted-foreground">
                If you don't see the email, please check your spam folder.
              </p>
            </div>
          </Card>
        ) : (
          <Card className="p-6">
            <form action={handleSubmit} className="space-y-6">
              <div className="space-y-4">
                <div>
                  <Label htmlFor="name">Name</Label>
                  <Input id="name" name="name" required />
                </div>
                
                <div>
                  <Label htmlFor="email">Email Address</Label>
                  <Input id="email" name="email" type="email" required />
                </div>
                
                <div>
                  <Label>Areas of Interest</Label>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                    <div className="flex items-center space-x-2">
                      <Checkbox id="interest-ml" name="interests" value="machine-learning" />
                      <label htmlFor="interest-ml">Machine Learning</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox id="interest-forecasting" name="interests" value="forecasting" />
                      <label htmlFor="interest-forecasting">Economic Forecasting</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox id="interest-policy" name="interests" value="policy" />
                      <label htmlFor="interest-policy">Policy Analysis</label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox id="interest-time-series" name="interests" value="time-series" />
                      <label htmlFor="interest-time-series">Time Series Analysis</label>
                    </div>
                  </div>
                </div>
                
                <div>
                  <Label>Email Frequency</Label>
                  <RadioGroup defaultValue="weekly" name="frequency" className="mt-2">
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="immediate" id="frequency-immediate" />
                      <Label htmlFor="frequency-immediate">Immediate (as articles are published)</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="weekly" id="frequency-weekly" />
                      <Label htmlFor="frequency-weekly">Weekly Digest</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <RadioGroupItem value="monthly" id="frequency-monthly" />
                      <Label htmlFor="frequency-monthly">Monthly Digest</Label>
                    </div>
                  </RadioGroup>
                </div>
              </div>
              
              <Button type="submit" className="w-full" disabled={isSubmitting}>
                {isSubmitting ? "Processing..." : "Subscribe"}
              </Button>
              
              <p className="text-xs text-muted-foreground text-center">
                By subscribing, you agree to receive emails from us. You can unsubscribe at any time.
              </p>
            </form>
          </Card>
        )}
      </div>
      
      <Toaster />
    </div>
  )
}
