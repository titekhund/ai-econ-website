'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'
import { Button } from '@/components/ui/button'
import { confirmSubscription } from '../../actions'
import { useState, useEffect } from 'react'

export default function ConfirmSubscriptionPage() {
  const params = useParams()
  const token = params.token as string
  
  const [isConfirming, setIsConfirming] = useState(true)
  const [isSuccess, setIsSuccess] = useState(false)
  const [error, setError] = useState('')
  
  useEffect(() => {
    const confirmToken = async () => {
      try {
        const result = await confirmSubscription(token)
        setIsSuccess(result.success)
      } catch (err) {
        setError('Failed to confirm subscription. The token may be invalid or expired.')
      } finally {
        setIsConfirming(false)
      }
    }
    
    confirmToken()
  }, [token])

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-md mx-auto text-center">
        <h1 className="text-3xl font-bold mb-6">Email Confirmation</h1>
        
        {isConfirming ? (
          <div className="space-y-4">
            <p>Confirming your subscription...</p>
            <div className="flex justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          </div>
        ) : isSuccess ? (
          <div className="space-y-6">
            <div className="bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-300 p-4 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Subscription Confirmed!</h2>
              <p>Thank you for confirming your email address. You will now receive updates about new articles.</p>
            </div>
            <Button asChild>
              <Link href="/">Return to Homepage</Link>
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-300 p-4 rounded-lg">
              <h2 className="text-xl font-semibold mb-2">Confirmation Failed</h2>
              <p>{error || 'There was an error confirming your subscription.'}</p>
            </div>
            <Button asChild>
              <Link href="/subscribe">Try Again</Link>
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
