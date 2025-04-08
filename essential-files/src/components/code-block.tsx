'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Download, Copy, Check } from 'lucide-react'

interface CodeBlockProps {
  code: string
  language: string
  filename?: string
}

export default function CodeBlock({ code, language, filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  
  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  const handleDownload = () => {
    const element = document.createElement('a')
    const file = new Blob([code], { type: 'text/plain' })
    element.href = URL.createObjectURL(file)
    element.download = filename || `code.${language === 'python' ? 'py' : language}`
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }
  
  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between bg-muted px-4 py-2 border-b">
        <div className="flex items-center space-x-2">
          <span className="font-mono text-sm">
            {filename || `code.${language === 'python' ? 'py' : language}`}
          </span>
          <span className="text-xs px-2 py-1 rounded-full bg-primary/10 text-primary">
            {language}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={handleCopy}
            title="Copy code"
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
          {filename && (
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={handleDownload}
              title="Download code"
            >
              <Download className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
      <pre className="p-4 overflow-x-auto">
        <code className="text-sm font-mono">{code}</code>
      </pre>
    </Card>
  )
}
