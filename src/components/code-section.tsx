'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Download, FileDown, FileText } from 'lucide-react'
import CodeBlock from '@/components/code-block'

interface CodeSectionProps {
  files: {
    name: string
    language: string
    code: string
  }[]
  dataFiles?: {
    name: string
    description: string
    size: string
    url: string
  }[]
}

export default function CodeSection({ files, dataFiles = [] }: CodeSectionProps) {
  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Code & Data</h2>
      
      <Tabs defaultValue="code" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="code">Code Files</TabsTrigger>
          {dataFiles.length > 0 && (
            <TabsTrigger value="data">Data Files</TabsTrigger>
          )}
        </TabsList>
        
        <TabsContent value="code" className="space-y-6">
          {files.map((file, index) => (
            <div key={index} className="space-y-2">
              <CodeBlock 
                code={file.code}
                language={file.language}
                filename={file.name}
              />
            </div>
          ))}
          
          {files.length > 1 && (
            <Button variant="outline" className="mt-4">
              <Download className="mr-2 h-4 w-4" />
              Download All Code Files
            </Button>
          )}
        </TabsContent>
        
        {dataFiles.length > 0 && (
          <TabsContent value="data">
            <div className="space-y-4">
              {dataFiles.map((file, index) => (
                <Card key={index} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <div className="flex items-center">
                        <FileText className="mr-2 h-4 w-4 text-muted-foreground" />
                        <h3 className="font-medium">{file.name}</h3>
                      </div>
                      <p className="text-sm text-muted-foreground">{file.description}</p>
                      <p className="text-xs text-muted-foreground">{file.size}</p>
                    </div>
                    <Button variant="outline" size="sm" asChild>
                      <a href={file.url} download>
                        <FileDown className="mr-2 h-4 w-4" />
                        Download
                      </a>
                    </Button>
                  </div>
                </Card>
              ))}
              
              {dataFiles.length > 1 && (
                <Button variant="outline" className="mt-4">
                  <Download className="mr-2 h-4 w-4" />
                  Download All Data Files
                </Button>
              )}
            </div>
          </TabsContent>
        )}
      </Tabs>
    </div>
  )
}
