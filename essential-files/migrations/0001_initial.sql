-- Initialize subscribers table
CREATE TABLE IF NOT EXISTS subscribers (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  interests TEXT,
  frequency TEXT NOT NULL DEFAULT 'weekly',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  confirmed BOOLEAN DEFAULT FALSE,
  confirmation_token TEXT,
  last_notification_sent TIMESTAMP
);

-- Initialize articles table
CREATE TABLE IF NOT EXISTS articles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  excerpt TEXT,
  author TEXT,
  category TEXT,
  tags TEXT,
  date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  has_code BOOLEAN DEFAULT FALSE,
  code_snippet TEXT
);

-- Initialize notifications table
CREATE TABLE IF NOT EXISTS notifications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  article_id INTEGER NOT NULL,
  sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  recipient_count INTEGER DEFAULT 0,
  FOREIGN KEY (article_id) REFERENCES articles(id)
);
