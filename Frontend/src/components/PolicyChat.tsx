import React, { useState } from 'react';
import axios from 'axios';

// If needed, set default base URL
axios.defaults.baseURL = 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: {
    policy: string;
    section: string;
  }[];
  screenshots?: string[];
}

const PolicyChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    const formData = new FormData();
    const uploadedFiles = Array.from(e.target.files);

    uploadedFiles.forEach((file) => {
      formData.append('files', file);
    });

    try {
      console.log('Uploading files...');
      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log('Upload response:', response.data);

      setFiles((prev) => [...prev, ...uploadedFiles]);
      alert('Files uploaded successfully');
    } catch (error: any) {
      console.error('Upload error:', error.response?.data || error.message);
      alert(`Failed to upload files: ${error.response?.data?.detail || error.message}`);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const newMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput('');
    setIsLoading(true);

    try {
      console.log('Sending chat message:', input);
      const response = await axios.post('/api/chat', {
        message: newMessage.content,
        history: messages,
      });

      console.log('Chat response:', response.data);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.data.content,
          citations: response.data.citations,
          screenshots: response.data.screenshots,
        },
      ]);
    } catch (error: any) {
      console.error('Chat error:', error.response?.data || error.message);
      alert(`Failed to send message: ${error.response?.data?.detail || error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
      <div style={{ marginBottom: '20px' }}>
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileUpload}
          style={{ marginBottom: '10px' }}
        />
        <div>{files.length} files uploaded</div>
      </div>

      <div
        style={{
          height: '500px',
          overflowY: 'auto',
          border: '1px solid #ccc',
          padding: '10px',
          marginBottom: '20px',
          backgroundColor: '#fff',
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            style={{
              marginBottom: '10px',
              padding: '10px',
              backgroundColor: msg.role === 'user' ? '#e3f2fd' : '#f5f5f5',
              borderRadius: '4px',
            }}
          >
            <div style={{ fontWeight: 'bold' }}>{msg.role === 'user' ? 'You' : 'Assistant'}:</div>
            <div style={{ marginTop: '5px' }}>{msg.content}</div>

            {msg.citations && (
              <div style={{ marginTop: '10px', fontSize: '0.9em' }}>
                <div style={{ fontWeight: 'bold' }}>Citations:</div>
                {msg.citations.map((cite, i) => (
                  <div key={i}>
                    {cite.policy} - {cite.section}
                  </div>
                ))}
              </div>
            )}

            {msg.screenshots && (
              <div
                style={{
                  marginTop: '10px',
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '10px',
                }}
              >
                {msg.screenshots.map((screenshot, i) => (
                  <img
                    key={i}
                    src={`data:image/png;base64,${screenshot}`}
                    alt={`Policy screenshot ${i + 1}`}
                    style={{ width: '100%', border: '1px solid #ddd' }}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '10px' }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about policy..."
          style={{
            flex: 1,
            padding: '8px',
            borderRadius: '4px',
            border: '1px solid #ccc',
          }}
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          style={{
            padding: '8px 16px',
            backgroundColor: isLoading ? '#ccc' : '#1976d2',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
          }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default PolicyChat;

