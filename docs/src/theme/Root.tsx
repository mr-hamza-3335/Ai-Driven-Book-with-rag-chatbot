import React from 'react';
import Chatbot from '../components/Chatbot';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  return (
    <>
      {children}
      <Chatbot apiUrl="http://localhost:8000" />
    </>
  );
}
