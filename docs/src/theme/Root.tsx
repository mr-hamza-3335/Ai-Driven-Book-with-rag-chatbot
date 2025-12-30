import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Chatbot from '../components/Chatbot';

interface RootProps {
  children: React.ReactNode;
}

export default function Root({ children }: RootProps): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  const apiUrl = (siteConfig.customFields?.apiUrl as string) || 'http://localhost:8000';

  return (
    <>
      {children}
      <Chatbot apiUrl={apiUrl} />
    </>
  );
}
