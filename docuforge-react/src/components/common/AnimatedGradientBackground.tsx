import React from 'react';

interface Props {
  children?: React.ReactNode;
  className?: string;
}

export function AnimatedGradientBackground({ children, className = '' }: Props) {
  return (
    <div className={`relative min-h-screen w-full overflow-hidden ${className}`} style={{ backgroundColor: '#0A0E21' }}>
      {/* Blob 1: Cyan - top right */}
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: '80vw',
          height: '80vw',
          maxWidth: '800px',
          maxHeight: '800px',
          top: '10%',
          right: '-10%',
          background: 'radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%)',
          animation: 'blobDrift1 12s ease-in-out infinite',
        }}
      />
      {/* Blob 2: Violet - center left */}
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: '90vw',
          height: '90vw',
          maxWidth: '900px',
          maxHeight: '900px',
          top: '30%',
          left: '-15%',
          background: 'radial-gradient(circle, rgba(123,47,190,0.06) 0%, transparent 70%)',
          animation: 'blobDrift2 12s ease-in-out infinite',
        }}
      />
      {/* Blob 3: Pink - bottom right */}
      <div
        className="absolute rounded-full pointer-events-none"
        style={{
          width: '70vw',
          height: '70vw',
          maxWidth: '700px',
          maxHeight: '700px',
          bottom: '5%',
          right: '0%',
          background: 'radial-gradient(circle, rgba(255,45,120,0.04) 0%, transparent 70%)',
          animation: 'blobDrift3 12s ease-in-out infinite',
        }}
      />
      <div className="relative z-10">{children}</div>
    </div>
  );
}
