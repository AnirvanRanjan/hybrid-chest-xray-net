import React from 'react';

interface Props {
  onPressed: () => void;
  label?: string;
  icon?: React.ReactNode;
  gradient?: [string, string];
  isLoading?: boolean;
  className?: string;
}

export function AnimatedUploadButton({
  onPressed,
  label = 'Upload File',
  icon,
  gradient = ['#00D4FF', '#7B2FBE'],
  isLoading = false,
  className = '',
}: Props) {
  return (
    <button
      onClick={isLoading ? undefined : onPressed}
      disabled={isLoading}
      className={`animate-pulse-glow inline-flex items-center gap-3 px-8 py-4 rounded-[20px] font-semibold text-[15px] transition-all duration-300 disabled:opacity-70 disabled:cursor-not-allowed ${className}`}
      style={{
        background: `linear-gradient(135deg, ${gradient[0]}, ${gradient[1]})`,
        color: '#0A0E21',
        boxShadow: `0 4px 20px 2px ${gradient[0]}4D`,
      }}
    >
      {isLoading ? (
        <div className="w-5 h-5 border-2 border-[#0A0E21] border-t-transparent rounded-full animate-spin" />
      ) : (
        icon
      )}
      <span>{isLoading ? 'Processing...' : label}</span>
    </button>
  );
}
