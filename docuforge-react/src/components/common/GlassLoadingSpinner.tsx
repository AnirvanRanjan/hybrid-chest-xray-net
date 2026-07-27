/* eslint-disable @next/next/no-img-element */

interface Props {
  message?: string;
  size?: number;
  className?: string;
}

export function GlassLoadingSpinner({ message = 'Processing...', size = 80, className = '' }: Props) {
  const containerSize = size + 40;

  return (
    <div className={`flex flex-col items-center gap-4 ${className}`}>
      <div
        className="flex items-center justify-center"
        style={{
          width: containerSize,
          height: containerSize,
          borderRadius: 24,
          background: 'linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.08))',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.15)',
        }}
      >
        <svg width={size} height={size} className="animate-spin-gradient">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={size / 2 - 4}
            fill="none"
            stroke="rgba(255,255,255,0.15)"
            strokeWidth="3"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={size / 2 - 4}
            fill="none"
            stroke="url(#spinnerGradient)"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray={`${1.8 * (size / 2 - 4)} ${2 * Math.PI * (size / 2 - 4)}`}
            style={{ transformOrigin: 'center', transform: 'rotate(-90deg)' }}
          />
          <defs>
            <linearGradient id="spinnerGradient" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#00D4FF" />
              <stop offset="100%" stopColor="#7B2FBE" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <p className="text-sm font-medium" style={{ color: 'rgba(240,240,245,0.7)' }}>
        {message}
      </p>
    </div>
  );
}
