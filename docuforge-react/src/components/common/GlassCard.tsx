import { useState } from 'react';

interface Props {
  children: React.ReactNode;
  className?: string;
  padding?: string;
  margin?: string;
  borderRadius?: string;
  borderGradient?: string[];
  onTap?: () => void;
  style?: React.CSSProperties;
}

export function GlassCard({
  children,
  className = '',
  padding = '20px',
  margin,
  borderRadius = '24px',
  borderGradient,
  onTap,
  style,
}: Props) {
  const [isHovered, setIsHovered] = useState(false);

  const baseStyle: React.CSSProperties = {
    borderRadius,
    padding,
    background: isHovered
      ? 'linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.08))'
      : 'linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.08))',
    backdropFilter: 'blur(18px)',
    WebkitBackdropFilter: 'blur(18px)',
    border: borderGradient
      ? undefined
      : `1px solid ${isHovered ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.15)'}`,
    transition: 'all 0.25s ease-out',
    transform: onTap && isHovered ? 'scale(1.025)' : 'scale(1)',
    boxShadow: onTap && isHovered
      ? `0 0 24px 2px ${borderGradient ? borderGradient[0] : '#00D4FF'}26`
      : 'none',
    cursor: onTap ? 'pointer' : 'default',
    ...style,
  };

  if (borderGradient) {
    const gradStyle: React.CSSProperties = {
      position: 'relative',
      ...baseStyle,
      border: '1.5px solid transparent',
      backgroundClip: 'padding-box',
    };
    return (
      <div
        style={gradStyle}
        className={className}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        onClick={onTap}
      >
        <div
          style={{
            position: 'absolute',
            inset: 0,
            borderRadius,
            padding: '1.5px',
            background: `linear-gradient(135deg, ${borderGradient[0]}${isHovered ? '8C' : '66'}, ${borderGradient[1]}${isHovered ? '40' : '26'})`,
            WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
            WebkitMaskComposite: 'xor',
            maskComposite: 'exclude',
            pointerEvents: 'none',
          }}
        />
        {children}
      </div>
    );
  }

  return (
    <div
      style={margin ? { margin, ...baseStyle } : baseStyle}
      className={className}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onTap}
    >
      {children}
    </div>
  );
}
