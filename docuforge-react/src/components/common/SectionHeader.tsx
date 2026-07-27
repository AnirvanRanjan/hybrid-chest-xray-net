import React from 'react';

interface Props {
  title: string;
  subtitle?: string;
  prefixIcon?: React.ReactNode;
  trailing?: React.ReactNode;
  accentColor?: string;
}

export function SectionHeader({
  title,
  subtitle,
  prefixIcon,
  trailing,
  accentColor = '#00D4FF',
}: Props) {
  return (
    <div className="flex items-center gap-3 py-2">
      {prefixIcon && (
        <div
          className="flex items-center justify-center w-9 h-9 rounded-[10px]"
          style={{
            backgroundColor: `${accentColor}1A`,
            border: `1px solid ${accentColor}33`,
          }}
        >
          <span style={{ color: accentColor }}>{prefixIcon}</span>
        </div>
      )}
      <div className="flex-1">
        <h3 className="text-lg font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: '#F0F0F5', lineHeight: 1.2 }}>
          {title}
        </h3>
        {subtitle && (
          <p className="text-xs mt-1" style={{ color: 'rgba(240,240,245,0.49)', lineHeight: 1.3 }}>
            {subtitle}
          </p>
        )}
      </div>
      {trailing}
    </div>
  );
}
