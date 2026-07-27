import { useState } from 'react';
import { GlassCard } from './GlassCard';
import { COLORS } from '../../constants';

interface Props {
  fileName: string;
  outputType: string;
  pageCount?: number;
  accentColor?: string;
  extraInfo?: string;
  downloadUrl?: string;
}

export function DocumentPreviewCard({
  fileName,
  outputType,
  pageCount = 1,
  accentColor = COLORS.accentCyan,
  extraInfo,
  downloadUrl,
}: Props) {
  const [zoomed, setZoomed] = useState(false);

  const outputIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'pdf': return '📄';
      case 'word': return '📝';
      case 'image': return '🖼';
      case 'signed pdf': return '✅';
      case 'compressed': return '📦';
      case 'enhanced': return '✨';
      default: return '📋';
    }
  };

  return (
    <GlassCard borderGradient={[accentColor, COLORS.accentCyan]} className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
      {/* Header */}
      <div className="flex items-center gap-3">
        <div
          className="w-9 h-9 rounded-[10px] flex items-center justify-center"
          style={{ background: `linear-gradient(135deg, ${accentColor}40, ${COLORS.accentCyan}26)` }}
        >
          <span style={{ color: accentColor, fontSize: '16px' }}>{outputIcon(outputType)}</span>
        </div>
        <div className="flex-1">
          <p className="text-[15px] font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>
            Output Preview
          </p>
          <p className="text-xs" style={{ color: COLORS.textTertiary }}>
            {outputType} · {pageCount} page{pageCount === 1 ? '' : 's'}
          </p>
        </div>
        <button
          onClick={() => setZoomed(!zoomed)}
          className="flex items-center gap-1 px-3 py-1.5 rounded-[10px] text-xs font-medium transition-all"
          style={{
            backgroundColor: `${accentColor}1A`,
            border: `1px solid ${accentColor}4D`,
            color: accentColor,
          }}
        >
          {zoomed ? '⊟' : '⊞'} {zoomed ? 'Zoom Out' : 'Zoom In'}
        </button>
      </div>

      {/* Simulated Document Page */}
      <div
        className="mt-4 rounded-xl overflow-hidden transition-all duration-300"
        style={{
          height: zoomed ? 500 : 280,
          backgroundColor: '#F8F9FA',
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
        }}
      >
        <div className="p-5 h-full flex flex-col gap-3">
          {/* Title bar */}
          <div className="w-full h-3.5 rounded-sm" style={{ backgroundColor: 'rgba(26,26,46,0.8)' }} />
          <div className="w-[180px] h-2.5 rounded-sm" style={{ backgroundColor: 'rgba(26,26,46,0.35)' }} />
          <div className="w-full h-px my-1" style={{ backgroundColor: 'rgba(26,26,46,0.1)' }} />
          {/* Text lines */}
          {[1, 0.85, 0.92, 0.78, 0.88, 0.65].map((w, i) => (
            <div
              key={i}
              className="rounded-sm"
              style={{
                width: `${w * 100}%`,
                height: 10,
                backgroundColor: `rgba(26,26,46,${0.12 + (i % 2 === 0 ? 0.05 : 0)})`,
              }}
            />
          ))}
          {/* File type badge */}
          <div className="absolute bottom-3 right-3">
            <span
              className="px-2 py-1 rounded-md text-[10px] font-bold tracking-wider"
              style={{ backgroundColor: `${accentColor}26`, border: `1px solid ${accentColor}66`, color: accentColor }}
            >
              {outputType}
            </span>
          </div>
        </div>
      </div>

      {/* File info */}
      <div className="flex items-center gap-2 mt-3">
        <span style={{ color: COLORS.accentGreen }}>✓</span>
        <span className="text-xs flex-1" style={{ color: COLORS.textSecondary }}>
          {extraInfo || fileName}
        </span>
        {downloadUrl && (
          <span className="flex items-center gap-1 text-[11px]" style={{ color: COLORS.accentCyan }}>
            ☁ Ready to download
          </span>
        )}
      </div>
    </GlassCard>
  );
}
