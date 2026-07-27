import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedGradientBackground } from '../components/common/AnimatedGradientBackground';
import { GlassCard } from '../components/common/GlassCard';
import { FloatingWrapper } from '../components/common/FloatingWrapper';
import { COLORS } from '../constants';
import type { ToolItem } from '../types';
import {
  Rocket, ArrowRight, Zap, ArrowLeftRight,
  Image, PenTool, Merge
} from 'lucide-react';

const TOOLS: ToolItem[] = [
  {
    name: 'AI Resume Builder',
    description: 'Craft compelling resumes with AI-powered assistance and live preview',
    icon: 'Rocket',
    route: '/resume-builder',
    gradientColors: ['#00D4FF', '#7B2FBE'],
  },
  {
    name: 'Merge & Split PDF',
    description: 'Combine multiple PDFs or extract specific pages effortlessly',
    icon: 'Merge',
    route: '/merge-split',
    gradientColors: ['#FF9100', '#FF6D00'],
  },
  {
    name: 'Format Converter',
    description: 'Convert, compress, and enhance documents with pixel-perfect quality',
    icon: 'ArrowLeftRight',
    route: '/format-converter',
    gradientColors: ['#00E676', '#00BFA5'],
  },
  {
    name: 'Image to PDF',
    description: 'Transform photos into polished PDFs with optional OCR text extraction',
    icon: 'Image',
    route: '/image-to-pdf',
    gradientColors: ['#448AFF', '#7B2FBE'],
  },
  {
    name: 'Digital Signature',
    description: 'Draw, apply, and seal your signature onto any PDF document',
    icon: 'PenTool',
    route: '/digital-signature',
    gradientColors: ['#FF2D78', '#FF9100'],
  },
];

const iconMap: Record<string, React.ReactNode> = {
  Rocket: <Rocket size={26} color="white" />,
  Merge: <Merge size={26} color="white" />,
  ArrowLeftRight: <ArrowLeftRight size={26} color="white" />,
  Image: <Image size={26} color="white" />,
  PenTool: <PenTool size={26} color="white" />,
};

export function DashboardScreen() {
  const navigate = useNavigate();
  const [windowWidth, setWindowWidth] = useState(
    typeof window !== 'undefined' ? window.innerWidth : 1200
  );
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    setTimeout(() => setVisible(true), 100);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const columns = windowWidth > 600 ? 2 : 1;

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen flex justify-center">
        <div className="w-full max-w-[1000px] px-6 py-12">
          {/* Brand Header */}
          <div className={`text-center mb-10 transition-all duration-700 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-8'}`}>
            {/* Brand chip */}
            <div className="inline-flex items-center gap-3 px-5 py-2.5 rounded-full mb-7"
              style={{
                background: `linear-gradient(135deg, ${COLORS.accentCyan}26, ${COLORS.accentViolet}26)`,
                border: `1px solid ${COLORS.accentCyan}33`,
              }}
            >
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center"
                style={{
                  background: `linear-gradient(135deg, ${COLORS.accentCyan}, ${COLORS.accentViolet})`,
                  boxShadow: `0 0 12px ${COLORS.accentCyan}4D`,
                }}
              >
                <Zap size={18} color="white" />
              </div>
              <span
                className="text-[22px] font-bold tracking-tight"
                style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}
              >
                DocuForge
              </span>
            </div>

            {/* Tagline */}
            <h1
              className="text-[32px] font-extrabold leading-tight tracking-tight"
              style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}
            >
              Smart Document Suite
            </h1>

            {/* Subtitle */}
            <p className="text-[15px] mt-3" style={{ color: COLORS.textTertiary, lineHeight: 1.5 }}>
              Fast, secure, and AI-powered document processing
            </p>

            {/* Gradient accent line */}
            <div className="flex justify-center mt-6">
              <div
                className="w-[60px] h-[3px] rounded-sm"
                style={{ background: `linear-gradient(135deg, ${COLORS.accentCyan}, ${COLORS.accentViolet})` }}
              />
            </div>
          </div>

          {/* Tool Grid */}
          <div
            className="grid gap-5"
            style={{
              gridTemplateColumns: `repeat(${columns}, 1fr)`,
            }}
          >
            {TOOLS.map((tool, index) => (
              <FloatingWrapper
                key={tool.route}
                amplitude={5}
                duration={4 + (index % 2)}
                delay={300 * index}
              >
                <GlassCard
                  onTap={() => navigate(tool.route)}
                  padding="20px"
                  className={`transition-all duration-500 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}
                  style={{ animationDelay: `${500 + 120 * index}ms`, minHeight: columns === 1 ? 200 : 230 }}
                >
                  <div className="flex flex-col h-full" style={{ minHeight: columns === 1 ? 160 : 190 }}>
                    {/* Icon + Arrow */}
                    <div className="flex items-start justify-between">
                      <div
                        className="w-[52px] h-[52px] rounded-[14px] flex items-center justify-center"
                        style={{
                          background: `linear-gradient(135deg, ${tool.gradientColors[0]}, ${tool.gradientColors[1]})`,
                          boxShadow: `0 6px 16px ${tool.gradientColors[0]}4D`,
                        }}
                      >
                        {iconMap[tool.icon]}
                      </div>
                      <div
                        className="w-8 h-8 rounded-full flex items-center justify-center"
                        style={{
                          backgroundColor: 'rgba(255,255,255,0.05)',
                          border: '1px solid rgba(255,255,255,0.15)',
                        }}
                      >
                        <ArrowRight size={16} style={{ color: COLORS.textTertiary }} />
                      </div>
                    </div>

                    {/* Spacer */}
                    <div className="flex-1" />

                    {/* Name + Description */}
                    <div>
                      <h3
                        className="text-lg font-bold leading-snug mb-2"
                        style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}
                      >
                        {tool.name}
                      </h3>
                      <p className="text-[13px] leading-relaxed line-clamp-2" style={{ color: COLORS.textTertiary }}>
                        {tool.description}
                      </p>
                    </div>
                  </div>
                </GlassCard>
              </FloatingWrapper>
            ))}
          </div>

          {/* Footer */}
          <div className="text-center mt-10 py-10 opacity-50">
            <p className="text-xs tracking-wider" style={{ color: COLORS.textTertiary }}>
              Built with React & Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </AnimatedGradientBackground>
  );
}
