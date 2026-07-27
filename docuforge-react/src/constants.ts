export const BASE_URL = 'http://localhost:8000';
export const STATUS_ENDPOINT = '/api/v1/status';
export const UPLOAD_ENDPOINT = '/api/v1/upload';
export const POLL_INTERVAL = 3000;
export const MOCK_TASK_DURATION = 5000;

export const FLOATING_DURATION = 3;
export const ENTRY_ANIMATION_DURATION = 600;
export const STAGGER_DELAY = 100;
export const PAGE_TRANSITION_DURATION = 400;

export const GLASS_BLUR_SIGMA = 18;
export const GLASS_BORDER_OPACITY = 0.15;
export const GLASS_FILL_OPACITY = 0.08;

export const CARD_BORDER_RADIUS = 24;
export const INPUT_BORDER_RADIUS = 16;
export const BUTTON_BORDER_RADIUS = 20;

export const FLOATING_AMPLITUDE = 8;

// Colors
export const COLORS = {
  bgDark: '#0A0E21',
  bgMedium: '#111632',
  surfaceDark: '#161B3A',
  accentCyan: '#00D4FF',
  accentViolet: '#7B2FBE',
  accentPink: '#FF2D78',
  accentGreen: '#00E676',
  accentOrange: '#FF9100',
  accentBlue: '#448AFF',
  glassBorder: 'rgba(255, 255, 255, 0.15)',
  glassFill: 'rgba(255, 255, 255, 0.08)',
  glassHighlight: 'rgba(255, 255, 255, 0.10)',
  textPrimary: '#F0F0F5',
  textSecondary: 'rgba(240, 240, 245, 0.7)',
  textTertiary: 'rgba(240, 240, 245, 0.5)',
} as const;

export const TOOL_GRADIENTS: [string, string][] = [
  ['#00D4FF', '#7B2FBE'],
  ['#FF2D78', '#FF9100'],
  ['#00E676', '#00D4FF'],
  ['#448AFF', '#7B2FBE'],
  ['#FF9100', '#FF2D78'],
  ['#7B2FBE', '#FF2D78'],
];