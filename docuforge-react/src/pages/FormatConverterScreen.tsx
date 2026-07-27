import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedGradientBackground } from '../components/common/AnimatedGradientBackground';
import { GlassCard } from '../components/common/GlassCard';
import { GlassLoadingSpinner } from '../components/common/GlassLoadingSpinner';
import { AnimatedUploadButton } from '../components/common/AnimatedUploadButton';
import { DocumentPreviewCard } from '../components/common/DocumentPreviewCard';
import { useTaskManager } from '../hooks/useTaskManager';
import { useToast } from '../hooks/useToast';
import { ToastContainer } from '../components/common/Toast';
import { COLORS } from '../constants';
import { ChevronLeft, ArrowRight, Upload, FileText } from 'lucide-react';

const ALL_FORMATS = ['PDF', 'Word', 'Excel', 'Images', 'Text', 'JPG', 'PNG', 'WebP', 'BMP', 'TIFF', 'GIF'];

const CONVERSION_MAP: Record<string, string[]> = {
  Word: ['PDF', 'Text'], Excel: ['PDF', 'Text'], PDF: ['Word', 'Excel', 'Text'],
  Text: ['PDF', 'Word', 'Excel'], Images: ['PDF', 'Word', 'Text'],
  JPG: ['PNG', 'WebP', 'BMP', 'TIFF', 'GIF', 'PDF'],
  PNG: ['JPG', 'WebP', 'BMP', 'TIFF', 'GIF', 'PDF'],
  WebP: ['JPG', 'PNG', 'BMP', 'TIFF', 'GIF', 'PDF'],
  BMP: ['JPG', 'PNG', 'WebP', 'TIFF', 'GIF', 'PDF'],
  TIFF: ['JPG', 'PNG', 'WebP', 'BMP', 'GIF', 'PDF'],
  GIF: ['JPG', 'PNG', 'WebP', 'BMP', 'TIFF', 'PDF'],
};

const FORMAT_COLORS: Record<string, string> = {
  Word: '#448AFF', Excel: '#217346', PDF: '#FF2D78', Images: '#00E676',
  Text: '#7B2FBE', JPG: '#FF6F00', PNG: '#1565C0', WebP: '#00897B',
  BMP: '#5D4037', TIFF: '#6A1B9A', GIF: '#C62828',
};

const FORMAT_ICONS: Record<string, string> = {
  Word: '📝', Excel: '📊', PDF: '📄', Images: '🖼', Text: '📃',
  JPG: '🖼', PNG: '🖼', WebP: '🖼', BMP: '🖼', TIFF: '🖼', GIF: '🎞',
};

const MODES = [
  { icon: '↔', label: 'Convert' },
  { icon: '📦', label: 'Compress' },
  { icon: '✨', label: 'Enhance' },
];

export function FormatConverterScreen() {
  const navigate = useNavigate();
  const { task, loading, error, reset, uploadFile, isPending, isCompleted, isFailed } = useTaskManager();
  const { toasts, removeToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState(0);
  const [fromFormat, setFromFormat] = useState('Word');
  const [toFormat, setToFormat] = useState('PDF');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [enhanceOptions, setEnhanceOptions] = useState({ sharpen: true, upscale: false, denoise: false });

  useEffect(() => { reset(); }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  };

  const handleConvert = () => {
    if (!selectedFile) return;
    uploadFile({ file: selectedFile, fileName: selectedFile.name, endpoint: `/convert/${fromFormat.toLowerCase()}-to-${toFormat.toLowerCase()}` });
  };

  const handleCompress = () => {
    if (!selectedFile) return;
    uploadFile({ file: selectedFile, fileName: selectedFile.name, endpoint: '/compress' });
  };

  const handleEnhance = () => {
    if (!selectedFile) return;
    const { sharpen, upscale, denoise } = enhanceOptions;
    uploadFile({ file: selectedFile, fileName: selectedFile.name, endpoint: `/enhance?sharpen=${sharpen}&upscale=${upscale}&denoise=${denoise}` });
  };

  const resetState = () => {
    reset();
    setSelectedFile(null);
  };

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen">
        <div className="flex items-center gap-4 px-5 py-4">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl hover:bg-white/5"><ChevronLeft size={24} color={COLORS.textPrimary} /></button>
          <h1 className="text-xl font-semibold" style={{ fontFamily: "'Outfit', sans-serif" }}>Format Converter</h1>
        </div>

        <div className="px-5 pb-10 max-w-[1000px] mx-auto space-y-5">
          {/* Mode Selector */}
          <GlassCard padding="8px" className="flex animate-fade-in-up">
            {MODES.map((m, i) => (
              <button key={i} onClick={() => { setMode(i); reset(); setSelectedFile(null); }}
                className="flex-1 py-3 rounded-2xl flex flex-col items-center gap-1.5 transition-all"
                style={{
                  backgroundColor: mode === i ? `${COLORS.accentCyan}26` : 'transparent',
                  border: `1px solid ${mode === i ? `${COLORS.accentCyan}66` : 'transparent'}`,
                  color: mode === i ? COLORS.accentCyan : COLORS.textTertiary,
                }}>
                <span className="text-xl">{m.icon}</span>
                <span className="text-xs font-medium">{m.label}</span>
              </button>
            ))}
          </GlassCard>

          {mode === 0 && (
            <>
              {/* Convert Mode */}
              <GlassCard className="animate-fade-in-up">
                <p className="text-[15px] font-semibold mb-4" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Conversion Type</p>
                <div className="flex items-center gap-4">
                  <div className="flex-1">
                    <label className="text-xs mb-1.5 block" style={{ color: COLORS.textTertiary }}>From</label>
                    <select value={fromFormat} onChange={e => { setFromFormat(e.target.value); const opts = CONVERSION_MAP[e.target.value]; if (opts && !opts.includes(toFormat)) setToFormat(opts[0]); reset(); setSelectedFile(null); }}
                      className="w-full py-3 px-3.5 rounded-xl text-sm font-medium appearance-none cursor-pointer"
                      style={{ backgroundColor: COLORS.glassFill, border: `1px solid ${COLORS.glassBorder}`, color: COLORS.textPrimary }}>
                      {ALL_FORMATS.map(f => <option key={f} value={f}>{FORMAT_ICONS[f]} {f}</option>)}
                    </select>
                  </div>
                  <ArrowRight size={24} style={{ color: FORMAT_COLORS[fromFormat] || COLORS.accentCyan, marginTop: 20 }} />
                  <div className="flex-1">
                    <label className="text-xs mb-1.5 block" style={{ color: COLORS.textTertiary }}>To</label>
                    <select value={toFormat} onChange={e => { setToFormat(e.target.value); reset(); setSelectedFile(null); }}
                      className="w-full py-3 px-3.5 rounded-xl text-sm font-medium appearance-none cursor-pointer"
                      style={{ backgroundColor: COLORS.glassFill, border: `1px solid ${COLORS.glassBorder}`, color: COLORS.textPrimary }}>
                      {(CONVERSION_MAP[fromFormat] || []).map(f => <option key={f} value={f}>{f}</option>)}
                    </select>
                  </div>
                </div>
              </GlassCard>

              {/* Upload Zone */}
              <GlassCard padding="0" className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <div onClick={() => fileInputRef.current?.click()} className="p-9 text-center cursor-pointer" style={{ border: `1.5px dashed ${FORMAT_COLORS[fromFormat]}66`, borderRadius: 24 }}>
                  {selectedFile ? <FileText size={40} color={`${FORMAT_COLORS[fromFormat]}B3`} className="mx-auto mb-3" /> :
                    <Upload size={40} color={`${FORMAT_COLORS[fromFormat]}B3`} className="mx-auto mb-3" />}
                  <p className="text-[15px] font-medium" style={{ color: selectedFile ? COLORS.textPrimary : COLORS.textSecondary }}>
                    {selectedFile?.name || `Upload ${fromFormat} file`}
                  </p>
                  <p className="text-xs mt-1" style={{ color: COLORS.textTertiary }}>
                    Supports: {(CONVERSION_MAP[fromFormat] || []).map(f => `.${f.toLowerCase()}`).join(', ')}
                  </p>
                </div>
                <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileSelect} />
              </GlassCard>

              {selectedFile && <AnimatedUploadButton onPressed={handleConvert} label={`Convert ${fromFormat} to ${toFormat}`} isLoading={loading} gradient={[FORMAT_COLORS[fromFormat] || COLORS.accentCyan, COLORS.accentCyan]} />}
            </>
          )}

          {mode === 1 && (
            <>
              <GlassCard borderGradient={[COLORS.accentOrange, COLORS.accentPink]} className="animate-fade-in-up">
                <div className="flex items-center gap-2.5 mb-2">
                  <span className="text-xl">📦</span>
                  <p className="text-[17px] font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>File Compression</p>
                </div>
                <p className="text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>
                  Reduce your file size without format conversion. Upload a PDF or Word document.
                </p>
              </GlassCard>

              <GlassCard padding="0" className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <div onClick={() => fileInputRef.current?.click()} className="p-9 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentOrange}66`, borderRadius: 24 }}>
                  <Upload size={40} color={`${COLORS.accentOrange}B3`} className="mx-auto mb-3" />
                  <p className="text-[15px] font-medium" style={{ color: selectedFile ? COLORS.textPrimary : COLORS.textSecondary }}>
                    {selectedFile?.name || 'Upload file to compress'}
                  </p>
                  <p className="text-xs mt-1" style={{ color: COLORS.textTertiary }}>Supports .pdf, .docx, .doc formats</p>
                </div>
                <input ref={fileInputRef} type="file" accept=".pdf,.docx,.doc" className="hidden" onChange={handleFileSelect} />
              </GlassCard>

              {selectedFile && <AnimatedUploadButton onPressed={handleCompress} label="Compress Now" isLoading={loading} gradient={[COLORS.accentOrange, COLORS.accentPink]} />}
            </>
          )}

          {mode === 2 && (
            <>
              <GlassCard borderGradient={[COLORS.accentViolet, COLORS.accentCyan]} className="animate-fade-in-up">
                <div className="flex items-center gap-2.5 mb-2">
                  <span className="text-xl">✨</span>
                  <p className="text-[17px] font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Pixel Improvement</p>
                </div>
                <p className="text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>
                  Enhance the visual quality of scanned or low-resolution PDFs. Sharpen text, upscale images, and remove noise.
                </p>
              </GlassCard>

              <GlassCard padding="0" className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <div onClick={() => fileInputRef.current?.click()} className="p-9 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentViolet}66`, borderRadius: 24 }}>
                  <Upload size={40} color={`${COLORS.accentViolet}B3`} className="mx-auto mb-3" />
                  <p className="text-[15px] font-medium" style={{ color: selectedFile ? COLORS.textPrimary : COLORS.textSecondary }}>
                    {selectedFile?.name || 'Upload PDF to enhance'}
                  </p>
                </div>
                <input ref={fileInputRef} type="file" accept=".pdf" className="hidden" onChange={handleFileSelect} />
              </GlassCard>

              <GlassCard className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                <p className="text-[15px] font-semibold mb-4" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Enhancement Options</p>
                <div className="flex flex-wrap gap-3">
                  {[
                    { key: 'sharpen', label: 'Sharpen Text', icon: 'T', color: COLORS.accentCyan },
                    { key: 'upscale', label: 'Upscale Images', icon: '⊞', color: COLORS.accentGreen },
                    { key: 'denoise', label: 'Remove Noise', icon: '◎', color: COLORS.accentOrange },
                  ].map(opt => (
                    <button key={opt.key} onClick={() => setEnhanceOptions(prev => ({ ...prev, [opt.key]: !(prev as any)[opt.key] }))}
                      className="flex items-center gap-2 px-4 py-2.5 rounded-xl transition-all"
                      style={{
                        backgroundColor: (enhanceOptions as any)[opt.key] ? `${opt.color}26` : 'transparent',
                        border: `1px solid ${(enhanceOptions as any)[opt.key] ? `${opt.color}80` : COLORS.glassBorder}`,
                        color: (enhanceOptions as any)[opt.key] ? opt.color : COLORS.textTertiary,
                      }}>
                      <span className="text-sm">{(enhanceOptions as any)[opt.key] ? '✓' : opt.icon}</span>
                      <span className="text-[13px] font-medium">{opt.label}</span>
                    </button>
                  ))}
                </div>
              </GlassCard>

              {selectedFile && <AnimatedUploadButton onPressed={handleEnhance} label="Enhance Document" isLoading={loading} gradient={[COLORS.accentViolet, COLORS.accentCyan]} />}
            </>
          )}

          {/* Task Result */}
          <TaskResult task={task} loading={loading} error={error} isPending={isPending} isCompleted={isCompleted} isFailed={isFailed}
            reset={resetState} mode={mode} fromFormat={fromFormat} toFormat={toFormat} selectedFile={selectedFile} />
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </AnimatedGradientBackground>
  );
}

function TaskResult({ task, loading, error, isPending, isCompleted, isFailed, reset, mode, fromFormat, toFormat, selectedFile }: any) {
  if (loading) return <div className="mt-5 flex justify-center"><GlassLoadingSpinner message="Processing..." /></div>;
  if (error) return (
    <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 animate-fade-in-up">
      <div className="flex items-center gap-3"><span style={{ color: COLORS.accentPink }}>⚠</span><p className="text-sm" style={{ color: COLORS.textSecondary }}>{error}</p></div>
    </GlassCard>
  );
  if (isPending) return <div className="mt-5 flex justify-center"><GlassLoadingSpinner message={`Processing... ${Math.round((task?.progress || 0) * 100)}%`} /></div>;
  if (isCompleted) {
    const outputType = mode === 0 ? toFormat : mode === 1 ? 'Compressed' : 'Enhanced';
    const accentColor = mode === 0 ? (COLORS as any)[`accent${toFormat}`] || COLORS.accentCyan : mode === 1 ? COLORS.accentOrange : COLORS.accentViolet;
    return (
      <div className="mt-5 space-y-4 animate-scale-in">
        {task?.aiSummary && (
          <GlassCard borderGradient={[COLORS.accentViolet, COLORS.accentCyan]}>
            <div className="flex items-center gap-2 mb-3">
              <span>✨</span>
              <p className="text-[15px] font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.accentViolet }}>AI Document Summary</p>
            </div>
            <p className="text-[13px] leading-relaxed" style={{ color: COLORS.textSecondary }}>{task.aiSummary}</p>
          </GlassCard>
        )}
        <DocumentPreviewCard fileName={selectedFile?.name || 'output'} outputType={outputType} accentColor={accentColor}
          extraInfo={mode === 0 ? `Converted from ${fromFormat}` : mode === 1 ? 'Compressed' : 'Enhanced'} downloadUrl={task?.downloadUrl} />
        <GlassCard borderGradient={[COLORS.accentGreen, COLORS.accentCyan]} className="text-center">
          <div className="text-4xl mb-2">✓</div>
          <p className="text-base font-semibold mb-4" style={{ color: COLORS.textPrimary }}>
            {mode === 0 ? 'Conversion Complete!' : mode === 1 ? 'Compression Complete!' : 'Enhancement Complete!'}
          </p>
          {task?.downloadUrl && <a href={task.downloadUrl} target="_blank" rel="noreferrer"
            className="block w-full py-3 rounded-[20px] font-semibold" style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgDark }}>Download</a>}
          <button onClick={reset} className="mt-3 text-sm font-medium" style={{ color: COLORS.accentCyan }}>
            {mode === 0 ? 'Convert Another' : mode === 1 ? 'Compress Another' : 'Enhance Another'}
          </button>
        </GlassCard>
      </div>
    );
  }
  if (isFailed) return (
    <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 text-center animate-fade-in-up">
      <p className="text-4xl mb-3">⚠</p>
      <p className="text-sm mb-3" style={{ color: COLORS.textSecondary }}>{task?.errorMessage || 'Operation failed'}</p>
      <button onClick={reset} className="text-sm font-medium" style={{ color: COLORS.accentCyan }}>Try Again</button>
    </GlassCard>
  );
  return null;
}
