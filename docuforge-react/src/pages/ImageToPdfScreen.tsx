import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedGradientBackground } from '../components/common/AnimatedGradientBackground';
import { GlassCard } from '../components/common/GlassCard';
import { GlassLoadingSpinner } from '../components/common/GlassLoadingSpinner';
import { AnimatedUploadButton } from '../components/common/AnimatedUploadButton';
import { useTaskManager } from '../hooks/useTaskManager';
import { useToast } from '../hooks/useToast';
import { ToastContainer } from '../components/common/Toast';
import { COLORS } from '../constants';
import { ChevronLeft, Plus, X, Image, FileText, Upload } from 'lucide-react';

const PLACEHOLDER_COLORS = [COLORS.accentCyan, COLORS.accentViolet, COLORS.accentPink, COLORS.accentGreen, COLORS.accentOrange, COLORS.accentBlue];

export function ImageToPdfScreen() {
  const navigate = useNavigate();
  const { task, loading, error, reset, uploadFile, uploadFiles, isPending, isCompleted, isFailed } = useTaskManager();
  const { toasts, removeToast } = useToast();
  const imageInputRef = useRef<HTMLInputElement>(null);
  const pdfInputRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState(0);
  const [images, setImages] = useState<File[]>([]);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [ocrEnabled, setOcrEnabled] = useState(false);

  useEffect(() => { reset(); }, []);

  const addImages = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) setImages(prev => [...prev, ...Array.from(files)]);
  };

  const selectPdf = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setPdfFile(file);
  };

  const uploadImageToPdf = () => {
    if (images.length === 0) return;
    uploadFiles({ files: images, endpoint: `/api/v1/convert/image-to-pdf${ocrEnabled ? '?ocr=true' : ''}` });
  };

  const uploadPdfToImage = () => {
    if (!pdfFile) return;
    uploadFile({ file: pdfFile, fileName: pdfFile.name, endpoint: '/api/v1/convert/pdf-to-image' });
  };

  const resetState = () => {
    reset();
    setImages([]);
    setPdfFile(null);
  };

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen">
        <div className="flex items-center gap-4 px-5 py-4">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl hover:bg-white/5"><ChevronLeft size={24} color={COLORS.textPrimary} /></button>
          <h1 className="text-xl font-semibold" style={{ fontFamily: "'Outfit', sans-serif" }}>Image & PDF Converter</h1>
        </div>

        <div className="px-5 pb-10 max-w-[1000px] mx-auto space-y-5">
          {/* Mode Selector */}
          <GlassCard padding="8px" className="flex animate-fade-in-up">
            {[{ icon: '🖼', label: 'Image to PDF' }, { icon: '📄', label: 'PDF to Image' }].map((m, i) => (
              <button key={i} onClick={() => { setMode(i); reset(); }}
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

          {mode === 0 ? (
            <>
              {/* OCR Toggle */}
              <GlassCard className="animate-fade-in-up">
                <div className="flex items-center gap-3.5">
                  <div className="w-[42px] h-[42px] rounded-xl flex items-center justify-center"
                    style={{ background: `linear-gradient(135deg, ${COLORS.accentBlue}33, ${COLORS.accentViolet}1A)` }}>
                    <span style={{ color: COLORS.accentBlue, fontSize: 18 }}>T</span>
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold" style={{ color: COLORS.textPrimary }}>AI Text Recognition (OCR)</p>
                    <p className="text-xs mt-0.5" style={{ color: COLORS.textTertiary }}>Make text selectable in the final PDF</p>
                  </div>
                  <button onClick={() => setOcrEnabled(!ocrEnabled)}
                    className="w-12 h-7 rounded-full transition-all relative"
                    style={{ backgroundColor: ocrEnabled ? `${COLORS.accentCyan}4D` : COLORS.glassFill }}>
                    <div className="w-5 h-5 rounded-full absolute top-1 transition-all"
                      style={{ backgroundColor: ocrEnabled ? COLORS.accentCyan : COLORS.textTertiary, left: ocrEnabled ? 26 : 4 }} />
                  </button>
                </div>
              </GlassCard>

              {/* Gallery Header */}
              <div className="flex items-center justify-between">
                <p className="text-lg font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Gallery</p>
                <button onClick={() => imageInputRef.current?.click()}
                  className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl"
                  style={{ backgroundColor: `${COLORS.accentBlue}1F`, border: `1px solid ${COLORS.accentBlue}4D`, color: COLORS.accentBlue }}>
                  <Plus size={18} /> <span className="text-[13px] font-semibold">Add Photo</span>
                </button>
              </div>
              <input ref={imageInputRef} type="file" accept="image/*" multiple className="hidden" onChange={addImages} />

              {/* Image Grid */}
              {images.length === 0 ? (
                <GlassCard className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                  <div className="py-10 text-center">
                    <p className="text-[40px] mb-3 opacity-50">🖼</p>
                    <p className="text-[14px]" style={{ color: COLORS.textTertiary }}>No images added yet</p>
                    <p className="text-xs mt-1" style={{ color: COLORS.textTertiary }}>Tap "Add Photo" to get started</p>
                  </div>
                </GlassCard>
              ) : (
                <div className="grid grid-cols-3 gap-3">
                  {images.map((img, i) => {
                    const color = PLACEHOLDER_COLORS[i % PLACEHOLDER_COLORS.length];
                    return (
                      <GlassCard key={i} padding="4px" borderRadius="16px" className="animate-fade-in-up relative overflow-hidden"
                        style={{ animationDelay: `${i * 100}ms` }}>
                        <div className="h-24 rounded-xl flex items-center justify-center"
                          style={{ background: `linear-gradient(135deg, ${color}4D, ${color}1A)` }}>
                          <Image size={28} color={`${color}80`} />
                        </div>
                        <button onClick={() => setImages(prev => prev.filter((_, j) => j !== i))}
                          className="absolute top-1 right-1 w-[22px] h-[22px] rounded-full flex items-center justify-center bg-black/40">
                          <X size={14} color="white" />
                        </button>
                        <p className="px-1.5 py-1.5 text-[9px] font-medium truncate" style={{ color: COLORS.textPrimary }}>{img.name}</p>
                      </GlassCard>
                    );
                  })}
                </div>
              )}

              {images.length > 0 && (
                <div className="flex justify-center">
                  <AnimatedUploadButton onPressed={uploadImageToPdf} label={`Convert ${images.length} Images to PDF`} isLoading={loading} gradient={[COLORS.accentBlue, COLORS.accentViolet]} />
                </div>
              )}
            </>
          ) : (
            <>
              {/* PDF to Image Mode */}
              <GlassCard padding="0" className="animate-fade-in-up" style={{ animationDelay: '100ms' }}>
                <div onClick={() => pdfInputRef.current?.click()} className="p-9 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentOrange}66`, borderRadius: 24 }}>
                  {pdfFile ? <FileText size={40} color={`${COLORS.accentOrange}B3`} className="mx-auto mb-3" /> :
                    <Upload size={40} color={`${COLORS.accentOrange}B3`} className="mx-auto mb-3" />}
                  <p className="text-[15px] font-medium" style={{ color: pdfFile ? COLORS.textPrimary : COLORS.textSecondary }}>
                    {pdfFile?.name || 'Upload PDF file'}
                  </p>
                  <p className="text-xs mt-1" style={{ color: COLORS.textTertiary }}>Supports .pdf format</p>
                </div>
                <input ref={pdfInputRef} type="file" accept=".pdf" className="hidden" onChange={selectPdf} />
              </GlassCard>

              {pdfFile && (
                <GlassCard className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                  <div className="flex items-center gap-3">
                    <span>✨</span>
                    <div>
                      <p className="text-sm font-medium" style={{ color: COLORS.textPrimary }}>{pdfFile.name}</p>
                      <p className="text-xs mt-0.5" style={{ color: COLORS.textTertiary }}>Best format will be auto-detected</p>
                    </div>
                  </div>
                </GlassCard>
              )}

              {pdfFile && (
                <div className="flex justify-center">
                  <AnimatedUploadButton onPressed={uploadPdfToImage} label="Convert to Images" isLoading={loading} gradient={[COLORS.accentOrange, COLORS.accentPink]} />
                </div>
              )}
            </>
          )}

          {/* Task Result */}
          {loading && <div className="mt-5 flex justify-center"><GlassLoadingSpinner message="Converting..." /></div>}
          {error && (
            <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 animate-fade-in-up">
              <div className="flex items-center gap-3"><span style={{ color: COLORS.accentPink }}>⚠</span><p className="text-sm" style={{ color: COLORS.textSecondary }}>{error}</p></div>
            </GlassCard>
          )}
          {isPending && <div className="mt-5 flex justify-center"><GlassLoadingSpinner message={`Processing... ${Math.round((task?.progress || 0) * 100)}%`} /></div>}
          {isCompleted && (
            <div className="mt-5 space-y-4 animate-scale-in">
              <GlassCard borderGradient={[COLORS.accentGreen, COLORS.accentBlue]} className="text-center">
                <div className="text-4xl mb-2">✓</div>
                <p className="text-base font-semibold mb-4" style={{ color: COLORS.textPrimary }}>
                  {mode === 0 ? (ocrEnabled ? 'PDF Created with OCR!' : 'PDF Created!') : 'Images Created!'}
                </p>
                {task?.downloadUrl && <a href={task.downloadUrl} target="_blank" rel="noreferrer"
                  className="block w-full py-3 rounded-[20px] font-semibold" style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgDark }}>
                  {mode === 0 ? 'Download PDF' : 'Download ZIP'}
                </a>}
                <button onClick={resetState} className="mt-3 text-sm font-medium" style={{ color: COLORS.accentCyan }}>Convert More</button>
              </GlassCard>
            </div>
          )}
          {isFailed && (
            <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 text-center animate-fade-in-up">
              <p className="text-4xl mb-3">⚠</p>
              <p className="text-sm mb-3" style={{ color: COLORS.textSecondary }}>{task?.errorMessage || 'Conversion failed'}</p>
              <button onClick={reset} className="text-sm font-medium" style={{ color: COLORS.accentCyan }}>Try Again</button>
            </GlassCard>
          )}
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </AnimatedGradientBackground>
  );
}
