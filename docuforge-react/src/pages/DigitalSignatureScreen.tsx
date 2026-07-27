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
import { ChevronLeft, Undo, Trash2, Download, Upload, FileText, Eye, CheckCircle } from 'lucide-react';
import SignaturePad from 'signature_pad';

const PEN_COLORS = ['#000000', '#1565C0', '#C62828', '#2E7D32', '#6A1B9A'];

export function DigitalSignatureScreen() {
  const navigate = useNavigate();
  const { task, loading, error, reset, uploadFileWithSignature, isPending, isCompleted, isFailed } = useTaskManager();
  const { toasts, addToast, removeToast } = useToast();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const signaturePadRef = useRef<SignaturePad | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedPdf, setSelectedPdf] = useState<File | null>(null);
  const [penColor, setPenColor] = useState('#000000');
  const [penWidth, setPenWidth] = useState(3);

  useEffect(() => {
    reset();
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
      const ctx = canvas.getContext('2d');
      if (ctx) { ctx.scale(2, 2); }
      signaturePadRef.current = new SignaturePad(canvas, {
        backgroundColor: 'rgb(255, 255, 255)',
        penColor: penColor,
      });
    }
  }, []);

  useEffect(() => {
    if (signaturePadRef.current) {
      signaturePadRef.current.penColor = penColor;
      (signaturePadRef.current as any).strokeWidth = penWidth;
    }
  }, [penColor, penWidth]);

  const selectPdf = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedPdf(file);
  };

  const canSign = () => {
    return signaturePadRef.current && !signaturePadRef.current.isEmpty() && selectedPdf;
  };

  const upload = async () => {
    if (!canSign() || !selectedPdf) {
      addToast('Please draw your signature and select a PDF', 'error');
      return;
    }
    const dataUrl = signaturePadRef.current!.toDataURL('image/png');
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    uploadFileWithSignature({ pdfFile: selectedPdf, signatureBlob: blob });
  };

  const undo = () => (signaturePadRef.current as any)?.undo();
  const clear = () => signaturePadRef.current?.clear();

  const downloadSignature = async () => {
    if (!signaturePadRef.current) return;
    const dataUrl = signaturePadRef.current.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataUrl; a.download = 'signature.png'; a.click();
  };

  const showPreview = () => {
    if (!signaturePadRef.current || signaturePadRef.current.isEmpty()) return;
    const dataUrl = signaturePadRef.current.toDataURL('image/png');
    const win = window.open('');
    if (win) {
      win.document.write(`<html><head><title>Signature Preview</title><style>body{display:flex;justify-content:center;align-items:center;min-height:100vh;background:#f0f0f5;margin:0;font-family:sans-serif;}img{max-width:90%;box-shadow:0 4px 20px rgba(0,0,0,0.15);border-radius:12px;}</style></head><body><img src="${dataUrl}" /></body></html>`);
    }
  };

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen">
        <div className="flex items-center gap-4 px-5 py-4">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl hover:bg-white/5"><ChevronLeft size={24} color={COLORS.textPrimary} /></button>
          <h1 className="text-xl font-semibold" style={{ fontFamily: "'Outfit', sans-serif" }}>Digital Signature</h1>
        </div>

        <div className="px-5 pb-10 max-w-[1000px] mx-auto space-y-5">
          {/* Draw Signature */}
          <GlassCard borderGradient={[COLORS.accentCyan, COLORS.accentViolet]} className="animate-fade-in-up">
            <div className="flex items-center gap-2.5 mb-4">
              <span style={{ color: `${COLORS.accentCyan}CC` }}>✏️</span>
              <p className="text-[17px] font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Draw Your Signature</p>
            </div>

            <div className="rounded-2xl overflow-hidden mb-4" style={{ border: `1px solid ${COLORS.glassBorder}` }}>
              <canvas ref={canvasRef} className="w-full" style={{ height: 200, touchAction: 'none' }} />
            </div>

            {/* Color Picker */}
            <div className="flex items-center gap-3 mb-3">
              <span className="text-[13px]" style={{ color: COLORS.textSecondary }}>Color</span>
              {PEN_COLORS.map(c => (
                <button key={c} onClick={() => setPenColor(c)}
                  className="w-7 h-7 rounded-full transition-all"
                  style={{
                    backgroundColor: c,
                    border: `${penColor === c ? 2.5 : 1}px solid ${penColor === c ? '#333' : '#ccc'}`,
                  }} />
              ))}
            </div>

            {/* Width Slider */}
            <div className="flex items-center gap-3 mb-4">
              <span className="text-[13px]" style={{ color: COLORS.textSecondary }}>Width</span>
              <input type="range" min="1" max="8" step="0.5" value={penWidth} onChange={e => setPenWidth(Number(e.target.value))}
                className="flex-1 accent-[#00D4FF]" />
              <span className="text-xs font-semibold" style={{ color: COLORS.textSecondary }}>{Math.round(penWidth)}px</span>
            </div>

            {/* Undo / Clear */}
            <div className="flex gap-3 mb-4">
              <button onClick={undo} className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium transition-colors hover:bg-white/5"
                style={{ border: '1px solid #ccc', color: 'rgba(0,0,0,0.54)' }}>
                <Undo size={18} /> Undo
              </button>
              <button onClick={clear} className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium transition-colors hover:bg-white/5"
                style={{ border: '1px solid #ffcdd2', color: '#ef5350' }}>
                <Trash2 size={18} /> Clear
              </button>
            </div>

            {/* Download eSignature */}
            <button onClick={downloadSignature}
              className="w-full flex items-center justify-center gap-2 py-3.5 rounded-2xl font-semibold text-[15px]"
              style={{ backgroundColor: '#1976D2', color: 'white' }}>
              <Download size={18} /> Download eSignature
            </button>
          </GlassCard>

          {/* PDF Upload */}
          <GlassCard padding="0" className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
            <div onClick={() => fileInputRef.current?.click()} className="p-7 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentOrange}66`, borderRadius: 24 }}>
              {selectedPdf ? <FileText size={32} color={`${COLORS.accentOrange}B3`} className="mx-auto mb-2.5" /> :
                <Upload size={32} color={`${COLORS.accentOrange}B3`} className="mx-auto mb-2.5" />}
              <p className="text-[14px] font-medium" style={{ color: selectedPdf ? COLORS.textPrimary : COLORS.textSecondary }}>
                {selectedPdf?.name || 'Select PDF to sign'}
              </p>
              {!selectedPdf && <p className="text-[11px] mt-1" style={{ color: COLORS.textTertiary }}>Signature will be placed at "Sign here" markers</p>}
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf" className="hidden" onChange={selectPdf} />
          </GlassCard>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button onClick={showPreview} disabled={!signaturePadRef.current || signaturePadRef.current?.isEmpty()}
              className="flex-1 flex items-center justify-center gap-2 py-4 rounded-2xl text-sm font-medium transition-all disabled:opacity-50"
              style={{ border: `1px solid ${COLORS.accentCyan}66`, color: COLORS.accentCyan, background: 'transparent' }}>
              <Eye size={18} /> Preview
            </button>
            <div className="flex-[2]">
              <AnimatedUploadButton onPressed={upload} label="Seal & Sign" isLoading={loading} gradient={[COLORS.accentOrange, COLORS.accentPink]} />
            </div>
          </div>

          {/* Task Result */}
          {loading && <div className="mt-5 flex justify-center"><GlassLoadingSpinner message="Scanning for signature locations..." /></div>}
          {error && (
            <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 animate-fade-in-up">
              <div className="flex items-center gap-3"><span style={{ color: COLORS.accentPink }}>⚠</span><p className="text-sm" style={{ color: COLORS.textSecondary }}>{error}</p></div>
            </GlassCard>
          )}
          {isPending && <div className="mt-5 flex justify-center"><GlassLoadingSpinner message={`Placing signature... ${Math.round((task?.progress || 0) * 100)}%`} /></div>}
          {isCompleted && (
            <GlassCard borderGradient={[COLORS.accentGreen, COLORS.accentOrange]} className="text-center animate-scale-in">
              <div className="w-14 h-14 rounded-full mx-auto flex items-center justify-center mb-3"
                style={{ background: `linear-gradient(135deg, ${COLORS.accentGreen}33, ${COLORS.accentOrange}1A)` }}>
                <CheckCircle size={32} color={COLORS.accentGreen} />
              </div>
              <p className="text-base font-semibold mb-1" style={{ color: COLORS.textPrimary }}>Document Signed & Sealed</p>
              <p className="text-xs mb-4" style={{ color: COLORS.textTertiary }}>{task?.aiSummary || 'Signature placed at detected locations'}</p>
              {task?.downloadUrl && <a href={task.downloadUrl} target="_blank" rel="noreferrer"
                className="block w-full py-3 rounded-[20px] font-semibold" style={{ backgroundColor: COLORS.accentGreen, color: 'white' }}>Download Signed PDF</a>}
              <button onClick={() => { reset(); signaturePadRef.current?.clear(); setSelectedPdf(null); }}
                className="mt-3 text-sm font-medium" style={{ color: COLORS.accentCyan }}>Sign Another</button>
            </GlassCard>
          )}
          {isFailed && (
            <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 text-center animate-fade-in-up">
              <p className="text-4xl mb-3">⚠</p>
              <p className="text-sm mb-3" style={{ color: COLORS.textSecondary }}>{task?.errorMessage || 'Signing failed'}</p>
              <button onClick={reset} className="text-sm font-medium" style={{ color: COLORS.accentCyan }}>Try Again</button>
            </GlassCard>
          )}
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </AnimatedGradientBackground>
  );
}
