import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedGradientBackground } from '../components/common/AnimatedGradientBackground';
import { GlassCard } from '../components/common/GlassCard';
import { GlassTextField } from '../components/common/GlassTextField';
import { GlassLoadingSpinner } from '../components/common/GlassLoadingSpinner';
import { AnimatedUploadButton } from '../components/common/AnimatedUploadButton';
import { DocumentPreviewCard } from '../components/common/DocumentPreviewCard';
import { useTaskManager } from '../hooks/useTaskManager';
import { useToast } from '../hooks/useToast';
import { ToastContainer } from '../components/common/Toast';
import { COLORS, UPLOAD_ENDPOINT } from '../constants';
import { ChevronLeft, Plus, X, FileText } from 'lucide-react';

export function MergeSplitScreen() {
  const navigate = useNavigate();
  const { task, loading, error, reset, uploadFile, uploadFiles, isPending, isCompleted, isFailed } = useTaskManager();
  const { toasts, removeToast } = useToast();

  const [activeTab, setActiveTab] = useState(0);
  const [mergeFiles, setMergeFiles] = useState<File[]>([]);
  const [splitFile, setSplitFile] = useState<File | null>(null);
  const [pageRange, setPageRange] = useState('');
  const [lastUploadedTab, setLastUploadedTab] = useState(0);
  const mergeInputRef = useRef<HTMLInputElement>(null);
  const splitInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { reset(); }, []);

  const addMergeFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) setMergeFiles(prev => [...prev, ...Array.from(files)]);
  };

  const selectSplitFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSplitFile(file);
  };

  const uploadMerge = () => {
    setLastUploadedTab(0);
    uploadFiles({ files: mergeFiles, endpoint: UPLOAD_ENDPOINT });
  };

  const uploadSplit = () => {
    if (!splitFile) return;
    setLastUploadedTab(1);
    uploadFile({ file: splitFile, fileName: splitFile.name, endpoint: UPLOAD_ENDPOINT });
  };

  const resetState = () => {
    reset();
    setMergeFiles([]);
    setSplitFile(null);
    setPageRange('');
  };

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen">
        {/* AppBar */}
        <div className="flex items-center gap-4 px-5 py-4">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl hover:bg-white/5"><ChevronLeft size={24} color={COLORS.textPrimary} /></button>
          <h1 className="text-xl font-semibold" style={{ fontFamily: "'Outfit', sans-serif" }}>Merge & Split PDF</h1>
        </div>

        {/* Tabs */}
        <div className="px-5 mb-4">
          <GlassCard padding="4px" borderRadius="14px" className="inline-flex">
            {['Merge', 'Split'].map((tab, i) => (
              <button key={i} onClick={() => setActiveTab(i)}
                className="px-6 py-2 rounded-[10px] text-sm font-medium transition-all"
                style={{
                  backgroundColor: activeTab === i ? `${COLORS.accentCyan}26` : 'transparent',
                  color: activeTab === i ? COLORS.accentCyan : COLORS.textTertiary,
                }}>
                {tab}
              </button>
            ))}
          </GlassCard>
        </div>

        <div className="px-5 pb-10 max-w-[1000px] mx-auto">
          {activeTab === 0 ? (
            /* Merge Tab */
            <div className="space-y-5">
              <GlassCard padding="0" className="animate-fade-in-up">
                <div onClick={() => mergeInputRef.current?.click()}
                  className="p-8 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentPink}66`, borderRadius: 24 }}>
                  <Plus size={36} color={`${COLORS.accentPink}B3`} className="mx-auto mb-3" />
                  <p className="text-[14px] font-medium" style={{ color: COLORS.textSecondary }}>Tap to add PDF files</p>
                  <p className="text-xs mt-1" style={{ color: COLORS.textTertiary }}>Files will be merged in the order listed</p>
                </div>
                <input ref={mergeInputRef} type="file" accept=".pdf" multiple className="hidden" onChange={addMergeFiles} />
              </GlassCard>

              {mergeFiles.length > 0 && (
                <GlassCard className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                  <div className="space-y-2">
                    {mergeFiles.map((f, i) => (
                      <div key={i} className="flex items-center gap-3 py-2 px-3 rounded-xl hover:bg-white/5 transition-colors">
                        <div className="w-9 h-9 rounded-[10px] flex items-center justify-center" style={{ backgroundColor: `${COLORS.accentPink}1A` }}>
                          <span className="text-sm font-semibold" style={{ color: COLORS.accentPink }}>{i + 1}</span>
                        </div>
                        <span className="flex-1 text-sm" style={{ color: COLORS.textPrimary }}>{f.name}</span>
                        <button onClick={() => setMergeFiles(prev => prev.filter((_, j) => j !== i))} className="p-1 hover:bg-white/10 rounded-lg">
                          <X size={16} style={{ color: COLORS.textTertiary }} />
                        </button>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              )}

              {mergeFiles.length > 0 && (
                <AnimatedUploadButton onPressed={uploadMerge} label={`Merge ${mergeFiles.length} Files`} isLoading={loading} gradient={[COLORS.accentPink, COLORS.accentOrange]} />
              )}
            </div>
          ) : (
            /* Split Tab */
            <div className="space-y-5">
              <GlassCard padding="0" className="animate-fade-in-up">
                <div onClick={() => splitInputRef.current?.click()}
                  className="p-8 text-center cursor-pointer" style={{ border: `1.5px dashed ${COLORS.accentCyan}66`, borderRadius: 24 }}>
                  {splitFile ? <FileText size={36} color={`${COLORS.accentCyan}B3`} className="mx-auto mb-3" /> :
                    <div className="w-9 h-9 rounded-lg mx-auto mb-3 flex items-center justify-center" style={{ backgroundColor: `${COLORS.accentCyan}1A` }}>📄</div>}
                  <p className="text-[14px] font-medium" style={{ color: splitFile ? COLORS.textPrimary : COLORS.textSecondary }}>
                    {splitFile?.name || 'Select a PDF to split'}
                  </p>
                </div>
                <input ref={splitInputRef} type="file" accept=".pdf" className="hidden" onChange={selectSplitFile} />
              </GlassCard>

              {splitFile && (
                <GlassCard className="animate-fade-in-up" style={{ animationDelay: '150ms' }}>
                  <p className="text-base font-semibold mb-1" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>Page Extraction</p>
                  <p className="text-xs mb-4" style={{ color: COLORS.textTertiary }}>Enter page numbers to extract (e.g. 1, 3, 5-8)</p>
                  <GlassTextField label="Page Numbers" hint="1, 3, 5-8" value={pageRange} onChange={e => setPageRange(e.target.value)} />
                  <div className="mt-5">
                    <AnimatedUploadButton onPressed={uploadSplit} label="Split PDF" isLoading={loading} />
                  </div>
                </GlassCard>
              )}
            </div>
          )}

          {/* Task Result */}
          <TaskResult task={task} loading={loading} error={error} isPending={isPending} isCompleted={isCompleted} isFailed={isFailed}
            reset={resetState} lastUploadedTab={lastUploadedTab} mergeFileCount={mergeFiles.length} splitFileName={splitFile?.name} />
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </AnimatedGradientBackground>
  );
}

function TaskResult({ task, loading, error, isPending, isCompleted, isFailed, reset, lastUploadedTab, mergeFileCount, splitFileName }: any) {
  if (loading) return <div className="mt-10 flex justify-center"><GlassLoadingSpinner message="Processing your PDF..." /></div>;
  if (error) return (
    <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 animate-fade-in-up">
      <div className="flex items-center gap-3"><span style={{ color: COLORS.accentPink }}>⚠</span><p className="text-sm" style={{ color: COLORS.textSecondary }}>{error}</p></div>
    </GlassCard>
  );
  if (isPending) return <div className="mt-10 flex justify-center"><GlassLoadingSpinner message={`Processing... ${Math.round((task?.progress || 0) * 100)}%`} /></div>;
  if (isCompleted) return (
    <div className="mt-6 space-y-4 animate-scale-in">
      <DocumentPreviewCard fileName={lastUploadedTab === 0 ? `${mergeFileCount} files merged` : (splitFileName || 'output.pdf')}
        outputType="PDF" pageCount={lastUploadedTab === 0 ? Math.min(mergeFileCount, 99) : 1}
        accentColor={lastUploadedTab === 0 ? COLORS.accentPink : COLORS.accentCyan}
        extraInfo={lastUploadedTab === 0 ? `Merged ${mergeFileCount} PDF files` : 'Split PDF ready'}
        downloadUrl={task?.downloadUrl} />
      <GlassCard borderGradient={[COLORS.accentGreen, COLORS.accentCyan]} className="text-center">
        <div className="text-4xl mb-3">✓</div>
        <p className="text-base font-semibold mb-4" style={{ color: COLORS.textPrimary }}>Processing Complete!</p>
        {task?.downloadUrl && <a href={task.downloadUrl} target="_blank" rel="noreferrer"
          className="block w-full py-3 rounded-[20px] font-semibold text-center" style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgDark }}>Download</a>}
        <button onClick={reset} className="mt-3 text-sm font-medium" style={{ color: COLORS.accentCyan }}>Process Another</button>
      </GlassCard>
    </div>
  );
  if (isFailed) return (
    <GlassCard borderGradient={[COLORS.accentPink, COLORS.accentOrange]} className="mt-5 text-center animate-fade-in-up">
      <p className="text-4xl mb-3">⚠</p>
      <p className="text-sm mb-3" style={{ color: COLORS.textSecondary }}>{task?.errorMessage || 'Processing failed'}</p>
      <button onClick={reset} className="text-sm font-medium" style={{ color: COLORS.accentCyan }}>Try Again</button>
    </GlassCard>
  );
  return null;
}
