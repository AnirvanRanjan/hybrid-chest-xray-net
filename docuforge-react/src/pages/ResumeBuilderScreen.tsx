import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AnimatedGradientBackground } from '../components/common/AnimatedGradientBackground';
import { GlassCard } from '../components/common/GlassCard';
import { GlassTextField } from '../components/common/GlassTextField';
import { SectionHeader } from '../components/common/SectionHeader';
import { useTaskManager } from '../hooks/useTaskManager';
import { useToast } from '../hooks/useToast';
import { ToastContainer } from '../components/common/Toast';
import { COLORS } from '../constants';
import { extractPdfText, parseResume, improveField, checkAtsScore } from '../services/aiService';
import {
  Upload, Sparkles, User, Mail, Phone, Briefcase,
  FileText, Code, Award, Download, FileDown, CheckCircle,
  AlertCircle, TrendingUp, Zap, ChevronLeft
} from 'lucide-react';

const TEMPLATES = [
  { name: 'Modern', colors: [COLORS.accentCyan, COLORS.accentViolet] },
  { name: 'Classic', colors: [COLORS.accentBlue, COLORS.accentCyan] },
  { name: 'Minimal', colors: [COLORS.textSecondary, COLORS.textTertiary] },
  { name: 'Creative', colors: [COLORS.accentPink, COLORS.accentOrange] },
  { name: 'Elegant', colors: ['#D4AF37', '#1E1E2C'] },
];

export function ResumeBuilderScreen() {
  const navigate = useNavigate();
  const { reset } = useTaskManager();
  const { toasts, addToast, removeToast } = useToast();

  const [selectedTemplate, setSelectedTemplate] = useState(0);
  const [isAutoFilling, setIsAutoFilling] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isCheckingAts, setIsCheckingAts] = useState(false);
  const [atsResult, setAtsResult] = useState<any>(null);
  const [windowWidth, setWindowWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1200);

  const [formData, setFormData] = useState({
    name: '', email: '', phone: '', title: '', summary: '',
    experience: '', education: '', projectName: '', projectDesc: '',
    techSkills: '', softSkills: '',
  });

  useEffect(() => {
    reset();
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const updateField = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const pickAndParseResume = async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.txt,.pdf';
    input.onchange = async (e: any) => {
      const file = e.target.files?.[0];
      if (!file) return;
      setIsAutoFilling(true);
      try {
        let resumeText = '';
        if (file.name.endsWith('.pdf')) {
          resumeText = await extractPdfText(file, file.name);
        } else {
          resumeText = await file.text();
        }
        if (!resumeText.trim() || resumeText.trim().length < 10) {
          addToast('File appears empty or too short.', 'error');
          return;
        }
        const parsed = await parseResume(resumeText);
        setFormData({
          name: parsed.name || '',
          email: parsed.email || '',
          phone: parsed.phone || '',
          title: parsed.jobTitle || '',
          summary: parsed.summary || '',
          experience: parsed.experience || '',
          education: parsed.education || '',
          projectName: parsed.projectName || '',
          projectDesc: parsed.projectDesc || '',
          techSkills: parsed.techSkills || '',
          softSkills: parsed.softSkills || '',
        });
        const hasData = parsed.name || parsed.email || parsed.jobTitle;
        addToast(hasData ? 'Resume parsed and fields filled!' : 'AI could not extract data.', hasData ? 'success' : 'info');
      } catch (e: any) {
        addToast(`AI parsing failed: ${e.message}`, 'error');
      } finally {
        setIsAutoFilling(false);
      }
    };
    input.click();
  };

  const improve = async (fieldType: string, currentValue: string) => {
    if (!currentValue.trim()) return;
    addToast(`AI improving ${fieldType}...`, 'loading', 10000);
    try {
      const improved = await improveField({ fieldType, content: currentValue, jobTitle: formData.title });
      updateField(fieldType, improved);
      addToast(`${fieldType} improved by AI!`, 'success');
    } catch (e: any) {
      addToast(`AI improvement failed: ${e.message}`, 'error');
    }
  };

  const downloadPdf = async () => {
    if (!formData.name.trim()) { addToast('Please enter your name first.', 'error'); return; }
    setIsGenerating(true);
    try {
      const { jsPDF } = await import('jspdf');
      const doc = new jsPDF();
      const colors = TEMPLATES[selectedTemplate].colors;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(26);
      const [r1, g1, b1] = hexToRgb(colors[0]);
      doc.setTextColor(r1, g1, b1);
      doc.text(formData.name || 'Your Name', 40, 40);
      if (formData.title) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(13);
        const [r2, g2, b2] = hexToRgb(colors[1]);
        doc.setTextColor(r2, g2, b2);
        doc.text(formData.title, 40, 48);
      }
      doc.setFontSize(10);
      doc.setTextColor(120);
      const contact = [formData.email, formData.phone].filter(Boolean).join(' | ');
      if (contact) doc.text(contact, 40, 56);
      const [r3, g3, b3] = hexToRgb(colors[0]);
      doc.setDrawColor(r3, g3, b3);
      doc.setLineWidth(0.5);
      doc.line(40, 60, 170, 60);
      let y = 70;
      const addSection = (title: string, content: string) => {
        if (!content.trim()) return;
        if (y > 260) { doc.addPage(); y = 40; }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(12);
        const [r4, g4, b4] = hexToRgb(colors[0]);
        doc.setTextColor(r4, g4, b4);
        doc.text(title.toUpperCase(), 40, y);
        doc.setDrawColor(r4, g4, b4);
        doc.line(40, y + 2, 80, y + 2);
        y += 8;
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(10);
        doc.setTextColor(50);
        const lines = doc.splitTextToSize(content, 150);
        lines.forEach((line: string) => {
          if (y > 275) { doc.addPage(); y = 40; }
          doc.text(line, 40, y);
          y += 5;
        });
        y += 4;
      };
      addSection('Professional Summary', formData.summary);
      addSection('Work Experience', formData.experience);
      if (formData.projectName || formData.projectDesc) addSection('Projects', `${formData.projectName}\n${formData.projectDesc}`);
      addSection('Education', formData.education);
      if (formData.techSkills || formData.softSkills) {
        addSection('Skills', `Technical: ${formData.techSkills}\nSoft Skills: ${formData.softSkills}`);
      }
      doc.save(`${(formData.name || 'resume').replace(/\s+/g, '_')}_resume.pdf`);
      addToast('PDF downloaded successfully!', 'success');
    } catch (e: any) {
      addToast(`PDF generation failed: ${e.message}`, 'error');
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadDocx = () => {
    if (!formData.name.trim()) { addToast('Please enter your name first.', 'error'); return; }
    const html = `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><style>body{font-family:Calibri,sans-serif;font-size:10pt;color:#333;}h1{font-size:22pt;color:#0078D4;margin:0 0 4pt 0;}h2{font-size:11pt;color:#555;margin:0 0 8pt 0;font-weight:normal;}</style></head><body><h1>${formData.name || 'Your Name'}</h1>${formData.title ? `<h2>${formData.title}</h2>` : ''}<p>${formData.email} ${formData.phone ? '| ' + formData.phone : ''}</p><hr><p style="margin-top:16pt"><b style="color:#0078D4">PROFESSIONAL SUMMARY</b></p><p>${formData.summary}</p><p style="margin-top:16pt"><b style="color:#0078D4">EXPERIENCE</b></p><p>${formData.experience}</p><p style="margin-top:16pt"><b style="color:#0078D4">EDUCATION</b></p><p>${formData.education}</p><p style="margin-top:16pt"><b style="color:#0078D4">SKILLS</b></p><p>Technical: ${formData.techSkills}</p><p>Soft Skills: ${formData.softSkills}</p></body></html>`;
    const blob = new Blob([html], { type: 'application/msword' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `${(formData.name || 'resume').replace(/\s+/g, '_')}_resume.doc`;
    a.click(); URL.revokeObjectURL(url);
    addToast('DOC downloaded!', 'success');
  };

  const checkAts = async () => {
    if (!formData.name && !formData.summary) { addToast('Please fill in some fields first.', 'error'); return; }
    setIsCheckingAts(true);
    try {
      const result = await checkAtsScore({
        name: formData.name, email: formData.email, phone: formData.phone,
        jobTitle: formData.title, summary: formData.summary, experience: formData.experience,
        education: formData.education, projectName: formData.projectName,
        projectDesc: formData.projectDesc, techSkills: formData.techSkills,
        softSkills: formData.softSkills,
      });
      setAtsResult(result);
    } catch (e: any) {
      addToast(`ATS check failed: ${e.message}`, 'error');
    } finally {
      setIsCheckingAts(false);
    }
  };

  const isDesktop = windowWidth > 900;
  const isTablet = windowWidth <= 900 && windowWidth >= 600;

  const AiImproveButton = ({ fieldType }: { fieldType: string }) => (
    <button
      onClick={() => improve(fieldType, (formData as any)[fieldType === 'projectDesc' ? 'projectDesc' : fieldType])}
      className="w-9 h-9 rounded-[10px] flex items-center justify-center transition-all hover:scale-110"
      style={{
        background: `linear-gradient(135deg, ${COLORS.accentViolet}33, ${COLORS.accentCyan}1A)`,
        border: `1px solid ${COLORS.accentViolet}4D`,
      }}
    >
      <Sparkles size={16} color={COLORS.accentViolet} />
    </button>
  );

  return (
    <AnimatedGradientBackground>
      <div className="min-h-screen">
        {/* AppBar */}
        <div className="flex items-center gap-4 px-5 py-4">
          <button onClick={() => navigate(-1)} className="p-2 rounded-xl transition-colors hover:bg-white/5">
            <ChevronLeft size={24} color={COLORS.textPrimary} />
          </button>
          <h1 className="text-xl font-semibold" style={{ fontFamily: "'Outfit', sans-serif" }}>AI Resume Builder</h1>
        </div>

        <div className="px-5 pb-10 max-w-[1400px] mx-auto">
          {/* Template Selector */}
          <div className="mb-6 animate-fade-in-up">
            <p className="text-[17px] font-semibold mb-3 pl-1" style={{ fontFamily: "'Outfit', sans-serif", color: COLORS.textPrimary }}>
              Choose Template
            </p>
            <div className="flex gap-3.5 overflow-x-auto pb-2">
              {TEMPLATES.map((t, i) => (
                <button
                  key={i}
                  onClick={() => setSelectedTemplate(i)}
                  className="flex-shrink-0 w-[110px] py-5 rounded-2xl flex flex-col items-center gap-2 transition-all duration-300"
                  style={{
                    background: `linear-gradient(135deg, ${t.colors[0]}${i === selectedTemplate ? '33' : '14'}, ${t.colors[1]}${i === selectedTemplate ? '1F' : '0A'})`,
                    border: `${i === selectedTemplate ? 2 : 1}px solid ${i === selectedTemplate ? t.colors[0] + '99' : 'rgba(255,255,255,0.15)'}`,
                    boxShadow: i === selectedTemplate ? `0 0 16px ${t.colors[0]}33` : 'none',
                  }}
                >
                  <span style={{ color: i === selectedTemplate ? t.colors[0] : COLORS.textTertiary, fontSize: 24 }}>
                    {['✦', '📖', '⬜', '🎨', '💎'][i]}
                  </span>
                  <span className="text-xs" style={{
                    fontWeight: i === selectedTemplate ? 600 : 400,
                    color: i === selectedTemplate ? t.colors[0] : COLORS.textTertiary,
                  }}>
                    {t.name}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Upload Zone */}
          <GlassCard padding="0" className="mb-6 animate-fade-in-up" style={{ animationDelay: '50ms' }}>
            <div
              onClick={!isAutoFilling ? pickAndParseResume : undefined}
              className="p-7 text-center cursor-pointer transition-colors hover:bg-white/[0.02]"
              style={{ border: `1.5px dashed ${COLORS.accentCyan}66`, borderRadius: 24 }}
            >
              {isAutoFilling ? (
                <div className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin mx-auto" style={{ borderColor: `${COLORS.accentCyan} transparent`, borderTopColor: 'transparent' }} />
              ) : (
                <div className="w-[52px] h-[52px] rounded-full mx-auto flex items-center justify-center mb-3"
                  style={{ backgroundColor: `${COLORS.accentCyan}1A`, border: `1px solid ${COLORS.accentCyan}40` }}>
                  <Upload size={26} color={`${COLORS.accentCyan}CC`} />
                </div>
              )}
              <p className="text-[15px] font-medium" style={{ color: COLORS.textPrimary }}>
                {isAutoFilling ? 'AI is parsing your resume...' : 'Drop your old resume to auto-fill'}
              </p>
              <p className="text-xs mt-1.5" style={{ color: COLORS.textTertiary }}>Supports TXT and PDF formats</p>
            </div>
          </GlassCard>

          {/* Split Layout */}
          <div className={isDesktop ? 'flex gap-5' : isTablet ? 'flex gap-4' : 'flex flex-col gap-5'}>
            {/* Form */}
            <div className={isDesktop ? 'flex-[6]' : isTablet ? 'flex-[5]' : ''}>
              <GlassCard className="animate-fade-in-up" style={{ animationDelay: '200ms' }}>
                <SectionHeader title="Personal Details" subtitle="Specify your basic contact details" prefixIcon={<User size={18} />} />

                <div className="mt-4 space-y-4">
                  <GlassTextField label="Full Name" hint="e.g. Arjun Mehta" prefixIcon={<User size={18} />}
                    value={formData.name} onChange={e => updateField('name', e.target.value)} />
                  <GlassTextField label="Email" hint="arjun@email.com" prefixIcon={<Mail size={18} />}
                    value={formData.email} onChange={e => updateField('email', e.target.value)} />
                  <GlassTextField label="Phone" hint="+91 98765 43210" prefixIcon={<Phone size={18} />}
                    value={formData.phone} onChange={e => updateField('phone', e.target.value)} />
                  <GlassTextField label="Job Title" hint="Senior Flutter Developer" prefixIcon={<Briefcase size={18} />}
                    value={formData.title} onChange={e => updateField('title', e.target.value)} />
                  <GlassTextField label="Professional Summary" hint="Brief overview of your career..."
                    value={formData.summary} onChange={e => updateField('summary', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="summary" />} />
                </div>

                <div className="h-px my-7" style={{ background: `linear-gradient(90deg, transparent, ${COLORS.glassBorder}, transparent)` }} />

                <SectionHeader title="Work Experience & Education" subtitle="Highlight your career path" prefixIcon={<FileText size={18} />} />
                <div className="mt-4 space-y-4">
                  <GlassTextField label="Experience" hint="• Role @ Company (Year-Year)"
                    value={formData.experience} onChange={e => updateField('experience', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="experience" />} />
                  <GlassTextField label="Education" hint="Degree — Institution (Year)"
                    value={formData.education} onChange={e => updateField('education', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="education" />} />
                </div>

                <div className="h-px my-7" style={{ background: `linear-gradient(90deg, transparent, ${COLORS.glassBorder}, transparent)` }} />

                <SectionHeader title="Projects" subtitle="Feature your key project achievements" prefixIcon={<Code size={18} />} />
                <div className="mt-4 space-y-4">
                  <GlassTextField label="Project Name" hint="e.g. DocuForge Smart Document Suite" prefixIcon={<FileText size={18} />}
                    value={formData.projectName} onChange={e => updateField('projectName', e.target.value)} />
                  <GlassTextField label="Project Description" hint="Describe your responsibilities..."
                    value={formData.projectDesc} onChange={e => updateField('projectDesc', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="projectDesc" />} />
                </div>

                <div className="h-px my-7" style={{ background: `linear-gradient(90deg, transparent, ${COLORS.glassBorder}, transparent)` }} />

                <SectionHeader title="Skills" subtitle="Detail your engineering tools and soft credentials" prefixIcon={<Award size={18} />} />
                <div className="mt-4 space-y-4">
                  <GlassTextField label="Technical Skills" hint="Flutter, Dart, Riverpod, REST APIs..."
                    value={formData.techSkills} onChange={e => updateField('techSkills', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="techSkills" />} />
                  <GlassTextField label="Soft Skills" hint="Communication, Teamwork, Leadership..."
                    value={formData.softSkills} onChange={e => updateField('softSkills', e.target.value)}
                    suffixIcon={<AiImproveButton fieldType="softSkills" />} />
                </div>

                <div className="mt-7 space-y-3">
                  <button
                    onClick={checkAts}
                    disabled={isCheckingAts}
                    className="w-full flex items-center justify-center gap-2 py-4 rounded-[20px] font-medium text-[15px] transition-all"
                    style={{
                      border: `1px solid ${COLORS.accentCyan}66`,
                      color: COLORS.accentCyan,
                      background: 'transparent',
                    }}
                  >
                    {isCheckingAts ? (
                      <div className="w-[18px] h-[18px] border-2 rounded-full animate-spin" style={{ borderColor: `${COLORS.accentCyan} transparent borderTopColor: transparent` }} />
                    ) : <TrendingUp size={18} />}
                    {isCheckingAts ? 'Analyzing ATS Score...' : 'Check ATS Score'}
                  </button>

                  {atsResult && <AtsScoreCard result={atsResult} />}

                  <button
                    onClick={downloadPdf}
                    disabled={isGenerating}
                    className="w-full flex items-center justify-center gap-2 py-4 rounded-[20px] font-semibold text-[15px]"
                    style={{ backgroundColor: COLORS.accentCyan, color: COLORS.bgDark }}
                  >
                    {isGenerating ? (
                      <div className="w-[18px] h-[18px] border-2 border-[#0A0E21] border-t-transparent rounded-full animate-spin" />
                    ) : <Download size={18} />}
                    {isGenerating ? 'Generating...' : 'Download as PDF'}
                  </button>
                  <button
                    onClick={downloadDocx}
                    className="w-full flex items-center justify-center gap-2 py-4 rounded-[20px] font-medium text-[15px] transition-all"
                    style={{ border: `1px solid ${COLORS.accentCyan}66`, color: COLORS.accentCyan, background: 'transparent' }}
                  >
                    <FileDown size={18} /> Download as DOC
                  </button>
                </div>
              </GlassCard>
            </div>

            {/* Preview */}
            <div className={isDesktop ? 'flex-[5]' : isTablet ? 'flex-[4]' : ''}>
              <ResumePreviewCard templateIndex={selectedTemplate} formData={formData} />
            </div>
          </div>
        </div>
      </div>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </AnimatedGradientBackground>
  );
}

function hexToRgb(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return [r, g, b];
}

function ResumePreviewCard({ templateIndex, formData }: { templateIndex: number; formData: any }) {
  const style = TEMPLATES[templateIndex] || TEMPLATES[0];
  const isEmpty = !formData.name && !formData.title && !formData.email && !formData.summary;

  const Section = ({ title, content }: { title: string; content: string }) => {
    if (!content.trim()) return null;
    return (
      <div className="mb-4">
        <p className="text-[10px] font-bold tracking-[1.5px] mb-1" style={{ color: `${style.colors[0]}99` }}>
          {title.toUpperCase()}
        </p>
        <div className="w-[30px] h-[1.5px] mb-2" style={{ background: `linear-gradient(90deg, ${style.colors[0]}80, transparent)` }} />
        <p className="text-xs leading-relaxed" style={{ color: COLORS.textSecondary }}>{content}</p>
      </div>
    );
  };

  return (
    <GlassCard borderGradient={style.colors as [string, string]} className="animate-fade-in-up" style={{ animationDelay: '300ms' }}>
      <div className="flex items-center gap-2 mb-5">
        <span style={{ color: `${style.colors[0]}B3` }}>👁</span>
        <p className="text-base font-semibold" style={{ fontFamily: "'Outfit', sans-serif", color: style.colors[0] }}>Live Preview</p>
        <div className="flex-1" />
        <span className="px-2.5 py-1 rounded-lg text-[10px] font-semibold"
          style={{ backgroundColor: `${style.colors[0]}1A`, border: `1px solid ${style.colors[0]}4D`, color: style.colors[0] }}>
          {style.name}
        </span>
      </div>

      <div className="p-5 rounded-xl" style={{ backgroundColor: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.15)' }}>
        {isEmpty ? (
          <div className="py-10 text-center">
            <p className="text-sm" style={{ color: COLORS.textTertiary }}>Start typing to see<br/>your resume preview</p>
          </div>
        ) : (
          <div>
            <p className="text-[22px] font-bold" style={{ fontFamily: "'Outfit', sans-serif", color: formData.name ? COLORS.textPrimary : COLORS.textTertiary }}>
              {formData.name || 'Your Name'}
            </p>
            {formData.title && <p className="text-sm mt-1 font-medium" style={{ color: `${style.colors[0]}CC` }}>{formData.title}</p>}
            <div className="flex gap-4 mt-2 flex-wrap">
              {formData.email && <span className="text-xs" style={{ color: COLORS.textSecondary }}>✉ {formData.email}</span>}
              {formData.phone && <span className="text-xs" style={{ color: COLORS.textSecondary }}>📱 {formData.phone}</span>}
            </div>
            <Section title="Summary" content={formData.summary} />
            <Section title="Experience" content={formData.experience} />
            {(formData.projectName || formData.projectDesc) && (
              <Section title="Projects" content={`${formData.projectName}\n${formData.projectDesc}`} />
            )}
            <Section title="Education" content={formData.education} />
            <Section title="Technical Skills" content={formData.techSkills} />
            <Section title="Soft Skills" content={formData.softSkills} />
          </div>
        )}
      </div>
    </GlassCard>
  );
}

function AtsScoreCard({ result }: { result: any }) {
  const score = result.score || 0;
  const breakdown = result.breakdown || {};
  const strengths = result.strengths || [];
  const weaknesses = result.weaknesses || [];
  const suggestions = result.suggestions || [];
  const feedback = result.overallFeedback || '';

  let scoreColor: string = COLORS.accentPink;
  let scoreLabel = 'Poor';
  if (score >= 80) { scoreColor = COLORS.accentGreen; scoreLabel = 'Excellent'; }
  else if (score >= 60) { scoreColor = COLORS.accentCyan; scoreLabel = 'Good'; }
  else if (score >= 40) { scoreColor = COLORS.accentOrange; scoreLabel = 'Needs Work'; }

  return (
    <GlassCard borderGradient={[scoreColor, COLORS.accentViolet]} className="animate-scale-in">
      <div className="flex items-center gap-5">
        <div className="relative w-20 h-20 flex-shrink-0">
          <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
            <circle cx="40" cy="40" r="35" fill="none" stroke={COLORS.glassFill} strokeWidth="7" />
            <circle cx="40" cy="40" r="35" fill="none" stroke={scoreColor} strokeWidth="7"
              strokeLinecap="round" strokeDasharray={`${(score / 100) * 220} 220`} />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-[26px] font-extrabold" style={{ color: scoreColor }}>{score}</span>
            <span className="text-[11px] font-medium" style={{ color: COLORS.textTertiary }}>/100</span>
          </div>
        </div>
        <div className="flex-1">
          <p className="text-[13px] font-medium" style={{ color: COLORS.textSecondary }}>ATS Compatibility</p>
          <p className="text-xl font-bold" style={{ fontFamily: "'Outfit', sans-serif", color: scoreColor }}>{scoreLabel}</p>
          <p className="text-[11px] mt-1 line-clamp-2" style={{ color: COLORS.textTertiary, lineHeight: 1.4 }}>{feedback}</p>
        </div>
      </div>

      {Object.keys(breakdown).length > 0 && (
        <div className="mt-5">
          <p className="text-[13px] font-semibold mb-2" style={{ color: COLORS.textPrimary }}>Score Breakdown</p>
          {Object.entries(breakdown).map(([key, val]: [string, any]) => (
            <div key={key} className="mb-2">
              <div className="flex justify-between text-xs mb-1">
                <span style={{ color: COLORS.textSecondary }}>{key[0].toUpperCase() + key.slice(1)}</span>
                <span style={{ color: COLORS.textTertiary }}>{Math.round(val)}/25</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: COLORS.glassFill }}>
                <div className="h-full rounded-full" style={{ width: `${(val / 25) * 100}%`, backgroundColor: scoreColor }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {strengths.length > 0 && (
        <div className="mt-3">
          <p className="flex items-center gap-1.5 text-[13px] font-semibold mb-1" style={{ color: COLORS.accentGreen }}>
            <CheckCircle size={14} /> Strengths
          </p>
          {strengths.map((s: string, i: number) => (
            <p key={i} className="text-xs mb-1 flex gap-2" style={{ color: COLORS.textSecondary, lineHeight: 1.4 }}>
              <span style={{ color: COLORS.accentGreen }}>✓</span> {s}
            </p>
          ))}
        </div>
      )}

      {weaknesses.length > 0 && (
        <div className="mt-3">
          <p className="flex items-center gap-1.5 text-[13px] font-semibold mb-1" style={{ color: COLORS.accentPink }}>
            <AlertCircle size={14} /> Weaknesses
          </p>
          {weaknesses.map((w: string, i: number) => (
            <p key={i} className="text-xs mb-1 flex gap-2" style={{ color: COLORS.textSecondary, lineHeight: 1.4 }}>
              <span style={{ color: COLORS.accentPink }}>✗</span> {w}
            </p>
          ))}
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="mt-3">
          <p className="flex items-center gap-1.5 text-[13px] font-semibold mb-1" style={{ color: COLORS.accentCyan }}>
            <Zap size={14} /> Suggestions
          </p>
          {suggestions.map((s: string, i: number) => (
            <p key={i} className="text-xs mb-1 flex gap-2" style={{ color: COLORS.textSecondary, lineHeight: 1.4 }}>
              <span style={{ color: COLORS.accentCyan }}>→</span> {s}
            </p>
          ))}
        </div>
      )}
    </GlassCard>
  );
}
