import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { DashboardScreen } from './pages/DashboardScreen';
import { ResumeBuilderScreen } from './pages/ResumeBuilderScreen';
import { MergeSplitScreen } from './pages/MergeSplitScreen';
import { FormatConverterScreen } from './pages/FormatConverterScreen';
import { ImageToPdfScreen } from './pages/ImageToPdfScreen';
import { DigitalSignatureScreen } from './pages/DigitalSignatureScreen';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DashboardScreen />} />
        <Route path="/resume-builder" element={<ResumeBuilderScreen />} />
        <Route path="/merge-split" element={<MergeSplitScreen />} />
        <Route path="/format-converter" element={<FormatConverterScreen />} />
        <Route path="/image-to-pdf" element={<ImageToPdfScreen />} />
        <Route path="/digital-signature" element={<DigitalSignatureScreen />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
