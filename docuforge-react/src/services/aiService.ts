const _baseUrl = 'http://localhost:8000';

export async function extractPdfText(pdfBytes: File, filename: string): Promise<string> {
  const formData = new FormData();
  formData.append('file', pdfBytes, filename);
  const response = await fetch(`${_baseUrl}/api/v1/ai/extract-pdf-text`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error(`PDF extraction failed: ${await response.text()}`);
  const data = await response.json();
  return data.text as string;
}

export async function chat(systemPrompt: string, userMessage: string): Promise<string> {
  const response = await fetch(`${_baseUrl}/api/v1/ai/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
    }),
  });
  if (!response.ok) throw new Error(`AI API error ${response.status}: ${await response.text()}`);
  const data = await response.json();
  return data.choices[0].message.content as string;
}

export async function parseResume(resumeText: string): Promise<Record<string, string>> {
  const response = await fetch(`${_baseUrl}/api/v1/ai/parse-resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text: resumeText }),
  });
  if (!response.ok) throw new Error(`Parse failed: ${await response.text()}`);
  const data = await response.json();
  return Object.fromEntries(Object.entries(data).map(([k, v]) => [k, String(v ?? '')]));
}

export async function improveField({
  fieldType,
  content,
  jobTitle,
}: {
  fieldType: string;
  content: string;
  jobTitle: string;
}): Promise<string> {
  const response = await fetch(`${_baseUrl}/api/v1/ai/improve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ field_type: fieldType, content, job_title: jobTitle }),
  });
  if (!response.ok) throw new Error(`Improve failed: ${await response.text()}`);
  const data = await response.json();
  return data.result as string;
}

export async function checkAtsScore(data: {
  name: string; email: string; phone: string; jobTitle: string; summary: string;
  experience: string; education: string; projectName: string; projectDesc: string;
  techSkills: string; softSkills: string;
}): Promise<Record<string, any>> {
  const response = await fetch(`${_baseUrl}/api/v1/ai/ats-score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: data.name, email: data.email, phone: data.phone,
      job_title: data.jobTitle, summary: data.summary, experience: data.experience,
      education: data.education, project_name: data.projectName,
      project_desc: data.projectDesc, tech_skills: data.techSkills,
      soft_skills: data.softSkills,
    }),
  });
  if (response.ok) return response.json();
  return {
    score: 0, breakdown: { formatting: 0, keywords: 0, content: 0, impact: 0 },
    strengths: [], weaknesses: [], suggestions: [], overallFeedback: 'Could not analyze resume.',
  };
}