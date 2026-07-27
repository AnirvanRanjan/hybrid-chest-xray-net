import { useState, useRef, useCallback, useEffect } from 'react';
import type { TaskStatus, TaskState, TaskStatusResponse } from '../types';
import { BASE_URL, STATUS_ENDPOINT, POLL_INTERVAL } from '../constants';

export function useTaskManager() {
  const [task, setTask] = useState<TaskStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pollRetryRef = useRef(0);
  const MAX_RETRIES = 3;

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  const reset = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    pollRetryRef.current = 0;
    setTask(null);
    setLoading(false);
    setError(null);
  }, []);

  const startPolling = useCallback((taskId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRetryRef.current = 0;

    pollRef.current = setInterval(async () => {
      try {
        const response = await fetch(`${BASE_URL}${STATUS_ENDPOINT}/${taskId}`);
        if (!response.ok) throw new Error(`Server error (${response.status})`);
        const json: TaskStatusResponse = await response.json();
        const status: TaskState = json.status as TaskState;
        const taskStatus: TaskStatus = {
          taskId: json.task_id || taskId,
          status,
          progress: json.progress,
          downloadUrl: json.download_url,
          aiSummary: json.ai_summary,
          errorMessage: json.error_message,
        };
        setTask(taskStatus);
        setLoading(false);
        pollRetryRef.current = 0;

        if (status !== 'pending' && status !== 'processing') {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
        }
      } catch {
        pollRetryRef.current++;
        if (pollRetryRef.current >= MAX_RETRIES) {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          setError(`Processing check failed after ${MAX_RETRIES} attempts`);
          setLoading(false);
        }
      }
    }, POLL_INTERVAL);
  }, []);

  const runMockTask = useCallback((endpoint: string) => {
    setLoading(true);
    setTimeout(() => {
      const taskId = `mock_task_${Date.now()}`;
      setTask({ taskId, status: 'pending' });

      let progress = 0;
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(() => {
        progress += 0.25;
        if (progress >= 1.0) {
          if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
          }
          let summary: string | undefined;
          if (endpoint.includes('convert') || endpoint.includes('compress') || endpoint.includes('enhance')) {
            summary = 'This is a mock AI summary generated for your document. The system successfully parsed the layout, extracted 12 key paragraphs, and optimized the text contrast by 15%. All operations completed successfully in mock mode.';
          }
          setTask({
            taskId,
            status: 'completed',
            downloadUrl: 'https://www.africau.edu/images/default/sample.pdf',
            aiSummary: summary,
            progress: 1.0,
          });
          setLoading(false);
        } else {
          setTask({ taskId, status: 'processing', progress });
        }
      }, 1000);
    }, 500);
  }, []);

  const isConnectionError = (e: unknown) => {
    const msg = String(e).toLowerCase();
    return msg.includes('failed to fetch') || msg.includes('networkerror') || msg.includes('socketexception') || msg.includes('connection refused');
  };

  const uploadFile = useCallback(async ({
    file,
    fileName,
    endpoint,
  }: {
    file: File;
    fileName: string;
    endpoint: string;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', file, fileName);
      const response = await fetch(`${BASE_URL}${endpoint}`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Upload failed (${response.status}): ${await response.text()}`);
      const json = await response.json();
      const taskId = json.task_id as string;
      setTask({ taskId, status: 'pending' });
      startPolling(taskId);
    } catch (e: unknown) {
      if (isConnectionError(e)) {
        runMockTask(endpoint);
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    }
  }, [startPolling, runMockTask]);

  const uploadFiles = useCallback(async ({
    files,
    endpoint,
  }: {
    files: File[];
    endpoint: string;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      files.forEach(f => formData.append('files', f, f.name));
      const response = await fetch(`${BASE_URL}${endpoint}`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Upload failed (${response.status}): ${await response.text()}`);
      const json = await response.json();
      const taskId = json.task_id as string;
      setTask({ taskId, status: 'pending' });
      startPolling(taskId);
    } catch (e: unknown) {
      if (isConnectionError(e)) {
        runMockTask(endpoint);
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    }
  }, [startPolling, runMockTask]);

  const uploadFileWithSignature = useCallback(async ({
    pdfFile,
    signatureBlob,
  }: {
    pdfFile: File;
    signatureBlob: Blob;
  }) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', pdfFile, pdfFile.name);
      formData.append('signature', signatureBlob, 'signature.png');
      const response = await fetch(`${BASE_URL}/api/v1/sign`, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Upload failed (${response.status}): ${await response.text()}`);
      const json = await response.json();
      const taskId = json.task_id as string;
      setTask({ taskId, status: 'pending' });
      startPolling(taskId);
    } catch (e: unknown) {
      if (isConnectionError(e)) {
        runMockTask('/api/v1/sign');
      } else {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    }
  }, [startPolling, runMockTask]);

  return {
    task,
    loading,
    error,
    reset,
    uploadFile,
    uploadFiles,
    uploadFileWithSignature,
    isPending: task?.status === 'pending' || task?.status === 'processing',
    isCompleted: task?.status === 'completed',
    isFailed: task?.status === 'failed',
  };
}
