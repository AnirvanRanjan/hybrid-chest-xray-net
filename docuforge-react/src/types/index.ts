// ToolItem model
export interface ToolItem {
  name: string;
  description: string;
  icon: string; // Lucide icon name
  route: string;
  gradientColors: [string, string];
}

// TaskStatus model
export type TaskState = 'pending' | 'processing' | 'completed' | 'failed';

export interface TaskStatus {
  taskId: string;
  status: TaskState;
  downloadUrl?: string;
  aiSummary?: string;
  errorMessage?: string;
  progress?: number; // 0.0 to 1.0
}

export interface TaskStatusResponse {
  task_id: string;
  status: string;
  progress?: number;
  download_url?: string;
  ai_summary?: string;
  error_message?: string;
}