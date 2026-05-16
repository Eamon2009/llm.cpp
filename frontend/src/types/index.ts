export type Role = "user" | "assistant" | "system";
export type ModelBackend = "cpp" | "torch";

export interface Message {
  id: string;
  session_id: string;
  role: Role;
  text: string;
  prompt?: string | null;
  chars: number;
  seconds: number;
  error?: string | null;
  created_at: string;
  pending?: boolean;
}

export interface Session {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ChatRequest {
  session_id: string | null;
  prompt: string;
  max_tokens: number;
  temperature: number;
  stream: false;
  model_backend: ModelBackend;
}

export interface ChatResponse {
  id: string;
  session_id: string;
  prompt: string;
  text: string;
  chars: number;
  seconds: number;
  model: string;
  model_backend: ModelBackend;
  created_at: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  api: "ok";
  cpp_server: "ok" | "unreachable";
  torch_model: "ok" | "unavailable";
  model: string;
  vocab: number;
  params: number;
  uptime_seconds: number;
}

export interface ModelStats {
  model: string;
  architecture: string;
  parameters: number;
  vocabulary: number;
  val_loss: number;
  context: number;
  training: string;
  backend: string;
  backend_online: boolean;
  torch_checkpoint: string;
  torch_online: boolean;
  uptime_seconds: number;
}

export interface ApiError {
  error: string;
  message: string;
  code: number;
}
