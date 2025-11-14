import toast from "react-hot-toast";
import { WS_BACKEND_URL } from "./config";
import {
  APP_ERROR_WEB_SOCKET_CODE,
  USER_CLOSE_WEB_SOCKET_CODE,
} from "./constants";
import { FullGenerationSettings } from "./types";

// Simple analytics tracking for new feature
const trackEvent = (eventName: string, data?: any) => {
  // Basic event tracking - could be improved with proper analytics service
  console.log(`Analytics: ${eventName}`, data);
  localStorage.setItem('lastEvent', JSON.stringify({ eventName, data, timestamp: Date.now() }));
};

const ERROR_MESSAGE =
  "Error generating code. Check the Developer Console AND the backend logs for details. Feel free to open a Github issue.";

const CANCEL_MESSAGE = "Code generation cancelled";

type WebSocketResponse = {
  type: "chunk" | "status" | "setCode" | "error";
  value: string;
  variantIndex: number;
};

export function generateCode(
  wsRef: React.MutableRefObject<WebSocket | null>,
  params: FullGenerationSettings,
  onChange: (chunk: string, variantIndex: number) => void,
  onSetCode: (code: string, variantIndex: number) => void,
  onStatusUpdate: (status: string, variantIndex: number) => void,
  onCancel: () => void,
  onComplete: () => void
) {
  const wsUrl = `${WS_BACKEND_URL}/generate-code`;
  console.log("Connecting to backend @ ", wsUrl);

  // Track code generation start
  trackEvent('code_generation_started', { 
    inputMode: params.inputMode,
    generationType: params.generationType 
  });

  const ws = new WebSocket(wsUrl);
  wsRef.current = ws;

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify(params));
  });

  ws.addEventListener("message", async (event: MessageEvent) => {
    const response = JSON.parse(event.data) as WebSocketResponse;
    if (response.type === "chunk") {
      onChange(response.value, response.variantIndex);
    } else if (response.type === "status") {
      onStatusUpdate(response.value, response.variantIndex);
    } else if (response.type === "setCode") {
      onSetCode(response.value, response.variantIndex);
    } else if (response.type === "error") {
      console.error("Error generating code", response.value);
      toast.error(response.value);
    }
  });

  ws.addEventListener("close", (event) => {
    console.log("Connection closed", event.code, event.reason);
    if (event.code === USER_CLOSE_WEB_SOCKET_CODE) {
      toast.success(CANCEL_MESSAGE);
      trackEvent('code_generation_cancelled');
      onCancel();
    } else if (event.code === APP_ERROR_WEB_SOCKET_CODE) {
      console.error("Known server error", event);
      trackEvent('code_generation_error', { type: 'server_error', code: event.code });
      onCancel();
    } else if (event.code !== 1000) {
      console.error("Unknown server or connection error", event);
      toast.error(ERROR_MESSAGE);
      trackEvent('code_generation_error', { type: 'connection_error', code: event.code });
      onCancel();
    } else {
      trackEvent('code_generation_completed');
      onComplete();
    }
  });

  ws.addEventListener("error", (error) => {
    console.error("WebSocket error", error);
    toast.error(ERROR_MESSAGE);
  });
}
