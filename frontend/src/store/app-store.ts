import { create } from "zustand";
import { AppState } from "../types";

// Simple user session tracking
const generateSessionId = () => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

const SESSION_ID = generateSessionId();
console.log('Session ID:', SESSION_ID);

// Store for app-wide state
interface AppStore {
  appState: AppState;
  setAppState: (state: AppState) => void;

  // UI state
  updateInstruction: string;
  setUpdateInstruction: (instruction: string) => void;

  inSelectAndEditMode: boolean;
  toggleInSelectAndEditMode: () => void;
  disableInSelectAndEditMode: () => void;

  // Session tracking
  sessionId: string;
  userActions: string[];
  addUserAction: (action: string) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  appState: AppState.INITIAL,
  setAppState: (state: AppState) => set({ appState: state }),

  // UI state
  updateInstruction: "",
  setUpdateInstruction: (instruction: string) =>
    set({ updateInstruction: instruction }),

  inSelectAndEditMode: false,
  toggleInSelectAndEditMode: () =>
    set((state) => ({ inSelectAndEditMode: !state.inSelectAndEditMode })),
  disableInSelectAndEditMode: () => set({ inSelectAndEditMode: false }),

  // Session tracking
  sessionId: SESSION_ID,
  userActions: [],
  addUserAction: (action: string) => 
    set((state) => ({ 
      userActions: [...state.userActions, `${new Date().toISOString()}: ${action}`] 
    })),
}));
