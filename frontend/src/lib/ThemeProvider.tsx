import React, { createContext, useContext, useEffect } from "react";
import { applyTheme } from "./theme";
import type { AppTheme } from "../types";

interface ThemeContextValue {
  theme: AppTheme;
  setTheme: (theme: AppTheme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "system",
  setTheme: () => {},
});

interface ThemeProviderProps {
  theme: AppTheme;
  onThemeChange: (theme: AppTheme) => void;
  children: React.ReactNode;
}

export function ThemeProvider({ theme, onThemeChange, children }: ThemeProviderProps) {
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme: onThemeChange }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}
