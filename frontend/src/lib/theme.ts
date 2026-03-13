export type ThemeMode = "light" | "dark" | "system";

const STORAGE_KEY = "theme";

let systemMql: MediaQueryList | null = null;
let systemListener: ((e: MediaQueryListEvent) => void) | null = null;

export function applyTheme(theme: ThemeMode) {
  try {
    const root = document.documentElement; // <html> only — single top-level indicator

    // Clean up any previous system listener
    if (systemMql && systemListener) {
      systemMql.removeEventListener("change", systemListener);
      systemMql = null;
      systemListener = null;
    }

    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const prefersDark = mql.matches;
    const isDark = theme === "system" ? prefersDark : theme === "dark";

    // Persist choice for preload script in index.html
    if (theme === "system") {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, theme);
    }

    // Toggle Tailwind dark class on the root element only
    if (isDark) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }

    // React to OS theme changes while in "system" mode
    if (theme === "system") {
      systemMql = mql;
      systemListener = () => applyTheme("system");
      systemMql.addEventListener("change", systemListener);
    }
  } catch (e) {
    console.warn("[theme] Failed to apply theme:", e);
  }
}
