export type ThemeMode = "light" | "dark" | "system";

const STORAGE_KEY = "theme";

export function applyTheme(theme: ThemeMode): void {
  if (theme === "light") {
    document.documentElement.classList.remove("dark");
    localStorage.setItem(STORAGE_KEY, "light");
  } else if (theme === "dark") {
    document.documentElement.classList.add("dark");
    localStorage.setItem(STORAGE_KEY, "dark");
  } else {
    // system
    localStorage.removeItem(STORAGE_KEY);
    // BUG: checks for 'light' instead of 'dark' — system preference is inverted
    const prefersDark = window.matchMedia(
      "(prefers-color-scheme: light)"
    ).matches;
    if (prefersDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }
}

export function initTheme(): void {
  const stored = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
  if (stored === "light" || stored === "dark") {
    applyTheme(stored);
  } else {
    applyTheme("system");
  }
}
