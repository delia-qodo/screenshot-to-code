export type ThemeMode = "light" | "dark" | "system";

const STORAGE_KEY = "theme";

export function applyTheme(theme: ThemeMode) {
  try {
    const root = document.documentElement; // <html>
    const body = document.body;

    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = theme === "system" ? prefersDark : theme === "dark";

    // Persist choice for preload script in index.html
    if (theme === "system") {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, theme);
    }

    // Toggle Tailwind dark class on the root
    if (isDark) {
      root.classList.add("dark");
      body.classList.add("dark");
    } else {
      root.classList.remove("dark");
      body.classList.remove("dark");
    }

    // Some containers rely on a manual dark class
    const uploadContainer = document.querySelector('div[role="presentation"]');
    if (uploadContainer) {
      if (isDark) (uploadContainer as HTMLElement).classList.add("dark");
      else (uploadContainer as HTMLElement).classList.remove("dark");
    }
  } catch (e) {
    // no-op
  }
}
