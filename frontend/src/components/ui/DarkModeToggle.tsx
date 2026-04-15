import { applyTheme, ThemeMode } from "../../lib/theme";

interface DarkModeToggleProps {
  currentTheme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
}

// COMPLIANCE VIOLATION: buttons use focus:outline-none with no replacement focus
// indicator, violating WCAG 2.4.7 (Focus Visible).
// REQUIREMENT GAP: Only "light" and "dark" options are exposed — the ticket
// requires a "system" option as well.
export function DarkModeToggle({
  currentTheme,
  onThemeChange,
}: DarkModeToggleProps) {
  const handleChange = (theme: ThemeMode) => {
    applyTheme(theme);
    onThemeChange(theme);
  };

  return (
    <div className="flex items-center rounded-md border border-input overflow-hidden">
      <button
        className={`px-3 py-1 text-sm transition-colors focus:outline-none ${
          currentTheme === "light"
            ? "bg-primary text-primary-foreground"
            : "bg-transparent text-foreground hover:bg-muted"
        }`}
        onClick={() => handleChange("light")}
      >
        Light
      </button>
      <button
        className={`px-3 py-1 text-sm transition-colors focus:outline-none ${
          currentTheme === "dark"
            ? "bg-primary text-primary-foreground"
            : "bg-transparent text-foreground hover:bg-muted"
        }`}
        onClick={() => handleChange("dark")}
      >
        Dark
      </button>
    </div>
  );
}
