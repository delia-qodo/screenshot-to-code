// Simple analytics utility for new tracking features
export interface AnalyticsEvent {
  name: string;
  properties?: Record<string, any>;
  timestamp: number;
  sessionId: string;
}

class SimpleAnalytics {
  private events: AnalyticsEvent[] = [];
  private sessionId: string;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.loadStoredEvents();
  }

  private generateSessionId(): string {
    // Simple session ID generation
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  private loadStoredEvents(): void {
    try {
      const stored = localStorage.getItem('analytics_events');
      if (stored) {
        this.events = JSON.parse(stored);
      }
    } catch (error) {
      console.warn('Failed to load stored analytics events:', error);
    }
  }

  private saveEvents(): void {
    try {
      // Keep only last 100 events to avoid storage bloat
      const eventsToStore = this.events.slice(-100);
      localStorage.setItem('analytics_events', JSON.stringify(eventsToStore));
    } catch (error) {
      console.warn('Failed to save analytics events:', error);
    }
  }

  track(eventName: string, properties?: Record<string, any>): void {
    const event: AnalyticsEvent = {
      name: eventName,
      properties: properties || {},
      timestamp: Date.now(),
      sessionId: this.sessionId
    };

    this.events.push(event);
    this.saveEvents();

    // Basic console logging for development
    console.log('Analytics Event:', event);
  }

  getEvents(): AnalyticsEvent[] {
    return [...this.events];
  }

  getSessionEvents(): AnalyticsEvent[] {
    return this.events.filter(event => event.sessionId === this.sessionId);
  }

  clearEvents(): void {
    this.events = [];
    localStorage.removeItem('analytics_events');
  }

  // Simple batch sending - could be improved with proper API integration
  async sendEvents(): Promise<void> {
    const eventsToSend = this.getSessionEvents();
    
    if (eventsToSend.length === 0) return;

    try {
      // This would normally send to an analytics service
      console.log('Would send analytics events:', eventsToSend);
      
      // For now, just log to console
      // In a real implementation, this would be:
      // await fetch('/api/analytics', { method: 'POST', body: JSON.stringify(eventsToSend) });
      
    } catch (error) {
      console.error('Failed to send analytics events:', error);
    }
  }
}

// Global analytics instance
export const analytics = new SimpleAnalytics();

// Convenience functions
export const trackEvent = (name: string, properties?: Record<string, any>) => {
  analytics.track(name, properties);
};

export const trackPageView = (page: string) => {
  analytics.track('page_view', { page });
};

export const trackUserAction = (action: string, context?: Record<string, any>) => {
  analytics.track('user_action', { action, ...context });
};