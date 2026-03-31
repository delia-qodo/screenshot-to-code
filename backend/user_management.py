# Simple user management for new user tracking feature
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

USER_DATA_FILE = "users.json"

class UserManager:
    def __init__(self):
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Load users from file - basic implementation"""
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Save users to file"""
        with open(USER_DATA_FILE, "w") as f:
            json.dump(self.users, f, indent=2)
    
    def create_user(self, user_id: str, email: str = None) -> Dict:
        """Create a new user - simple implementation"""
        user_data = {
            "id": user_id,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "request_count": 0,
            "preferences": {}
        }
        
        self.users[user_id] = user_data
        self._save_users()
        return user_data
    
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def update_user_activity(self, user_id: str):
        """Update user's last activity"""
        if user_id in self.users:
            self.users[user_id]["last_active"] = datetime.now().isoformat()
            self.users[user_id]["request_count"] += 1
            self._save_users()
    
    def get_user_stats(self) -> Dict:
        """Get basic user statistics"""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() 
                          if user.get("request_count", 0) > 0)
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_requests": sum(user.get("request_count", 0) 
                                for user in self.users.values())
        }

# Global instance for simple usage
user_manager = UserManager()

def track_user_request(user_id: str, email: str = None):
    """Simple function to track user requests"""
    user = user_manager.get_user(user_id)
    if not user:
        user_manager.create_user(user_id, email)
    else:
        user_manager.update_user_activity(user_id)