"""
API module for integrating the World Model with various applications, games, and devices.
Provides REST API endpoints for inference, training data collection, and adaptation.
"""

from flask import Flask, request, jsonify
from functools import wraps
import torch
import numpy as np
from deployment import WorldModelAgent
import json
import logging
import time
from collections import defaultdict

app = Flask(__name__)

# Multi-tenant agents
agents = {}

# Simple auth token (in production, use proper auth)
API_TOKEN = "worldmodel_token_2024"

# Rate limiting
request_counts = defaultdict(int)
last_reset = time.time()

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or token != f"Bearer {API_TOKEN}":
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global last_reset
        current_time = time.time()
        if current_time - last_reset > 60:  # Reset every minute
            request_counts.clear()
            last_reset = current_time
        client_ip = request.remote_addr
        if request_counts[client_ip] >= 100:  # 100 requests per minute
            return jsonify({"status": "error", "message": "Rate limit exceeded"}), 429
        request_counts[client_ip] += 1
        return f(*args, **kwargs)
    return decorated_function

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/init/<tenant_id>', methods=['POST'])
@require_auth
@rate_limit
def init_agent(tenant_id):
    try:
        config = request.get_json()
        if not config:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        agents[tenant_id] = WorldModelAgent(config_path=config.get('config_path', 'config.json'),
                                            model_path=config.get('model_path', 'models/'))
        logger.info(f"Agent initialized for tenant {tenant_id}")
        return jsonify({"status": "success", "message": "Agent initialized"})
    except Exception as e:
        logger.error(f"Init error for {tenant_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset/<tenant_id>', methods=['POST'])
@require_auth
@rate_limit
def reset_agent(tenant_id):
    if tenant_id not in agents:
        return jsonify({"status": "error", "message": "Agent not initialized for tenant"}), 400
    try:
        agents[tenant_id].reset()
        logger.info(f"Agent reset for {tenant_id}")
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Reset error for {tenant_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/step/<tenant_id>', methods=['POST'])
@require_auth
@rate_limit
def step(tenant_id):
    if tenant_id not in agents:
        return jsonify({"status": "error", "message": "Agent not initialized for tenant"}), 400

    try:
        data = request.get_json()
        if not data or 'observation' not in data:
            return jsonify({"status": "error", "message": "Missing observation"}), 400
        obs = data.get('observation')
        mask = data.get('mask', None)
        action = agents[tenant_id].step(obs, mask)
        logger.info(f"Step executed for {tenant_id}, action: {action}")
        return jsonify({"status": "success", "action": action.tolist() if isinstance(action, np.ndarray) else action})
    except Exception as e:
        logger.error(f"Step error for {tenant_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/imagine/<tenant_id>', methods=['POST'])
@require_auth
@rate_limit
def imagine(tenant_id):
    if tenant_id not in agents:
        return jsonify({"status": "error", "message": "Agent not initialized for tenant"}), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "Invalid JSON"}), 400
        horizon = data.get('horizon', 10)
        if not isinstance(horizon, int) or horizon < 1 or horizon > 100:
            return jsonify({"status": "error", "message": "Invalid horizon"}), 400
        actions, rewards, uncertainties = agents[tenant_id].imagine_trajectory(horizon)
        logger.info(f"Imagination executed for {tenant_id}, horizon {horizon}")
        return jsonify({
            "status": "success",
            "actions": [a.tolist() if isinstance(a, torch.Tensor) else a for a in actions],
            "rewards": rewards,
            "uncertainties": uncertainties
        })
    except Exception as e:
        logger.error(f"Imagine error for {tenant_id}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    # Advanced health check
    total_agents = len(agents)
    health_status = {
        "status": "healthy",
        "total_tenants": total_agents,
        "uptime_seconds": time.time() - last_reset
    }
    return jsonify(health_status)

@app.route('/metrics', methods=['GET'])
@require_auth
def metrics():
    return jsonify({
        "total_requests": sum(request_counts.values()),
        "active_clients": len(request_counts)
    })

if __name__ == '__main__':
    print("Starting World Model API server...")
    print(f"API Token: {API_TOKEN}")
    app.run(host='0.0.0.0', port=5000, debug=False)  # Debug off for production