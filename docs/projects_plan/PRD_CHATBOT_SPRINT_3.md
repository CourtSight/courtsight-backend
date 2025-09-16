# 📋 Sprint 3 - Production Deployment & Optimization

## CourtSight Legal Chatbot - Sprint 3 Implementation Plan

**Sprint:** 3 (Production Deployment & Optimization)  
**Tanggal:** October 14, 2025  
**Durasi:** 2 minggu  
**Status:** Sprint Planning  

---

## Sprint 3 Overview

### 🎯 Sprint Goals
**Production-ready deployment** dengan comprehensive monitoring, security hardening, dan final optimization untuk CourtSight Legal Chatbot system.

### ✅ Sprint Objectives
1. **Production Deployment** - GKE/Cloud Run deployment dengan auto-scaling
2. **Security Hardening** - Comprehensive security implementation
3. **Monitoring & Observability** - Full stack monitoring dengan alerting
4. **Performance Optimization** - Final performance tuning dan optimization
5. **User Acceptance Testing** - UAT dengan real users dan stakeholder approval

---

## 1. Sprint 3 Production Architecture

### 1.1 Production System Architecture

```mermaid
graph TB
    subgraph "Production Environment"
        subgraph "Load Balancer Layer"
            LB[Google Cloud Load Balancer]
            CDN[Cloud CDN]
            WAF[Web Application Firewall]
        end

        subgraph "Application Layer (GKE)"
            INGRESS[Nginx Ingress Controller]
            CHATBOT_PODS[Chatbot Pods (3 replicas)]
            BG_WORKERS[Background Worker Pods]
            HPA[Horizontal Pod Autoscaler]
        end

        subgraph "Data Layer"
            POSTGRES_CLUSTER[(PostgreSQL Cluster)]
            REDIS_CLUSTER[(Redis Cluster)]
            GCS_BUCKET[Google Cloud Storage]
        end

        subgraph "Monitoring Stack"
            PROMETHEUS[Prometheus]
            GRAFANA[Grafana]
            ALERTMANAGER[AlertManager]
            JAEGER[Jaeger Tracing]
        end

        subgraph "External Services"
            EXISTING_LLM[GOOGLE_GENAI LLM]
            CLOUD_LOGGING[Google Cloud Logging]
            CLOUD_MONITORING[Google Cloud Monitoring]
        end

        subgraph "Security Layer"
            VAULT[HashiCorp Vault]
            IAM[Google Cloud IAM]
            CERT_MANAGER[Cert Manager]
        end
    end

    LB --> CDN
    CDN --> WAF
    WAF --> INGRESS
    INGRESS --> CHATBOT_PODS
    CHATBOT_PODS --> HPA
    CHATBOT_PODS --> POSTGRES_CLUSTER
    CHATBOT_PODS --> REDIS_CLUSTER
    BG_WORKERS --> POSTGRES_CLUSTER
    BG_WORKERS --> GCS_BUCKET
    
    PROMETHEUS --> CHATBOT_PODS
    PROMETHEUS --> ALERTMANAGER
    GRAFANA --> PROMETHEUS
    JAEGER --> CHATBOT_PODS
    
    CHATBOT_PODS --> EXISTING_LLM
    CHATBOT_PODS --> GOOGLE_AI
    CHATBOT_PODS --> CLOUD_LOGGING
    
    VAULT --> CHATBOT_PODS
    IAM --> CHATBOT_PODS
    CERT_MANAGER --> INGRESS
```

### 1.2 Production Dependencies & Configuration

```yaml
# kubernetes/production/values.yaml
global:
  environment: production
  replicas: 3
  resources:
    requests:
      memory: "2Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi" 
      cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
  jaeger:
    enabled: true

security:
  vault:
    enabled: true
  networkPolicies:
    enabled: true
  podSecurityPolicies:
    enabled: true
```

---

## 2. Production Deployment Strategy

### 2.1 Infrastructure as Code

#### Terraform Configuration
```hcl
# infrastructure/terraform/main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

# GKE Cluster
resource "google_container_cluster" "legal_chatbot_cluster" {
  name     = "legal-chatbot-${var.environment}"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
    horizontal_pod_autoscaling {
      disabled = false
    }
  }
}

# Node Pool
resource "google_container_node_pool" "primary_nodes" {
  name       = "primary-node-pool"
  location   = var.region
  cluster    = google_container_cluster.legal_chatbot_cluster.name
  node_count = var.initial_node_count

  node_config {
    preemptible  = false
    machine_type = "e2-standard-4"

    service_account = google_service_account.gke_service_account.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      application = "legal-chatbot"
    }

    tags = ["legal-chatbot-node"]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# PostgreSQL Instance
resource "google_sql_database_instance" "postgres" {
  name             = "legal-chatbot-db-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-custom-4-16384"
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 30
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }

    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements,pgaudit,vector"
    }
  }

  deletion_protection = true
}

# Redis Instance
resource "google_redis_instance" "cache" {
  name           = "legal-chatbot-cache-${var.environment}"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  region         = var.region

  authorized_network = google_compute_network.vpc.id
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }
}
```

#### Kubernetes Deployment
```yaml
# kubernetes/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: legal-chatbot
  namespace: production
  labels:
    app: legal-chatbot
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: legal-chatbot
  template:
    metadata:
      labels:
        app: legal-chatbot
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: legal-chatbot-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 2000
      containers:
      - name: legal-chatbot
        image: gcr.io/courtsight-project/legal-chatbot:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: EXISTING_LLM_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: credentials-volume
          mountPath: /app/credentials
          readOnly: true
      volumes:
      - name: config-volume
        configMap:
          name: legal-chatbot-config
      - name: credentials-volume
        secret:
          secretName: gcp-credentials
---
apiVersion: v1
kind: Service
metadata:
  name: legal-chatbot-service
  namespace: production
  labels:
    app: legal-chatbot
spec:
  selector:
    app: legal-chatbot
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: legal-chatbot-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: legal-chatbot
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 2.2 CI/CD Pipeline Production

```yaml
# .github/workflows/production-deploy.yml
name: Production Deployment

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  PROJECT_ID: courtsight-project
  GKE_CLUSTER: legal-chatbot-production
  GKE_ZONE: asia-southeast1-a
  DEPLOYMENT_NAME: legal-chatbot
  IMAGE: legal-chatbot

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security scan
      uses: securecodewarrior/github-action-add-sarif@v1
      with:
        sarif-file: security-scan-results.sarif
    
    - name: Container security scan
      run: |
        docker build -t $IMAGE:$GITHUB_SHA .
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          -v $(pwd):/tmp aquasec/trivy image --exit-code 1 $IMAGE:$GITHUB_SHA

  test:
    runs-on: ubuntu-latest
    needs: security-scan
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
    
    - name: Run tests
      run: |
        pytest tests/ --cov=app --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  build-and-push:
    runs-on: ubuntu-latest
    needs: test
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Google Cloud CLI
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ env.PROJECT_ID }}
    
    - name: Configure Docker to use gcloud as credential helper
      run: gcloud auth configure-docker
    
    - name: Build Docker image
      run: |
        docker build -t gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA \
          --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
          --build-arg VCS_REF=$GITHUB_SHA \
          --build-arg VERSION=$GITHUB_REF_NAME .
    
    - name: Push Docker image
      run: docker push gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA

  deploy:
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Google Cloud CLI
      uses: google-github-actions/setup-gcloud@v1
      with:
        service_account_key: ${{ secrets.GCP_SA_KEY }}
        project_id: ${{ env.PROJECT_ID }}
    
    - name: Get GKE credentials
      run: |
        gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE
    
    - name: Deploy to GKE
      run: |
        # Update deployment with new image
        kubectl set image deployment/$DEPLOYMENT_NAME \
          legal-chatbot=gcr.io/$PROJECT_ID/$IMAGE:$GITHUB_SHA \
          --namespace=production
        
        # Wait for rollout to complete
        kubectl rollout status deployment/$DEPLOYMENT_NAME --namespace=production
        
        # Verify deployment
        kubectl get services -o wide --namespace=production

  smoke-tests:
    runs-on: ubuntu-latest
    needs: deploy
    steps:
    - name: Run smoke tests
      run: |
        # Basic health check
        curl -f https://api.courtsight.id/chat/health || exit 1
        
        # API functionality test
        response=$(curl -s -X POST https://api.courtsight.id/chat/ \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer ${{ secrets.TEST_API_TOKEN }}" \
          -d '{"message": "Test question"}')
        
        echo $response | jq -e '.message' || exit 1
```

---

## 3. Security Implementation

### 3.1 Security Hardening

#### Network Security
```yaml
# kubernetes/security/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: legal-chatbot-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: legal-chatbot
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS outbound
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
```

#### Pod Security Policy
```yaml
# kubernetes/security/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: legal-chatbot-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

#### Secret Management dengan Vault
```python
# src/app/core/security/vault_client.py
import hvac
from typing import Dict, Any
import os

class VaultClient:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_URL'),
            token=os.getenv('VAULT_TOKEN')
        )
        
    async def get_secret(self, path: str) -> Dict[str, Any]:
        """Get secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data']
        except Exception as e:
            logger.error(f"Failed to retrieve secret from {path}: {e}")
            raise
    
    async def get_database_credentials(self) -> Dict[str, str]:
        """Get database credentials from Vault"""
        return await self.get_secret('legal-chatbot/database')
    
    async def get_api_keys(self) -> Dict[str, str]:
        """Get API keys from Vault"""
        return await self.get_secret('legal-chatbot/api-keys')
```

### 3.2 Authentication & Authorization Enhancement

```python
# src/app/core/security/enhanced_auth.py
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from typing import List, Optional
import redis
import time

class EnhancedJWTAuth:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.security = HTTPBearer()
    
    async def verify_token(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> dict:
        """Enhanced token verification with blacklist check"""
        token = credentials.credentials
        
        # Check if token is blacklisted
        if await self._is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked"
            )
        
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Check token expiration
            if payload.get("exp", 0) < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            # Rate limiting check
            await self._check_rate_limit(payload.get("user_id"))
            
            return payload
            
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is in blacklist"""
        return await self.redis_client.exists(f"blacklist:{token}")
    
    async def _check_rate_limit(self, user_id: str) -> None:
        """Check user rate limits"""
        key = f"rate_limit:{user_id}"
        current = await self.redis_client.get(key)
        
        if current and int(current) > settings.RATE_LIMIT_PER_MINUTE:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

# Role-based access control
class RBACManager:
    ROLES = {
        "admin": ["read", "write", "delete", "admin"],
        "judge": ["read", "write", "legal_analysis"],
        "lawyer": ["read", "write", "case_comparison"],
        "researcher": ["read", "legal_search"],
        "public": ["read"]
    }
    
    @classmethod
    def check_permission(cls, user_role: str, required_permission: str) -> bool:
        """Check if user role has required permission"""
        return required_permission in cls.ROLES.get(user_role, [])
    
    @classmethod
    def require_permission(cls, permission: str):
        """Decorator for permission checking"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Get current user from context
                current_user = kwargs.get('current_user')
                if not current_user or not cls.check_permission(
                    current_user.get('role'), permission
                ):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission}' required"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator
```

---

## 4. Monitoring & Observability

### 4.1 Comprehensive Monitoring Stack

#### Prometheus Configuration
```yaml
# monitoring/prometheus/config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "legal_chatbot_rules.yml"

scrape_configs:
  - job_name: 'legal-chatbot'
    static_configs:
      - targets: ['legal-chatbot-service:80']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'nginx-ingress'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - ingress-nginx
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Custom Metrics Implementation
```python
# src/app/core/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps

# Application metrics
CHAT_REQUESTS_TOTAL = Counter(
    'legal_chatbot_chat_requests_total',
    'Total number of chat requests',
    ['user_role', 'endpoint']
)

CHAT_RESPONSE_TIME = Histogram(
    'legal_chatbot_response_time_seconds',
    'Chat response time in seconds',
    ['tool_used', 'complexity']
)

ACTIVE_CONVERSATIONS = Gauge(
    'legal_chatbot_active_conversations',
    'Number of active conversations'
)

TOOL_USAGE_TOTAL = Counter(
    'legal_chatbot_tool_usage_total',
    'Total tool usage count',
    ['tool_name', 'success']
)

REASONING_STEPS = Histogram(
    'legal_chatbot_reasoning_steps',
    'Number of reasoning steps per query',
    buckets=[1, 2, 3, 5, 8, 10, 15, 20]
)

LLM_TOKEN_USAGE = Counter(
    'legal_chatbot_llm_tokens_total',
    'Total LLM tokens used',
    ['provider', 'model', 'type']  # type: input/output
)

ERROR_RATE = Counter(
    'legal_chatbot_errors_total',
    'Total number of errors',
    ['error_type', 'endpoint']
)

# Application info
APP_INFO = Info(
    'legal_chatbot_app',
    'Legal Chatbot application info'
)

class MetricsCollector:
    @staticmethod
    def track_chat_request(user_role: str, endpoint: str):
        """Track chat request"""
        CHAT_REQUESTS_TOTAL.labels(user_role=user_role, endpoint=endpoint).inc()
    
    @staticmethod
    def track_response_time(tool_used: str, complexity: str, duration: float):
        """Track response time"""
        CHAT_RESPONSE_TIME.labels(tool_used=tool_used, complexity=complexity).observe(duration)
    
    @staticmethod
    def track_tool_usage(tool_name: str, success: bool):
        """Track tool usage"""
        TOOL_USAGE_TOTAL.labels(tool_name=tool_name, success=success).inc()
    
    @staticmethod
    def track_reasoning_steps(steps: int):
        """Track reasoning steps"""
        REASONING_STEPS.observe(steps)
    
    @staticmethod
    def track_llm_tokens(provider: str, model: str, input_tokens: int, output_tokens: int):
        """Track LLM token usage"""
        LLM_TOKEN_USAGE.labels(provider=provider, model=model, type="input").inc(input_tokens)
        LLM_TOKEN_USAGE.labels(provider=provider, model=model, type="output").inc(output_tokens)
    
    @staticmethod
    def track_error(error_type: str, endpoint: str):
        """Track errors"""
        ERROR_RATE.labels(error_type=error_type, endpoint=endpoint).inc()

def monitor_performance(tool_name: str = None, complexity: str = "medium"):
    """Decorator for performance monitoring"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                MetricsCollector.track_error(type(e).__name__, func.__name__)
                raise
            finally:
                duration = time.time() - start_time
                MetricsCollector.track_response_time(
                    tool_name or func.__name__, 
                    complexity, 
                    duration
                )
                if tool_name:
                    MetricsCollector.track_tool_usage(tool_name, success)
        
        return wrapper
    return decorator
```

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Legal Chatbot Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(legal_chatbot_chat_requests_total[5m])",
            "legendFormat": "{{user_role}} - {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time P95",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(legal_chatbot_response_time_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Active Conversations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "legal_chatbot_active_conversations"
          }
        ]
      },
      {
        "title": "Tool Usage Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "increase(legal_chatbot_tool_usage_total[1h])",
            "legendFormat": "{{tool_name}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(legal_chatbot_errors_total[5m])",
            "legendFormat": "{{error_type}}"
          }
        ]
      }
    ]
  }
}
```

### 4.2 Alerting Configuration

```yaml
# monitoring/alertmanager/rules.yaml
groups:
  - name: legal_chatbot_alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(legal_chatbot_response_time_seconds_bucket[5m])) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: HighErrorRate
        expr: rate(legal_chatbot_errors_total[5m]) > 0.1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors/second"

      - alert: LowCacheHitRate
        expr: rate(redis_cache_hits_total[5m]) / rate(redis_cache_requests_total[5m]) < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is {{ $value }}"

      - alert: DatabaseConnectionHigh
        expr: postgres_active_connections > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connections"
          description: "{{ $value }} active database connections"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} is restarting frequently"
```

---

## 5. Performance Optimization Final

### 5.1 Database Optimization

```sql
-- Database optimization scripts
-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_created 
ON conversations(user_id, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_created 
ON messages(conversation_id, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_reasoning_gin 
ON messages USING gin(reasoning_steps);

-- Optimize pgvector queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_cosine 
ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Partitioning for large tables
CREATE TABLE messages_y2025m10 PARTITION OF messages
FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
SELECT pg_reload_conf();
```

### 5.2 Application Performance Optimization

```python
# src/app/core/performance/optimizer.py
import asyncio
from functools import lru_cache
from typing import List, Dict, Any
import aioredis
import time

class PerformanceOptimizer:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.request_cache = {}
    
    @lru_cache(maxsize=1000)
    def get_cached_prompt_template(self, template_name: str) -> str:
        """Cache prompt templates in memory"""
        # Load from database or file
        return self._load_template(template_name)
    
    async def batch_tool_execution(self, tool_calls: List[Dict]) -> List[Any]:
        """Execute multiple tools in parallel"""
        tasks = []
        
        for tool_call in tool_calls:
            if self._can_run_parallel(tool_call):
                tasks.append(self._execute_tool_async(tool_call))
            else:
                # Sequential execution for dependent tools
                if tasks:
                    await asyncio.gather(*tasks)
                    tasks = []
                await self._execute_tool_async(tool_call)
        
        if tasks:
            return await asyncio.gather(*tasks)
    
    async def optimize_llm_calls(self, messages: List[Dict]) -> Dict:
        """Optimize LLM API calls"""
        # Combine multiple short messages
        if self._should_combine_messages(messages):
            combined_message = self._combine_messages(messages)
            return await self._call_llm_optimized(combined_message)
        
        # Use different models for different complexities
        complexity = self._assess_complexity(messages)
        model = self._select_optimal_model(complexity)
        
        return await self._call_llm_with_model(messages, model)
    
    async def cache_frequent_queries(self, query: str, result: Any, ttl: int = 3600):
        """Cache frequent legal queries"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"query_cache:{query_hash}"
        
        await self.redis.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str)
        )
    
    async def preload_common_data(self):
        """Preload commonly accessed data"""
        # Preload frequent legal documents
        await self._preload_popular_documents()
        
        # Preload user contexts
        await self._preload_active_user_contexts()
        
        # Warm up model connections
        await self._warmup_llm_connections()

class ConnectionPoolOptimizer:
    """Optimize database and external API connections"""
    
    def __init__(self):
        self.db_pool_config = {
            'min_size': 5,
            'max_size': 20,
            'max_queries': 50000,
            'max_inactive_connection_lifetime': 300
        }
        
        self.redis_pool_config = {
            'max_connections': 50,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
    
    async def optimize_database_pool(self):
        """Optimize database connection pool"""
        # Monitor connection usage
        active_connections = await self._get_active_connections()
        
        if active_connections > self.db_pool_config['max_size'] * 0.8:
            # Scale up pool size
            self.db_pool_config['max_size'] = min(
                self.db_pool_config['max_size'] + 5, 50
            )
        
        return self.db_pool_config
```

---

## 6. User Acceptance Testing

### 6.1 UAT Test Cases

```python
# tests/uat/test_user_acceptance.py
import pytest
import asyncio
from typing import List, Dict

class TestUserAcceptance:
    """User Acceptance Test scenarios"""
    
    @pytest.mark.uat
    async def test_judge_workflow(self):
        """Test judge's typical workflow"""
        # Scenario: Judge researching case precedents
        
        # Step 1: Login as judge
        auth_response = await self.login_as_judge()
        assert auth_response.status_code == 200
        
        # Step 2: Start new conversation
        conversation = await self.start_conversation(
            "Analisis preseden untuk kasus perceraian dengan kekerasan domestik"
        )
        assert conversation['conversation_id'] is not None
        
        # Step 3: Ask complex legal question
        response = await self.ask_question(
            conversation['conversation_id'],
            "Apa preseden terkuat untuk membuktikan KDRT dalam perkara perceraian?"
        )
        
        # Assertions
        assert response['processing_time'] < 8.0  # Within performance target
        assert len(response['citations']) >= 3    # Adequate citations
        assert response['confidence_score'] > 0.8 # High confidence
        assert 'PrecedentExplorer' in [tc['tool_name'] for tc in response['tool_calls']]
        
        # Step 4: Follow-up question
        followup = await self.ask_question(
            conversation['conversation_id'],
            "Bandingkan dengan putusan serupa di 5 tahun terakhir"
        )
        
        assert 'CaseComparator' in [tc['tool_name'] for tc in followup['tool_calls']]
        
        # Step 5: Export citations
        citations = await self.export_citations(
            conversation['conversation_id'], 
            format="ma_indonesia"
        )
        assert len(citations) > 0
    
    @pytest.mark.uat
    async def test_lawyer_workflow(self):
        """Test lawyer's typical workflow"""
        # Scenario: Lawyer preparing legal brief
        
        auth_response = await self.login_as_lawyer()
        
        # Research phase
        research_query = "Dasar hukum gugatan wanprestasi kontrak bisnis"
        response = await self.ask_question(None, research_query)
        
        # Should use multiple tools
        tools_used = [tc['tool_name'] for tc in response['tool_calls']]
        assert 'LegalRetriever' in tools_used
        assert 'LawLookup' in tools_used
        
        # Comparative analysis
        comparison_query = "Bandingkan pendekatan MA terhadap wanprestasi vs perbuatan melawan hukum"
        comp_response = await self.ask_question(
            response['conversation_id'], 
            comparison_query
        )
        
        assert comp_response['reasoning_steps'] is not None
        assert len(comp_response['reasoning_steps']) >= 4  # Multi-step reasoning
    
    @pytest.mark.uat
    async def test_public_user_workflow(self):
        """Test public user's typical workflow"""
        # Scenario: Public user seeking legal information
        
        auth_response = await self.login_as_public()
        
        # Simple legal question
        simple_query = "Bagaimana cara mengurus surat cerai?"
        response = await self.ask_question(None, simple_query)
        
        # Should be accessible and understandable
        assert response['processing_time'] < 6.0  # Faster for simple queries
        assert len(response['answer'].split()) > 50  # Detailed explanation
        
        # Should not have access to advanced tools
        tools_used = [tc['tool_name'] for tc in response['tool_calls']]
        assert 'CaseComparator' not in tools_used  # Restricted tool
    
    @pytest.mark.uat
    async def test_system_load_handling(self):
        """Test system behavior under load"""
        # Simulate 20 concurrent users
        async def simulate_user():
            response = await self.ask_question(
                None, 
                "Apa itu hukum pidana?"
            )
            return response['processing_time']
        
        tasks = [simulate_user() for _ in range(20)]
        response_times = await asyncio.gather(*tasks)
        
        # Performance assertions
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 10.0  # Average within target
        assert max(response_times) < 15.0  # No response exceeds 15s
        
        # All requests should succeed
        assert len(response_times) == 20
    
    @pytest.mark.uat
    async def test_conversation_memory(self):
        """Test conversation memory and context"""
        auth_response = await self.login_as_judge()
        
        # Start conversation
        q1 = "Jelaskan tentang hukum perdata Indonesia"
        r1 = await self.ask_question(None, q1)
        conversation_id = r1['conversation_id']
        
        # Follow-up that requires context
        q2 = "Apa perbedaannya dengan hukum pidana?"
        r2 = await self.ask_question(conversation_id, q2)
        
        # Should maintain context
        assert 'perdata' in r2['answer'].lower()
        assert conversation_id == r2['conversation_id']
        
        # Check conversation history
        history = await self.get_conversation_history(conversation_id)
        assert len(history['messages']) >= 4  # 2 questions + 2 responses

class PerformanceUAT:
    """Performance-focused UAT scenarios"""
    
    @pytest.mark.performance
    async def test_response_time_requirements(self):
        """Test response time requirements by user type"""
        
        test_cases = [
            ("judge", "Analisis mendalam kasus korupsi", 8.0),
            ("lawyer", "Preseden hukum kontrak", 6.0),
            ("public", "Cara mengurus KTP", 4.0)
        ]
        
        for user_type, query, max_time in test_cases:
            auth_response = await self.login_as(user_type)
            
            start_time = time.time()
            response = await self.ask_question(None, query)
            actual_time = time.time() - start_time
            
            assert actual_time < max_time, f"{user_type} query took {actual_time}s, expected < {max_time}s"
    
    @pytest.mark.performance
    async def test_concurrent_conversation_handling(self):
        """Test handling multiple conversations simultaneously"""
        
        async def conversation_session(user_id: str):
            """Simulate a complete conversation session"""
            auth = await self.login_with_user_id(user_id)
            
            conversation_id = None
            for i in range(5):  # 5 questions per conversation
                query = f"Legal question {i+1} from user {user_id}"
                response = await self.ask_question(conversation_id, query)
                conversation_id = response['conversation_id']
                
                # Random delay between questions
                await asyncio.sleep(random.uniform(1, 3))
            
            return conversation_id
        
        # Run 10 concurrent conversation sessions
        user_ids = [f"user_{i}" for i in range(10)]
        tasks = [conversation_session(uid) for uid in user_ids]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # All conversations should complete successfully
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Should complete within reasonable time
        assert total_time < 120  # 2 minutes for all conversations
```

### 6.2 UAT Performance Benchmarks

```python
# tests/uat/benchmarks.py
import pytest
import asyncio
import statistics
from typing import List, Dict

class LegalChatbotBenchmarks:
    """Performance benchmarks for UAT"""
    
    @pytest.mark.benchmark
    async def test_tool_performance_benchmarks(self):
        """Benchmark individual tool performance"""
        
        benchmarks = {
            'LegalRetriever': {
                'query': 'putusan mahkamah agung korupsi',
                'max_time': 3.0,
                'min_results': 5
            },
            'CaseComparator': {
                'query': 'bandingkan putusan korupsi 2023',
                'max_time': 8.0,
                'min_results': 2
            },
            'PrecedentExplorer': {
                'query': 'preseden hukum perdata',
                'max_time': 5.0,
                'min_results': 3
            }
        }
        
        results = {}
        
        for tool_name, benchmark in benchmarks.items():
            # Run tool 10 times
            times = []
            for _ in range(10):
                start_time = time.time()
                result = await self.run_tool_directly(tool_name, benchmark['query'])
                end_time = time.time()
                
                times.append(end_time - start_time)
                assert len(result) >= benchmark['min_results']
            
            # Statistical analysis
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
            
            results[tool_name] = {
                'avg_time': avg_time,
                'p95_time': p95_time,
                'max_time': max(times),
                'min_time': min(times)
            }
            
            # Assertions
            assert avg_time < benchmark['max_time']
            assert p95_time < benchmark['max_time'] * 1.2
        
        # Print benchmark results
        self._print_benchmark_results(results)
    
    @pytest.mark.benchmark
    async def test_end_to_end_latency(self):
        """Test end-to-end response latency"""
        
        query_complexities = {
            'simple': [
                "Apa itu hukum pidana?",
                "Bagaimana cara mengurus KTP?",
                "Apa itu perceraian?"
            ],
            'medium': [
                "Jelaskan prosedur gugatan perdata",
                "Apa perbedaan hukum pidana dan perdata?",
                "Bagaimana cara mengajukan banding?"
            ],
            'complex': [
                "Analisis preseden kasus korupsi dengan membandingkan 3 putusan terakhir",
                "Bandingkan pendekatan MA terhadap kasus KDRT sebelum dan sesudah UU TPKS",
                "Bagaimana evolusi yurisprudensi MA tentang wanprestasi kontrak bisnis?"
            ]
        }
        
        latency_targets = {
            'simple': 4.0,
            'medium': 6.0,
            'complex': 10.0
        }
        
        results = {}
        
        for complexity, queries in query_complexities.items():
            times = []
            
            for query in queries:
                # Run each query 3 times
                for _ in range(3):
                    start_time = time.time()
                    response = await self.ask_question(None, query)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                    
                    # Verify response quality
                    assert response['answer'] is not None
                    assert len(response['answer']) > 100  # Substantial answer
            
            avg_latency = statistics.mean(times)
            p95_latency = statistics.quantiles(times, n=20)[18]
            
            results[complexity] = {
                'avg_latency': avg_latency,
                'p95_latency': p95_latency,
                'target': latency_targets[complexity]
            }
            
            # Assertions
            assert avg_latency < latency_targets[complexity]
            assert p95_latency < latency_targets[complexity] * 1.3
        
        return results
```

---

## 7. Go-Live Preparation

### 7.1 Pre-Go-Live Checklist

#### Technical Readiness
- [ ] All production infrastructure provisioned (GKE, PostgreSQL, Redis)
- [ ] SSL/TLS certificates configured dan valid
- [ ] Load balancer dan autoscaling configured
- [ ] Monitoring dan alerting fully operational
- [ ] Backup dan disaster recovery procedures tested
- [ ] Security scanning completed (SAST, DAST, container scanning)
- [ ] Performance benchmarks met
- [ ] UAT completed dan signed off

#### Operational Readiness
- [ ] Production deployment runbook completed
- [ ] Rollback procedures documented dan tested
- [ ] Support escalation procedures defined
- [ ] Production access controls configured
- [ ] Log aggregation dan analysis tools configured
- [ ] Capacity planning completed
- [ ] Cost monitoring dan budgets set up

#### Business Readiness
- [ ] User training materials completed
- [ ] API documentation published
- [ ] Legal compliance review completed
- [ ] Terms of service dan privacy policy updated
- [ ] Stakeholder communication plan executed
- [ ] Launch announcement prepared

### 7.2 Production Launch Plan

#### Phase 1: Soft Launch (Week 1)
- **Scope:** Limited user group (10-20 internal users)
- **Monitoring:** 24/7 monitoring dengan immediate response
- **Goals:** Verify production stability, identify any critical issues

#### Phase 2: Controlled Rollout (Week 2) 
- **Scope:** Expand to 100 selected users (judges, senior lawyers)
- **Monitoring:** Continuous monitoring dengan daily reviews
- **Goals:** Validate performance under realistic load

#### Phase 3: Public Launch (Week 3)
- **Scope:** Full public availability
- **Monitoring:** Standard production monitoring
- **Goals:** Achieve full production capability

### 7.3 Post-Launch Support Plan

#### Immediate Support (First 30 days)
- **Response Time:** < 2 hours for critical issues
- **Coverage:** 24/7 monitoring dan support
- **Team:** Dedicated launch support team
- **Focus:** Stability, performance, user adoption

#### Ongoing Support
- **Response Time:** < 4 hours for critical, < 24 hours for normal
- **Coverage:** Business hours primary, 24/7 for critical systems
- **Team:** Standard operations team
- **Focus:** Feature requests, optimization, maintenance

---

## 8. Sprint 3 Deliverables

### 8.1 Production Infrastructure
- [ ] GKE cluster dengan auto-scaling configured
- [ ] PostgreSQL cluster dengan high availability
- [ ] Redis cluster untuk caching dan sessions
- [ ] Complete monitoring stack (Prometheus, Grafana, AlertManager)
- [ ] Security hardening (NetworkPolicies, PodSecurityPolicies, Vault)

### 8.2 Production Application
- [ ] Production-ready Docker images dengan security scanning
- [ ] Complete CI/CD pipeline dengan automated testing
- [ ] API rate limiting dan authentication enhancements  
- [ ] Comprehensive error handling dan logging
- [ ] Performance optimizations implemented

### 8.3 Documentation & Training
- [ ] Production deployment guide
- [ ] Operations runbook
- [ ] User training materials
- [ ] API documentation complete
- [ ] Troubleshooting guide

---

## 9. Success Criteria Sprint 3

### 9.1 Production Readiness Criteria
- ✅ System successfully deployed to production environment
- ✅ All monitoring dan alerting systems operational
- ✅ Security audit passed dengan no critical vulnerabilities
- ✅ Performance benchmarks met under production load
- ✅ UAT completed dengan 95%+ test case pass rate

### 9.2 Business Success Criteria
- ✅ System available 99.9% uptime during launch period
- ✅ Average response time < 6 seconds under normal load
- ✅ User satisfaction score > 4.0/5.0 in initial feedback
- ✅ No critical security incidents during launch
- ✅ Support ticket resolution within SLA targets

---

**Sprint 3 Document Version:**
- v1.0 (2025-09-16): Production deployment dan optimization plan
- Target completion: Go-live ready after 2 weeks