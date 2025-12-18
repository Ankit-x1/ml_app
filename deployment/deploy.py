"""
Production Deployment Configuration

Deployment scripts and configurations for industrial production environments.
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import docker
import subprocess
import os
from enum import Enum


class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentMode(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""

    environment: EnvironmentType
    mode: DeploymentMode
    version: str
    replicas: int
    resources: Dict[str, Any]
    network: Dict[str, Any]
    storage: Dict[str, Any]
    security: Dict[str, Any]
    monitoring: Dict[str, Any]
    scaling: Dict[str, Any]
    backup: Dict[str, Any]


@dataclass
class ResourceLimits:
    """Resource limits configuration"""

    cpu: str
    memory: str
    gpu: Optional[str] = None
    disk: Optional[str] = None


class DeploymentManager:
    """Manages deployment of RL systems"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.deployment_config = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("DeploymentManager")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def load_config(self) -> None:
        """Load deployment configuration"""
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        self.deployment_config = DeploymentConfig(**config_data)
        self.logger.info(
            f"Loaded configuration for {self.deployment_config.environment}"
        )

    def deploy(self) -> bool:
        """Deploy the system"""
        if not self.deployment_config:
            self.load_config()

        try:
            if self.deployment_config.mode == DeploymentMode.LOCAL:
                return self._deploy_local()
            elif self.deployment_config.mode == DeploymentMode.DOCKER:
                return self._deploy_docker()
            elif self.deployment_config.mode == DeploymentMode.KUBERNETES:
                return self._deploy_kubernetes()
            else:
                raise ValueError(
                    f"Unsupported deployment mode: {self.deployment_config.mode}"
                )

        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            return False

    def _deploy_local(self) -> bool:
        """Deploy locally"""
        self.logger.info("Deploying locally...")

        # Setup environment variables
        env_file = self.config_path.parent / "env.local"
        if env_file.exists():
            self._load_environment_file(env_file)

        # Run health checks
        if self._run_health_checks():
            self.logger.info("Local deployment successful")
            return True

        return False

    def _deploy_docker(self) -> bool:
        """Deploy using Docker"""
        self.logger.info("Deploying with Docker...")

        try:
            client = docker.from_env()

            # Build image
            dockerfile_path = self.config_path.parent
            image_tag = f"rl-robotics:{self.deployment_config.version}"

            client.images.build(path=str(dockerfile_path), tag=image_tag, rm=True)

            # Run containers
            for i in range(self.deployment_config.replicas):
                container_name = f"rl-robotics-{i}"

                container = client.containers.run(
                    image_tag,
                    name=container_name,
                    detach=True,
                    ports={"8000": f"{8000 + i}"},
                    environment=self._get_docker_environment(),
                    volumes={
                        str(self.config_path.parent / "logs"): {
                            "bind": "/app/logs",
                            "mode": "rw",
                        },
                        str(self.config_path.parent / "data"): {
                            "bind": "/app/data",
                            "mode": "rw",
                        },
                    },
                    restart_policy={"Name": "unless-stopped"},
                )

                self.logger.info(f"Started container: {container.id}")

            return True

        except Exception as e:
            self.logger.error(f"Docker deployment failed: {e}")
            return False

    def _deploy_kubernetes(self) -> bool:
        """Deploy using Kubernetes"""
        self.logger.info("Deploying with Kubernetes...")

        try:
            # Generate Kubernetes manifests
            self._generate_kubernetes_manifests()

            # Apply manifests
            kubectl_cmd = [
                "kubectl",
                "apply",
                "-f",
                str(self.config_path.parent / "k8s"),
            ]

            result = subprocess.run(kubectl_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info("Kubernetes deployment successful")
                return True
            else:
                self.logger.error(f"Kubernetes deployment failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Kubernetes deployment failed: {e}")
            return False

    def _load_environment_file(self, env_file: Path) -> None:
        """Load environment variables from file"""
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value

    def _get_docker_environment(self) -> Dict[str, str]:
        """Get environment variables for Docker"""
        env_vars = {
            "PYTHONPATH": "/app",
            "LOG_LEVEL": "INFO",
            "ENVIRONMENT": self.deployment_config.environment.value,
        }

        # Add security environment variables
        security_vars = self.deployment_config.security.get("env_vars", {})
        env_vars.update(security_vars)

        return env_vars

    def _run_health_checks(self) -> bool:
        """Run deployment health checks"""
        # Implement health check logic
        return True

    def _generate_kubernetes_manifests(self) -> None:
        """Generate Kubernetes manifests"""
        k8s_dir = self.config_path.parent / "k8s"
        k8s_dir.mkdir(exist_ok=True)

        # Generate deployment manifest
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "rl-robotics", "namespace": "production"},
            "spec": {
                "replicas": self.deployment_config.replicas,
                "selector": {"matchLabels": {"app": "rl-robotics"}},
                "template": {
                    "metadata": {"labels": {"app": "rl-robotics"}},
                    "spec": {
                        "containers": [
                            {
                                "name": "rl-robotics",
                                "image": f"rl-robotics:{self.deployment_config.version}",
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "requests": self.deployment_config.resources.get(
                                        "requests", {}
                                    ),
                                    "limits": self.deployment_config.resources.get(
                                        "limits", {}
                                    ),
                                },
                                "env": self._generate_kubernetes_env(),
                            }
                        ]
                    },
                },
            },
        }

        # Write deployment manifest
        with open(k8s_dir / "deployment.yaml", "w") as f:
            yaml.dump(deployment_manifest, f)

        # Generate service manifest
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "rl-robotics-service", "namespace": "production"},
            "spec": {
                "selector": {"app": "rl-robotics"},
                "ports": [{"protocol": "TCP", "port": 80, "targetPort": 8000}],
                "type": "LoadBalancer",
            },
        }

        # Write service manifest
        with open(k8s_dir / "service.yaml", "w") as f:
            yaml.dump(service_manifest, f)

    def _generate_kubernetes_env(self) -> List[Dict[str, str]]:
        """Generate environment variables for Kubernetes"""
        env_vars = [
            {"name": "PYTHONPATH", "value": "/app"},
            {"name": "LOG_LEVEL", "value": "INFO"},
            {"name": "ENVIRONMENT", "value": self.deployment_config.environment.value},
        ]

        # Add security environment variables
        security_vars = self.deployment_config.security.get("env_vars", {})
        for key, value in security_vars.items():
            env_vars.append(
                {
                    "name": key,
                    "valueFrom": {
                        "secretKeyRef": {"name": "rl-robotics-secrets", "key": key}
                    },
                }
            )

        return env_vars


class MonitoringManager:
    """Manages monitoring and observability"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("MonitoringManager")
        logger.setLevel(logging.INFO)
        return logger

    def setup_monitoring(self) -> bool:
        """Setup monitoring infrastructure"""
        try:
            # Setup Prometheus
            self._setup_prometheus()

            # Setup Grafana
            self._setup_grafana()

            # Setup log aggregation
            self._setup_log_aggregation()

            self.logger.info("Monitoring setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            return False

    def _setup_prometheus(self) -> None:
        """Setup Prometheus monitoring"""
        prometheus_config = {
            "global": {"scrape_interval": "15s"},
            "scrape_configs": [
                {
                    "job_name": "rl-robotics",
                    "static_configs": [{"targets": ["localhost:8000"]}],
                }
            ],
        }

        config_dir = self.config_path.parent / "monitoring" / "prometheus"
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(config_dir / "prometheus.yml", "w") as f:
            yaml.dump(prometheus_config, f)

    def _setup_grafana(self) -> None:
        """Setup Grafana dashboards"""
        dashboard_config = {
            "dashboard": {
                "title": "RL Robotics Metrics",
                "panels": [
                    {
                        "title": "Training Progress",
                        "type": "graph",
                        "targets": [{"expr": "rate(training_episodes_total[5m])"}],
                    },
                    {
                        "title": "Inference Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(inference_latency_seconds_bucket[5m]))"
                            }
                        ],
                    },
                ],
            }
        }

        config_dir = self.config_path.parent / "monitoring" / "grafana"
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(config_dir / "dashboard.json", "w") as f:
            json.dump(dashboard_config, f)

    def _setup_log_aggregation(self) -> None:
        """Setup log aggregation"""
        log_config = {
            "version": 1,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
                "file": {
                    "level": "INFO",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "/app/logs/rl-robotics.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "formatter": "standard",
                },
            },
            "loggers": {
                "rl_robotics": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                }
            },
        }

        config_dir = self.config_path.parent / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        with open(config_dir / "logging.yaml", "w") as f:
            yaml.dump(log_config, f)


class SecurityManager:
    """Manages security configurations"""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("SecurityManager")
        logger.setLevel(logging.INFO)
        return logger

    def setup_security(self) -> bool:
        """Setup security configurations"""
        try:
            # Generate secrets
            self._generate_secrets()

            # Setup RBAC
            self._setup_rbac()

            # Configure network policies
            self._setup_network_policies()

            self.logger.info("Security setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Security setup failed: {e}")
            return False

    def _generate_secrets(self) -> None:
        """Generate application secrets"""
        import secrets

        app_secrets = {
            "SECRET_KEY": secrets.token_urlsafe(32),
            "DB_PASSWORD": secrets.token_urlsafe(16),
            "API_KEY": secrets.token_urlsafe(24),
        }

        secrets_dir = self.config_path.parent / "secrets"
        secrets_dir.mkdir(exist_ok=True)

        with open(secrets_dir / "app-secrets.yaml", "w") as f:
            yaml.dump(app_secrets, f)

    def _setup_rbac(self) -> None:
        """Setup Role-Based Access Control"""
        rbac_config = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {"name": "rl-robotics-role", "namespace": "production"},
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods", "services"],
                    "verbs": ["get", "list", "watch"],
                }
            ],
        }

        k8s_dir = self.config_path.parent / "k8s" / "rbac"
        k8s_dir.mkdir(parents=True, exist_ok=True)

        with open(k8s_dir / "role.yaml", "w") as f:
            yaml.dump(rbac_config, f)

    def _setup_network_policies(self) -> None:
        """Setup network policies"""
        network_policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": "rl-robotics-network-policy",
                "namespace": "production",
            },
            "spec": {
                "podSelector": {"matchLabels": {"app": "rl-robotics"}},
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"name": "monitoring"}
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": 8000}],
                    }
                ],
            },
        }

        k8s_dir = self.config_path.parent / "k8s" / "network"
        k8s_dir.mkdir(parents=True, exist_ok=True)

        with open(k8s_dir / "network-policy.yaml", "w") as f:
            yaml.dump(network_policy, f)


def create_deployment_configs(base_path: str) -> None:
    """Create deployment configuration files"""
    base_path = Path(base_path)

    # Development config
    dev_config = {
        "environment": "development",
        "mode": "local",
        "version": "1.0.0",
        "replicas": 1,
        "resources": {
            "requests": {"cpu": "500m", "memory": "512Mi"},
            "limits": {"cpu": "1000m", "memory": "1Gi"},
        },
        "network": {"port": 8000},
        "storage": {"type": "local", "path": "./data"},
        "security": {"auth_required": False},
        "monitoring": {"enabled": True},
        "scaling": {"min_replicas": 1, "max_replicas": 1},
        "backup": {"enabled": False},
    }

    # Production config
    prod_config = {
        "environment": "production",
        "mode": "kubernetes",
        "version": "1.0.0",
        "replicas": 3,
        "resources": {
            "requests": {"cpu": "2000m", "memory": "4Gi", "gpu": "1"},
            "limits": {"cpu": "4000m", "memory": "8Gi", "gpu": "1"},
        },
        "network": {"port": 80, "tls": True},
        "storage": {"type": "persistent", "size": "100Gi"},
        "security": {
            "auth_required": True,
            "env_vars": {"SECRET_KEY": "secret-key-name"},
        },
        "monitoring": {"enabled": True, "prometheus": True, "grafana": True},
        "scaling": {"min_replicas": 2, "max_replicas": 10, "target_cpu": 70},
        "backup": {"enabled": True, "schedule": "0 2 * * *"},
    }

    config_dir = base_path / "config"
    config_dir.mkdir(exist_ok=True)

    with open(config_dir / "development.yaml", "w") as f:
        yaml.dump(dev_config, f)

    with open(config_dir / "production.yaml", "w") as f:
        yaml.dump(prod_config, f)
