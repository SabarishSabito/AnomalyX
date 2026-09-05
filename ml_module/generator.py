import time
import random
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any

class TelemetryGenerator:
    """
    Generates realistic multi-metric telemetry streams (Server Metrics, Financial Streams, IoT Logs)
    with controllable synthetic anomaly injections for real-time demonstration.
    """
    def __init__(self, category: str = "server"):
        self.category = category
        self.step_count = 0
        self._init_baseline()

    def _init_baseline(self):
        """Initializes baseline state for continuous time-series stream."""
        if self.category == "server":
            self.cpu = 45.0
            self.memory = 60.0
            self.network = 350.0
            self.latency = 22.0
            self.error_rate = 0.2
        elif self.category == "financial":
            self.amount = 120.0
            self.tx_frequency = 5.0
            self.risk_score = 12.0
            self.location_delta = 2.0
        else: # IoT / Industrial
            self.temperature = 65.0
            self.vibration = 1.2
            self.pressure = 101.3
            self.power_draw = 450.0

    def generate_next_frame(self, force_anomaly: bool = False, anomaly_prob: float = 0.10) -> Dict[str, Any]:
        """Generates the next time-series frame."""
        self.step_count += 1
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        # Determine if this frame is an anomaly
        is_injected_anomaly = force_anomaly or (random.random() < anomaly_prob)

        if self.category == "server":
            # Smooth noise random walk
            self.cpu = max(10.0, min(95.0, self.cpu + np.random.normal(0, 3.0)))
            self.memory = max(20.0, min(95.0, self.memory + np.random.normal(0.2, 1.5))) # slow memory creep
            self.network = max(50.0, min(2000.0, self.network + np.random.normal(0, 25.0)))
            self.latency = max(5.0, min(100.0, self.latency + np.random.normal(0, 2.0)))
            self.error_rate = max(0.0, min(2.0, self.error_rate + np.random.normal(0, 0.1)))

            # Inject point anomaly spike
            if is_injected_anomaly:
                anomaly_type = random.choice(["cpu_spike", "memory_leak", "ddos_attack", "latency_timeout"])
                if anomaly_type == "cpu_spike":
                    self.cpu = float(random.uniform(92.0, 99.8))
                elif anomaly_type == "memory_leak":
                    self.memory = float(random.uniform(95.0, 99.9))
                elif anomaly_type == "ddos_attack":
                    self.network = float(random.uniform(1800.0, 3500.0))
                    self.error_rate = float(random.uniform(8.5, 25.0))
                elif anomaly_type == "latency_timeout":
                    self.latency = float(random.uniform(450.0, 1200.0))
                    self.error_rate = float(random.uniform(5.0, 18.0))

            return {
                "timestamp": now_str,
                "step": self.step_count,
                "cpu_usage": round(float(self.cpu), 2),
                "memory_usage": round(float(self.memory), 2),
                "network_io": round(float(self.network), 2),
                "latency_ms": round(float(self.latency), 2),
                "error_rate": round(float(self.error_rate), 2)
            }

        elif self.category == "financial":
            self.amount = max(5.0, min(500.0, self.amount + np.random.normal(0, 15.0)))
            self.tx_frequency = max(1.0, min(10.0, self.tx_frequency + np.random.normal(0, 0.5)))
            self.risk_score = max(0.0, min(30.0, self.risk_score + np.random.normal(0, 2.0)))
            self.location_delta = max(0.0, min(10.0, self.location_delta + np.random.normal(0, 0.5)))

            if is_injected_anomaly:
                self.amount = float(random.uniform(5000.0, 25000.0))
                self.tx_frequency = float(random.uniform(45.0, 120.0))
                self.risk_score = float(random.uniform(85.0, 99.0))
                self.location_delta = float(random.uniform(500.0, 3000.0))

            return {
                "timestamp": now_str,
                "step": self.step_count,
                "transaction_amount": round(float(self.amount), 2),
                "tx_frequency_per_min": round(float(self.tx_frequency), 2),
                "risk_score": round(float(self.risk_score), 2),
                "location_delta_km": round(float(self.location_delta), 2)
            }

        else: # IoT
            self.temperature = max(40.0, min(80.0, self.temperature + np.random.normal(0, 1.0)))
            self.vibration = max(0.5, min(3.0, self.vibration + np.random.normal(0, 0.1)))
            self.pressure = max(95.0, min(105.0, self.pressure + np.random.normal(0, 0.5)))
            self.power_draw = max(300.0, min(600.0, self.power_draw + np.random.normal(0, 10.0)))

            if is_injected_anomaly:
                self.temperature = float(random.uniform(105.0, 140.0))
                self.vibration = float(random.uniform(8.5, 18.0))
                self.power_draw = float(random.uniform(1400.0, 2500.0))

            return {
                "timestamp": now_str,
                "step": self.step_count,
                "temperature_celsius": round(float(self.temperature), 2),
                "vibration_g": round(float(self.vibration), 2),
                "pressure_kpa": round(float(self.pressure), 2),
                "power_draw_watts": round(float(self.power_draw), 2)
            }

    def generate_batch(self, num_samples: int = 100, anomaly_rate: float = 0.08) -> pd.DataFrame:
        """Generates a historical batch DataFrame with synthetic anomalies."""
        records = []
        for i in range(num_samples):
            frame = self.generate_next_frame(anomaly_prob=anomaly_rate)
            records.append(frame)
        return pd.DataFrame(records)
