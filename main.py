import asyncio
import json
import logging
import os
import secrets
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Literal, Optional, Set

import numpy as np
import websockets
from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================================================================
# CONFIGURACIÓN Y SEGURIDAD (AISLAMIENTO ESTRICTO - ISO 27001)
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("quant-engine-core")

ENGINE_SECRET = os.getenv("ENGINE_SECRET", "CHANGE_ME_IN_RENDER_ENV_VARS")
TREASURY_ADDRESS = os.getenv("TREASURY_ADDRESS", "0x0000000000000000000000000000000000000000")
ALLOWED_ORIGIN = os.getenv("HUD_ORIGIN", "*")

# Nodos WSS de Polygon PoS con failover automático
POLYGON_WSS_NODES = [
    os.getenv("POLYGON_WSS_RPC", "wss://polygon-bor-rpc.publicnode.com"),
    "wss://polygon.drpc.org",
]
TARGET_POOL_ADDRESS = os.getenv("TARGET_POOL_ADDRESS", "0x6e7a5FAFcec6BB1e78bAE2A1F0B612012BF14827").lower()
SYNC_EVENT_TOPIC = "0x1c411e9a96e071241c2f21f7726b17ae89e3cab4c78be50e062b03a9fffbbad1"

MAX_POSITION_PCT = 0.15       # Tope máximo de exposición por operación (15%)
BASE_KELLY_FRACTION = 0.25    # Quarter-Kelly base
MC_SIMULATIONS = 2000         # Trayectorias Monte Carlo por tick
MC_HORIZON_STEPS = 12         # Pasos futuros estimados
WINDOW_SIZE = 120             # Ventana deslizante de retornos logarítmicos


# ==============================================================================
# PERFILES DE LAS 7 WALLETS OPERATIVAS (RABBY)
# ==============================================================================
@dataclass
class WalletStrategyProfile:
    slot_id: int
    name: str
    strategy_tag: str
    address: str
    allocated_capital_usdc: float
    current_equity_usdc: float
    kelly_multiplier: float
    max_slippage_bps: int
    min_edge_threshold: float
    status: str = "IDLE"
    active_position_size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl_session: float = 0.0
    trades_count: int = 0


def init_rabby_wallets() -> List[WalletStrategyProfile]:
    profiles_spec = [
        (1, "Rabby-01 | Ultra-Scalp",    "DEX_ARB_MICRO",   1000.0, 0.80, 15, 0.0012),
        (2, "Rabby-02 | Mean-Rev Low",   "MEAN_REVERSION",  1500.0, 0.90, 20, 0.0018),
        (3, "Rabby-03 | Core Quarter",   "MOMENTUM_FLOW",   2500.0, 1.00, 25, 0.0022),
        (4, "Rabby-04 | Vol-Breakout",   "VOL_EXPANSION",   2000.0, 1.10, 35, 0.0030),
        (5, "Rabby-05 | Liquidity-Snipe","POOL_IMBALANCE",  1500.0, 0.85, 30, 0.0025),
        (6, "Rabby-06 | Trend-Guard",    "EMA_CROSS_QUANT", 1000.0, 0.95, 25, 0.0020),
        (7, "Rabby-07 | Tail-Hedge",     "CONVEX_HEDGE",     500.0, 0.60, 40, 0.0040),
    ]
    wallets = []
    for slot, name, tag, cap, k_mult, slip, edge in profiles_spec:
        addr = os.getenv(f"RABBY_WALLET_{slot}_ADDR", f"0xRabby0{slot}...{secrets.token_hex(2)}")
        wallets.append(
            WalletStrategyProfile(
                slot_id=slot,
                name=name,
                strategy_tag=tag,
                address=addr,
                allocated_capital_usdc=cap,
                current_equity_usdc=cap,
                kelly_multiplier=k_mult,
                max_slippage_bps=slip,
                min_edge_threshold=edge,
            )
        )
    return wallets


# ==============================================================================
# NÚCLEO MATEMÁTICO: MONTE CARLO + QUARTER-KELLY + SL/TP DINÁMICO
# ==============================================================================
class QuantMathCore:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.prices: deque[float] = deque(maxlen=window_size + 1)
        self.log_returns: deque[float] = deque(maxlen=window_size)
        self.rng = np.random.default_rng()

    def update_price(self, price: float) -> Optional[dict]:
        if price <= 0:
            return None
        if self.prices:
            prev = self.prices[-1]
            ret = float(np.log(price / prev))
            self.log_returns.append(ret)
        self.prices.append(price)

        if len(self.log_returns) < 20:
            return {
                "price": round(price, 6),
                "sigma": 0.0025,
                "mu": 0.0,
                "var_95": 0.0041,
                "kelly_f_star": 0.0,
                "quarter_kelly": 0.0,
                "suggested_sl": round(price * 0.992, 6),
                "suggested_tp": round(price * 1.015, 6),
                "warmup_progress": f"{len(self.log_returns)}/20"
            }

        returns_arr = np.array(self.log_returns, dtype=np.float64)
        mu = float(np.mean(returns_arr))
        sigma = max(float(np.std(returns_arr, ddof=1)), 1e-6)

        shocks = self.rng.normal(
            loc=(mu - 0.5 * sigma ** 2),
            scale=sigma,
            size=(MC_SIMULATIONS, MC_HORIZON_STEPS)
        )
        terminal_log_returns = np.sum(shocks, axis=1)
        simulated_prices = price * np.exp(terminal_log_returns)

        var_95_price = float(np.percentile(simulated_prices, 5))
        var_95_pct = max(0.0, (price - var_95_price) / price)

        raw_kelly = mu / (sigma ** 2) if mu > 0 else 0.0
        scaled_kelly = np.tanh(raw_kelly * 0.05)
        quarter_kelly = float(np.clip(scaled_kelly * BASE_KELLY_FRACTION, 0.0, MAX_POSITION_PCT))

        horizon_vol = sigma * np.sqrt(MC_HORIZON_STEPS)
        sl_distance = np.clip(2.0 * horizon_vol, 0.002, 0.05)
        tp_distance = np.clip(3.5 * horizon_vol, 0.0035, 0.10)

        return {
            "price": round(price, 6),
            "sigma": round(sigma, 6),
            "mu": round(mu, 7),
            "var_95": round(var_95_pct, 5),
            "kelly_f_star": round(float(scaled_kelly), 4),
            "quarter_kelly": round(quarter_kelly, 4),
            "suggested_sl": round(price * (1.0 - sl_distance), 6),
            "suggested_tp": round(price * (1.0 + tp_distance), 6),
            "warmup_progress": "READY"
        }


# ==============================================================================
# ESTADO GLOBAL Y WORKERS ASÍNCRONOS
# ==============================================================================
class EngineState:
    def __init__(self):
        self.running: bool = False
        self.killed: bool = False
        self.mode: Literal["PAPER", "LIVE"] = "PAPER"
        self.rpc_connected: bool = False
        self.rpc_latency_ms: float = 0.0
        self.last_block: int = 0
        self.ticks_processed: int = 0
        self.math_core = QuantMathCore()
        self.latest_metrics: dict = {
            "price": 0.0, "sigma": 0.0, "mu": 0.0, "var_95": 0.0,
            "kelly_f_star": 0.0, "quarter_kelly": 0.0,
            "suggested_sl": 0.0, "suggested_tp": 0.0, "warmup_progress": "0/20"
        }
        self.wallets: List[WalletStrategyProfile] = init_rabby_wallets()
        self.hud_clients: Set[WebSocket] = set()
        self.bg_tasks: List[asyncio.Task] = []

    def snapshot(self) -> dict:
        total_equity = sum(w.current_equity_usdc for w in self.wallets)
        total_pnl = sum(w.pnl_session for w in self.wallets)
        return {
            "timestamp": int(time.time() * 1000),
            "engine": {
                "running": self.running,
                "killed": self.killed,
                "mode": self.mode,
                "rpc_connected": self.rpc_connected,
                "rpc_latency_ms": round(self.rpc_latency_ms, 2),
                "last_block": self.last_block,
                "ticks_processed": self.ticks_processed,
                "treasury_address": TREASURY_ADDRESS,
                "total_equity_usdc": round(total_equity, 2),
                "total_pnl_usdc": round(total_pnl, 2),
            },
            "quant": self.latest_metrics,
            "wallets": [
                {
                    "slot_id": w.slot_id,
                    "name": w.name,
                    "strategy_tag": w.strategy_tag,
                    "address": w.address,
                    "equity_usdc": round(w.current_equity_usdc, 2),
                    "kelly_mult": w.kelly_multiplier,
                    "max_slippage_bps": w.max_slippage_bps,
                    "status": w.status,
                    "active_pos_usdc": round(w.active_position_size, 2),
                    "sl": w.stop_loss,
                    "tp": w.take_profit,
                    "pnl_session": round(w.pnl_session, 2),
                    "trades": w.trades_count,
                }
                for w in self.wallets
            ],
        }


engine = EngineState()


async def polygon_wss_listener():
    """Suscripción directa de latencia mínima a reservas Sync del pool DEX en Polygon PoS."""
    node_idx = 0
    while True:
        if engine.killed:
            await asyncio.sleep(1.0)
            continue

        current_rpc = POLYGON_WSS_NODES[node_idx % len(POLYGON_WSS_NODES)]
        try:
            t0 = time.perf_counter()
            async with websockets.connect(
                current_rpc,
                ping_interval=15,
                ping_timeout=8,
                max_queue=64
            ) as ws:
                sub_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": [
                        "logs",
                        {
                            "address": TARGET_POOL_ADDRESS,
                            "topics": [SYNC_EVENT_TOPIC]
                        }
                    ]
                }
                await ws.send(json.dumps(sub_payload))
                sub_resp = json.loads(await ws.recv())
                engine.rpc_connected = True
                engine.rpc_latency_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(f"Nodo activo: {current_rpc} | Sub: {sub_resp.get('result')} | {engine.rpc_latency_ms:.1f}ms")

                async for raw_msg in ws:
                    if engine.killed:
                        break
                    msg_t0 = time.perf_counter()
                    data = json.loads(raw_msg)
                    if "params" not in data:
                        continue

                    log_entry = data["params"]["result"]
                    engine.last_block = int(log_entry.get("blockNumber", "0x0"), 16)
                    engine.ticks_processed += 1

                    hex_data = log_entry.get("data", "0x")[2:]
                    if len(hex_data) >= 128:
                        reserve0 = int(hex_data[0:64], 16)
                        reserve1 = int(hex_data[64:128], 16)
                        if reserve0 > 0 and reserve1 > 0:
                            spot_price = (reserve1 / reserve0) * 1e12
                            metrics = engine.math_core.update_price(spot_price)
                            if metrics:
                                engine.latest_metrics = metrics

                    engine.rpc_latency_ms = (time.perf_counter() - msg_t0) * 1000.0

        except Exception as exc:
            engine.rpc_connected = False
            node_idx += 1
            logger.warning(f"Failover RPC ({exc}). Rotando nodo...")
            await asyncio.sleep(1.0)


async def wallet_strategy_worker(wallet: WalletStrategyProfile):
    while True:
        await asyncio.sleep(0.5)

        if engine.killed:
            wallet.status = "KILLED"
            wallet.active_position_size = 0.0
            continue

        if not engine.running:
            if wallet.status not in ("PAUSED", "KILLED"):
                wallet.status = "PAUSED"
            continue

        metrics = engine.latest_metrics
        if metrics.get("warmup_progress") != "READY":
            wallet.status = "WARMUP"
            continue

        price = metrics["price"]
        q_kelly = metrics["quarter_kelly"]

        if wallet.active_position_size > 0:
            wallet.status = "IN_POSITION"
            if price <= wallet.stop_loss or price >= wallet.take_profit:
                pnl_pct = (price - wallet.entry_price) / wallet.entry_price
                realized_pnl = wallet.active_position_size * pnl_pct
                wallet.current_equity_usdc += realized_pnl
                wallet.pnl_session += realized_pnl
                wallet.active_position_size = 0.0
                wallet.entry_price = 0.0
                wallet.status = "SCANNING"
        else:
            wallet.status = "SCANNING"
            effective_kelly = min(MAX_POSITION_PCT, q_kelly * wallet.kelly_multiplier)
            if effective_kelly >= wallet.min_edge_threshold and metrics["mu"] > 0:
                pos_size = wallet.current_equity_usdc * effective_kelly
                if pos_size >= 10.0:
                    wallet.active_position_size = pos_size
                    wallet.entry_price = price
                    wallet.stop_loss = metrics["suggested_sl"]
                    wallet.take_profit = metrics["suggested_tp"]
                    wallet.trades_count += 1
                    wallet.status = "IN_POSITION"


async def telemetry_broadcaster():
    while True:
        await asyncio.sleep(0.25)
        if not engine.hud_clients:
            continue
        payload = json.dumps(engine.snapshot())
        disconnected = []
        for ws in list(engine.hud_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            engine.hud_clients.discard(ws)


# ==============================================================================
# LIFESPAN Y ENDPOINTS FASTAPI
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando quant-engine-core...")
    engine.bg_tasks.append(asyncio.create_task(polygon_wss_listener()))
    engine.bg_tasks.append(asyncio.create_task(telemetry_broadcaster()))
    for w in engine.wallets:
        engine.bg_tasks.append(asyncio.create_task(wallet_strategy_worker(w)))
    yield
    for task in engine.bg_tasks:
        task.cancel()


app = FastAPI(title="quant-engine-core", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN] if ALLOWED_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def verify_secret(secret_header: Optional[str]):
    if not secret_header or not secrets.compare_digest(secret_header, ENGINE_SECRET):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid ENGINE_SECRET")


class ControlAction(BaseModel):
    action: Literal["start", "pause", "kill", "mode_paper", "mode_live"]


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "rpc_connected": engine.rpc_connected,
        "killed": engine.killed,
        "mode": engine.mode
    }


@app.get("/state")
async def get_state(x_engine_secret: Optional[str] = Header(default=None)):
    verify_secret(x_engine_secret)
    return engine.snapshot()


@app.post("/control")
async def post_control(
    cmd: ControlAction,
    x_engine_secret: Optional[str] = Header(default=None)
):
    verify_secret(x_engine_secret)

    if cmd.action == "kill":
        engine.killed = True
        engine.running = False
        for w in engine.wallets:
            w.active_position_size = 0.0
            w.status = "KILLED"
        logger.critical("KILL SWITCH ACTIVADO MANUALMENTE.")
    elif cmd.action == "start":
        engine.killed = False
        engine.running = True
        logger.info(f"Motor INICIADO en modo {engine.mode}.")
    elif cmd.action == "pause":
        engine.running = False
        logger.info("Motor PAUSADO.")
    elif cmd.action == "mode_paper":
        engine.mode = "PAPER"
    elif cmd.action == "mode_live":
        engine.mode = "LIVE"

    return {"ok": True, "action": cmd.action, "state": engine.snapshot()["engine"]}


@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket, token: str = Query(default="")):
    if not token or not secrets.compare_digest(token, ENGINE_SECRET):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    engine.hud_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps(engine.snapshot()))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        engine.hud_clients.discard(websocket)