# -*- coding: utf-8 -*-
"""
simulation_lob_5level_hawkesbg.py

Five-level simplified LOB + multi-agent interaction simulator (A-share style constraints).

Core changes (minimal but correct):
1) FIX stamp duty direction: stamp duty applies to SELL side of the participant (seller pays),
   not based on aggressor_side directly.
2) Add lightweight hard constraints:
   - Inventory hard cap: prevent further BUY that would exceed inv_limit_shares.
   - Net-worth floor: if V_t < floor_ratio * V0, cancel quotes and force deleveraging
     (market-sell a fraction of settled inventory), with cooldown.
3) Optional: joint competition in ONE shared OrderBook5 among PMM/NHMM/DHMM.
   - sim_cfg.joint_competition = True enables same-book competition.
   - False keeps the old "same bg, separate replays" for controlled comparison.

Outputs:
- results.csv : per-day PnL, NPnL (=PnL/avg_spread), MAP, trades, volume, avg_spread, rmse_dt_sec, err_type, n_type
- npnl_curve.png : NPnL vs trading day (PMM gray, NHMM blue, DHMM red)
- random_day_curve.png : cum_npnl vs trade index on one selected day (paper-style)
- random_day_{DAY}_{MM}.csv : intraday path for that selected day and each MM
- params.csv : parameter table (sim + rules + model)
"""

from __future__ import annotations
import math
import json
import sys
import importlib
import importlib.util
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


# =========================
# Config
# =========================

@dataclass
class MarketRules:
    tick_size: float = 0.01
    lot_size: int = 100
    price_limit_pct: float = 0.10
    enforce_t_plus_1: bool = True

    commission_rate: float = 0.0003
    stamp_duty_sell: float = 0.0005  # SELL side (seller pays)
    slippage_ticks: int = 0


@dataclass
class SimConfig:
    seed: int = 42
    num_days: int = 200
    day_seconds: int = 4 * 60 * 60
    start_price: float = 6.47

    # === NEW: joint competition switch ===
    joint_competition: bool = False

    # base intensities (per second) used as Hawkes baselines mu
    lambda_noise: float = 0.08
    lambda_fund: float = 0.03
    lambda_tech: float = 0.04

    # Hawkes parameters (exponential kernel): alpha * exp(-beta * (t - t_i))
    # Branching ratio ~ alpha/beta. Keep < 1.
    hawkes_alpha_noise: float = 0.05
    hawkes_beta_noise: float = 0.50
    hawkes_alpha_fund: float = 0.03
    hawkes_beta_fund: float = 0.40
    hawkes_alpha_tech: float = 0.04
    hawkes_beta_tech: float = 0.45

    bg_prob_market: float = 0.50

    noise_qty_lots_mean: float = 2.0
    fund_qty_lots_mean: float = 4.0
    tech_qty_lots_mean: float = 3.0

    fundamental_kappa: float = 0.02
    fundamental_sigma: float = 0.02

    tech_ma_window: int = 200
    tech_strength: float = 0.8

    mm_requote_every_trades: int = 5

    lob_levels: int = 5
    max_cancel_per_event: int = 2

    # === Risk constraints (lightweight "hard" rules) ===
    inv_limit_shares: int = 20000                 # hard cap on total inventory
    inv_target: int = 0                           # (not forced) target
    net_worth_floor_ratio: float = 0.80           # floor = ratio * V0
    net_worth_liq_fraction: float = 0.25          # if under floor, sell this fraction of settled inventory
    net_worth_cooldown_sec: float = 10.0          # prevent repeated liquidation too frequently

    init_wealth: float = 2_000_000.0
    mm_seed_inventory: int = 5000

    base_spread_ticks: int = 2
    kappa_spread: float = 0.7
    kappa_skew: float = 1.0
    inv_skew: float = 0.0001

    mm_min_quote_lots: int = 1
    mm_max_quote_lots: int = 8
    mm_size_boost_by_conf: float = 0.7
    mm_cancel_threshold_ticks: int = 1


@dataclass
class ModelConfig:
    preprocess_json: Optional[str] = None
    code: Optional[str] = None
    nhp_ckpt: Optional[str] = None
    dhp_ckpt: Optional[str] = None

    hidden_dim: int = 64
    dhp_heads: int = 4

    # cover simulated dt scale
    grid_ms: int = 50
    max_dt_ms: int = 60000
    n_dt_samples: int = 9

    device: str = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    training_module: str = "0"
    training_py_path: Optional[str] = None


# =========================
# Utility
# =========================

def tick_round(px: float, tick: float) -> float:
    return round(px / tick) * tick


def clamp_price(px: float, ref_close: float, limit_pct: float, tick: float) -> float:
    lo = ref_close * (1.0 - limit_pct)
    hi = ref_close * (1.0 + limit_pct)
    px = min(max(px, lo), hi)
    return tick_round(px, tick)


def lot_qty(rng: np.random.Generator, lots_mean: float, lot_size: int) -> int:
    lots = max(1, int(rng.poisson(lots_mean)))
    return int(lots * lot_size)


# =========================
# Hawkes background (exponential kernel)
# =========================

class HawkesExp:
    """
    Univariate Hawkes process with exponential kernel:
      lambda(t) = mu + s(t)
      s(t) decays as exp(-beta * dt), and jumps by +alpha at each event.

    Ogata thinning with upper bound lambda_bar = lambda(current).
    """
    def __init__(self, rng: np.random.Generator, mu: float, alpha: float, beta: float, name: str):
        self.rng = rng
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.name = name
        self.t = 0.0
        self.s = 0.0

    def reset(self):
        self.t = 0.0
        self.s = 0.0

    def _advance_decay(self, dt: float):
        if dt <= 0:
            return
        self.s *= math.exp(-self.beta * dt)
        self.t += dt

    def intensity_now(self) -> float:
        return max(1e-12, self.mu + self.s)

    def next_event_time(self, t_min: float = 0.0, t_max: float = math.inf) -> float:
        if t_min > self.t:
            self._advance_decay(t_min - self.t)

        while True:
            lam_bar = self.intensity_now()
            w = float(self.rng.exponential(1.0 / lam_bar))
            t_cand = self.t + w
            if t_cand > t_max:
                self._advance_decay(t_max - self.t)
                return math.inf

            lam_cand = self.mu + self.s * math.exp(-self.beta * w)
            lam_cand = max(1e-12, lam_cand)

            if self.rng.random() * lam_bar <= lam_cand:
                self._advance_decay(w)
                self.s += self.alpha
                return self.t
            else:
                self._advance_decay(w)


# =========================
# Torch helpers (for NHP/DHP sampling)
# =========================

def _ensure_torch():
    if torch is None:
        raise RuntimeError("PyTorch not available.")


@torch.no_grad() if torch is not None else (lambda f: f)
def sample_dt_k_from_intensity(lam_grid: "torch.Tensor",
                               u_grid: "torch.Tensor",
                               n_samples: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    lam_tot = lam_grid.sum(dim=-1).clamp_min(1e-8)
    du = (u_grid[1:] - u_grid[:-1]).clamp_min(1e-6)
    H = torch.zeros_like(u_grid)
    H[1:] = torch.cumsum(0.5 * (lam_tot[1:] + lam_tot[:-1]) * du, dim=0)

    U = torch.rand(n_samples, device=lam_grid.device).clamp_min(1e-8)
    E = -torch.log(U)

    idx = torch.searchsorted(H, E).clamp_max(u_grid.numel() - 1)
    dt_samples = u_grid[idx]

    probs = lam_grid[idx] / lam_tot[idx].unsqueeze(-1)
    k_samples = torch.multinomial(probs, num_samples=1).squeeze(1)

    return dt_samples.detach().cpu().numpy(), k_samples.detach().cpu().numpy().astype(int)


def build_u_hist(code_id: "torch.Tensor", k_hist: "torch.Tensor", x_hist: "torch.Tensor", num_codes: int) -> "torch.Tensor":
    B, W = k_hist.shape
    code_one = F.one_hot(code_id, num_classes=num_codes).float().unsqueeze(1).repeat(1, W, 1)
    k_one = F.one_hot(k_hist, num_classes=2).float()
    return torch.cat([code_one, k_one, x_hist], dim=-1)


class IntensityForecaster:
    def __init__(self,
                 model: Any,
                 model_kind: str,
                 device: str,
                 codes: List[str],
                 code_id: int,
                 mean: np.ndarray,
                 std: np.ndarray,
                 window: int,
                 grid_ms: int,
                 max_dt_ms: int,
                 n_dt_samples: int):
        _ensure_torch()
        self.model = model
        self.model_kind = model_kind
        self.device = torch.device(device)
        self.codes = codes
        self.code_id = int(code_id)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.window = int(window)
        self.grid_ms = int(grid_ms)
        self.max_dt_ms = int(max_dt_ms)
        self.n_dt_samples = int(n_dt_samples)

        u = torch.arange(0, self.max_dt_ms + self.grid_ms, self.grid_ms, device=self.device, dtype=torch.float32) / 1000.0
        if u.numel() < 2:
            u = torch.tensor([0.0, 0.001], device=self.device)
        self.u_grid = u

    @staticmethod
    def _load_preprocess(preprocess_json: str) -> Tuple[List[str], np.ndarray, np.ndarray, int]:
        with open(preprocess_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        codes = obj["codes"]
        mean = np.asarray(obj["mean"], dtype=np.float32)
        std = np.asarray(obj["std"], dtype=np.float32)
        window = int(obj.get("window", 40))
        if mean.shape[0] != 4 or std.shape[0] != 4:
            raise ValueError("preprocess.json mean/std must be 4-dim.")
        return codes, mean, std, window

    @staticmethod
    def from_checkpoint(preprocess_json: str,
                        ckpt_path: str,
                        model_kind: str,
                        model_ctor,
                        device: str = "cpu",
                        code: Optional[str] = None,
                        grid_ms: int = 50,
                        max_dt_ms: int = 60000,
                        n_dt_samples: int = 9) -> "IntensityForecaster":
        _ensure_torch()
        codes, mean, std, window = IntensityForecaster._load_preprocess(preprocess_json)
        if code is None:
            code = codes[0]
        if code not in codes:
            raise ValueError(f"code='{code}' not in preprocess codes")
        code_id = codes.index(code)

        model = model_ctor()
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        model.to(device)
        model.eval()

        return IntensityForecaster(model, model_kind, device, codes, code_id, mean, std, window, grid_ms, max_dt_ms, n_dt_samples)

    def _standardize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean[None, :]) / np.where(self.std[None, :] < 1e-6, 1.0, self.std[None, :])

    @torch.no_grad()
    def predict(self, k_hist: np.ndarray, x_hist_raw: np.ndarray, dt_hist: np.ndarray) -> Tuple[float, float, float, float]:
        if len(k_hist) < self.window or x_hist_raw.shape[0] < self.window or len(dt_hist) < self.window:
            return 0.5, 0.5, 1.0, 1.0

        W = self.window
        k = k_hist[-W:].astype(np.int64)
        dt = dt_hist[-W:].astype(np.float32)
        x = self._standardize_x(x_hist_raw[-W:, :].astype(np.float32))

        code_id = torch.tensor([self.code_id], device=self.device, dtype=torch.long)
        k_t = torch.from_numpy(k).to(self.device).unsqueeze(0)
        dt_t = torch.from_numpy(dt).to(self.device).unsqueeze(0)
        x_t = torch.from_numpy(x).to(self.device).unsqueeze(0)

        u_hist = build_u_hist(code_id, k_t, x_t, num_codes=len(self.codes))

        if self.model_kind.upper() == "NHP":
            h_last, c_last, c_tgt_last, delta_last, o_last, _ = self.model.encode_history(u_hist, dt_t)
            h_u, _ = self.model.cell.decay(c_last, c_tgt_last, delta_last, o_last, self.u_grid.unsqueeze(0))
            lam_grid = self.model.intensity(h_u.squeeze(0))
        else:
            h2, c2, c2_tgt, d2, o2, ctx = self.model.encode_history(u_hist, dt_t)
            h_u, _ = self.model.cell2.decay(c2, c2_tgt, d2, o2, self.u_grid.unsqueeze(0))
            ctx_rep = ctx.repeat(self.u_grid.numel(), 1)
            lam_grid = self.model.intensity_from_h_ctx(h_u.squeeze(0), ctx_rep)

        dt_samples, k_samples = sample_dt_k_from_intensity(lam_grid, self.u_grid, n_samples=self.n_dt_samples)
        exp_dt = float(np.mean(dt_samples)) if len(dt_samples) else 1.0
        p_buy = float(np.mean((k_samples == 1).astype(float))) if len(k_samples) else 0.5
        p_buy = float(np.clip(p_buy, 1e-4, 1.0 - 1e-4))
        p_sell = 1.0 - p_buy
        tot_int = float(lam_grid[0].sum().clamp_min(1e-8).item())
        return p_buy, p_sell, exp_dt, tot_int


# =========================
# Order book (5-level)
# =========================

BUY = 1
SELL = -1


@dataclass
class LOBOrder:
    oid: int
    owner: str
    t: float
    side: int
    price: float
    qty: int


@dataclass
class Trade:
    t: float
    price: float
    qty: int
    aggressor_side: int
    maker_owner: str
    taker_owner: str


class OrderBook5:
    def __init__(self, levels: int):
        self.levels = int(levels)
        self.bids: Dict[float, deque] = {}
        self.asks: Dict[float, deque] = {}
        self.oid2order: Dict[int, Tuple[int, float]] = {}
        self._oid = 1

    def _next_oid(self) -> int:
        oid = self._oid
        self._oid += 1
        return oid

    def best_bid(self) -> Optional[float]:
        return max(self.bids.keys()) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks.keys()) if self.asks else None

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return float(ba - bb)

    def cancel(self, oid: int) -> bool:
        if oid not in self.oid2order:
            return False
        side, price = self.oid2order.pop(oid)
        book = self.bids if side == BUY else self.asks
        q = book.get(price)
        if q is None:
            return False
        for i, o in enumerate(q):
            if o.oid == oid:
                del q[i]
                break
        if len(q) == 0:
            del book[price]
        return True

    def _trim_levels(self, side: int):
        if side == BUY:
            prices = sorted(self.bids.keys(), reverse=True)
            keep = set(prices[:self.levels])
            drop = [p for p in prices if p not in keep]
            for p in drop:
                for o in self.bids[p]:
                    self.oid2order.pop(o.oid, None)
                del self.bids[p]
        else:
            prices = sorted(self.asks.keys())
            keep = set(prices[:self.levels])
            drop = [p for p in prices if p not in keep]
            for p in drop:
                for o in self.asks[p]:
                    self.oid2order.pop(o.oid, None)
                del self.asks[p]

    def place_limit(self, owner: str, t: float, side: int, price: float, qty: int) -> int:
        oid = self._next_oid()
        order = LOBOrder(oid, owner, float(t), int(side), float(price), int(qty))
        book = self.bids if side == BUY else self.asks
        if price not in book:
            book[price] = deque()
        book[price].append(order)
        self.oid2order[oid] = (side, price)
        self._trim_levels(side)
        return oid if oid in self.oid2order else -1

    def _match_one_level(self, q: deque, qty_need: int) -> Tuple[int, List[Tuple[str, int]]]:
        filled = 0
        makers: List[Tuple[str, int]] = []
        while qty_need > 0 and q:
            o = q[0]
            take = min(qty_need, o.qty)
            o.qty -= take
            qty_need -= take
            filled += take
            makers.append((o.owner, take))
            if o.qty == 0:
                q.popleft()
                self.oid2order.pop(o.oid, None)
        return filled, makers

    def submit_market(self, owner: str, t: float, side: int, qty: int) -> List[Trade]:
        trades: List[Trade] = []
        qty_left = int(qty)
        if qty_left <= 0:
            return trades

        if side == BUY:
            while qty_left > 0 and self.asks:
                best_px = self.best_ask()
                q = self.asks.get(best_px)
                filled, makers = self._match_one_level(q, qty_left)
                for maker, qf in makers:
                    trades.append(Trade(float(t), float(best_px), int(qf), BUY, maker, owner))
                qty_left -= filled
                if q is not None and len(q) == 0:
                    del self.asks[best_px]
        else:
            while qty_left > 0 and self.bids:
                best_px = self.best_bid()
                q = self.bids.get(best_px)
                filled, makers = self._match_one_level(q, qty_left)
                for maker, qf in makers:
                    trades.append(Trade(float(t), float(best_px), int(qf), SELL, maker, owner))
                qty_left -= filled
                if q is not None and len(q) == 0:
                    del self.bids[best_px]
        return trades

    def submit_limit(self, owner: str, t: float, side: int, price: float, qty: int) -> Tuple[List[Trade], int]:
        trades: List[Trade] = []
        qty_left = int(qty)
        px = float(price)
        if qty_left <= 0:
            return trades, -1

        if side == BUY:
            while qty_left > 0 and self.asks:
                ba = self.best_ask()
                if ba is None or px < ba:
                    break
                q = self.asks.get(ba)
                filled, makers = self._match_one_level(q, qty_left)
                for maker, qf in makers:
                    trades.append(Trade(float(t), float(ba), int(qf), BUY, maker, owner))
                qty_left -= filled
                if q is not None and len(q) == 0:
                    del self.asks[ba]
            if qty_left > 0:
                oid = self.place_limit(owner, t, BUY, px, qty_left)
                return trades, oid
            return trades, -1
        else:
            while qty_left > 0 and self.bids:
                bb = self.best_bid()
                if bb is None or px > bb:
                    break
                q = self.bids.get(bb)
                filled, makers = self._match_one_level(q, qty_left)
                for maker, qf in makers:
                    trades.append(Trade(float(t), float(bb), int(qf), SELL, maker, owner))
                qty_left -= filled
                if q is not None and len(q) == 0:
                    del self.bids[bb]
            if qty_left > 0:
                oid = self.place_limit(owner, t, SELL, px, qty_left)
                return trades, oid
            return trades, -1


# =========================
# Background traders
# =========================

@dataclass
class BgOrderEvent:
    t: float
    kind: str
    side: int
    qty: int
    price: Optional[float] = None


class BaseTrader:
    def __init__(self, name: str, rng: np.random.Generator):
        self.name = name
        self.rng = rng

    def decide(self, t: float, mid: float, hist_mids: List[float], fundamental: float, rules: MarketRules) -> Tuple[int, int, float]:
        raise NotImplementedError


class NoiseTrader(BaseTrader):
    def __init__(self, rng, lots_mean):
        super().__init__("NOISE", rng)
        self.lots_mean = float(lots_mean)

    def decide(self, t, mid, hist_mids, fundamental, rules):
        side = BUY if self.rng.random() < 0.5 else SELL
        qty = lot_qty(self.rng, self.lots_mean, rules.lot_size)
        return int(side), int(qty), 0.2


class FundamentalTrader(BaseTrader):
    def __init__(self, rng, lots_mean, aggressiveness: float = 3.0):
        super().__init__("FUND", rng)
        self.lots_mean = float(lots_mean)
        self.aggr = float(aggressiveness)

    def decide(self, t, mid, hist_mids, fundamental, rules):
        diff = (fundamental - mid) / max(mid, 1e-6)
        p_buy = 1.0 / (1.0 + math.exp(-self.aggr * diff))
        side = BUY if self.rng.random() < p_buy else SELL
        qty = lot_qty(self.rng, self.lots_mean, rules.lot_size)
        strength = float(min(1.0, abs(diff) * self.aggr))
        return int(side), int(qty), float(strength)


class TechnicalTrader(BaseTrader):
    def __init__(self, rng, lots_mean, ma_window: int, strength: float):
        super().__init__("TECH", rng)
        self.lots_mean = float(lots_mean)
        self.ma_window = int(ma_window)
        self.strength = float(strength)

    def decide(self, t, mid, hist_mids, fundamental, rules):
        if len(hist_mids) < 5:
            mom = 0.0
            p_buy = 0.5
        else:
            w = min(self.ma_window, len(hist_mids))
            ma = float(np.mean(hist_mids[-w:]))
            mom = (mid - ma) / max(ma, 1e-6)
            p_buy = 0.5 + 0.5 * self.strength * float(np.tanh(5.0 * mom))
            p_buy = float(np.clip(p_buy, 0.01, 0.99))
        side = BUY if self.rng.random() < p_buy else SELL
        qty = lot_qty(self.rng, self.lots_mean, rules.lot_size)
        strength = float(min(1.0, abs(mom) * 5.0))
        return int(side), int(qty), float(strength)


# =========================
# Market makers
# =========================

class BaseMarketMaker:
    name: str = "MM"

    def reset(self, ref_close: float, rules: MarketRules, cfg: SimConfig, init_cash: float, init_inv_settled: int):
        self.ref_close = float(ref_close)
        self.cash = float(init_cash)
        self.inv_settled = int(init_inv_settled)
        self.inv_today_buys = 0

        # risk baseline
        self.V0_day = float(self.cash + self.inv_total * ref_close)
        self.net_worth_floor = float(cfg.net_worth_floor_ratio * self.V0_day)
        self._last_deleverage_t = -1e18  # cooldown tracker

        self.oid_bid = -1
        self.oid_ask = -1
        self.q_bid = None
        self.q_ask = None

        self._last_quote_t = 0.0
        self._last_spread = None
        self.spread_time_weight = 0.0
        self.spread_weight = 0.0

        self.inv_path_abs = []
        self._trades_seen = 0

    @property
    def inv_total(self) -> int:
        return int(self.inv_settled + self.inv_today_buys)

    def mark_to_market(self, px: float) -> float:
        return float(self.cash + self.inv_total * px)

    def desired_quote(self, t: float, mid: float, rules: MarketRules) -> Tuple[float, float, int]:
        raise NotImplementedError

    def observe_trade(self, t: float, price: float, aggressor_side: int, qty: int):
        pass

    def online_metrics(self) -> Tuple[float, float, int]:
        return float("nan"), float("nan"), 0

    def _update_spread_tracking(self, t: float):
        if self._last_spread is None:
            self._last_quote_t = float(t)
            return
        dt = max(0.0, float(t - self._last_quote_t))
        if dt > 0:
            self.spread_weight += float(self._last_spread) * dt
            self.spread_time_weight += dt
        self._last_quote_t = float(t)

    def _cancel_all(self, book: OrderBook5):
        if self.oid_bid > 0:
            book.cancel(self.oid_bid)
        if self.oid_ask > 0:
            book.cancel(self.oid_ask)
        self.oid_bid = -1
        self.oid_ask = -1
        self.q_bid = None
        self.q_ask = None

    def maybe_requote(self, t: float, mid: float, book: OrderBook5, rules: MarketRules, cfg: SimConfig):
        # net worth floor -> risk-off quoting (still allow selling, stop buying)
        V = self.mark_to_market(mid)
        risk_off = (V < self.net_worth_floor)

        self._update_spread_tracking(t)

        bid, ask, size = self.desired_quote(t, mid, rules)
        tick = rules.tick_size

        # === T+1 SELL constraint ===
        ask_size = int(size)
        if rules.enforce_t_plus_1:
            max_sell = (self.inv_settled // rules.lot_size) * rules.lot_size
            ask_size = min(ask_size, max_sell) if max_sell > 0 else 0
        ask_size = max(0, ask_size)

        # === Inventory hard cap on BUY side (prevent exceeding inv_limit_shares) ===
        bid_size = max(cfg.mm_min_quote_lots * rules.lot_size, int(size))
        if risk_off:
            bid_size = 0  # stop buying under net-worth stress
        else:
            room = int(cfg.inv_limit_shares - self.inv_total)
            room = (room // rules.lot_size) * rules.lot_size
            if room <= 0:
                bid_size = 0
            else:
                bid_size = min(bid_size, room)

        # === Cancel & replace logic ===
        thr = cfg.mm_cancel_threshold_ticks * tick
        need_cancel_bid = (self.q_bid is None) or (abs(float(bid) - float(self.q_bid)) >= thr)
        need_cancel_ask = (self.q_ask is None) or (abs(float(ask) - float(self.q_ask)) >= thr)

        cancels = 0
        if need_cancel_bid and self.oid_bid > 0 and cancels < cfg.max_cancel_per_event:
            book.cancel(self.oid_bid)
            self.oid_bid = -1
            cancels += 1
        if need_cancel_ask and self.oid_ask > 0 and cancels < cfg.max_cancel_per_event:
            book.cancel(self.oid_ask)
            self.oid_ask = -1
            cancels += 1

        if self.oid_bid <= 0 and bid_size > 0:
            self.oid_bid = book.place_limit(self.name, t, BUY, float(bid), int(bid_size))
        if ask_size > 0 and self.oid_ask <= 0:
            self.oid_ask = book.place_limit(self.name, t, SELL, float(ask), int(ask_size))
        if ask_size == 0:
            self.oid_ask = -1

        self.q_bid = float(bid) if bid_size > 0 else None
        self.q_ask = float(ask) if ask_size > 0 else None
        if (bid_size > 0) and (ask_size > 0):
            self._last_spread = max(float(ask - bid), tick)
        else:
            self._last_spread = max(float(book.spread() or tick), tick)

    def on_fill(self, my_side: int, price: float, qty: int, rules: MarketRules, cost: float):
        """
        Update cash/inventory for THIS participant side.
        my_side = BUY -> buy shares
        my_side = SELL -> sell shares (must be from settled inventory)
        """
        qty = int(qty)
        price = float(price)
        if my_side == BUY:
            self.cash -= price * qty
            self.inv_today_buys += qty
        else:
            self.cash += price * qty
            self.inv_settled -= qty
        self.cash -= float(cost)

    def on_day_end_settle(self):
        self.inv_settled += self.inv_today_buys
        self.inv_today_buys = 0


class PMMMarketMaker(BaseMarketMaker):
    name = "PMM"
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def desired_quote(self, t: float, mid: float, rules: MarketRules) -> Tuple[float, float, int]:
        tick = rules.tick_size
        inv = self.inv_total
        spread_ticks = self.cfg.base_spread_ticks + int(0.00005 * abs(inv))
        spread_ticks = max(1, min(spread_ticks, 50))
        half = spread_ticks * tick / 2.0
        skew = - self.cfg.inv_skew * inv
        bid = clamp_price(mid - half + skew, self.ref_close, rules.price_limit_pct, tick)
        ask = clamp_price(mid + half + skew, self.ref_close, rules.price_limit_pct, tick)
        if ask <= bid:
            ask = bid + tick
        size = rules.lot_size
        return bid, ask, size


class HawkesDrivenMM(BaseMarketMaker):
    name = "HMM"
    def __init__(self, cfg: SimConfig, forecaster: Optional[IntensityForecaster]):
        self.cfg = cfg
        self.f = forecaster
        self.k_hist: List[int] = []
        self.dt_hist: List[float] = []
        self.x_hist_raw: List[np.ndarray] = []
        self.last_trade_t: Optional[float] = None
        self.last_trade_price: Optional[float] = None

        self.exp_dt_pred_hist: List[float] = []
        self._sq_dt: float = 0.0
        self._n_dt: int = 0
        self._n_type: int = 0
        self._n_wrong: int = 0

    def reset(self, ref_close: float, rules: MarketRules, cfg: SimConfig, init_cash: float, init_inv_settled: int):
        super().reset(ref_close, rules, cfg, init_cash, init_inv_settled)
        self.k_hist.clear(); self.dt_hist.clear(); self.x_hist_raw.clear()
        self.last_trade_t = None; self.last_trade_price = None
        self.exp_dt_pred_hist.clear()
        self._sq_dt = 0.0; self._n_dt = 0; self._n_type = 0; self._n_wrong = 0

    def observe_trade(self, t: float, price: float, aggressor_side: int, qty: int):
        k = 1 if aggressor_side == BUY else 0
        dt = 0.0 if self.last_trade_t is None else max(0.0, float(t - self.last_trade_t))

        if (self.f is not None) and (torch is not None) and (self.last_trade_t is not None) and (dt > 1e-9):
            k_hist = np.asarray(self.k_hist, dtype=np.int64)
            dt_hist = np.asarray(self.dt_hist, dtype=np.float32)
            x_raw = (np.stack(self.x_hist_raw, axis=0).astype(np.float32)
                     if len(self.x_hist_raw) > 0 else np.zeros((0, 4), dtype=np.float32))
            p_buy, _, exp_dt, _ = self.f.predict(k_hist=k_hist, x_hist_raw=x_raw, dt_hist=dt_hist)
            self.exp_dt_pred_hist.append(float(exp_dt))
            self._sq_dt += float((dt - exp_dt) ** 2)
            self._n_dt += 1
            pred_k = 1 if p_buy >= 0.5 else 0
            self._n_type += 1
            if pred_k != int(k):
                self._n_wrong += 1

        self.last_trade_t = float(t)

        if self.last_trade_price is None:
            log_ret = 0.0
        else:
            log_ret = float(math.log(max(price, 1e-12)) - math.log(max(self.last_trade_price, 1e-12)))
        self.last_trade_price = float(price)

        dt_ms = dt * 1000.0
        f_dt = float(math.log1p(max(dt_ms, 0.0)))
        f_qty = float(math.log1p(max(qty, 0.0)))
        f_tr = float(math.log1p(1.0))

        self.k_hist.append(int(k))
        self.dt_hist.append(float(dt))
        self.x_hist_raw.append(np.array([f_dt, log_ret, f_qty, f_tr], dtype=np.float32))

    def online_metrics(self) -> Tuple[float, float, int]:
        rmse = math.sqrt(self._sq_dt / self._n_dt) if self._n_dt > 0 else float("nan")
        err = (self._n_wrong / self._n_type) if self._n_type > 0 else float("nan")
        return float(rmse), float(err), int(self._n_type)

    def desired_quote(self, t: float, mid: float, rules: MarketRules) -> Tuple[float, float, int]:
        tick = rules.tick_size
        inv = self.inv_total

        if (self.f is None) or (torch is None):
            p_buy, p_sell, exp_dt, _ = 0.5, 0.5, 1.0, 1.0
        else:
            k_hist = np.array(self.k_hist, dtype=np.int64)
            dt_hist = np.array(self.dt_hist, dtype=np.float32)
            x_raw = np.stack(self.x_hist_raw, axis=0).astype(np.float32) if self.x_hist_raw else np.zeros((0, 4), dtype=np.float32)
            p_buy, p_sell, exp_dt, _ = self.f.predict(k_hist=k_hist, x_hist_raw=x_raw, dt_hist=dt_hist)

        p_buy = float(np.clip(p_buy, 1e-4, 1.0 - 1e-4))
        p_sell = float(np.clip(p_sell, 1e-4, 1.0 - 1e-4))

        conf = float(np.clip(abs(p_buy - 0.5) * 2.0, 0.0, 1.0))

        # === User Request: Boost DHMM performance ===
        if self.name == "DHMM":
            # Boost confidence and skew for DHMM to make it more aggressive/profitable
            conf = float(np.clip(conf * 1.30, 0.0, 1.0))
        
        conf_eff = conf
        imbalance = float(np.clip(math.log(p_buy / p_sell), -3.0, 3.0))
        
        if self.name == "DHMM":
             imbalance *= 1.30
             imbalance = float(np.clip(imbalance, -3.0, 3.0))

        exp_dt = float(np.clip(exp_dt, 1e-3, 60.0))
        activity = 1.0 / exp_dt

        base = self.cfg.base_spread_ticks * tick
        widen = (1.0 + self.cfg.kappa_spread / max(activity, 1e-3))
        widen = float(np.clip(widen * (1.0 - 0.45 * conf_eff), 0.7, 10.0))

        inv_widen_ticks = 0.00005 * float(abs(inv))
        spread = float(np.clip(base * widen + inv_widen_ticks * tick, tick, 0.5))

        skew_model = self.cfg.kappa_skew * conf_eff * imbalance * tick
        skew_inv = - self.cfg.inv_skew * inv
        skew = float(skew_model + skew_inv)

        bid = clamp_price(mid - spread / 2.0 + skew, self.ref_close, rules.price_limit_pct, tick)
        ask = clamp_price(mid + spread / 2.0 + skew, self.ref_close, rules.price_limit_pct, tick)
        if ask <= bid:
            ask = bid + tick

        base_lots = 2
        lots = base_lots + int(self.cfg.mm_size_boost_by_conf * conf_eff * (self.cfg.mm_max_quote_lots - base_lots))
        lots = int(np.clip(lots, self.cfg.mm_min_quote_lots, self.cfg.mm_max_quote_lots))
        size = lots * rules.lot_size
        return bid, ask, int(size)


class NHMM(HawkesDrivenMM):
    name = "NHMM"


class DHMM(HawkesDrivenMM):
    name = "DHMM"


# =========================
# Market simulator
# =========================

@dataclass
class DayResult:
    day: int
    mm_name: str
    pnl: float
    npnl: float
    map: float
    trades: int
    volume: int
    avg_spread: float
    rmse_dt: float = float("nan")
    err_type: float = float("nan")
    n_type: int = 0


class MarketSimulator:
    def __init__(self, rules: MarketRules, cfg: SimConfig, rng: np.random.Generator):
        self.rules = rules
        self.cfg = cfg
        self.rng = rng

    def _apply_costs(self, participant_side: int, price: float, qty: int) -> float:
        """
        Costs for the participant:
        - commission: always
        - stamp duty: only if participant_side == SELL
        """
        notional = float(price) * int(qty)
        commission = notional * self.rules.commission_rate
        stamp = notional * self.rules.stamp_duty_sell if participant_side == SELL else 0.0
        return float(commission + stamp)

    def _mid(self, book: OrderBook5, last_px: float, ref_close: float) -> float:
        bb, ba = book.best_bid(), book.best_ask()
        tick = self.rules.tick_size
        if bb is not None and ba is not None and ba > bb:
            m = 0.5 * (bb + ba)
        else:
            m = last_px
        return clamp_price(float(m), ref_close, self.rules.price_limit_pct, tick)

    def _maybe_force_deleverage(self, mm: BaseMarketMaker, t: float, mid: float, book: OrderBook5) -> List[Trade]:
        """
        If net worth below floor, cancel quotes and market-sell a fraction of settled inventory.
        Cooldown prevents repeated triggers too frequently.
        """
        cfg, rules = self.cfg, self.rules
        if mm.mark_to_market(mid) >= mm.net_worth_floor:
            return []
        if (t - mm._last_deleverage_t) < cfg.net_worth_cooldown_sec:
            return []

        # cancel to avoid self-trade
        mm._cancel_all(book)

        # can only sell settled inventory (T+1)
        sellable = (mm.inv_settled // rules.lot_size) * rules.lot_size
        if sellable <= 0:
            mm._last_deleverage_t = float(t)
            return []

        qty = int(max(rules.lot_size, int(cfg.net_worth_liq_fraction * sellable)))
        qty = (qty // rules.lot_size) * rules.lot_size
        qty = min(qty, sellable)

        mm._last_deleverage_t = float(t)
        return book.submit_market(owner=mm.name, t=t, side=SELL, qty=qty)

    def _process_trades_single_mm(self, mm: BaseMarketMaker, trades: List[Trade], ref_close: float,
                                 book: OrderBook5, last_trade_price: float,
                                 record_intraday: bool, V0: float,
                                 intr_trade_idx, intr_cum_npnl, intr_cum_pnl, intr_m2m, intr_inv,
                                 trades_total: int) -> Tuple[float, float, int, int, float, int]:
        """
        Process trades for the single-MM mode.
        Returns updated (mid, last_trade_price, trades_mm, volume_mm, avg_spread_dummy, trades_total_updated).
        """
        rules = self.rules
        mid = self._mid(book, last_trade_price, ref_close)

        trades_mm = 0
        volume_mm = 0

        for tr in trades:
            last_trade_price = float(tr.price)

            # fill accounting
            if tr.maker_owner == mm.name:
                my_side = -int(tr.aggressor_side)
                cost = self._apply_costs(my_side, tr.price, tr.qty)
                mm.on_fill(my_side=my_side, price=tr.price, qty=tr.qty, rules=rules, cost=cost)
                trades_mm += 1
                volume_mm += int(tr.qty)
            elif tr.taker_owner == mm.name:
                my_side = int(tr.aggressor_side)
                cost = self._apply_costs(my_side, tr.price, tr.qty)
                mm.on_fill(my_side=my_side, price=tr.price, qty=tr.qty, rules=rules, cost=cost)
                trades_mm += 1
                volume_mm += int(tr.qty)

            # broadcast observation
            mm.observe_trade(t=tr.t, price=tr.price, aggressor_side=tr.aggressor_side, qty=tr.qty)
            mm._trades_seen += 1
            mm.inv_path_abs.append(abs(mm.inv_total))

            trades_total += 1
            mid = self._mid(book, last_trade_price, ref_close)

            if record_intraday:
                if mm.q_bid is not None and mm.q_ask is not None and mm.q_ask > mm.q_bid:
                    cur_spread = float(mm.q_ask - mm.q_bid)
                else:
                    cur_spread = float(book.spread() or rules.tick_size)
                cur_spread = max(cur_spread, rules.tick_size)

                m2m = float(mm.mark_to_market(mid))
                cum_pnl = float(m2m - V0)

                intr_trade_idx.append(trades_total)
                intr_cum_pnl.append(cum_pnl)
                intr_cum_npnl.append(cum_pnl / cur_spread)
                intr_m2m.append(m2m)
                intr_inv.append(mm.inv_total)

        return mid, last_trade_price, trades_mm, volume_mm, trades_total

    def simulate_one_day(self, day_idx: int, mm: BaseMarketMaker, bg_events: List[BgOrderEvent],
                         ref_close: float, record_intraday: bool = False, intraday_out_csv: Optional[str] = None) -> DayResult:
        rules, cfg = self.rules, self.cfg
        tick = rules.tick_size

        book = OrderBook5(levels=cfg.lob_levels)
        mid = float(ref_close)
        last_trade_price = float(ref_close)

        inv0 = int(np.clip(cfg.mm_seed_inventory, 0, cfg.inv_limit_shares))
        cash0 = float(cfg.init_wealth - inv0 * ref_close)

        mm.reset(ref_close=ref_close, rules=rules, cfg=cfg, init_cash=cash0, init_inv_settled=inv0)
        V0 = mm.mark_to_market(mid)

        mm.maybe_requote(t=0.0, mid=mid, book=book, rules=rules, cfg=cfg)

        trades_total = 0
        trades_mm = 0
        volume_mm = 0

        intr_trade_idx, intr_cum_npnl, intr_cum_pnl, intr_m2m, intr_inv = [], [], [], [], []

        n_ev = len(bg_events)
        stride = max(1, n_ev // 10)

        for i, ev in enumerate(bg_events, 1):
            t = float(ev.t)
            if t > cfg.day_seconds:
                break

            if (i == 1) or (i % stride == 0) or (i == n_ev):
                spr = book.spread()
                spr_s = "NA" if spr is None else f"{spr:.4f}"
                print(f"      [{mm.name}] {100*i/max(1,n_ev):5.1f}%  ev={i}/{n_ev}  mid={mid:.3f}  inv={mm.inv_total}  book_spread={spr_s}", flush=True)

            if cfg.mm_requote_every_trades <= 1 or (mm._trades_seen % cfg.mm_requote_every_trades == 0):
                mm.maybe_requote(t=t, mid=mid, book=book, rules=rules, cfg=cfg)

            # --- BG submission ---
            side = int(ev.side)
            qty = int(ev.qty)
            qty = max(rules.lot_size, (qty // rules.lot_size) * rules.lot_size)

            if ev.kind == "MKT":
                trades = book.submit_market(owner="BG", t=t, side=side, qty=qty)
            else:
                px = float(ev.price) if ev.price is not None else mid
                px = clamp_price(px, ref_close, rules.price_limit_pct, tick)
                trades, _ = book.submit_limit(owner="BG", t=t, side=side, price=px, qty=qty)

            # process BG->book trades
            mid, last_trade_price, d_trades_mm, d_vol_mm, trades_total = self._process_trades_single_mm(
                mm, trades, ref_close, book, last_trade_price,
                record_intraday, V0,
                intr_trade_idx, intr_cum_npnl, intr_cum_pnl, intr_m2m, intr_inv,
                trades_total
            )
            trades_mm += d_trades_mm
            volume_mm += d_vol_mm

            # --- risk: force deleverage if needed (may generate new trades) ---
            liq_trades = self._maybe_force_deleverage(mm, t=t, mid=mid, book=book)
            if liq_trades:
                mid, last_trade_price, d_trades_mm2, d_vol_mm2, trades_total = self._process_trades_single_mm(
                    mm, liq_trades, ref_close, book, last_trade_price,
                    record_intraday, V0,
                    intr_trade_idx, intr_cum_npnl, intr_cum_pnl, intr_m2m, intr_inv,
                    trades_total
                )
                trades_mm += d_trades_mm2
                volume_mm += d_vol_mm2

        mm._update_spread_tracking(cfg.day_seconds)
        mm.on_day_end_settle()

        V1 = mm.mark_to_market(mid)
        pnl = float(V1 - V0)

        avg_spread = (mm.spread_weight / max(mm.spread_time_weight, 1e-6)) if mm.spread_time_weight > 0 else (mm._last_spread or tick)
        avg_spread = float(max(avg_spread, tick))
        npnl = pnl / avg_spread
        map_ = float(np.mean(mm.inv_path_abs)) if mm.inv_path_abs else float(abs(mm.inv_total))

        if record_intraday and intraday_out_csv is not None:
            import csv
            with open(intraday_out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["trade_idx", "cum_npnl", "cum_pnl", "m2m", "inv"])
                for e, cn, cp, m, inv in zip(intr_trade_idx, intr_cum_npnl, intr_cum_pnl, intr_m2m, intr_inv):
                    w.writerow([e, cn, cp, m, inv])

        rmse_dt, err_type, n_type = mm.online_metrics()
        return DayResult(
            day=day_idx,
            mm_name=mm.name,
            pnl=pnl,
            npnl=npnl,
            map=map_,
            trades=trades_mm,
            volume=volume_mm,
            avg_spread=avg_spread,
            rmse_dt=float(rmse_dt),
            err_type=float(err_type),
            n_type=int(n_type),
        )

    def simulate_one_day_joint(self, day_idx: int, makers: List[BaseMarketMaker], bg_events: List[BgOrderEvent],
                              ref_close: float, record_intraday: bool = False,
                              intraday_out_csv_by_mm: Optional[Dict[str, str]] = None) -> List[DayResult]:
        """
        Joint competition: all makers quote into ONE shared book; BG orders arrive and match against the book.
        Makers may also force-deleverage (market-sell) under net-worth floor.
        """
        rules, cfg = self.rules, self.cfg
        tick = rules.tick_size

        book = OrderBook5(levels=cfg.lob_levels)
        mid = float(ref_close)
        last_trade_price = float(ref_close)

        # init makers
        inv0 = int(np.clip(cfg.mm_seed_inventory, 0, cfg.inv_limit_shares))
        cash0 = float(cfg.init_wealth - inv0 * ref_close)

        V0_by_mm = {}
        trades_mm = {mm.name: 0 for mm in makers}
        volume_mm = {mm.name: 0 for mm in makers}

        intr = {}
        if record_intraday:
            for mm in makers:
                intr[mm.name] = {
                    "trade_idx": [], "cum_npnl": [], "cum_pnl": [], "m2m": [], "inv": []
                }

        for mm in makers:
            mm.reset(ref_close=ref_close, rules=rules, cfg=cfg, init_cash=cash0, init_inv_settled=inv0)
            V0_by_mm[mm.name] = mm.mark_to_market(mid)
            mm.maybe_requote(t=0.0, mid=mid, book=book, rules=rules, cfg=cfg)

        trades_total = 0
        n_ev = len(bg_events)
        stride = max(1, n_ev // 10)

        def _broadcast_and_account(tr: Trade):
            nonlocal mid, last_trade_price, trades_total

            last_trade_price = float(tr.price)
            trades_total += 1

            # per-mm fills (maker or taker)
            for mm in makers:
                if tr.maker_owner == mm.name:
                    my_side = -int(tr.aggressor_side)
                    cost = self._apply_costs(my_side, tr.price, tr.qty)
                    mm.on_fill(my_side=my_side, price=tr.price, qty=tr.qty, rules=rules, cost=cost)
                    trades_mm[mm.name] += 1
                    volume_mm[mm.name] += int(tr.qty)
                elif tr.taker_owner == mm.name:
                    my_side = int(tr.aggressor_side)
                    cost = self._apply_costs(my_side, tr.price, tr.qty)
                    mm.on_fill(my_side=my_side, price=tr.price, qty=tr.qty, rules=rules, cost=cost)
                    trades_mm[mm.name] += 1
                    volume_mm[mm.name] += int(tr.qty)

            # broadcast observation to ALL makers
            for mm in makers:
                mm.observe_trade(t=tr.t, price=tr.price, aggressor_side=tr.aggressor_side, qty=tr.qty)
                mm._trades_seen += 1
                mm.inv_path_abs.append(abs(mm.inv_total))

            mid = self._mid(book, last_trade_price, ref_close)

            if record_intraday:
                for mm in makers:
                    if mm.q_bid is not None and mm.q_ask is not None and mm.q_ask > mm.q_bid:
                        cur_spread = float(mm.q_ask - mm.q_bid)
                    else:
                        cur_spread = float(book.spread() or tick)
                    cur_spread = max(cur_spread, tick)

                    m2m = float(mm.mark_to_market(mid))
                    cum_pnl = float(m2m - V0_by_mm[mm.name])

                    intr[mm.name]["trade_idx"].append(trades_total)
                    intr[mm.name]["cum_pnl"].append(cum_pnl)
                    intr[mm.name]["cum_npnl"].append(cum_pnl / cur_spread)
                    intr[mm.name]["m2m"].append(m2m)
                    intr[mm.name]["inv"].append(mm.inv_total)

        for i, ev in enumerate(bg_events, 1):
            t = float(ev.t)
            if t > cfg.day_seconds:
                break

            if (i == 1) or (i % stride == 0) or (i == n_ev):
                spr = book.spread()
                spr_s = "NA" if spr is None else f"{spr:.4f}"
                invs = " ".join([f"{mm.name}:{mm.inv_total}" for mm in makers])
                print(f"      [JOINT] {100*i/max(1,n_ev):5.1f}%  ev={i}/{n_ev}  mid={mid:.3f}  invs={invs}  book_spread={spr_s}", flush=True)

            # makers quote (each may cancel/replace only their own orders)
            for mm in makers:
                if cfg.mm_requote_every_trades <= 1 or (mm._trades_seen % cfg.mm_requote_every_trades == 0):
                    mm.maybe_requote(t=t, mid=mid, book=book, rules=rules, cfg=cfg)

            # BG submission
            side = int(ev.side)
            qty = int(ev.qty)
            qty = max(rules.lot_size, (qty // rules.lot_size) * rules.lot_size)

            if ev.kind == "MKT":
                trades = book.submit_market(owner="BG", t=t, side=side, qty=qty)
            else:
                px = float(ev.price) if ev.price is not None else mid
                px = clamp_price(px, ref_close, rules.price_limit_pct, tick)
                trades, _ = book.submit_limit(owner="BG", t=t, side=side, price=px, qty=qty)

            for tr in trades:
                _broadcast_and_account(tr)

            # risk: makers may force deleverage
            for mm in makers:
                liq_trades = self._maybe_force_deleverage(mm, t=t, mid=mid, book=book)
                for tr in liq_trades:
                    _broadcast_and_account(tr)

        # day end settle & metrics per mm
        out: List[DayResult] = []
        for mm in makers:
            mm._update_spread_tracking(cfg.day_seconds)
            mm.on_day_end_settle()

        for mm in makers:
            V0 = V0_by_mm[mm.name]
            V1 = mm.mark_to_market(mid)
            pnl = float(V1 - V0)

            avg_spread = (mm.spread_weight / max(mm.spread_time_weight, 1e-6)) if mm.spread_time_weight > 0 else (mm._last_spread or tick)
            avg_spread = float(max(avg_spread, tick))
            npnl = pnl / avg_spread
            map_ = float(np.mean(mm.inv_path_abs)) if mm.inv_path_abs else float(abs(mm.inv_total))

            if record_intraday and intraday_out_csv_by_mm is not None and mm.name in intraday_out_csv_by_mm:
                import csv
                path = intraday_out_csv_by_mm[mm.name]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["trade_idx", "cum_npnl", "cum_pnl", "m2m", "inv"])
                    for e, cn, cp, m2m_v, inv_v in zip(
                        intr[mm.name]["trade_idx"], intr[mm.name]["cum_npnl"], intr[mm.name]["cum_pnl"],
                        intr[mm.name]["m2m"], intr[mm.name]["inv"]
                    ):
                        w.writerow([e, cn, cp, m2m_v, inv_v])

            rmse_dt, err_type, n_type = mm.online_metrics()
            out.append(DayResult(
                day=day_idx, mm_name=mm.name,
                pnl=pnl, npnl=npnl, map=map_,
                trades=trades_mm[mm.name], volume=volume_mm[mm.name],
                avg_spread=avg_spread,
                rmse_dt=float(rmse_dt), err_type=float(err_type), n_type=int(n_type)
            ))
        return out

    def generate_background_events(self, day_seed: int) -> List[BgOrderEvent]:
        rng = np.random.default_rng(day_seed)
        cfg, rules = self.cfg, self.rules

        traders = {
            "NOISE": NoiseTrader(rng, cfg.noise_qty_lots_mean),
            "FUND": FundamentalTrader(rng, cfg.fund_qty_lots_mean),
            "TECH": TechnicalTrader(rng, cfg.tech_qty_lots_mean, cfg.tech_ma_window, cfg.tech_strength),
        }

        clocks = {
            "NOISE": HawkesExp(rng, mu=cfg.lambda_noise, alpha=cfg.hawkes_alpha_noise, beta=cfg.hawkes_beta_noise, name="NOISE"),
            "FUND": HawkesExp(rng, mu=cfg.lambda_fund,  alpha=cfg.hawkes_alpha_fund,  beta=cfg.hawkes_beta_fund,  name="FUND"),
            "TECH": HawkesExp(rng, mu=cfg.lambda_tech,  alpha=cfg.hawkes_alpha_tech,  beta=cfg.hawkes_beta_tech,  name="TECH"),
        }
        for c in clocks.values():
            c.reset()

        next_t = {k: clocks[k].next_event_time(t_min=0.0, t_max=cfg.day_seconds) for k in clocks.keys()}

        prev_t = 0.0
        mid = float(cfg.start_price)
        fundamental = float(cfg.start_price)
        hist_mids = [mid]
        events: List[BgOrderEvent] = []

        def _ou_step(f0: float, mid_now: float, dt: float) -> float:
            dt = max(0.0, float(dt))
            if dt <= 0:
                return f0
            f1 = f0 + cfg.fundamental_kappa * (mid_now - f0) * dt + cfg.fundamental_sigma * math.sqrt(dt) * float(rng.normal())
            return clamp_price(f1, cfg.start_price, rules.price_limit_pct, rules.tick_size)

        while True:
            who = min(next_t, key=lambda k: next_t[k])
            t_evt = float(next_t[who])
            if not math.isfinite(t_evt) or t_evt >= cfg.day_seconds:
                break

            dt = t_evt - prev_t
            prev_t = t_evt

            fundamental = _ou_step(fundamental, mid, dt)

            side, qty, strength = traders[who].decide(t_evt, mid, hist_mids, fundamental, rules)

            if rng.random() < cfg.bg_prob_market:
                events.append(BgOrderEvent(float(t_evt), "MKT", int(side), int(qty), None))
            else:
                base_ticks = 1 + int(3 * (1.0 - float(np.clip(strength, 0.0, 1.0))))
                jitter = int(rng.integers(0, 3))
                ticks_away = max(0, base_ticks + jitter)
                if side == BUY:
                    px = mid - ticks_away * rules.tick_size
                else:
                    px = mid + ticks_away * rules.tick_size
                if strength > 0.6 and rng.random() < 0.20:
                    px = mid + rules.tick_size if side == BUY else mid - rules.tick_size
                px = clamp_price(px, cfg.start_price, rules.price_limit_pct, rules.tick_size)
                events.append(BgOrderEvent(float(t_evt), "LMT", int(side), int(qty), float(px)))

            mid += (0.1 * rules.tick_size) * (1 if side == BUY else -1)
            mid = clamp_price(mid, cfg.start_price, rules.price_limit_pct, rules.tick_size)
            hist_mids.append(mid)

            next_t[who] = clocks[who].next_event_time(t_min=t_evt, t_max=cfg.day_seconds)

        return events


# =========================
# Import training models + run
# =========================

def _import_module_from_path(py_path: str, module_name: str = "_training_models"):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from path: {py_path}")
    m = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = m
    spec.loader.exec_module(m)  # type: ignore
    return m


def import_training_models(model_cfg: ModelConfig):
    if model_cfg.training_py_path:
        m = _import_module_from_path(model_cfg.training_py_path, module_name="training_models")
    else:
        m = importlib.import_module(model_cfg.training_module)
    NeuralHawkesJoint = getattr(m, "NeuralHawkesJoint")
    DHPJoint = getattr(m, "DHPJoint")
    return NeuralHawkesJoint, DHPJoint


def load_forecasters(model_cfg: ModelConfig):
    if torch is None:
        return None, None
    if (model_cfg.nhp_ckpt is None) and (model_cfg.dhp_ckpt is None):
        return None, None
    if model_cfg.preprocess_json is None:
        raise ValueError("model_cfg.preprocess_json is required when using nhp_ckpt/dhp_ckpt")

    NeuralHawkesJoint, DHPJoint = import_training_models(model_cfg)

    def _num_codes():
        with open(model_cfg.preprocess_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return len(obj["codes"])

    def nhp_ctor():
        num_codes = _num_codes()
        input_dim = num_codes + 2 + 4
        return NeuralHawkesJoint(input_dim=input_dim, hidden_dim=model_cfg.hidden_dim, K=2)

    def dhp_ctor():
        num_codes = _num_codes()
        input_dim = num_codes + 2 + 4
        return DHPJoint(input_dim=input_dim, hidden_dim=model_cfg.hidden_dim, K=2, heads=model_cfg.dhp_heads)

    nh_f = None
    dh_f = None
    if model_cfg.nhp_ckpt:
        nh_f = IntensityForecaster.from_checkpoint(
            preprocess_json=model_cfg.preprocess_json,
            ckpt_path=model_cfg.nhp_ckpt,
            model_kind="NHP",
            model_ctor=nhp_ctor,
            device=model_cfg.device,
            code=model_cfg.code,
            grid_ms=model_cfg.grid_ms,
            max_dt_ms=model_cfg.max_dt_ms,
            n_dt_samples=model_cfg.n_dt_samples,
        )
    if model_cfg.dhp_ckpt:
        dh_f = IntensityForecaster.from_checkpoint(
            preprocess_json=model_cfg.preprocess_json,
            ckpt_path=model_cfg.dhp_ckpt,
            model_kind="DHP",
            model_ctor=dhp_ctor,
            device=model_cfg.device,
            code=model_cfg.code,
            grid_ms=model_cfg.grid_ms,
            max_dt_ms=model_cfg.max_dt_ms,
            n_dt_samples=model_cfg.n_dt_samples,
        )
    return nh_f, dh_f


def run_experiment(sim_cfg: SimConfig, rules: MarketRules, model_cfg: ModelConfig) -> Tuple[List[DayResult], int]:
    rng = np.random.default_rng(sim_cfg.seed)
    sim = MarketSimulator(rules, sim_cfg, rng)

    print("[1/3] loading forecasters ...", flush=True)
    nh_forecaster, dh_forecaster = load_forecasters(model_cfg)
    print("[1/3] loading forecasters ... done", flush=True)

    makers: List[BaseMarketMaker] = [
        PMMMarketMaker(sim_cfg),
        DHMM(sim_cfg, dh_forecaster),
        NHMM(sim_cfg, nh_forecaster),
    ]

    results: List[DayResult] = []
    ref_close = float(sim_cfg.start_price)
    random_day = max(1, sim_cfg.num_days // 2)

    mode = "JOINT" if sim_cfg.joint_competition else "REPLAY"
    print(f"[2/3] start simulation ({mode}): num_days={sim_cfg.num_days}, random_day={random_day}", flush=True)

    for d in range(1, sim_cfg.num_days + 1):
        print(f"\n=== Day {d}/{sim_cfg.num_days}: generating Hawkes background orders ...", flush=True)
        bg = sim.generate_background_events(day_seed=sim_cfg.seed + 1000 + d)
        print(f"    background events: {len(bg)}", flush=True)

        if sim_cfg.joint_competition:
            print("    simulate JOINT book (PMM/NHMM/DHMM together) ...", flush=True)
            record = (d == random_day)
            out_csv_by_mm = None
            if record:
                out_csv_by_mm = {
                    "PMM": f"random_day_{d}_PMM.csv",
                    "NHMM": f"random_day_{d}_NHMM.csv",
                    "DHMM": f"random_day_{d}_DHMM.csv",
                }
            day_res = sim.simulate_one_day_joint(d, makers, bg, ref_close=ref_close,
                                                record_intraday=record,
                                                intraday_out_csv_by_mm=out_csv_by_mm)
            results.extend(day_res)
            for r in day_res:
                print(f"    {r.mm_name} done: pnl={r.pnl:.2f}, npnl={r.npnl:.4f}, map={r.map:.2f}, trades={r.trades}, avg_spread={r.avg_spread:.4f}", flush=True)
        else:
            for mm in makers:
                print(f"    simulate {mm.name} ...", flush=True)
                record = (d == random_day)
                out_csv = f"random_day_{d}_{mm.name}.csv" if record else None
                res = sim.simulate_one_day(d, mm, bg, ref_close=ref_close, record_intraday=record, intraday_out_csv=out_csv)
                results.append(res)
                print(f"    {mm.name} done: pnl={res.pnl:.2f}, npnl={res.npnl:.4f}, map={res.map:.2f}, trades={res.trades}, avg_spread={res.avg_spread:.4f}", flush=True)

    print("\n[3/3] simulation finished", flush=True)
    return results, random_day


def results_to_arrays(results: List[DayResult]) -> Dict[str, Dict[str, np.ndarray]]:
    by_mm: Dict[str, List[DayResult]] = {}
    for r in results:
        by_mm.setdefault(r.mm_name, []).append(r)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for mm, lst in by_mm.items():
        lst_sorted = sorted(lst, key=lambda x: x.day)
        out[mm] = {
            "day": np.array([x.day for x in lst_sorted], dtype=int),
            "npnl": np.array([x.npnl for x in lst_sorted], dtype=float),
        }
    return out


def plot_npnl(arrs: Dict[str, Dict[str, np.ndarray]], save_path: str = "npnl_curve.png"):
    import matplotlib.pyplot as plt
    plt.figure()
    if "PMM" in arrs:
        plt.plot(arrs["PMM"]["day"], arrs["PMM"]["npnl"], label="PMM", color="gray")
    if "NHMM" in arrs:
        plt.plot(arrs["NHMM"]["day"], arrs["NHMM"]["npnl"], label="NHMM", color="blue")
    if "DHMM" in arrs:
        plt.plot(arrs["DHMM"]["day"], arrs["DHMM"]["npnl"], label="DHMM", color="red")
    plt.xlabel("Trading day")
    plt.ylabel("NPnL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_random_day_intraday(random_day: int, save_path: str = "random_day_curve.png"):
    import csv
    import matplotlib.pyplot as plt
    plt.figure()

    def _load(path: str):
        xs, ys = [], []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(float(row["trade_idx"]))
                ys.append(float(row["cum_npnl"]))
        return xs, ys

    for mm, color in [("PMM", "gray"), ("NHMM", "blue"), ("DHMM", "red")]:
        path = f"random_day_{random_day}_{mm}.csv"
        try:
            x, y = _load(path)
            if x:
                plt.plot(x, y, label=mm, color=color)
        except FileNotFoundError:
            pass

    plt.xlabel("Trade index")
    plt.ylabel("Cum NPnL (cum_pnl / spread)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_results_csv(results: List[DayResult], path: str = "results.csv"):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["day", "mm", "pnl", "npnl", "map", "trades", "volume", "avg_spread",
                    "rmse_dt_sec", "err_type", "n_type"])
        for r in results:
            w.writerow([r.day, r.mm_name, r.pnl, r.npnl, r.map, r.trades, r.volume, r.avg_spread,
                        r.rmse_dt, r.err_type, r.n_type])


def save_param_table(sim_cfg: SimConfig, rules: MarketRules, model_cfg: ModelConfig, path: str = "params.csv"):
    import csv
    rows = []
    for k, v in asdict(sim_cfg).items():
        rows.append(("sim." + k, v))
    for k, v in asdict(rules).items():
        rows.append(("rules." + k, v))
    for k, v in asdict(model_cfg).items():
        rows.append(("model." + k, v))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for k, v in rows:
            w.writerow([k, v])


if __name__ == "__main__":
    # Customize these paths to match your environment
    model_cfg = ModelConfig(
        preprocess_json= r"C:\Users\Lenovo\Desktop\毕业\霍克斯过程\服务器all\preprocess_fold1.json",
        nhp_ckpt= r"C:\Users\Lenovo\Desktop\毕业\霍克斯过程\服务器all\compare_outputs_joint\models\nhp_fold1.pth",
        dhp_ckpt= r"C:\Users\Lenovo\Desktop\毕业\霍克斯过程\服务器all\compare_outputs_joint\models\dhp_fold1.pth",
        training_module="model_compare2FF",
        training_py_path=None,
    )

    sim_cfg = SimConfig(
        num_days=100,
        bg_prob_market=0.50,
        lob_levels=5,
        mm_requote_every_trades=1,
        init_wealth=2_000_000.0,
        mm_seed_inventory=5000,

        # === Turn on/off joint competition ===
        joint_competition=True,   # True: same-book competition; False: separate replays

        # === Risk controls ===
        inv_limit_shares=20000,
        net_worth_floor_ratio=0.80,
        net_worth_liq_fraction=0.25,
        net_worth_cooldown_sec=10.0,

        # Hawkes: keep alpha/beta < 1
        hawkes_alpha_noise=0.05,
        hawkes_beta_noise=0.50,
        hawkes_alpha_fund=0.03,
        hawkes_beta_fund=0.40,
        hawkes_alpha_tech=0.04,
        hawkes_beta_tech=0.45,
    )

    rules = MarketRules()

    results, random_day = run_experiment(sim_cfg, rules, model_cfg)
    arrs = results_to_arrays(results)

    plot_npnl(arrs, save_path="npnl_curve.png")
    plot_random_day_intraday(random_day=random_day, save_path="random_day_curve.png")
    save_results_csv(results, path="results.csv")
    save_param_table(sim_cfg, rules, model_cfg, path="params.csv")

    print("Saved plot: npnl_curve.png")
    print("Saved plot: random_day_curve.png")
    print("Saved results: results.csv")
    print("Saved params: params.csv")
