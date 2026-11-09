"""GBSM and American binomial pricing, following section 12 conventions."""
import numpy as np
from scipy import stats


def _validate(call, underlying, strike, ttm, rf, carry, volatility):
    if not isinstance(call, (bool, np.bool_)):
        raise ValueError("call must be a boolean")
    if (not np.isfinite([underlying, strike, ttm, rf, carry, volatility]).all()
            or min(underlying, strike, ttm, volatility) <= 0):
        raise ValueError("spot, strike, maturity, and volatility must be positive and finite")


def gbsm(call, underlying, strike, ttm, rf, carry, volatility):
    _validate(call, underlying, strike, ttm, rf, carry, volatility)
    s, k, t, v = underlying, strike, ttm, volatility
    z = 1 if call else -1
    d1 = (np.log(s / k) + (carry + v*v/2)*t) / (v*np.sqrt(t))
    d2 = d1 - v*np.sqrt(t)
    growth, discount = np.exp((carry-rf)*t), np.exp(-rf*t)
    density = stats.norm.pdf(d1)
    return {
        "Value": z*(s*growth*stats.norm.cdf(z*d1) - k*discount*stats.norm.cdf(z*d2)),
        "Delta": z*growth*stats.norm.cdf(z*d1),
        "Gamma": growth*density/(s*v*np.sqrt(t)),
        "Vega": s*growth*density*np.sqrt(t),
        # GBSM rho varies rf and carry together (fixed dividend yield).
        "Rho": z*t*k*discount*stats.norm.cdf(z*d2),
        "Theta": (-s*growth*density*v/(2*np.sqrt(t))
                  - z*(carry-rf)*s*growth*stats.norm.cdf(z*d1)
                  - z*rf*k*discount*stats.norm.cdf(z*d2)),
    }


def _tree(call, spots, strike, ttm, rf, carry, volatility, steps, dividends=()):
    """Price multiple initial spots together; recurse only at cash dividends."""
    spots = np.atleast_1d(spots)
    dt = ttm/steps
    u = np.exp(volatility*np.sqrt(dt))
    probability = (np.exp(carry*dt) - 1/u)/(u - 1/u)
    if not 0 <= probability <= 1:
        raise ValueError("binomial probability is outside [0, 1]; increase steps")
    discount = np.exp(-rf*dt)
    sign = 1 if call else -1
    end = dividends[0][0] if dividends else steps
    prices = spots[:, None] * u**(2*np.arange(end+1) - end)
    values = np.maximum(0, sign*(prices-strike))
    if dividends:
        date, amount = dividends[0]
        after = np.maximum(prices-amount, np.finfo(float).tiny)
        if date == steps:
            continuation = np.maximum(0, sign*(after-strike))
        else:
            remaining = tuple((d-date, a) for d, a in dividends[1:])
            # Bound memory while batching terminal subtrees. A cash dividend
            # destroys recombination across the dividend boundary.
            flat = after.ravel()
            continuation = np.empty_like(flat)
            for start in range(0, len(flat), 256):
                continuation[start:start+256] = _tree(
                    call, flat[start:start+256], strike, ttm-date*dt, rf, rf,
                    volatility, steps-date, remaining)
            continuation = continuation.reshape(after.shape)
        values = np.maximum(values, continuation)
    for j in range(end-1, -1, -1):
        continuation = discount*((1-probability)*values[:, :-1] + probability*values[:, 1:])
        prices = spots[:, None] * u**(2*np.arange(j+1) - j)
        values = np.maximum(np.maximum(0, sign*(prices-strike)), continuation)
    return values[:, 0]


def american(call, underlying, strike, ttm, rf, carry, volatility, steps=500,
             dividend_times=None, dividend_amounts=None):
    _validate(call, underlying, strike, ttm, rf, carry, volatility)
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (dividend_times is None) != (dividend_amounts is None):
        raise ValueError("dividend times and amounts must be supplied together")
    dividends = ()
    if dividend_times is not None:
        times, amounts = np.asarray(dividend_times), np.asarray(dividend_amounts, dtype=float)
        if (times.ndim != 1 or amounts.shape != times.shape
                or not np.isfinite(times).all() or not np.isfinite(amounts).all()
                or np.any(times != np.floor(times)) or np.any(times <= 0)
                or np.any(np.diff(times) <= 0) or np.any(amounts < 0)):
            raise ValueError("cash dividends need increasing positive grid dates and nonnegative amounts")
        dividends = tuple((int(d), a) for d, a in zip(times, amounts) if d <= steps)
        carry = rf
    return float(_tree(call, underlying, strike, ttm, rf, carry, volatility, steps, dividends)[0])


def american_greeks(call, underlying, strike, ttm, rf, carry, volatility, steps=500):
    parms = np.array([underlying, strike, ttm, rf, carry, volatility], dtype=float)

    def value(p):
        return american(call, *p, steps=steps)

    base = value(parms)
    gradient = []
    # FiniteDiff's default central gradient: cbrt(eps)*max(abs(x), 1).
    for i in (0, 5, 3, 2):
        shifted = parms.copy()
        h = np.cbrt(np.finfo(float).eps)*max(abs(parms[i]), 1)
        shifted[i] += h
        upper = value(shifted)
        shifted[i] -= 2*h
        gradient.append((upper-value(shifted))/(2*h))
    above, below = parms.copy(), parms.copy()
    above[0] += 1.5
    below[0] -= 1.5
    return {"Value": base, "Delta": gradient[0],
            "Gamma": (value(above)+value(below)-2*base)/1.5**2,
            "Vega": gradient[1],
            # Reference holds carry fixed for rho and reports dV/dT as theta.
            "Rho": gradient[2],
            "Theta": gradient[3]}
