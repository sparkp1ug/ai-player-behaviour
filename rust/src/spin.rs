//! The payout maths, ported from `src/generate_synthetic_data.py`.
//!
//! This is a deliberate line-by-line translation of `spin_from_uniforms`, not
//! a reimplementation. The browser cannot call Python, so the maths has to
//! exist twice — which is exactly the drift risk that `slot_machine.py` was
//! written to avoid by importing `_spin` instead of copying it.
//!
//! The defence is that randomness is an *input* on both sides. Feed the same
//! three uniforms to both implementations and the outputs must match exactly,
//! which `tests/golden_vectors.rs` asserts over 1,230 recorded cases. That is
//! a much stronger guarantee than comparing histograms, which is all you can
//! do when each side generates its own randomness.
//!
//! --- Rust notes ---
//!
//! `pub` marks an item visible outside this module. Without it, everything is
//! private to the module — the opposite default to Python, where you opt *out*
//! of visibility by convention (`_name`) rather than opting in.

/// Volatility tier. Only the *shape* (variance) of a win differs between
/// tiers; every tier's unscaled shape has mean exactly 1.0, which is what lets
/// `payout_scale` be solved in closed form on the Python side.
///
/// `derive` asks the compiler to write trait impls for us:
///   Debug      — printing with `{:?}`, which the test failure messages use
///   Clone/Copy — this is two bytes of tag; copying beats borrowing
///   PartialEq  — allows `==`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Volatility {
    Low,
    Medium,
    High,
}

impl Volatility {
    /// Parse the string form used in `games.csv` and the JSON fixture.
    ///
    /// Returns `Option` rather than panicking or returning a default: an
    /// unrecognised tier is a real error, and `Option` makes the caller decide
    /// what to do about it instead of silently getting `Low`.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Volatility::Low),
            "medium" => Some(Volatility::Medium),
            "high" => Some(Volatility::High),
            _ => None,
        }
    }
}

/// One game's parameters. Mirrors the `SlotGame` dataclass in `src/schema.py`.
///
/// Note `rtp` is carried for display only — it does not take part in any
/// payout calculation. `payout_scale` is the field that actually moves money,
/// and it was solved so that the realised mean matches `rtp`.
#[derive(Debug, Clone)]
pub struct SlotGame {
    pub game_id: String,
    pub name: String,
    pub theme: String,
    pub volatility: Volatility,
    pub rtp: f64,
    pub base_win_probability: f64,
    pub bonus_frequency: f64,
    pub max_multiplier: f64,
    pub payout_scale: f64,
    pub min_bet: f64,
    pub max_bet: f64,
}

/// What one spin produced.
///
/// A struct rather than the Python version's `(bool, f64, bool)` tuple. Rust
/// would let us return a tuple too, but then every call site reads
/// `result.0`/`result.2` and a swapped pair of bools compiles happily. Named
/// fields make that mistake impossible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinOutcome {
    pub is_win: bool,
    /// Payout as a multiple of the bet. Zero on a loss. Not rounded — money
    /// rounding is a display concern and is deliberately kept out of the
    /// cross-language comparison, since the two languages break exact .5 ties
    /// differently.
    pub multiplier: f64,
    pub is_bonus: bool,
}

impl SpinOutcome {
    /// The losing outcome, named so the two `return` sites cannot disagree.
    pub const LOSS: SpinOutcome = SpinOutcome {
        is_win: false,
        multiplier: 0.0,
        is_bonus: false,
    };
}

/// `b` in the bonus payout's Beta(1, b). Must match `_BONUS_BETA_B` in Python.
pub const BONUS_BETA_B: f64 = 12.0;

/// Inverse CDF of the unscaled non-bonus shape, for `u` in [0, 1).
///
/// ```text
///   Low     U(0.5, 1.5)  ->  0.5 + u
///   Medium  U(0.2, 1.8)  ->  0.2 + 1.6u
///   High    Exp(1)       ->  -ln(1 - u)
/// ```
///
/// `match` on an enum is exhaustive: add a fourth tier and this stops
/// compiling until it is handled. A Python `if/elif` chain with a fallback
/// `return` would silently treat the new tier as High.
///
/// `(-u).ln_1p()` computes ln(1 + (-u)) = ln(1 - u). Using the dedicated
/// `ln_1p` rather than `(1.0 - u).ln()` preserves precision for small `u`,
/// where `1.0 - u` loses low-order bits to cancellation. This mirrors
/// `math.log1p(-u)` in the Python.
pub fn shape_quantile(volatility: Volatility, u: f64) -> f64 {
    match volatility {
        Volatility::Low => 0.5 + u,
        Volatility::Medium => 0.2 + 1.6 * u,
        Volatility::High => -(-u).ln_1p(),
    }
}

/// The payout maths. Pure: same inputs, same outputs, no RNG, no state.
///
/// Three uniforms are always consumed even though at most one branch reads
/// `u_shape`. That is distributionally identical to drawing lazily, since the
/// three are independent, and it keeps the signature fixed — which is what the
/// golden-vector fixture depends on.
///
/// `game: &SlotGame` borrows rather than takes ownership: the caller keeps its
/// game and we only read it. That is the Rust equivalent of "pass by reference
/// and don't mutate", except the compiler enforces the "don't mutate" half.
pub fn spin_from_uniforms(game: &SlotGame, u_bonus: f64, u_win: f64, u_shape: f64) -> SpinOutcome {
    // Bonus is tested first and wins outright, so the base-game test below
    // only ever sees spins that did not trigger a bonus. Getting this nesting
    // wrong is the likeliest porting error and shifts the hit rate by roughly
    // 30 sigma at the sample sizes in tests/distribution.rs.
    if u_bonus < game.bonus_frequency {
        // Beta(1, b) quantile: 1 - (1-u)^(1/b).
        let beta = 1.0 - (1.0 - u_shape).powf(1.0 / BONUS_BETA_B);
        return SpinOutcome {
            is_win: true,
            multiplier: beta * game.max_multiplier * game.payout_scale,
            is_bonus: true,
        };
    }

    if u_win < game.base_win_probability {
        let multiplier = shape_quantile(game.volatility, u_shape) * game.payout_scale;
        return SpinOutcome {
            is_win: true,
            // `f64::max` rather than a branch; mirrors Python's `max(x, 0.0)`.
            multiplier: multiplier.max(0.0),
            is_bonus: false,
        };
    }

    SpinOutcome::LOSS
}

/// Closed-form E[payout/bet] for one spin, at the game's own `payout_scale`.
///
/// The Rust twin of `_expected_raw_return` x `payout_scale`. Every branch is a
/// probability times a known mean: E[Beta(1,12)] = 1/13, and every tier's
/// shape has mean exactly 1.0.
///
/// Used by the RTP test as the reference to compare realised returns against —
/// comparing against the catalogue's `rtp` field instead would fold the
/// generator's own rounding into the tolerance.
pub fn expected_multiplier(game: &SlotGame) -> f64 {
    let beta_mean = 1.0 / (1.0 + BONUS_BETA_B);
    let raw = game.bonus_frequency * beta_mean * game.max_multiplier
        + (1.0 - game.bonus_frequency) * game.base_win_probability;
    raw * game.payout_scale
}

// `#[cfg(test)]` compiles this module only under `cargo test`, so unit tests
// live beside the code they test without shipping in the binary.
#[cfg(test)]
mod tests {
    use super::*;

    fn game(volatility: Volatility) -> SlotGame {
        SlotGame {
            game_id: "test".into(),
            name: "Test".into(),
            theme: "test".into(),
            volatility,
            rtp: 0.95,
            base_win_probability: 0.3,
            bonus_frequency: 0.02,
            max_multiplier: 100.0,
            payout_scale: 2.0,
            min_bet: 0.1,
            max_bet: 100.0,
        }
    }

    #[test]
    fn losing_spin_pays_nothing() {
        let g = game(Volatility::Medium);
        assert_eq!(spin_from_uniforms(&g, 1.0, 1.0, 0.5), SpinOutcome::LOSS);
    }

    #[test]
    fn bonus_takes_priority_over_the_base_game() {
        let g = game(Volatility::Medium);
        let out = spin_from_uniforms(&g, g.bonus_frequency - 1e-12, 1.0, 0.5);
        assert!(out.is_win && out.is_bonus);
    }

    #[test]
    fn shape_quantiles_hit_their_endpoints() {
        assert!((shape_quantile(Volatility::Low, 0.0) - 0.5).abs() < 1e-15);
        assert!((shape_quantile(Volatility::Low, 1.0) - 1.5).abs() < 1e-15);
        assert!((shape_quantile(Volatility::Medium, 0.0) - 0.2).abs() < 1e-15);
        assert!((shape_quantile(Volatility::Medium, 1.0) - 1.8).abs() < 1e-15);
        assert_eq!(shape_quantile(Volatility::High, 0.0), 0.0);
    }

    #[test]
    fn volatility_parses_and_rejects() {
        assert_eq!(Volatility::from_str("high"), Some(Volatility::High));
        assert_eq!(Volatility::from_str("HIGH"), None);
    }
}
