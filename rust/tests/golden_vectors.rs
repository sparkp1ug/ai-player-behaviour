//! Exact agreement between the Rust payout maths and the Python reference.
//!
//! The fixture is written by `tools/export_golden_vectors.py`, which records
//! what `spin_from_uniforms` in `src/generate_synthetic_data.py` returns for
//! 1,230 (u_bonus, u_win, u_shape) triples across all 15 games. This test
//! feeds the same triples to the Rust port and requires the same answers.
//!
//! Why this is possible at all: the randomness is an *input* on both sides.
//! If each language generated its own, the strongest available claim would be
//! "the distributions look the same", which cannot distinguish a correct port
//! from one with a subtly wrong constant.
//!
//! Why the multiplier tolerance is relative 1e-12 and not bit equality:
//! `powf` and `ln_1p` are libm-dependent, and wasm32 uses Rust's bundled libm
//! while macOS uses Apple's. A few ulps of difference is expected and
//! harmless. A genuine logic error — a swapped constant, the wrong branch
//! order, a missing `payout_scale` — is O(1) wrong, not O(1e-16), so 1e-12
//! leaves four thousand ulps of headroom and still catches everything real.
//!
//! The booleans are compared exactly. There is no tolerance on which branch
//! the maths took.
//!
//! Regenerate the fixture after any change to the Python spin maths:
//!     python tools/export_golden_vectors.py

use std::path::PathBuf;

use serde::Deserialize;

// `#[path]` reaches into src/ because integration tests in tests/ are separate
// crates and cannot see a binary crate's private modules. For a library crate
// this would be a plain `use player_behaviour_demo::spin;`.
#[path = "../src/spin.rs"]
mod spin;

use spin::{spin_from_uniforms, SlotGame, Volatility};

/// Relative tolerance on the payout multiplier. See the module docs.
const RELATIVE_TOLERANCE: f64 = 1e-12;

#[derive(Deserialize)]
struct Fixture {
    games: Vec<GameRow>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct GameRow {
    game_id: String,
    name: String,
    theme: String,
    volatility: String,
    rtp: f64,
    base_win_probability: f64,
    bonus_frequency: f64,
    max_multiplier: f64,
    payout_scale: f64,
    min_bet: f64,
    max_bet: f64,
}

#[derive(Deserialize)]
struct Case {
    game_id: String,
    u_bonus: f64,
    u_win: f64,
    u_shape: f64,
    is_win: bool,
    is_bonus: bool,
    multiplier: f64,
}

impl GameRow {
    fn into_game(self) -> SlotGame {
        SlotGame {
            volatility: Volatility::from_str(&self.volatility)
                .unwrap_or_else(|| panic!("unknown volatility {:?}", self.volatility)),
            game_id: self.game_id,
            name: self.name,
            theme: self.theme,
            rtp: self.rtp,
            base_win_probability: self.base_win_probability,
            bonus_frequency: self.bonus_frequency,
            max_multiplier: self.max_multiplier,
            payout_scale: self.payout_scale,
            min_bet: self.min_bet,
            max_bet: self.max_bet,
        }
    }
}

fn load() -> Fixture {
    let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "data", "spin_golden.json"]
        .iter()
        .collect();
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "could not read {}: {e}\nrun: python tools/export_golden_vectors.py",
            path.display()
        )
    });
    serde_json::from_str(&raw).expect("fixture is not valid JSON")
}

/// Relative difference, falling back to absolute when both sides are ~zero.
fn relative_error(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale < 1e-300 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

#[test]
fn rust_reproduces_python_exactly() {
    let fixture = load();
    let games: Vec<SlotGame> = fixture.games.into_iter().map(GameRow::into_game).collect();

    let mut worst_error = 0.0_f64;
    let mut worst_case = String::new();
    let mut checked = 0usize;

    for case in &fixture.cases {
        let game = games
            .iter()
            .find(|g| g.game_id == case.game_id)
            .unwrap_or_else(|| panic!("fixture references unknown game {}", case.game_id));

        let got = spin_from_uniforms(game, case.u_bonus, case.u_win, case.u_shape);

        // Branch selection is compared exactly — no tolerance on which path
        // the maths took.
        assert_eq!(
            got.is_win, case.is_win,
            "is_win differs for {} at u=({}, {}, {})",
            case.game_id, case.u_bonus, case.u_win, case.u_shape
        );
        assert_eq!(
            got.is_bonus, case.is_bonus,
            "is_bonus differs for {} at u=({}, {}, {})",
            case.game_id, case.u_bonus, case.u_win, case.u_shape
        );

        let error = relative_error(got.multiplier, case.multiplier);
        if error > worst_error {
            worst_error = error;
            worst_case = format!(
                "{} u=({}, {}, {}) rust={} python={}",
                case.game_id, case.u_bonus, case.u_win, case.u_shape, got.multiplier, case.multiplier
            );
        }
        assert!(
            error < RELATIVE_TOLERANCE,
            "multiplier differs by {error:.3e} (> {RELATIVE_TOLERANCE:.0e}) for {}\n  \
             u=({}, {}, {})\n  rust   = {}\n  python = {}",
            case.game_id,
            case.u_bonus,
            case.u_win,
            case.u_shape,
            got.multiplier,
            case.multiplier
        );
        checked += 1;
    }

    println!(
        "checked {checked} cases across {} games; worst relative error {worst_error:.3e}\n  {worst_case}",
        games.len()
    );
}

#[test]
fn fixture_covers_every_branch() {
    // A green test over cases that never trigger a bonus would prove very
    // little, so assert the fixture is actually exercising all three paths.
    let fixture = load();
    let wins = fixture.cases.iter().filter(|c| c.is_win).count();
    let bonuses = fixture.cases.iter().filter(|c| c.is_bonus).count();
    let losses = fixture.cases.len() - wins;

    assert!(bonuses >= 20, "only {bonuses} bonus cases — too few to be meaningful");
    assert!(wins - bonuses >= 100, "too few base-game wins");
    assert!(losses >= 100, "too few losing spins");
}
