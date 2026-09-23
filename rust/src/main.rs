//! Step 0 of the Rust demo: the smallest thing that proves the toolchain works.
//!
//! Deliberately has no project logic in it yet. The point of this file is to
//! isolate the WASM unknowns — target, bundler, GitHub Pages base path,
//! canvas wiring — from anything that could also be wrong for a reason of its
//! own. Once this renders at the deployed URL, every later failure is a bug
//! in the demo rather than in the build.
//!
//! The same source runs natively (`cargo run`) and in the browser
//! (`trunk serve`), which is the fallback if the WASM route stalls.
//!
//! --- Rust notes, since this crate is also a way to learn the language ---
//!
//! `//!` is a doc comment for the *enclosing* item (here, the whole module).
//! `///` documents the item that follows it. Both are Markdown and both end
//! up in `cargo doc`.

#![cfg_attr(not(debug_assertions), deny(warnings))]

/// All the state the app has. In egui's *immediate mode* model there is no
/// widget tree and no callbacks: the whole UI is re-declared from this struct
/// every frame, and a click is just a bool returned by the call that drew the
/// button. That is why everything mutable lives in one place.
#[derive(Default)] // asks the compiler to write `Default::default()` for us,
                   // which for u64 means 0. Beats a hand-written impl.
struct DemoApp {
    spins: u64,
}

impl eframe::App for DemoApp {
    /// Called every frame.
    ///
    /// `&mut self` is why we can mutate `spins` — Rust makes mutability part
    /// of the signature rather than something a method may quietly do.
    ///
    /// Note for anyone reading old tutorials: eframe 0.36 replaced
    /// `update(&mut self, ctx: &Context, ..)` with this. The app is handed a
    /// `Ui` directly and no longer creates its own `CentralPanel`.
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("Player Behaviour Demo");
        ui.label("Toolchain spike — the real demo replaces this.");
        ui.separator();

        // The button is drawn and tested in one expression. There is no
        // onclick handler to register and no state to keep in sync.
        if ui.button("Pull the lever").clicked() {
            self.spins += 1;
        }
        ui.label(format!("lever pulls: {}", self.spins));

        ui.separator();
        // `cfg!` is evaluated at compile time, so this collapses to a literal.
        ui.small(format!(
            "target: {}",
            if cfg!(target_arch = "wasm32") { "wasm32 (browser)" } else { "native" }
        ));
    }
}

// --- native -----------------------------------------------------------------
//
// `#[cfg(...)]` strips the item entirely when the predicate is false, so the
// native and browser entry points below never coexist in one binary.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Player Behaviour Demo",
        eframe::NativeOptions::default(),
        // `Box::new` heap-allocates the closure; eframe needs an owned value
        // it can keep past the end of this function.
        Box::new(|_cc| Ok(Box::<DemoApp>::default())),
    )
}

// --- browser ----------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

#[cfg(target_arch = "wasm32")]
fn main() {
    // Without this a Rust panic in wasm is a silently frozen canvas rather
    // than anything you can read in the console.
    console_error_panic_hook::set_once();

    let document = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document");

    // Drop the "Loading…" placeholder from index.html now that we are alive.
    if let Some(el) = document.get_element_by_id("loading") {
        el.remove();
    }

    let canvas = document
        .get_element_by_id("the_canvas_id")
        .expect("index.html must contain #the_canvas_id")
        // `dyn_into` is a checked downcast from a generic Element to the
        // concrete canvas type: it returns a Result rather than assuming.
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("#the_canvas_id is not a <canvas>");

    // Starting eframe is async on the web, and main cannot be async here, so
    // the future is handed to the browser's event loop to drive.
    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| Ok(Box::<DemoApp>::default())),
            )
            .await
            .expect("failed to start eframe");
    });
}
