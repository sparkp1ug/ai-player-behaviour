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

#![cfg_attr(not(debug_assertions), deny(warnings))]

struct DemoApp {
    spins: u64,
}

impl Default for DemoApp {
    fn default() -> Self {
        Self { spins: 0 }
    }
}

impl eframe::App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Player Behaviour Demo");
            ui.label("Toolchain spike — the real demo replaces this.");
            ui.separator();

            // A button and a counter: enough to prove the event loop, the
            // canvas and state mutation all work, which is everything the
            // real app depends on.
            if ui.button("Pull the lever").clicked() {
                self.spins += 1;
            }
            ui.label(format!("lever pulls: {}", self.spins));

            ui.separator();
            ui.small(format!(
                "target: {}",
                if cfg!(target_arch = "wasm32") { "wasm32 (browser)" } else { "native" }
            ));
        });
    }
}

// --- native -----------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    eframe::run_native(
        "Player Behaviour Demo",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Ok(Box::<DemoApp>::default())),
    )
}

// --- browser ----------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
fn main() {
    // Without this a panic is a silently frozen canvas.
    console_error_panic_hook::set_once();

    let document = web_sys::window()
        .expect("no window")
        .document()
        .expect("no document");

    // Drop the "Loading…" placeholder from index.html once we are alive.
    if let Some(el) = document.get_element_by_id("loading") {
        el.remove();
    }

    let canvas = document
        .get_element_by_id("the_canvas_id")
        .expect("index.html must contain #the_canvas_id")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("#the_canvas_id is not a <canvas>");

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

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
