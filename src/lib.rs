#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Use wee_alloc as the global allocator for smaller WASM size
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub mod tokenizer;
pub mod tfidf_lsa;
pub mod incremental;
pub mod stable_hash;
pub mod utils;

// Re-export main types
pub use incremental::IncrementalEmbedder;
pub use stable_hash::StableHashEmbedder;

// Set up console error panic hook for better debugging in browser
#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn init() {
    #[cfg(target_arch = "wasm32")]
    {
        utils::set_panic_hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Basic test will be added after implementation
        assert_eq!(1 + 1, 2);
    }
}
