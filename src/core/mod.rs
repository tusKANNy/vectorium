pub mod dataset;
pub mod distances;
pub mod flat_index;
pub mod index;
pub mod storage;
pub mod vector;
pub mod vector_encoder;

pub(crate) mod sealed {
    pub trait Sealed {}
}
