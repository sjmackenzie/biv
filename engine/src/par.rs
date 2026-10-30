//! Data parallelism: rayon's parallel iterators with the `rayon` feature, else
//! sequential iterators under the same names, so the compute code is written once.

#[cfg(feature = "rayon")]
pub(crate) use rayon::prelude::*;

/// Threads the parallel iterators run on
#[cfg(feature = "rayon")]
pub(crate) fn threads() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "rayon"))]
pub(crate) use seq::*;

#[cfg(not(feature = "rayon"))]
mod seq {
    use core::slice::{Chunks, ChunksMut, Iter, IterMut};

    pub(crate) fn threads() -> usize {
        1
    }

    pub(crate) trait ParallelSlice<T> {
        fn par_iter(&self) -> Iter<'_, T>;
        fn par_chunks(&self, size: usize) -> Chunks<'_, T>;
    }

    impl<T> ParallelSlice<T> for [T] {
        fn par_iter(&self) -> Iter<'_, T> {
            self.iter()
        }
        fn par_chunks(&self, size: usize) -> Chunks<'_, T> {
            self.chunks(size)
        }
    }

    pub(crate) trait ParallelSliceMut<T> {
        fn par_iter_mut(&mut self) -> IterMut<'_, T>;
        fn par_chunks_mut(&mut self, size: usize) -> ChunksMut<'_, T>;
    }

    impl<T> ParallelSliceMut<T> for [T] {
        fn par_iter_mut(&mut self) -> IterMut<'_, T> {
            self.iter_mut()
        }
        fn par_chunks_mut(&mut self, size: usize) -> ChunksMut<'_, T> {
            self.chunks_mut(size)
        }
    }

    pub(crate) trait IntoParallelIterator: IntoIterator + Sized {
        fn into_par_iter(self) -> Self::IntoIter {
            self.into_iter()
        }
    }

    impl<I: IntoIterator> IntoParallelIterator for I {}
}
