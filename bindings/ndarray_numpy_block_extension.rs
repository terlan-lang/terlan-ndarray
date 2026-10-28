//! NumPy-compatible recursively nested block assembly.

use super::native_support::{delete_arrays, shape, Array, INVALID_ARGUMENT, OK, SHAPE_MISMATCH};
use super::numpy_shape_extension::handles;

/// Maximum admitted nested-list depth before native recursion is rejected.
const MAX_BLOCK_DEPTH: usize = 64;

/// One parsed leaf or nested list in a NumPy block expression.
enum BlockNode {
    /// One borrowed package array.
    Leaf(Array),
    /// One nonempty nested block list.
    Group(Vec<BlockNode>),
}

/// Cursor over the package-neutral prefix encoding of one block tree.
struct BlockParser<'a> {
    /// Borrowed leaves in source traversal order.
    leaves: &'a [Array],
    /// Prefix tokens where `-1` is a leaf and positive values are child counts.
    tokens: &'a [i64],
    /// Next leaf index to consume.
    leaf_index: usize,
    /// Next prefix token to consume.
    token_index: usize,
}

impl<'a> BlockParser<'a> {
    /// Parses one node while enforcing a bounded native recursion depth.
    fn parse(&mut self, depth: usize) -> Result<BlockNode, i32> {
        if depth > MAX_BLOCK_DEPTH {
            return Err(INVALID_ARGUMENT);
        }
        let token = *self.tokens.get(self.token_index).ok_or(INVALID_ARGUMENT)?;
        self.token_index += 1;
        if token == -1 {
            let array = *self.leaves.get(self.leaf_index).ok_or(INVALID_ARGUMENT)?;
            self.leaf_index += 1;
            return Ok(BlockNode::Leaf(array));
        }
        if token == 0 {
            return Err(SHAPE_MISMATCH);
        }
        let count = usize::try_from(token).map_err(|_| INVALID_ARGUMENT)?;
        if count > self.tokens.len().saturating_sub(self.token_index) {
            return Err(INVALID_ARGUMENT);
        }
        let mut children = Vec::new();
        children
            .try_reserve_exact(count)
            .map_err(|_| INVALID_ARGUMENT)?;
        for _ in 0..count {
            children.push(self.parse(depth + 1)?);
        }
        Ok(BlockNode::Group(children))
    }

    /// Parses exactly one tree and rejects unused handles or encoding tokens.
    fn finish(mut self) -> Result<BlockNode, i32> {
        let node = self.parse(0)?;
        if self.leaf_index != self.leaves.len() || self.token_index != self.tokens.len() {
            Err(INVALID_ARGUMENT)
        } else {
            Ok(node)
        }
    }
}

/// Rank and nesting properties shared by every leaf in one valid block tree.
struct BlockMetrics {
    /// Uniform list depth at which leaves occur.
    leaf_depth: usize,
    /// Greatest source-array rank.
    maximum_rank: usize,
}

/// Measures one block tree and rejects leaves found at different list depths.
unsafe fn measure(node: &BlockNode, depth: usize) -> Result<BlockMetrics, i32> {
    match node {
        BlockNode::Leaf(array) => Ok(BlockMetrics {
            leaf_depth: depth,
            maximum_rank: unsafe { shape(*array)? }.len(),
        }),
        BlockNode::Group(children) => {
            let mut children = children.iter();
            let first = unsafe { measure(children.next().ok_or(SHAPE_MISMATCH)?, depth + 1)? };
            let mut maximum_rank = first.maximum_rank;
            for child in children {
                let metrics = unsafe { measure(child, depth + 1)? };
                if metrics.leaf_depth != first.leaf_depth {
                    return Err(SHAPE_MISMATCH);
                }
                maximum_rank = maximum_rank.max(metrics.maximum_rank);
            }
            Ok(BlockMetrics {
                leaf_depth: first.leaf_depth,
                maximum_rank,
            })
        }
    }
}

/// Returns whether one source array has exclusively Fortran-contiguous layout.
unsafe fn is_fortran_only(array: Array) -> Result<bool, i32> {
    let mut c_contiguous = 0;
    let mut fortran_contiguous = 0;
    let c_status = unsafe { crate::ffi::terlan_ndarray_v1_is_contiguous(array, &mut c_contiguous) };
    if c_status != OK {
        return Err(c_status);
    }
    let fortran_status = unsafe {
        crate::ffi::terlan_ndarray_v1_is_fortran_contiguous(array, &mut fortran_contiguous)
    };
    if fortran_status != OK {
        Err(fortran_status)
    } else {
        Ok(fortran_contiguous != 0 && c_contiguous == 0)
    }
}

/// Returns whether every leaf has exclusively Fortran-contiguous layout.
unsafe fn all_fortran(node: &BlockNode) -> Result<bool, i32> {
    match node {
        BlockNode::Leaf(array) => unsafe { is_fortran_only(*array) },
        BlockNode::Group(children) => {
            for child in children {
                if !unsafe { all_fortran(child)? } {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}

/// Copies one leaf and prepends singleton axes until it reaches `result_rank`.
unsafe fn promote_leaf(array: Array, result_rank: usize) -> Result<Array, i32> {
    let rank = unsafe { shape(array)? }.len();
    if rank > result_rank {
        return Err(SHAPE_MISMATCH);
    }
    let mut current = std::ptr::null_mut();
    let status = unsafe { crate::ffi::terlan_ndarray_v1_copy(array, &mut current) };
    if status != OK {
        return Err(status);
    }
    for _ in rank..result_rank {
        let mut expanded = std::ptr::null_mut();
        let status =
            unsafe { crate::ffi::terlan_ndarray_v1_expand_dims(current, 0, &mut expanded) };
        unsafe { delete_arrays(&[current]) };
        if status != OK {
            if !expanded.is_null() {
                unsafe { delete_arrays(&[expanded]) };
            }
            return Err(status);
        }
        current = expanded;
    }
    Ok(current)
}

/// Recursively concatenates a validated tree along its depth-selected axes.
unsafe fn assemble(
    node: &BlockNode,
    depth: usize,
    list_depth: usize,
    result_rank: usize,
) -> Result<Array, i32> {
    match node {
        BlockNode::Leaf(array) => unsafe { promote_leaf(*array, result_rank) },
        BlockNode::Group(children) => {
            let mut arrays = Vec::new();
            arrays
                .try_reserve_exact(children.len())
                .map_err(|_| INVALID_ARGUMENT)?;
            for child in children {
                match unsafe { assemble(child, depth + 1, list_depth, result_rank) } {
                    Ok(array) => arrays.push(array),
                    Err(status) => {
                        unsafe { delete_arrays(&arrays) };
                        return Err(status);
                    }
                }
            }
            let identities = arrays
                .iter()
                .map(|array| *array as usize as u64)
                .collect::<Vec<_>>();
            let axis = result_rank
                .checked_sub(list_depth)
                .and_then(|base| base.checked_add(depth))
                .and_then(|axis| i64::try_from(axis).ok())
                .ok_or(SHAPE_MISMATCH)?;
            let count = i64::try_from(arrays.len()).map_err(|_| INVALID_ARGUMENT)?;
            let mut output = std::ptr::null_mut();
            let status = unsafe {
                crate::ffi::terlan_ndarray_v1_concatenate(
                    identities.as_ptr(),
                    count,
                    axis,
                    &mut output,
                )
            };
            unsafe { delete_arrays(&arrays) };
            if status == OK {
                Ok(output)
            } else {
                if !output.is_null() {
                    unsafe { delete_arrays(&[output]) };
                }
                Err(status)
            }
        }
    }
}

/// Reads one checked `i64` slice from a C pointer/count pair.
unsafe fn layout_tokens<'a>(tokens: *const i64, count: i64) -> Result<&'a [i64], i32> {
    let count = usize::try_from(count).map_err(|_| INVALID_ARGUMENT)?;
    if count > 0 && tokens.is_null() {
        return Err(INVALID_ARGUMENT);
    }
    Ok(if count == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(tokens, count) }
    })
}

/// Assembles an owned array from a prefix-encoded recursively nested block tree.
#[no_mangle]
pub unsafe extern "C" fn terlan_ndarray_v1_block(
    leaves: *const u64,
    leaf_count: i64,
    layout: *const i64,
    layout_count: i64,
    output: *mut Array,
) -> i32 {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    unsafe { *output = std::ptr::null_mut() };
    let leaves = match unsafe { handles(leaves, leaf_count) } {
        Ok(leaves) => leaves,
        Err(status) => return status,
    };
    let tokens = match unsafe { layout_tokens(layout, layout_count) } {
        Ok(tokens) => tokens,
        Err(status) => return status,
    };
    let node = match (BlockParser {
        leaves: &leaves,
        tokens,
        leaf_index: 0,
        token_index: 0,
    })
    .finish()
    {
        Ok(node) => node,
        Err(status) => return status,
    };
    let metrics = match unsafe { measure(&node, 0) } {
        Ok(metrics) => metrics,
        Err(status) => return status,
    };
    let result_rank = metrics.maximum_rank.max(metrics.leaf_depth);
    let preserve_fortran = match unsafe { all_fortran(&node) } {
        Ok(value) => value,
        Err(status) => return status,
    };
    let mut result = match unsafe { assemble(&node, 0, metrics.leaf_depth, result_rank) } {
        Ok(result) => result,
        Err(status) => return status,
    };
    if preserve_fortran {
        let mut ordered = std::ptr::null_mut();
        let status = unsafe { crate::ffi::terlan_ndarray_v1_copy_ordered(result, 1, &mut ordered) };
        unsafe { delete_arrays(&[result]) };
        if status != OK {
            if !ordered.is_null() {
                unsafe { delete_arrays(&[ordered]) };
            }
            return status;
        }
        result = ordered;
    }
    unsafe { *output = result };
    OK
}

#[cfg(test)]
#[path = "ndarray_numpy_block_tests.rs"]
mod tests;
