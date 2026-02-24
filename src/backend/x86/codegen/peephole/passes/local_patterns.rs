//! Local peephole pattern matching passes.
//!
//! Merges 7 simple local passes into a single linear scan (`combined_local_pass`)
//! to avoid redundant iteration over the lines array. Also includes
//! `fuse_movq_ext_truncation` which fuses movq + extension/truncation patterns.
//!
//! Merged passes in `combined_local_pass`:
//!   1. eliminate_redundant_movq_self: movq %reg, %reg (same src/dst)
//!   2. eliminate_reverse_move: movq %A,%B + movq %B,%A -> remove second
//!   3. eliminate_redundant_jumps: jmp to the immediately following label
//!   4. eliminate_cond_branch_inversion: jCC+jmp+label -> j!CC (inverted)
//!   5. eliminate_adjacent_store_load: store/load at same %rbp offset
//!   6. eliminate_redundant_zero_extend: redundant zero/sign extensions
//!   7. eliminate_redundant_xorl_zero: xorl %eax,%eax when %rax already zero

use super::super::types::*;
use super::helpers::is_valid_gp_reg;

pub(super) fn combined_local_pass(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    // Track whether %rax is known to be zero for redundant xorl elimination.
    // This is set to true after `xorl %eax, %eax` and stays true across
    // StoreRbp instructions (which don't modify register values), but is
    // invalidated by anything that writes %rax, or by control flow barriers.
    let mut rax_is_zero = false;

    let mut i = 0;
    while i < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        // --- Pattern: redundant xorl %eax, %eax elimination ---
        // When %rax is already known to be zero (from a previous xorl %eax, %eax),
        // and only StoreRbp instructions intervene (which read but don't modify
        // registers), the repeated xorl is redundant.
        //
        // Common pattern from codegen zeroing multiple local variables:
        //   xorl %eax, %eax          # sets rax = 0
        //   movq %rax, -N(%rbp)      # stores 0, rax still 0
        //   xorl %eax, %eax          # REDUNDANT
        //   movq %rax, -M(%rbp)      # stores 0, rax still 0
        if rax_is_zero {
            if let LineKind::Other { dest_reg: 0 } = infos[i].kind {
                let trimmed = infos[i].trimmed(store.get(i));
                if trimmed == "xorl %eax, %eax" {
                    mark_nop(&mut infos[i]);
                    changed = true;
                    i += 1;
                    continue;
                }
            }
        }

        // Update rax_is_zero tracking based on current instruction.
        match infos[i].kind {
            LineKind::StoreRbp { .. } => {
                // Stores to stack don't modify registers, rax_is_zero unchanged.
            }
            LineKind::Other { dest_reg: 0 } => {
                // Something writes to %rax. Check if it's xorl %eax, %eax.
                let trimmed = infos[i].trimmed(store.get(i));
                rax_is_zero = trimmed == "xorl %eax, %eax";
            }
            LineKind::Other { dest_reg } if dest_reg != 0 => {
                // Writes to a non-rax register, rax_is_zero unchanged.
                // But check if it also reads/clobbers rax implicitly.
                // Most Other instructions only write their dest_reg.
                // Conservative: only keep rax_is_zero if the instruction
                // doesn't reference rax at all (via reg_refs).
                if infos[i].reg_refs & 1 != 0 {
                    // References rax - could be a read or write, invalidate
                    // But actually a read of rax is fine for rax_is_zero.
                    // Only a write to rax matters. Since dest_reg != 0,
                    // rax is not the destination, so it's a read - OK.
                    // Exception: instructions like div/idiv/mul/cqto that
                    // implicitly clobber rax through dest_reg rdx.
                    let trimmed = infos[i].trimmed(store.get(i));
                    if trimmed.starts_with("div") || trimmed.starts_with("idiv")
                        || trimmed.starts_with("mul") || trimmed.starts_with("imul")
                        || trimmed == "cqto" || trimmed == "cqo" || trimmed == "cdq"
                        || trimmed.starts_with("xchg") || trimmed.starts_with("cmpxchg") {
                        rax_is_zero = false;
                    }
                    // Otherwise rax is only read, not written - keep tracking.
                }
            }
            LineKind::LoadRbp { reg: 0, .. } => {
                // Load to rax - rax is no longer zero
                rax_is_zero = false;
            }
            LineKind::LoadRbp { .. } => {
                // Load to non-rax register, rax_is_zero unchanged.
            }
            LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect
            | LineKind::CondJmp | LineKind::Ret | LineKind::Call => {
                // Control flow or label - invalidate tracking
                rax_is_zero = false;
            }
            LineKind::Pop { reg: 0 } | LineKind::SetCC { reg: 0 } => {
                rax_is_zero = false;
            }
            LineKind::Pop { .. } | LineKind::SetCC { .. }
            | LineKind::Push { .. } | LineKind::Cmp | LineKind::Directive => {
                // Don't affect rax
            }
            _ => {
                // Conservative: invalidate
                rax_is_zero = false;
            }
        }

        // --- Pattern: self-move elimination (movq %reg, %reg) ---
        // Pre-classified as SelfMove during classify_line, avoiding string parsing.
        if infos[i].kind == LineKind::SelfMove {
            mark_nop(&mut infos[i]);
            changed = true;
            i += 1;
            continue;
        }

        // --- Pattern: reverse-move elimination ---
        // Detects `movq %regA, %regB` followed by `movq %regB, %regA` and
        // eliminates the second mov (since %regA still holds the original value).
        //
        // Safety: We only skip NOPs and StoreRbp between the two instructions.
        // StoreRbp reads registers but never modifies any GP register value.
        // Any other instruction type causes the search to stop via `break`.
        if let LineKind::Other { dest_reg: dest_a } = infos[i].kind {
            if is_valid_gp_reg(dest_a) {
                let line_i = infos[i].trimmed(store.get(i));
                // Parse "movq %srcReg, %dstReg" pattern
                if let Some(rest) = line_i.strip_prefix("movq ") {
                    if let Some((src_str, dst_str)) = rest.split_once(',') {
                        let src = src_str.trim();
                        let dst = dst_str.trim();
                        let src_fam = register_family_fast(src);
                        let dst_fam = register_family_fast(dst);
                        // Both must be GP registers, different families, both register operands
                        if is_valid_gp_reg(src_fam) && is_valid_gp_reg(dst_fam)
                            && src_fam != dst_fam
                            && src.starts_with('%') && dst.starts_with('%')
                        {
                            // Find the next non-NOP, non-StoreRbp instruction.
                            // Limit search to 8 lines to avoid pathological scanning.
                            let mut j = i + 1;
                            let search_limit = (i + 8).min(len);
                            while j < search_limit {
                                if infos[j].is_nop() {
                                    j += 1;
                                    continue;
                                }
                                if matches!(infos[j].kind, LineKind::StoreRbp { .. }) {
                                    j += 1;
                                    continue;
                                }
                                break;
                            }
                            if j < search_limit {
                                // Check if line j is the reverse: movq %dstReg, %srcReg
                                if let LineKind::Other { dest_reg: dest_b } = infos[j].kind {
                                    if dest_b == src_fam {
                                        let line_j = infos[j].trimmed(store.get(j));
                                        if let Some(rest_j) = line_j.strip_prefix("movq ") {
                                            if let Some((src_j, dst_j)) = rest_j.split_once(',') {
                                                let src_j = src_j.trim();
                                                let dst_j = dst_j.trim();
                                                let src_j_fam = register_family_fast(src_j);
                                                let dst_j_fam = register_family_fast(dst_j);
                                                if src_j_fam == dst_fam && dst_j_fam == src_fam {
                                                    mark_nop(&mut infos[j]);
                                                    changed = true;
                                                    i += 1;
                                                    continue;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: redundant jump to next label ---
        if infos[i].kind == LineKind::Jmp {
            let jmp_line = infos[i].trimmed(store.get(i));
            if let Some(target) = jmp_line.strip_prefix("jmp ") {
                let target = target.trim();
                // Find the next non-NOP, non-empty line
                let mut found_redundant = false;
                for j in (i + 1)..len {
                    if infos[j].is_nop() || infos[j].kind == LineKind::Empty {
                        continue;
                    }
                    if infos[j].kind == LineKind::Label {
                        let next = infos[j].trimmed(store.get(j));
                        if let Some(label) = next.strip_suffix(':') {
                            if label == target {
                                mark_nop(&mut infos[i]);
                                changed = true;
                                found_redundant = true;
                            }
                        }
                    }
                    break;
                }
                if found_redundant {
                    i += 1;
                    continue;
                }
            }
        }

        // --- Pattern: conditional branch inversion for fall-through ---
        // Detects:
        //   jCC .Ltrue        (conditional jump)
        //   jmp .Lfalse       (unconditional jump)
        //   .Ltrue:           (label matching the conditional target)
        //
        // Transforms to:
        //   j!CC .Lfalse      (inverted condition, jump to false target)
        //   .Ltrue:           (fall through naturally)
        if infos[i].kind == LineKind::CondJmp {
            let cond_line = infos[i].trimmed(store.get(i));
            // Parse: "jCC target" -> extract CC and target
            if let Some(space_pos) = cond_line.find(' ') {
                let cc = &cond_line[1..space_pos]; // e.g., "l", "ge", "ne"
                let cond_target = cond_line[space_pos + 1..].trim();
                // Find the next non-NOP line (should be jmp)
                let mut j = i + 1;
                while j < len && infos[j].is_nop() {
                    j += 1;
                }
                if j < len && infos[j].kind == LineKind::Jmp {
                    let jmp_line = infos[j].trimmed(store.get(j));
                    if let Some(jmp_target) = jmp_line.strip_prefix("jmp ") {
                        let jmp_target = jmp_target.trim();
                        // Find the next non-NOP/non-empty line after jmp (should be a label)
                        let mut k = j + 1;
                        while k < len && (infos[k].is_nop() || infos[k].kind == LineKind::Empty) {
                            k += 1;
                        }
                        if k < len && infos[k].kind == LineKind::Label {
                            let label_line = infos[k].trimmed(store.get(k));
                            if let Some(label_name) = label_line.strip_suffix(':') {
                                if label_name == cond_target {
                                    let inv_cc = invert_cc(cc);
                                    if inv_cc != cc {
                                        let new_line = format!("    j{} {}", inv_cc, jmp_target);
                                        replace_line(store, &mut infos[i], i, new_line);
                                        mark_nop(&mut infos[j]); // Remove the jmp
                                        changed = true;
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Pattern: adjacent store/load at same %rbp offset ---
        if let LineKind::StoreRbp { reg: sr, offset: so, size: ss } = infos[i].kind {
            if i + 1 < len && !infos[i + 1].is_nop() {
                if let LineKind::LoadRbp { reg: lr, offset: lo, size: ls } = infos[i + 1].kind {
                    // Different register cases are handled by global_store_forwarding
                    if so == lo && ss == ls && sr == lr && sr != REG_NONE {
                        // Same register: load is redundant
                        mark_nop(&mut infos[i + 1]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        // --- Pattern: redundant zero/sign extension (including cltq) ---
        // Uses pre-classified ExtKind to avoid repeated starts_with/ends_with
        // string comparisons on every iteration.
        let mut ext_idx = i + 1;
        while ext_idx < len && ext_idx < i + 10 {
            if infos[ext_idx].is_nop() {
                ext_idx += 1;
                continue;
            }
            if matches!(infos[ext_idx].kind, LineKind::StoreRbp { .. }) {
                ext_idx += 1;
                continue;
            }
            // Skip non-rax-writing instructions: these don't change %rax,
            // so an extension on %rax further ahead is still redundant.
            // This catches patterns like: movsbq (%r15), %rax; movq %rax, %r13; movsbq %al, %rax
            match infos[ext_idx].kind {
                LineKind::Other { dest_reg } if dest_reg != 0 => {
                    ext_idx += 1;
                    continue;
                }
                LineKind::LoadRbp { reg, .. } if reg != 0 => {
                    ext_idx += 1;
                    continue;
                }
                _ => break,
            }
        }

        if ext_idx < len && !infos[ext_idx].is_nop() {
            let next_ext = infos[ext_idx].ext_kind;
            let prev_ext = infos[i].ext_kind;

            let is_redundant_ext = match next_ext {
                ExtKind::MovzbqAlRax => matches!(prev_ext, ExtKind::ProducerMovzbqToRax | ExtKind::MovzbqAlRax),
                ExtKind::MovzwqAxRax => matches!(prev_ext, ExtKind::ProducerMovzwqToRax | ExtKind::MovzwqAxRax),
                ExtKind::MovsbqAlRax => matches!(prev_ext, ExtKind::ProducerMovsbqToRax | ExtKind::MovsbqAlRax),
                ExtKind::MovslqEaxRax => matches!(prev_ext, ExtKind::ProducerMovslqToRax | ExtKind::MovslqEaxRax),
                ExtKind::Cltq => matches!(prev_ext,
                    ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                    ExtKind::MovslqEaxRax | ExtKind::Cltq |
                    // Zero-extend producers always produce values with bit 31 = 0,
                    // so cltq (sign-extend from 32 to 64) is a no-op after them.
                    ExtKind::ProducerMovzbToEax | ExtKind::ProducerMovzwToEax |
                    ExtKind::ProducerMovzbqToRax | ExtKind::ProducerMovzwqToRax |
                    ExtKind::MovzbqAlRax | ExtKind::MovzwqAxRax),
                ExtKind::MovlEaxEax => matches!(prev_ext,
                    ExtKind::ProducerArith32 | ExtKind::ProducerMovlToEax |
                    ExtKind::ProducerMovzbToEax | ExtKind::ProducerMovzbqToRax |
                    ExtKind::ProducerMovzwToEax | ExtKind::ProducerMovzwqToRax |
                    ExtKind::ProducerDiv32 |
                    ExtKind::MovlEaxEax),
                _ => false,
            };

            if is_redundant_ext {
                mark_nop(&mut infos[ext_idx]);
                changed = true;
                i += 1;
                continue;
            }

            // --- Extended scan: cltq past non-rax-clobbering instructions ---
            if next_ext == ExtKind::Cltq && !is_redundant_ext {
                let i_writes_rax = match infos[i].kind {
                    LineKind::Other { dest_reg } => dest_reg == 0,
                    LineKind::LoadRbp { reg, .. } => reg == 0,
                    LineKind::StoreRbp { .. } => false,
                    LineKind::Nop | LineKind::Empty => false,
                    _ => true, // conservative: barriers, calls, etc. may write rax
                };

                if !i_writes_rax && i > 0 {
                    let mut found_producer = false;
                    let scan_limit = i.saturating_sub(6);
                    let mut k = i - 1;
                    while k >= scan_limit {
                        if infos[k].is_nop() {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        if matches!(infos[k].kind, LineKind::StoreRbp { .. }) {
                            if k == 0 { break; }
                            k -= 1;
                            continue;
                        }
                        // Stop at barriers (labels, calls, jumps, ret)
                        if infos[k].is_barrier() {
                            break;
                        }
                        // Check if this instruction is a sign-extension producer for rax
                        let k_ext = infos[k].ext_kind;
                        if matches!(k_ext,
                            ExtKind::ProducerMovslqToRax | ExtKind::ProducerMovqConstRax |
                            ExtKind::MovslqEaxRax | ExtKind::Cltq)
                        {
                            found_producer = true;
                            break;
                        }
                        // Check if this instruction writes to %rax (family 0)
                        let writes_rax = match infos[k].kind {
                            LineKind::Other { dest_reg } => dest_reg == 0,
                            LineKind::LoadRbp { reg, .. } => reg == 0,
                            _ => true, // conservative: treat unknown as writing rax
                        };
                        if writes_rax {
                            break;
                        }
                        if k == 0 { break; }
                        k -= 1;
                    }
                    if found_producer {
                        mark_nop(&mut infos[ext_idx]);
                        changed = true;
                        i += 1;
                        continue;
                    }
                }
            }
        }

        i += 1;
    }
    changed
}

// ── Movq + extension/truncation fusion ───────────────────────────────────────
//
// Fuses `movq %REG, %rax` followed by a cast instruction into a single
// instruction. The two-instruction pattern arises from the accumulator-based
// codegen model: emit_load_operand loads a 64-bit value into %rax, then
// emit_cast_instrs emits an extension/truncation on %rax/%eax/%ax/%al.
//
// Fused patterns (all require REG != rax, no intervening non-NOP instructions):
//   movq %REG, %rax + movl %eax, %eax   -> movl %REGd, %eax    (truncate to u32)
//   movq %REG, %rax + movslq %eax, %rax -> movslq %REGd, %rax  (sign-extend i32->i64)
//   movq %REG, %rax + cltq              -> movslq %REGd, %rax   (sign-extend i32->i64)
//   movq %REG, %rax + movzbq %al, %rax  -> movzbl %REGb, %eax  (zero-extend u8->i64)
//   movq %REG, %rax + movzwq %ax, %rax  -> movzwl %REGw, %eax  (zero-extend u16->i64)
//   movq %REG, %rax + movsbq %al, %rax  -> movsbq %REGb, %rax  (sign-extend i8->i64)

pub(super) fn fuse_movq_ext_truncation(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();

    let mut i = 0;
    while i + 1 < len {
        // Look for ProducerMovqRegToRax or ProducerMovqMemToRax
        let is_reg_src = infos[i].ext_kind == ExtKind::ProducerMovqRegToRax;
        let is_mem_src = infos[i].ext_kind == ExtKind::ProducerMovqMemToRax;
        if !is_reg_src && !is_mem_src {
            i += 1;
            continue;
        }

        // Find next non-NOP instruction (skip only NOPs, not stores)
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        // Check if next instruction is a fusable extension/truncation on %rax
        let next_ext = infos[j].ext_kind;
        let fusable = matches!(next_ext,
            ExtKind::MovlEaxEax | ExtKind::MovslqEaxRax | ExtKind::Cltq |
            ExtKind::MovzbqAlRax | ExtKind::MovzwqAxRax |
            ExtKind::MovsbqAlRax);
        if !fusable {
            i += 1;
            continue;
        }

        let movq_line = infos[i].trimmed(store.get(i));

        if is_mem_src {
            // Memory source: movq N(%rbp), %rax + cltq -> movslq N(%rbp), %rax
            // Extract the memory operand (everything between "movq " and ", %rax")
            let mem_operand = if let Some(rest) = movq_line.strip_prefix("movq ") {
                if let Some((src, _)) = rest.rsplit_once(", %rax") {
                    Some(src.trim().to_string())
                } else { None }
            } else { None };

            if let Some(mem_op) = mem_operand {
                let new_text = match next_ext {
                    ExtKind::MovslqEaxRax | ExtKind::Cltq => {
                        // movq N(%rbp), %rax + cltq -> movslq N(%rbp), %rax
                        format!("    movslq {}, %rax", mem_op)
                    }
                    ExtKind::MovlEaxEax => {
                        // movq N(%rbp), %rax + movl %eax, %eax -> movl N(%rbp), %eax
                        format!("    movl {}, %eax", mem_op)
                    }
                    ExtKind::MovzbqAlRax => {
                        // movq N(%rbp), %rax + movzbq %al, %rax -> movzbl N(%rbp), %eax
                        format!("    movzbl {}, %eax", mem_op)
                    }
                    ExtKind::MovzwqAxRax => {
                        // movq N(%rbp), %rax + movzwq %ax, %rax -> movzwl N(%rbp), %eax
                        format!("    movzwl {}, %eax", mem_op)
                    }
                    ExtKind::MovsbqAlRax => {
                        // movq N(%rbp), %rax + movsbq %al, %rax -> movsbl N(%rbp), %eax
                        format!("    movsbq {}, %rax", mem_op)
                    }
                    _ => unreachable!(),
                };
                replace_line(store, &mut infos[i], i, new_text);
                mark_nop(&mut infos[j]);
                changed = true;
                i = j + 1;
                continue;
            }
            i += 1;
            continue;
        }

        // Register source: extract source register family from the movq instruction
        let src_family = if let Some(rest) = movq_line.strip_prefix("movq ") {
            if let Some((src, _dst)) = rest.split_once(',') {
                let src = src.trim();
                let fam = register_family_fast(src);
                if fam != REG_NONE && fam != 0 { fam } else { REG_NONE }
            } else { REG_NONE }
        } else { REG_NONE };

        if src_family == REG_NONE {
            i += 1;
            continue;
        }

        // Build the fused instruction based on the extension type
        let new_text = match next_ext {
            ExtKind::MovlEaxEax => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movl {}, %eax", src_32)
            }
            ExtKind::MovslqEaxRax | ExtKind::Cltq => {
                let src_32 = REG_NAMES[1][src_family as usize];
                format!("    movslq {}, %rax", src_32)
            }
            ExtKind::MovzbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movzbl {}, %eax", src_8)
            }
            ExtKind::MovzwqAxRax => {
                let src_16 = REG_NAMES[2][src_family as usize];
                format!("    movzwl {}, %eax", src_16)
            }
            ExtKind::MovsbqAlRax => {
                let src_8 = REG_NAMES[3][src_family as usize];
                format!("    movsbq {}, %rax", src_8)
            }
            _ => unreachable!("mov+ext fusion matched unexpected ExtKind"),
        };

        replace_line(store, &mut infos[i], i, new_text);
        mark_nop(&mut infos[j]);
        changed = true;
        i = j + 1;
        continue;
    }
    changed
}

// ── XMM-through-accumulator folding ──────────────────────────────────────────
//
// Folds `movq %xmm0, %rax` + `movq %rax, <dest>` into `movq %xmm0, <dest>`.
// The accumulator-based codegen routes FP values through %rax when storing
// double/float results to stack slots or callee-saved registers. This pattern
// is safe because `movq %xmm0, <gp_reg>` and `movq %xmm0, <memory>` are both
// valid x86-64 instructions (SSE2 MOVQ encoding).
//
// Also handles `movd %xmm0, %eax` + `movl %eax, <dest>` → `movd %xmm0, <dest>`.

/// Fold `movq %xmm0, %rax; movq %rax, <dest>` into `movq %xmm0, <dest>`.
///
/// The accumulator-based codegen routes floating-point values through %rax,
/// producing two-move chains. This fold eliminates the intermediate step.
///
/// Safety: The fold removes the definition of %rax. We must verify that %rax
/// is dead after the second move (not read before being overwritten or before
/// a control flow boundary).
pub(super) fn fold_xmm_through_accumulator(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    const RAX: u8 = 0; // register family 0 = rax/eax/ax/al

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        let trimmed_i = infos[i].trimmed(store.get(i));

        // Match `movq %xmm0, %rax`
        if trimmed_i != "movq %xmm0, %rax" {
            i += 1;
            continue;
        }

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));

        // Match `movq %rax, <dest>` where dest is a register or memory
        if let Some(dest) = trimmed_j.strip_prefix("movq %rax, ") {
            let dest = dest.trim();
            if dest == "%rax" {
                i += 1;
                continue;
            }

            // Check that %rax is dead after line j.
            // Scan forward from j+1: if %rax is referenced before being
            // purely overwritten (or before a control flow boundary), the fold
            // is unsafe.
            if !is_reg_dead_after(infos, store, j + 1, len, RAX) {
                i += 1;
                continue;
            }

            let new_text = format!("    movq %xmm0, {}", dest);
            replace_line(store, &mut infos[i], i, new_text);
            mark_nop(&mut infos[j]);
            changed = true;
            i = j + 1;
            continue;
        }

        i += 1;
    }
    changed
}

/// Check if a register is dead (not read before being overwritten) starting
/// from position `start`. Scans at most 16 instructions forward and gives up
/// conservatively (returns false) at control flow boundaries.
fn is_reg_dead_after(infos: &[LineInfo], store: &LineStore, start: usize, len: usize, reg: u8) -> bool {
    let reg_bit = 1u16 << reg;
    let mut scanned = 0;
    let mut k = start;
    while k < len && scanned < 16 {
        if infos[k].is_nop() {
            k += 1;
            continue;
        }

        // Control flow boundary: label, jump, conditional jump — conservatively unsafe
        match infos[k].kind {
            LineKind::Label | LineKind::Jmp | LineKind::JmpIndirect | LineKind::CondJmp => return false,
            // calls clobber rax (caller-saved), so if reg==rax it's dead after call
            LineKind::Call => return reg == 0,
            LineKind::Ret => return false,
            _ => {}
        }

        let refs_reg = infos[k].reg_refs & reg_bit != 0;
        if refs_reg {
            // This line references the register. Check if it's a pure overwrite
            // (writes the reg without reading it).
            let dest = super::helpers::get_dest_reg(&infos[k]);
            if dest == reg {
                // dest_reg == our reg. But read-modify-write instructions
                // (addq %rax, ...; subq ..., %rax) also read it.
                // If it's a simple mov/lea with reg as dest only, it's a pure overwrite.
                let trimmed = infos[k].trimmed(store.get(k));
                if trimmed.starts_with("movq ") || trimmed.starts_with("movl ")
                    || trimmed.starts_with("movb ") || trimmed.starts_with("movw ")
                    || trimmed.starts_with("leaq ") || trimmed.starts_with("leal ")
                    || trimmed.starts_with("movabs")
                    || trimmed.starts_with("xorl %eax, %eax")
                {
                    // Pure overwrite — reg is dead here
                    return true;
                }
            }
            // Referenced but not a pure overwrite → reg is read, fold unsafe
            return false;
        }

        // Implicit rax usage by div/mul/cltq etc. that reg_refs might miss
        if reg == 0 {
            let trimmed = infos[k].trimmed(store.get(k));
            if super::helpers::has_implicit_reg_usage(trimmed) {
                return false;
            }
        }

        scanned += 1;
        k += 1;
    }

    // Reached scan limit without finding a definitive answer — conservatively unsafe
    false
}

// ── Address-through-secondary register folding ──────────────────────────────
//
// Folds `movq %rN, %rcx; <mem-op> (%rcx), ...` into `<mem-op> (%rN), ...`
// and NOP's the movq. The accumulator-based codegen routes all pointer
// dereferences through %rcx (the secondary register), producing two-instruction
// chains where a single instruction suffices.
//
// Handles both loads and stores through (%rcx), including displacement forms
// like `N(%rcx)`:
//   movq %r15, %rcx; movsbq (%rcx), %rax  → movsbq (%r15), %rax
//   movq %rax, %rcx; movq (%rcx), %rax    → movq (%rax), %rax
//   movq %r15, %rcx; movb %dl, (%rcx)     → movb %dl, (%r15)
//   movq %r14, %rcx; leaq 8(%rcx), %rax   → leaq 8(%r14), %rax
//
// Safety: the fold removes the definition of %rcx. We verify that %rcx is
// dead after the memory operation (not read before being overwritten).
// We also verify %rcx is not used as a register operand (outside parentheses)
// in the memory instruction.

pub(super) fn fold_address_through_secondary(store: &mut LineStore, infos: &mut [LineInfo]) -> bool {
    let mut changed = false;
    let len = store.len();
    const RCX: u8 = 1; // register family 1 = rcx/ecx/cx/cl

    let mut i = 0;
    while i + 1 < len {
        if infos[i].is_nop() {
            i += 1;
            continue;
        }

        let trimmed_i = infos[i].trimmed(store.get(i));

        // Match: movq %rN, %rcx (where %rN is a GP register, not %rcx itself)
        let src_reg = match trimmed_i.strip_prefix("movq ") {
            Some(rest) => match rest.strip_suffix(", %rcx") {
                Some(src) if src.starts_with('%') && src != "%rcx" => {
                    let fam = register_family_fast(src);
                    if !is_valid_gp_reg(fam) || fam == RCX {
                        i += 1;
                        continue;
                    }
                    src
                }
                _ => { i += 1; continue; }
            },
            None => { i += 1; continue; }
        };

        // Find next non-NOP instruction
        let mut j = i + 1;
        while j < len && infos[j].is_nop() {
            j += 1;
        }
        if j >= len {
            i += 1;
            continue;
        }

        let trimmed_j = infos[j].trimmed(store.get(j));

        // Check that the instruction uses (%rcx) as a memory operand
        // (contains "%rcx)" as a substring — covers (%rcx), N(%rcx), etc.)
        if !trimmed_j.contains("%rcx)") {
            i += 1;
            continue;
        }

        // Trial replacement: replace %rcx) with %src) in the instruction text.
        // Then verify %rcx doesn't appear elsewhere (as a non-memory operand).
        let new_instr = trimmed_j.replace("%rcx)", &format!("{})", src_reg));
        if new_instr.contains("%rcx") || new_instr.contains("%ecx")
            || new_instr.contains("%cx") || new_instr.contains("%cl")
        {
            // %rcx still appears — used as a register operand too, can't fold
            i += 1;
            continue;
        }

        // Check that %rcx is dead after the memory instruction
        if !is_reg_dead_after(infos, store, j + 1, len, RCX) {
            i += 1;
            continue;
        }

        // Safe to fold: NOP the movq, rewrite the memory instruction
        mark_nop(&mut infos[i]);
        let new_text = format!("    {}", new_instr);
        replace_line(store, &mut infos[j], j, new_text);
        changed = true;
        i = j + 1;
    }
    changed
}
