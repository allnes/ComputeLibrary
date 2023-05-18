/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifdef __aarch64__

template<>
void interleave_block<8, 8, VLType::None, true>(
  int8_t * &out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      "ldr x28, [%x[in], #0x0]\n"
      "ldr x27, [%x[in], #0x8]\n"
      "cmp %x[height], #0x8\n"
      "mov x20, #0x0\n"
      "ldr x26, [%x[in], #0x10]\n"
      "ldr x25, [%x[in], #0x18]\n"
      "movi v5.8h, #0x0\n"
      "movi v4.8h, #0x0\n"
      "ldr x24, [%x[in], #0x20]\n"
      "ldr x23, [%x[in], #0x28]\n"
      "movi v3.8h, #0x0\n"
      "movi v2.8h, #0x0\n"
      "ldr x22, [%x[in], #0x30]\n"
      "ldr x21, [%x[in], #0x38]\n"
      "movi v1.4s, #0x0\n"
      "movi v0.4s, #0x0\n"
      "movi v31.4s, #0x0\n"
      "movi v30.4s, #0x0\n"
      "add x28, x28, %x[row_offset]\n"
      "add x27, x27, %x[row_offset]\n"
      "add x26, x26, %x[row_offset]\n"
      "add x25, x25, %x[row_offset]\n"
      "add x24, x24, %x[row_offset]\n"
      "add x23, x23, %x[row_offset]\n"
      "add x22, x22, %x[row_offset]\n"
      "add x21, x21, %x[row_offset]\n"
      "beq 1f\n"
      "cmp %x[height], #0x2\n"
      "csel x27, x27, x28, GE\n"
      "csel x26, x26, x28, GT\n"
      "cmp %x[height], #0x4\n"
      "csel x25, x25, x28, GE\n"
      "csel x24, x24, x28, GT\n"
      "cmp %x[height], #0x6\n"
      "mov x21, x28\n"
      "csel x23, x23, x28, GE\n"
      "csel x22, x22, x28, GT\n"
      "1:"  // no_pointer_adj
      "prfm pldl1keep, [x28, #0x0]\n"
      "prfm pldl1keep, [x27, #0x0]\n"
      "movi v29.4s, #0x0\n"
      "movi v28.4s, #0x0\n"
      "prfm pldl1keep, [x26, #0x0]\n"
      "prfm pldl1keep, [x25, #0x0]\n"
      "prfm pldl1keep, [x24, #0x0]\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x28, #0x40]\n"
      "prfm pldl1keep, [x27, #0x40]\n"
      "prfm pldl1keep, [x26, #0x40]\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "prfm pldl1keep, [x24, #0x40]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "cbnz %w[first], 2f\n"
      "sub %x[out_ptr], %x[out_ptr], #0x20\n"
      "ld1 { v29.4s }, [%x[out_ptr]]\n"
      "ldr q28, [%x[out_ptr], #0x10]\n"
      "2:"  // first_pass
      "cmp %x[width], #0x10\n"
      "blt 5f\n"
      "3:"  // Main loop head
      "cmp x20, #0x3e\n"
      "ble 4f\n"
      "sadalp v1.4s, v5.8h\n"
      "movi v5.8h, #0x0\n"
      "mov x20, #0x0\n"
      "sadalp v0.4s, v4.8h\n"
      "movi v4.8h, #0x0\n"
      "sadalp v31.4s, v3.8h\n"
      "movi v3.8h, #0x0\n"
      "sadalp v30.4s, v2.8h\n"
      "movi v2.8h, #0x0\n"
      "4:"  // no_accumulate_16
      "ldr q27, [x28], #0x10\n"
      "ldr q19, [x27], #0x10\n"
      "zip1 v26.2d, v27.2d, v19.2d\n"
      "sadalp v5.8h, v26.16b\n"
      "ldr q25, [x26], #0x10\n"
      "ldr q18, [x25], #0x10\n"
      "zip1 v24.2d, v25.2d, v18.2d\n"
      "sadalp v4.8h, v24.16b\n"
      "ldr q23, [x24], #0x10\n"
      "ldr q17, [x23], #0x10\n"
      "zip1 v22.2d, v23.2d, v17.2d\n"
      "sadalp v3.8h, v22.16b\n"
      "ldr q21, [x22], #0x10\n"
      "ldr q16, [x21], #0x10\n"
      "zip1 v20.2d, v21.2d, v16.2d\n"
      "sadalp v2.8h, v20.16b\n"
      "zip2 v19.2d, v27.2d, v19.2d\n"
      "zip2 v18.2d, v25.2d, v18.2d\n"
      "subs %x[width], %x[width], #0x10\n"
      "cmp %x[width], #0x10\n"
      "zip2 v17.2d, v23.2d, v17.2d\n"
      "zip2 v16.2d, v21.2d, v16.2d\n"
      "prfm pldl1keep, [x28, #0x70]\n"
      "prfm pldl1keep, [x27, #0x70]\n"
      "prfm pldl1keep, [x26, #0x70]\n"
      "prfm pldl1keep, [x25, #0x70]\n"
      "str q26, [%x[out_ptr], #0x0]\n"
      "sadalp v5.8h, v19.16b\n"
      "prfm pldl1keep, [x24, #0x70]\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "str q24, [%x[out_ptr], #0x10]\n"
      "sadalp v4.8h, v18.16b\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "str q22, [%x[out_ptr], #0x20]\n"
      "sadalp v3.8h, v17.16b\n"
      "str q20, [%x[out_ptr], #0x30]\n"
      "sadalp v2.8h, v16.16b\n"
      "add x20, x20, #0x1\n"
      "str q19, [%x[out_ptr], #0x40]\n"
      "str q18, [%x[out_ptr], #0x50]\n"
      "str q17, [%x[out_ptr], #0x60]\n"
      "str q16, [%x[out_ptr], #0x70]\n"
      "add %x[out_ptr], %x[out_ptr], #0x80\n"
      "bge 3b\n"
      "5:"  // Main loop skip
      "cbz %x[width], 14f\n"
      "tbz %x[width], #3, 9f\n"
      "ldr d27, [x28], #0x8\n"
      "ldr d19, [x27], #0x8\n"
      "ldr d25, [x26], #0x8\n"
      "ldr d18, [x25], #0x8\n"
      "ldr d23, [x24], #0x8\n"
      "ldr d17, [x23], #0x8\n"
      "ldr d21, [x22], #0x8\n"
      "ldr d16, [x21], #0x8\n"
      "tbz %x[width], #2, 7f\n"
      "ld1 { v27.s }[2], [x28], #0x4\n"
      "ld1 { v19.s }[2], [x27], #0x4\n"
      "ld1 { v25.s }[2], [x26], #0x4\n"
      "ld1 { v18.s }[2], [x25], #0x4\n"
      "ld1 { v23.s }[2], [x24], #0x4\n"
      "ld1 { v17.s }[2], [x23], #0x4\n"
      "ld1 { v21.s }[2], [x22], #0x4\n"
      "ld1 { v16.s }[2], [x21], #0x4\n"
      "tbz %x[width], #1, 6f\n"
      "ld1 { v27.h }[6], [x28], #0x2\n"
      "ld1 { v19.h }[6], [x27], #0x2\n"
      "mov x20, #0x2\n"
      "ld1 { v25.h }[6], [x26], #0x2\n"
      "ld1 { v18.h }[6], [x25], #0x2\n"
      "ld1 { v23.h }[6], [x24], #0x2\n"
      "ld1 { v17.h }[6], [x23], #0x2\n"
      "ld1 { v21.h }[6], [x22], #0x2\n"
      "ld1 { v16.h }[6], [x21], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[14], [x28]\n"
      "ld1 { v19.b }[14], [x27]\n"
      "ld1 { v25.b }[14], [x26]\n"
      "ld1 { v18.b }[14], [x25]\n"
      "ld1 { v23.b }[14], [x24]\n"
      "ld1 { v17.b }[14], [x23]\n"
      "ld1 { v21.b }[14], [x22]\n"
      "ld1 { v16.b }[14], [x21]\n"
      "b 13f\n"
      "6:"  // odd_loads_1_12
      "mov x20, #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[12], [x28]\n"
      "ld1 { v19.b }[12], [x27]\n"
      "ld1 { v25.b }[12], [x26]\n"
      "ld1 { v18.b }[12], [x25]\n"
      "ld1 { v23.b }[12], [x24]\n"
      "ld1 { v17.b }[12], [x23]\n"
      "ld1 { v21.b }[12], [x22]\n"
      "ld1 { v16.b }[12], [x21]\n"
      "b 13f\n"
      "7:"  // odd_loads_2_8
      "tbz %x[width], #1, 8f\n"
      "ld1 { v27.h }[4], [x28], #0x2\n"
      "ld1 { v19.h }[4], [x27], #0x2\n"
      "mov x20, #0x2\n"
      "ld1 { v25.h }[4], [x26], #0x2\n"
      "ld1 { v18.h }[4], [x25], #0x2\n"
      "ld1 { v23.h }[4], [x24], #0x2\n"
      "ld1 { v17.h }[4], [x23], #0x2\n"
      "ld1 { v21.h }[4], [x22], #0x2\n"
      "ld1 { v16.h }[4], [x21], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[10], [x28]\n"
      "ld1 { v19.b }[10], [x27]\n"
      "ld1 { v25.b }[10], [x26]\n"
      "ld1 { v18.b }[10], [x25]\n"
      "ld1 { v23.b }[10], [x24]\n"
      "ld1 { v17.b }[10], [x23]\n"
      "ld1 { v21.b }[10], [x22]\n"
      "ld1 { v16.b }[10], [x21]\n"
      "b 13f\n"
      "8:"  // odd_loads_1_8
      "mov x20, #0x1\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[8], [x28]\n"
      "ld1 { v19.b }[8], [x27]\n"
      "mov x20, #0x2\n"
      "ld1 { v25.b }[8], [x26]\n"
      "ld1 { v18.b }[8], [x25]\n"
      "ld1 { v23.b }[8], [x24]\n"
      "ld1 { v17.b }[8], [x23]\n"
      "ld1 { v21.b }[8], [x22]\n"
      "ld1 { v16.b }[8], [x21]\n"
      "b 13f\n"
      "9:"  // odd_loads_4_0
      "tbz %x[width], #2, 11f\n"
      "ldr s27, [x28], #0x4\n"
      "ldr s19, [x27], #0x4\n"
      "ldr s25, [x26], #0x4\n"
      "ldr s18, [x25], #0x4\n"
      "ldr s23, [x24], #0x4\n"
      "ldr s17, [x23], #0x4\n"
      "ldr s21, [x22], #0x4\n"
      "ldr s16, [x21], #0x4\n"
      "tbz %x[width], #1, 10f\n"
      "ld1 { v27.h }[2], [x28], #0x2\n"
      "ld1 { v19.h }[2], [x27], #0x2\n"
      "mov x20, #0x1\n"
      "ld1 { v25.h }[2], [x26], #0x2\n"
      "ld1 { v18.h }[2], [x25], #0x2\n"
      "ld1 { v23.h }[2], [x24], #0x2\n"
      "ld1 { v17.h }[2], [x23], #0x2\n"
      "ld1 { v21.h }[2], [x22], #0x2\n"
      "ld1 { v16.h }[2], [x21], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[6], [x28]\n"
      "ld1 { v19.b }[6], [x27]\n"
      "ld1 { v25.b }[6], [x26]\n"
      "ld1 { v18.b }[6], [x25]\n"
      "ld1 { v23.b }[6], [x24]\n"
      "ld1 { v17.b }[6], [x23]\n"
      "ld1 { v21.b }[6], [x22]\n"
      "ld1 { v16.b }[6], [x21]\n"
      "b 13f\n"
      "10:"  // odd_loads_1_4
      "mov x20, #0x1\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[4], [x28]\n"
      "ld1 { v19.b }[4], [x27]\n"
      "ld1 { v25.b }[4], [x26]\n"
      "ld1 { v18.b }[4], [x25]\n"
      "ld1 { v23.b }[4], [x24]\n"
      "ld1 { v17.b }[4], [x23]\n"
      "ld1 { v21.b }[4], [x22]\n"
      "ld1 { v16.b }[4], [x21]\n"
      "b 13f\n"
      "11:"  // odd_loads_2_0
      "tbz %x[width], #1, 12f\n"
      "ldr h27, [x28], #0x2\n"
      "ldr h19, [x27], #0x2\n"
      "mov x20, #0x1\n"
      "ldr h25, [x26], #0x2\n"
      "ldr h18, [x25], #0x2\n"
      "ldr h23, [x24], #0x2\n"
      "ldr h17, [x23], #0x2\n"
      "ldr h21, [x22], #0x2\n"
      "ldr h16, [x21], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v27.b }[2], [x28]\n"
      "ld1 { v19.b }[2], [x27]\n"
      "ld1 { v25.b }[2], [x26]\n"
      "ld1 { v18.b }[2], [x25]\n"
      "ld1 { v23.b }[2], [x24]\n"
      "ld1 { v17.b }[2], [x23]\n"
      "ld1 { v21.b }[2], [x22]\n"
      "ld1 { v16.b }[2], [x21]\n"
      "b 13f\n"
      "12:"  // odd_loads_1_0
      "ldr b27, [x28, #0x0]\n"
      "ldr b19, [x27, #0x0]\n"
      "mov x20, #0x1\n"
      "ldr b25, [x26, #0x0]\n"
      "ldr b18, [x25, #0x0]\n"
      "ldr b23, [x24, #0x0]\n"
      "ldr b17, [x23, #0x0]\n"
      "ldr b21, [x22, #0x0]\n"
      "ldr b16, [x21, #0x0]\n"
      "13:"  // Odd load end
      "zip1 v26.2d, v27.2d, v19.2d\n"
      "zip1 v24.2d, v25.2d, v18.2d\n"
      "subs x20, x20, #0x1\n"
      "str q26, [%x[out_ptr], #0x0]\n"
      "zip1 v22.2d, v23.2d, v17.2d\n"
      "zip1 v20.2d, v21.2d, v16.2d\n"
      "str q24, [%x[out_ptr], #0x10]\n"
      "sadalp v5.8h, v26.16b\n"
      "sadalp v4.8h, v24.16b\n"
      "str q22, [%x[out_ptr], #0x20]\n"
      "sadalp v3.8h, v22.16b\n"
      "str q20, [%x[out_ptr], #0x30]\n"
      "sadalp v2.8h, v20.16b\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "beq 14f\n"
      "zip2 v19.2d, v27.2d, v19.2d\n"
      "zip2 v18.2d, v25.2d, v18.2d\n"
      "str q19, [%x[out_ptr], #0x0]\n"
      "zip2 v17.2d, v23.2d, v17.2d\n"
      "zip2 v16.2d, v21.2d, v16.2d\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "sadalp v5.8h, v19.16b\n"
      "sadalp v4.8h, v18.16b\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "sadalp v3.8h, v17.16b\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "sadalp v2.8h, v16.16b\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "14:"  // Odds skip
      "sadalp v1.4s, v5.8h\n"
      "sadalp v0.4s, v4.8h\n"
      "sadalp v31.4s, v3.8h\n"
      "sadalp v30.4s, v2.8h\n"
      "addp v1.4s, v1.4s, v0.4s\n"
      "addp v0.4s, v31.4s, v30.4s\n"
      "add v1.4s, v1.4s, v29.4s\n"
      "add v0.4s, v0.4s, v28.4s\n"
      "str q1, [%x[out_ptr], #0x0]\n"
      "str q0, [%x[out_ptr], #0x10]\n"
      "add %x[out_ptr], %x[out_ptr], #0x20\n"
      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}


#endif // __aarch64__
