use core::cmp::Ordering;
use rand::Rng;
use crunchy::unroll;

use byteorder::{BigEndian, ByteOrder};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use bytemuck;
#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use risc0_bigint2::field;

/// 256-bit, stack allocated biginteger for use in prime field
/// arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct U256(pub [u128; 2]);

impl From<[u64; 4]> for U256 {
    fn from(d: [u64; 4]) -> Self {
        let mut a = [0u128; 2];
        a[0] = (d[1] as u128) << 64 | d[0] as u128;
        a[1] = (d[3] as u128) << 64 | d[2] as u128;
        U256(a)
    }
}

impl From<u64> for U256 {
    fn from(d: u64) -> Self {
        U256::from([d, 0, 0, 0])
    }
}

/// 512-bit, stack allocated biginteger for use in extension
/// field serialization and scalar interpretation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct U512(pub [u128; 4]);

impl From<[u64; 8]> for U512 {
    fn from(d: [u64; 8]) -> Self {
        let mut a = [0u128; 4];
        a[0] = (d[1] as u128) << 64 | d[0] as u128;
        a[1] = (d[3] as u128) << 64 | d[2] as u128;
        a[2] = (d[5] as u128) << 64 | d[4] as u128;
        a[3] = (d[7] as u128) << 64 | d[6] as u128;
        U512(a)
    }
}

impl U512 {
    /// Multiplies c1 by modulo, adds c0.
    pub fn new(c1: &U256, c0: &U256, modulo: &U256) -> U512 {
        let mut res = [0; 4];

        debug_assert_eq!(c1.0.len(), 2);
        unroll! {
            for i in 0..2 {
                mac_digit(i, &mut res, &modulo.0, c1.0[i]);
            }
        }

        let mut carry = 0;

        debug_assert_eq!(res.len(), 4);
        unroll! {
            for i in 0..2 {
                res[i] = adc(res[i], c0.0[i], &mut carry);
            }
        }

        unroll! {
            for i in 0..2 {
                let (a1, a0) = split_u128(res[i + 2]);
                let (c, r0) = split_u128(a0 + carry);
                let (c, r1) = split_u128(a1 + c);
                carry = c;

                res[i + 2] = combine_u128(r1, r0);
            }
        }

        debug_assert!(0 == carry);

        U512(res)
    }

     pub fn from_slice(s: &[u8]) -> Result<U512, Error> {
        if s.len() != 64 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        let mut n = [0; 4];
        for (l, i) in (0..4).rev().zip((0..4).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&s[i..]);
        }

        Ok(U512(n))
    }

    /// Get a random U512
    pub fn random<R: Rng>(rng: &mut R) -> U512 {
        U512(rng.gen())
    }

    pub fn get_bit(&self, n: usize) -> Option<bool> {
        if n >= 512 {
            None
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            Some(self.0[part] & (1 << bit) > 0)
        }
    }

    /// Divides self by modulo, returning remainder and, if
    /// possible, a quotient smaller than the modulus.
    pub fn divrem(&self, modulo: &U256) -> (Option<U256>, U256) {
        let mut q = Some(U256::zero());
        let mut r = U256::zero();

        for i in (0..512).rev() {
            // NB: modulo's first two bits are always unset
            // so this will never destroy information
            mul2(&mut r.0);
            assert!(r.set_bit(0, self.get_bit(i).unwrap()));
            if &r >= modulo {
                sub_noborrow(&mut r.0, &modulo.0);
                if q.is_some() && !q.as_mut().unwrap().set_bit(i, true) {
                    q = None
                }
            }
        }

        if q.is_some() && (q.as_ref().unwrap() >= modulo) {
            (None, r)
        } else {
            (q, r)
        }
    }

    pub fn interpret(buf: &[u8; 64]) -> U512 {
        let mut n = [0; 4];
        for (l, i) in (0..4).rev().zip((0..4).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&buf[i..]);
        }

        U512(n)
    }
}

impl Ord for U512 {
    #[inline]
    fn cmp(&self, other: &U512) -> Ordering {
        for (a, b) in self.0.iter().zip(other.0.iter()).rev() {
            if *a < *b {
                return Ordering::Less;
            } else if *a > *b {
                return Ordering::Greater;
            }
        }

        return Ordering::Equal;
    }
}

impl PartialOrd for U512 {
    #[inline]
    fn partial_cmp(&self, other: &U512) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for U256 {
    #[inline]
    fn cmp(&self, other: &U256) -> Ordering {
        for (a, b) in self.0.iter().zip(other.0.iter()).rev() {
            if *a < *b {
                return Ordering::Less;
            } else if *a > *b {
                return Ordering::Greater;
            }
        }

        return Ordering::Equal;
    }
}

impl PartialOrd for U256 {
    #[inline]
    fn partial_cmp(&self, other: &U256) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// U256/U512 errors
#[derive(Debug)]
pub enum Error {
    InvalidLength { expected: usize, actual: usize },
}

impl U256 {
    /// Initialize U256 from slice of bytes (big endian)
    pub fn from_slice(s: &[u8]) -> Result<U256, Error> {
        if s.len() != 32 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        let mut n = [0; 2];
        for (l, i) in (0..2).rev().zip((0..2).map(|i| i * 16)) {
            n[l] = BigEndian::read_u128(&s[i..]);
        }

        Ok(U256(n))
    }

    pub fn to_big_endian(&self, s: &mut [u8]) -> Result<(), Error> {
        if s.len() != 32 {
            return Err(Error::InvalidLength {
                expected: 32,
                actual: s.len(),
            });
        }

        for (l, i) in (0..2).rev().zip((0..2).map(|i| i * 16)) {
            BigEndian::write_u128(&mut s[i..], self.0[l]);
        }

        Ok(())
    }

    #[inline]
    pub fn zero() -> U256 {
        U256([0, 0])
    }

    #[inline]
    pub fn one() -> U256 {
        U256([1, 0])
    }

    /// Produce a random number (mod `modulo`)
    pub fn random<R: Rng>(rng: &mut R, modulo: &U256) -> U256 {
        U512::random(rng).divrem(modulo).1
    }

    pub fn is_zero(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0
    }

    pub fn set_bit(&mut self, n: usize, to: bool) -> bool {
        if n >= 256 {
            false
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            if to {
                self.0[part] |= 1 << bit;
            } else {
                self.0[part] &= !(1 << bit);
            }

            true
        }
    }

    pub fn get_bit(&self, n: usize) -> Option<bool> {
        if n >= 256 {
            None
        } else {
            let part = n / 128;
            let bit = n - (128 * part);

            Some(self.0[part] & (1 << bit) > 0)
        }
    }

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    /// Add `other` to `self` (mod `modulo`)
    pub fn add(&mut self, other: &U256, modulo: &U256) {
        let lhs: &[u32; 8] = bytemuck::cast_ref(&self.0);
        let rhs: &[u32; 8] = bytemuck::cast_ref(&other.0);
        let prime: &[u32; 8] = bytemuck::cast_ref(&modulo.0);
        let mut result = [0u128; 2];
        let result_mut: &mut [u32; 8] = bytemuck::cast_mut(&mut result);
        field::modadd_256(lhs, rhs, prime, result_mut);
        self.0 = result;
    }


    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    /// Add `other` to `self` (mod `modulo`)
    pub fn add(&mut self, other: &U256, modulo: &U256) {
        add_nocarry(&mut self.0, &other.0);

        if *self >= *modulo {
            sub_noborrow(&mut self.0, &modulo.0);
        }
    }


    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    /// Subtract `other` from `self` (mod `modulo`)
    pub fn sub(&mut self, other: &U256, modulo: &U256) {
        let lhs: &[u32; 8] = bytemuck::cast_ref(&self.0);
        let rhs: &[u32; 8] = bytemuck::cast_ref(&other.0);
        let prime: &[u32; 8] = bytemuck::cast_ref(&modulo.0);
        let mut result = [0u128; 2];
        let result_mut: &mut [u32; 8] = bytemuck::cast_mut(&mut result);
        field::modsub_256(lhs, rhs, prime, result_mut);
        self.0 = result;
    }


    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    /// Subtract `other` from `self` (mod `modulo`)
    pub fn sub(&mut self, other: &U256, modulo: &U256) {
        if *self < *other {
            add_nocarry(&mut self.0, &modulo.0);
        }

        sub_noborrow(&mut self.0, &other.0);
    }

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    /// Multiply `self` by `other` (mod `modulo`). Non-Montgomery form
    ///
    /// A standard modular multiplication that assumes none of the inputs are in Montgomery form.
    pub fn modmul(&mut self, other: &U256, modulo: &U256) {
        let lhs: &[u32; 8] = bytemuck::cast_ref(&self.0);
        let rhs: &[u32; 8] = bytemuck::cast_ref(&other.0);
        let prime: &[u32; 8] = bytemuck::cast_ref(&modulo.0);
        let mut result = [0u128; 2];
        let result_mut: &mut [u32; 8] = bytemuck::cast_mut(&mut result);
        field::modmul_256(lhs, rhs, prime, result_mut);
        self.0 = result;
    }

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    /// Montgomery multiply `self` by `other` (mod `modulo`)
    ///
    /// Note that this does not give the same answer as a "classical" multiply! Instead, it computes
    /// `self * other * R_inv % modulo` where `R_inv` is the inverse of 2^256 mod modulo.
    ///
    /// To match the non-precompile API, this accepts an `inv` parameter; we don't need it for
    /// computations, but we do verify that it matches its expected value under the assumption that
    /// R = (2^128)^2
    pub fn mul(&mut self, other: &U256, modulo: &U256, inv: u128) {
        let lhs: &[u32; 8] = bytemuck::cast_ref(&self.0);
        let rhs: &[u32; 8] = bytemuck::cast_ref(&other.0);
        let prime: &[u32; 8] = bytemuck::cast_ref(&modulo.0);
        let mut result = [0u128; 2];
        let result_mut: &mut [u32; 8] = bytemuck::cast_mut(&mut result);

        // Verifying the assumption R = (2^128)^2 (so b = 2^128, n = 2) against inv
        // To do so, we compute m_prime = -prime^(-1) mod 2^128 and confirm it is `inv`
        let mut neg_modulus_mod_2_218 = [0u32; 8];
        assert_ne!(prime[0], 0, "Modulus must be prime");
        for i in 0..4 {  // mod 2^128, so only need to do least significant half
            neg_modulus_mod_2_218[i] = u32::MAX - prime[i];
        }
        neg_modulus_mod_2_218[0] += 1;  // Safe to add 1 since prime[0] != 0
        let mut m_prime = [0u32; 8];
        let two_128 = [0u32, 0u32, 0u32, 0u32, 1u32, 0u32, 0u32, 0u32];
        field::modinv_256(&neg_modulus_mod_2_218, &two_128, &mut m_prime);
        let m_prime: u128 = m_prime[0] as u128 + (1 << 32) * (m_prime[1] as u128) + (1 << 64) * (m_prime[2] as u128) + (1 << 96) * (m_prime[3] as u128);
        assert_eq!(inv, m_prime);  // Otherwise we have an inconsistency in b^n used in the algorithm

        // This computes R = 2^256 % P = 2^256 - P via digit-wise subtraction, using the trick
        // that 2^256 - P will be the bit inverse of P plus 1.
        // Note: If prime < 2^128 this isn't normalized (i.e. we'll have R > prime), but we are
        // only using it as an input to a field operation, so it doesn't need to be
        let mut r = [0u32; 8];  // This gives a representation of 2^256 mod prime, which is R
        let mut carry_needed = true;
        for i in 0..8 {
            let val = prime[i];
            r[i] = u32::MAX - val;
            if carry_needed {
                if val == 0 {
                    // r[i] is zero and we still need to carry
                    r[i] = 0;
                    continue;
                }
                // Otherwise, we are done carrying
                r[i] += 1;
                carry_needed = false;
            }
        }
        assert!(!carry_needed, "Cannot use 0 for modulus in `mul`");
        let mut r_inv = [0u32; 8];
        field::modinv_256(&r, prime, &mut r_inv);

        let mut intermediate = [0u32; 8];
        field::modmul_256(lhs, rhs, prime, &mut intermediate);
        field::modmul_256(&intermediate, &r_inv, prime, result_mut);

        self.0 = result;
    }

    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    /// Multiply `self` by `other` (mod `modulo`) via the Montgomery
    /// multiplication method.
    pub fn mul(&mut self, other: &U256, modulo: &U256, inv: u128) {
        mul_reduce(&mut self.0, &other.0, &modulo.0, inv);

        if *self >= *modulo {
            sub_noborrow(&mut self.0, &modulo.0);
        }
    }

    /// Turn `self` into its additive inverse (mod `modulo`)
    pub fn neg(&mut self, modulo: &U256) {
        if *self > Self::zero() {
            let mut tmp = modulo.0;
            sub_noborrow(&mut tmp, &self.0);

            self.0 = tmp;
        }
    }

    #[inline]
    pub fn is_even(&self) -> bool {
        self.0[0] & 1 == 0
    }

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    /// Turn `self` into its multiplicative inverse (mod `modulo`)
    pub fn invert(&mut self, modulo: &U256) {
        let inp: &[u32; 8] = bytemuck::cast_ref(&self.0);
        let prime: &[u32; 8] = bytemuck::cast_ref(&modulo.0);
        let mut result = [0u128; 2];
        let result_mut: &mut [u32; 8] = bytemuck::cast_mut(&mut result);
        field::modinv_256(inp, prime, result_mut);
        self.0 = result;
    }

    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    /// Turn `self` into its multiplicative inverse (mod `modulo`)
    pub fn invert(&mut self, modulo: &U256) {
        // Guajardo Kumar Paar Pelzl
        // Efficient Software-Implementation of Finite Fields with Applications to Cryptography
        // Algorithm 16 (BEA for Inversion in Fp)

        let mut u = *self;
        let mut v = *modulo;
        let mut b = U256::one();
        let mut c = U256::zero();

        while u != U256::one() && v != U256::one() {
            while u.is_even() {
                div2(&mut u.0);

                if b.is_even() {
                    div2(&mut b.0);
                } else {
                    add_nocarry(&mut b.0, &modulo.0);
                    div2(&mut b.0);
                }
            }
            while v.is_even() {
                div2(&mut v.0);

                if c.is_even() {
                    div2(&mut c.0);
                } else {
                    add_nocarry(&mut c.0, &modulo.0);
                    div2(&mut c.0);
                }
            }

            if u >= v {
                sub_noborrow(&mut u.0, &v.0);
                b.sub(&c, modulo);
            } else {
                sub_noborrow(&mut v.0, &u.0);
                c.sub(&b, modulo);
            }
        }

        if u == U256::one() {
            self.0 = b.0;
        } else {
            self.0 = c.0;
        }
    }

    /// Return an Iterator<Item=bool> over all bits from
    /// MSB to LSB.
    pub fn bits(&self) -> BitIterator {
        BitIterator { int: &self, n: 256 }
    }

    pub const fn from_le_slice(bytes: &[u8]) -> Self {
        assert!(bytes.len() == 32);
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut idx = 0;

        while idx < 16 {
            lo[idx] = bytes[idx];
            hi[idx] = bytes[idx + 16];
            idx += 1;
        }

        U256([u128::from_le_bytes(lo), u128::from_le_bytes(hi)])
    }
}

pub struct BitIterator<'a> {
    int: &'a U256,
    n: usize,
}

impl<'a> Iterator for BitIterator<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        if self.n == 0 {
            None
        } else {
            self.n -= 1;

            self.int.get_bit(self.n)
        }
    }
}

/// Divide by two
#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
#[inline]
fn div2(a: &mut [u128; 2]) {
    let tmp = a[1] << 127;
    a[1] >>= 1;
    a[0] >>= 1;
    a[0] |= tmp;
}

/// Multiply by two
#[inline]
fn mul2(a: &mut [u128; 2]) {
    let tmp = a[0] >> 127;
    a[0] <<= 1;
    a[1] <<= 1;
    a[1] |= tmp;
}

#[inline(always)]
fn split_u128(i: u128) -> (u128, u128) {
    (i >> 64, i & 0xFFFFFFFFFFFFFFFF)
}

#[inline(always)]
fn combine_u128(hi: u128, lo: u128) -> u128 {
    (hi << 64) | lo
}

#[inline]
fn adc(a: u128, b: u128, carry: &mut u128) -> u128 {
    let (a1, a0) = split_u128(a);
    let (b1, b0) = split_u128(b);
    let (c, r0) = split_u128(a0 + b0 + *carry);
    let (c, r1) = split_u128(a1 + b1 + c);
    *carry = c;

    combine_u128(r1, r0)
}

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
#[inline]
fn add_nocarry(a: &mut [u128; 2], b: &[u128; 2]) {
    let mut carry = 0;

    for (a, b) in a.into_iter().zip(b.iter()) {
        *a = adc(*a, *b, &mut carry);
    }

    debug_assert!(0 == carry);
}

#[inline]
fn sub_noborrow(a: &mut [u128; 2], b: &[u128; 2]) {
    #[inline]
    fn sbb(a: u128, b: u128, borrow: &mut u128) -> u128 {
        let (a1, a0) = split_u128(a);
        let (b1, b0) = split_u128(b);
        let (b, r0) = split_u128((1 << 64) + a0 - b0 - *borrow);
        let (b, r1) = split_u128((1 << 64) + a1 - b1 - ((b == 0) as u128));

        *borrow = (b == 0) as u128;

        combine_u128(r1, r0)
    }

    let mut borrow = 0;

    for (a, b) in a.into_iter().zip(b.iter()) {
        *a = sbb(*a, *b, &mut borrow);
    }

    debug_assert!(0 == borrow);
}

// TODO: Make `from_index` a const param
#[inline(always)]
fn mac_digit(from_index: usize, acc: &mut [u128; 4], b: &[u128; 2], c: u128) {
    #[inline]
    fn mac_with_carry(a: u128, b: u128, c: u128, carry: &mut u128) -> u128 {
        let (b_hi, b_lo) = split_u128(b);
        let (c_hi, c_lo) = split_u128(c);

        let (a_hi, a_lo) = split_u128(a);
        let (carry_hi, carry_lo) = split_u128(*carry);
        let (x_hi, x_lo) = split_u128(b_lo * c_lo + a_lo + carry_lo);
        let (y_hi, y_lo) = split_u128(b_lo * c_hi);
        let (z_hi, z_lo) = split_u128(b_hi * c_lo);
        // Brackets to allow better ILP
        let (r_hi, r_lo) = split_u128((x_hi + y_lo) + (z_lo + a_hi) + carry_hi);

        *carry = (b_hi * c_hi) + r_hi + y_hi + z_hi;

        combine_u128(r_lo, x_lo)
    }

    if c == 0 {
        return;
    }

    let mut carry = 0;

    debug_assert_eq!(acc.len(), 4);
    unroll! {
        for i in 0..2 {
            let a_index = i + from_index;
            acc[a_index] = mac_with_carry(acc[a_index], b[i], c, &mut carry);
        }
    }
    unroll! {
        for i in 0..2 {
            let a_index = i + from_index + 2;
            if a_index < 4 {
                let (a_hi, a_lo) = split_u128(acc[a_index]);
                let (carry_hi, carry_lo) = split_u128(carry);
                let (x_hi, x_lo) = split_u128(a_lo + carry_lo);
                let (r_hi, r_lo) = split_u128(x_hi + a_hi + carry_hi);

                carry = r_hi;

                acc[a_index] = combine_u128(r_lo, x_lo);
            }
        }
    }

    debug_assert!(carry == 0);
}

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
#[inline]
fn mul_reduce(this: &mut [u128; 2], by: &[u128; 2], modulus: &[u128; 2], inv: u128) {
    // The Montgomery reduction here is based on Algorithm 14.32 in
    // Handbook of Applied Cryptography
    // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

    let mut res = [0; 2 * 2];
    unroll! {
        for i in 0..2 {
            mac_digit(i, &mut res, by, this[i]);
        }
    }

    unroll! {
        for i in 0..2 {
            let k = inv.wrapping_mul(res[i]);
            mac_digit(i, &mut res, modulus, k);
        }
    }

    this.copy_from_slice(&res[2..]);
}

#[test]
fn setting_bits() {
    let rng = &mut ::rand::thread_rng();
    let modulo = U256::from([0xffffffffffffffff; 4]);

    let a = U256::random(rng, &modulo);
    let mut e = U256::zero();
    for (i, b) in a.bits().enumerate() {
        assert!(e.set_bit(255 - i, b));
    }

    assert_eq!(a, e);
}

#[test]
fn from_slice() {
    let tst = U256::one();
    let mut s = [0u8; 32];
    s[31] = 1;

    let num =
        U256::from_slice(&s).expect("U256 should initialize ok from slice in `from_slice` test");
    assert_eq!(num, tst);
}

#[test]
fn to_big_endian() {
    let num = U256::one();
    let mut s = [0u8; 32];

    num.to_big_endian(&mut s)
        .expect("U256 should convert to bytes ok in `to_big_endian` test");
    assert_eq!(
        s,
        [
            0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
            0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8,
        ]
    );
}

#[test]
fn testing_divrem() {
    let rng = &mut ::rand::thread_rng();

    let modulo = U256::from([
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ]);

    for _ in 0..2 {
        let c0 = U256::random(rng, &modulo);
        let c1 = U256::random(rng, &modulo);

        let c1q_plus_c0 = U512::new(&c1, &c0, &modulo);

        let (new_c1, new_c0) = c1q_plus_c0.divrem(&modulo);

        assert!(c1 == new_c1.unwrap());
        assert!(c0 == new_c0);
    }

    {
        // Modulus should become 1*q + 0
        let a = U512::from([
            0x3c208c16d87cfd47,
            0x97816a916871ca8d,
            0xb85045b68181585d,
            0x30644e72e131a029,
            0,
            0,
            0,
            0,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert_eq!(c1.unwrap(), U256::one());
        assert_eq!(c0, U256::zero());
    }

    {
        // Modulus squared minus 1 should be (q-1) q + q-1
        let a = U512::from([
            0x3b5458a2275d69b0,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert_eq!(
            c1.unwrap(),
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
        assert_eq!(
            c0,
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
    }

    {
        // Modulus squared minus 2 should be (q-1) q + q-2
        let a = U512::from([
            0x3b5458a2275d69af,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);

        assert_eq!(
            c1.unwrap(),
            U256::from([
                0x3c208c16d87cfd46,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
        assert_eq!(
            c0,
            U256::from([
                0x3c208c16d87cfd45,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029
            ])
        );
    }

    {
        // Ridiculously large number should fail
        let a = U512::from([
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(
            c0,
            U256::from([
                0xf32cfc5b538afa88,
                0xb5e71911d44501fb,
                0x47ab1eff0a417ff6,
                0x06d89f71cab8351f
            ])
        );
    }

    {
        // Modulus squared should fail
        let a = U512::from([
            0x3b5458a2275d69b1,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(c0, U256::zero());
    }

    {
        // Modulus squared plus one should fail
        let a = U512::from([
            0x3b5458a2275d69b2,
            0xa602072d09eac101,
            0x4a50189c6d96cadc,
            0x04689e957a1242c8,
            0x26edfa5c34c6b38d,
            0xb00b855116375606,
            0x599a6f7c0348d21c,
            0x0925c4b8763cbf9c,
        ]);

        let (c1, c0) = a.divrem(&modulo);
        assert!(c1.is_none());
        assert_eq!(c0, U256::one());
    }

    {
        let modulo = U256::from([
            0x43e1f593f0000001,
            0x2833e84879b97091,
            0xb85045b68181585d,
            0x30644e72e131a029,
        ]);

        // Fr modulus masked off is valid
        let a = U512::from([
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x07ffffffffffffff,
        ]);

        let (c1, c0) = a.divrem(&modulo);

        assert!(c1.unwrap() < modulo);
        assert!(c0 < modulo);
    }
}

// TODO: Do we want to extend tests of U512?
// i.e. new, from_slice, random, get_bit, interpret

#[test]
fn r0_basic_arith() {
    let prime = U256::from([
        0x3C208C16D87CFD47,
        0x97816A916871CA8D,
        0xB85045B68181585D,
        0x30644E72E131A029,
    ]);
    let mut val = U256::zero();
    val.add(&U256::zero(), &prime);
    assert_eq!(val, U256::zero());

    let mut val = U256::one();
    val.add(&U256::zero(), &prime);
    assert_eq!(val, U256::one());

    let mut val = U256::zero();
    val.add(&U256::one(), &prime);
    assert_eq!(val, U256::one());

    let mut val = U256::one();
    val.sub(&U256::one(), &prime);
    assert_eq!(val, U256::zero());

    let mut val = U256::one();
    val.sub(&U256::zero(), &prime);
    assert_eq!(val, U256::one());

    let mut val = U256::zero();
    val.sub(&U256::one(), &prime);
    val.neg(&prime);
    assert_eq!(val, U256::one());

    let mut val = U256::zero();
    val.sub(&U256::zero(), &prime);
    assert_eq!(val, U256::zero());

    assert!(U256::zero().is_even());
    assert!(!U256::one().is_even());

    let mut val = U256::one();
    val.invert(&prime);
    assert_eq!(val, U256::one());
}

#[test]
fn r0_test_mul() {
    // Tests Montgomery multiplication with known values (5 * 10 = 50)
    let mut lhs = U256::from([
        0x5059E8694A6C0E5F,
        0x3F69A66D0E742784,
        0x625AF360AB101812,
        0x2E12444C75E1B101,
    ]);
    let rhs = U256::from([
        0x58E95CFC540200FD,
        0xB0E12FC6F4485774,
        0x338FF155E94B9396,
        0x13B994C81C2F53A6,
    ]);
    let expected = U256::from([
        0x3D084613B3A71993,
        0xDE7F9C7C725C25C1,
        0x69D81708926CB9CD,
        0x17472F351E407698,
    ]);
    let prime = U256::from([
        0x3C208C16D87CFD47,
        0x97816A916871CA8D,
        0xB85045B68181585D,
        0x30644E72E131A029,
    ]);
    let inv = 0x9ede7d651eca6ac987d20782e4866389;

    lhs.mul(&rhs, &prime, inv);

    assert_eq!(lhs, expected);
}

#[test]
fn r0_from_le_slice() {
    let one_bytes: [u8; 32] = [
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];
    assert_eq!(U256::from_le_slice(&one_bytes), U256::one());

    const le_bytes: [u8; 32] = [
        47, 24, 11, 13, 33, 22, 10, 20,
        0, 40, 0, 0, 0, 0, 0, 0,
        0, 0, 50, 0, 0, 0, 0, 0,
        0, 0, 0, 60, 0, 0, 0, 34,
    ];
    const val_from_le: U256 = U256::from_le_slice(&le_bytes);
    let be_bytes: [u8; 32] = [
        34, 0, 0, 0, 60, 0, 0, 0,
        0, 0, 0, 0, 0, 50, 0, 0,
        0, 0, 0, 0, 0, 0, 40, 0,
        20, 10, 22, 33, 13, 11, 24, 47,
    ];
    assert_eq!(val_from_le, U256::from_slice(&be_bytes).unwrap());
    assert_eq!(val_from_le, U256::from([
        0x140A16210D0B182F,
        0x0000000000002800,
        0x0000000000320000,
        0x220000003C000000,
    ]));

    assert_eq!(val_from_le.get_bit(0), Some(true));
    assert_eq!(val_from_le.get_bit(1), Some(true));
    assert_eq!(val_from_le.get_bit(2), Some(true));
    assert_eq!(val_from_le.get_bit(3), Some(true));
    assert_eq!(val_from_le.get_bit(4), Some(false));
    assert_eq!(val_from_le.get_bit(5), Some(true));
    assert_eq!(val_from_le.get_bit(6), Some(false));
    assert_eq!(val_from_le.get_bit(7), Some(false));
}
