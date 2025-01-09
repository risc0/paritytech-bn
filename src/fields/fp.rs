use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};
use rand::Rng;
use crate::fields::FieldElement;
use crate::arith::{U256, U512};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use risc0_bigint2::field;

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
macro_rules! field_impl {
    ($name:ident, $modulus:expr, $rsquared:expr, $rcubed:expr, $one:expr, $inv:expr) => {
        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        #[repr(C)]
        pub struct $name(U256);


        impl From<$name> for U256 {
            #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
            #[inline]
            fn from(mut a: $name) -> Self {
                a.0.mul(&U256::one(), &U256::from($modulus), $inv);

                a.0
            }

            #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
            #[inline]
            fn from(mut a: $name) -> Self {
                a.0
            }
        }

        impl $name {
            pub fn from_str(s: &str) -> Option<Self> {
                let ints: Vec<_> = {
                    let mut acc = Self::zero();
                    (0..11).map(|_| {let tmp = acc; acc = acc + Self::one(); tmp}).collect()
                };

                let mut res = Self::zero();
                for c in s.chars() {
                    match c.to_digit(10) {
                        Some(d) => {
                            res = res * ints[10];
                            res = res + ints[d as usize];
                        },
                        None => {
                            return None;
                        }
                    }
                }

                Some(res)
            }

            /// Converts a U256 to an Fp so long as it's below the modulus.
            pub fn new(mut a: U256) -> Option<Self> {
                if a < U256::from($modulus) {
                    a.mul(&U256::from($rsquared), &U256::from($modulus), $inv);

                    Some($name(a))
                } else {
                    None
                }
            }

            /// Converts a U256 to an Fr regardless of modulus.
            pub fn new_mul_factor(mut a: U256) -> Self {
                a.mul(&U256::from($rsquared), &U256::from($modulus), $inv);
                $name(a)
            }

            pub fn interpret(buf: &[u8; 64]) -> Self {
                $name::new(U512::interpret(buf).divrem(&U256::from($modulus)).1).unwrap()
            }

            /// Returns the modulus
            #[inline]
            #[allow(dead_code)]
            pub fn modulus() -> U256 {
                U256::from($modulus)
            }

            // TODO: Pretty pointless in the zkVM but I guess why not
            #[inline]
            #[allow(dead_code)]
            pub fn inv(&self) -> u128 {
                $inv
            }

            pub fn raw(&self) -> &U256 {
                &self.0
            }

            pub fn set_bit(&mut self, bit: usize, to: bool) {
                self.0.set_bit(bit, to);
            }
        }

        impl FieldElement for $name {
            #[inline]
            fn zero() -> Self {
                $name(U256::from([0, 0, 0, 0]))
            }

            #[inline]
            fn one() -> Self {
                $name(U256::from($one))
            }

            fn random<R: Rng>(rng: &mut R) -> Self {
                $name(U256::random(rng, &U256::from($modulus)))
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }

            fn inverse(mut self) -> Option<Self> {
                if self.is_zero() {
                    None
                } else {
                    self.0.invert(&U256::from($modulus));
                    self.0.mul(&U256::from($rcubed), &U256::from($modulus), $inv);

                    Some(self)
                }
            }
        }

        impl Add for $name {
            type Output = $name;

            #[inline]
            fn add(mut self, other: $name) -> $name {
                self.0.add(&other.0, &U256::from($modulus));

                self
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline]
            fn sub(mut self, other: $name) -> $name {
                self.0.sub(&other.0, &U256::from($modulus));

                self
            }
        }

        impl Mul for $name {
            type Output = $name;

            #[inline]
            fn mul(mut self, other: $name) -> $name {
                self.0.mul(&other.0, &U256::from($modulus), $inv);

                self
            }
        }

        impl Neg for $name {
            type Output = $name;

            #[inline]
            fn neg(mut self) -> $name {
                self.0.neg(&U256::from($modulus));

                self
            }
        }
    }
}

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
macro_rules! field_impl {
    ($name:ident, $modulus:expr, $rsquared:expr, $rcubed:expr, $one:expr, $inv:expr, $r:expr, $rinv:expr) => {
        #[derive(Copy, Clone, PartialEq, Eq, Debug)]
        #[repr(C)]
        pub struct $name(U256);


        impl From<$name> for U256 {
            #[inline]
            fn from(mut a: $name) -> Self {
                // TODO: Note: Can skip mul b/c raw form is simpler in zkvm
                a.0
            }
        }

        impl $name {
            pub fn from_str(s: &str) -> Option<Self> {
                // TODO: Same as base. Could make a little more efficient in the zkVM, but ... why bother?
                let ints: Vec<_> = {
                    let mut acc = Self::zero();
                    (0..11).map(|_| {let tmp = acc; acc = acc + Self::one(); tmp}).collect()
                };

                let mut res = Self::zero();
                for c in s.chars() {
                    match c.to_digit(10) {
                        Some(d) => {
                            res = res * ints[10];
                            res = res + ints[d as usize];
                        },
                        None => {
                            return None;
                        }
                    }
                }

                Some(res)
            }

            pub fn new(mut a: U256) -> Option<Self> {
                // TODO: Note: This is the simpler zkVM case
                if a < U256::from($modulus) {
                    Some($name(a))
                } else {
                    None
                }
            }

            pub fn new_mul_factor(mut a: U256) -> Self {
                // TODO: Note this is the simpler zkVM case
                // Note: This adds zero to force a reduce
                // TODO: Maybe make a reduce op?
                $name(a) + $name(U256::from([0, 0, 0, 0]))
            }

            // TODO: Do we actually even need these?
            const fn r() -> Self {
                Self(U256::from_le_bytes($r))
                // Self(U256::from($r))
            }

            const fn r_inv() -> Self {
                Self(U256::from_le_bytes($rinv))
                // Self(U256::from($rinv))
            }

            pub fn to_montgomery(mut self) -> Self {
                self.mul(Self::r())
            }

            pub fn from_montgomery(mut self) -> Self {
                self.mul(Self::r_inv())
            }

            /// Parse 64 bytes (big-endian) as a field element
            ///
            /// Includes reducing by the modulus if necessary
            pub fn interpret(buf: &[u8; 64]) -> Self {
                $name::new(U512::interpret(buf).divrem(&U256::from($modulus)).1).unwrap()
            }

            /// Returns the modulus
            #[inline]
            #[allow(dead_code)]
            pub fn modulus() -> U256 {
                // TODO: Note: Same as base case
                U256::from($modulus)
            }

            #[inline]
            #[allow(dead_code)]
            pub fn inv(&self) -> u128 {
                // TODO: Pretty pointless in the zkVM but I guess why not. Note: same as base case
                $inv
            }

            pub fn raw(&self) -> &U256 {
                unimplemented!("There is no `raw` Montgomery representation; consider `raw_nonmont`");
            }

            pub fn raw_nonmont(&self) -> &U256 {
                &self.0
            }

            pub fn set_bit(&mut self, bit: usize, to: bool) {
                // TODO: Maintaining set_bit semantics is annoying
                unimplemented!("TODO: Maitaining `set_bit` semantics is annoying");
            }

            // TODO: Test my new const initializing functions
            pub const fn from_le_slice(bytes: &[u8]) -> Self {
                $name(U256::from_le_slice(&bytes))
            }
        }

        impl FieldElement for $name {
            #[inline]
            fn zero() -> Self {
                // TODO: Note: Same as base case
                $name(U256::from([0, 0, 0, 0]))
            }

            #[inline]
            fn one() -> Self {
                // TODO: Note: Simpler than base case
                $name(U256::from([1, 0, 0, 0]))
            }

            // TODO: Does it matter that we get different random numbers (Montgomery vs. not)?
            fn random<R: Rng>(rng: &mut R) -> Self {
                $name(U256::random(rng, &U256::from($modulus)))
            }

            #[inline]
            fn is_zero(&self) -> bool {
                // TODO: Note: Same as base case
                self.0.is_zero()
            }

            fn inverse(mut self) -> Option<Self> {
                // TODO: Note: Simpler than base case
                if self.is_zero() {
                    None
                } else {
                    self.0.invert(&U256::from($modulus));
                    Some(self)
                }
            }
        }

        impl Add for $name {
            type Output = $name;

            #[inline]
            fn add(mut self, other: $name) -> $name {
                // TODO: Note: Same as base case
                self.0.add(&other.0, &U256::from($modulus));

                self
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline]
            fn sub(mut self, other: $name) -> $name {
                // TODO: Note: Simpler than base case
                self.0.sub(&other.0, &U256::from($modulus));

                self
            }
        }

        impl Mul for $name {
            type Output = $name;

            #[inline]
            fn mul(mut self, other: $name) -> $name {
                // TODO: Note: Simpler than base case
                self.0.modmul(&other.0, &U256::from($modulus));

                self
            }
        }

        impl Neg for $name {
            type Output = $name;

            #[inline]
            fn neg(mut self) -> $name {
                // TODO: Note: Same as base case
                self.0.neg(&U256::from($modulus));

                self
            }
        }
    }
}

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
field_impl!(
    Fr,
    [
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029
    ],
    [
        0x1bb8e645ae216da7,
        0x53fe3ab1e35c59e3,
        0x8c49833d53bb8085,
        0x0216d0b17f4e44a5
    ],
    [
        0x5e94d8e1b4bf0040,
        0x2a489cbe1cfbb6b8,
        0x893cc664a19fcfed,
        0x0cf8594b7fcc657c
    ],
    [
        0xac96341c4ffffffb,
        0x36fc76959f60cd29,
        0x666ea36f7879462e,
        0xe0a77c19a07df2f
    ],
    0x6586864b4c6911b3c2e1f593efffffff
);

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
field_impl!(
    Fq,
    [
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029
    ],
    [
        0xf32cfc5b538afa89,
        0xb5e71911d44501fb,
        0x47ab1eff0a417ff6,
        0x06d89f71cab8351f
    ],
    [
        0xb1cd6dafda1530df,
        0x62f210e6a7283db6,
        0xef7f0b0c0ada0afb,
        0x20fd6e902d592544
    ],
    [
        0xd35d438dc58f0d9d,
        0xa78eb28f5c70b3d,
        0x666ea36f7879462c,
        0xe0a77c19a07df2f
    ],
    0x9ede7d651eca6ac987d20782e4866389
);

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
field_impl!(
    Fr,
    [
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029
    ],
    [
        0x1bb8e645ae216da7,
        0x53fe3ab1e35c59e3,
        0x8c49833d53bb8085,
        0x0216d0b17f4e44a5
    ],
    [
        0x5e94d8e1b4bf0040,
        0x2a489cbe1cfbb6b8,
        0x893cc664a19fcfed,
        0x0cf8594b7fcc657c
    ],
    [
        0xac96341c4ffffffb,
        0x36fc76959f60cd29,
        0x666ea36f7879462e,
        0xe0a77c19a07df2f
    ],
    0x6586864b4c6911b3c2e1f593efffffff,
    [
        0xff, 0xff, 0xff, 0x0f, 0x6c, 0x0a, 0x1e, 0xbc,
        0x6e, 0x8f, 0x46, 0x86, 0xb7, 0x17, 0xcc, 0xd7,
        0xa2, 0xa7, 0x7e, 0x7e, 0x49, 0xba, 0xaf, 0x47,
        0xd6, 0x5f, 0xce, 0x1e, 0x8d, 0xb1, 0x9b, 0xcf,
    ],
    [
        0x4e, 0x19, 0xb1, 0x6d, 0x05, 0xa0, 0x5b, 0xdc,
        0x87, 0xec, 0x11, 0xe1, 0xa9, 0xf5, 0x0e, 0x09,
        0x5d, 0x5d, 0xb8, 0xae, 0xe4, 0x0d, 0x26, 0xc8,
        0x1c, 0x55, 0xc5, 0x82, 0x51, 0xf9, 0xeb, 0x15,
    ]
);

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
field_impl!(
    Fq,
    [
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029
    ],
    [
        0xf32cfc5b538afa89,
        0xb5e71911d44501fb,
        0x47ab1eff0a417ff6,
        0x06d89f71cab8351f
    ],
    [
        0xb1cd6dafda1530df,
        0x62f210e6a7283db6,
        0xef7f0b0c0ada0afb,
        0x20fd6e902d592544
    ],
    [
        0xd35d438dc58f0d9d,
        0xa78eb28f5c70b3d,
        0x666ea36f7879462c,
        0xe0a77c19a07df2f
    ],
    0x9ede7d651eca6ac987d20782e4866389,
    [
        0xb9, 0x02, 0x83, 0x27, 0xe9, 0x73, 0xdf, 0xc3,
        0x72, 0x35, 0x8e, 0x97, 0x6e, 0x95, 0x7e, 0x68,
        0xa2, 0xa7, 0x7e, 0x7e, 0x49, 0xba, 0xaf, 0x47,
        0xd6, 0x5f, 0xce, 0x1e, 0x8d, 0xba, 0x9b, 0xcf,
    ],
    [
        0x37, 0xfa, 0x4a, 0x01, 0x4a, 0x88, 0x84, 0xed,
        0xf8, 0xed, 0x78, 0x02, 0x85, 0x22, 0x20, 0xeb,
        0xd9, 0x92, 0x44, 0xb7, 0xcf, 0xe9, 0x63, 0xcf,
        0x39, 0xc6, 0xe5, 0x59, 0x71, 0x15, 0x67, 0x2e,
    ]
);

lazy_static::lazy_static! {

    static ref FQ: U256 = U256::from([
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029
    ]);

	pub static ref FQ_MINUS3_DIV4: Fq =
		Fq::new(3.into()).expect("3 is a valid field element and static; qed").neg() *
		Fq::new(4.into()).expect("4 is a valid field element and static; qed").inverse()
			.expect("4 has inverse in Fq and is static; qed");

	static ref FQ_MINUS1_DIV2: Fq =
		Fq::new(1.into()).expect("1 is a valid field element and static; qed").neg() *
		Fq::new(2.into()).expect("2 is a valid field element and static; qed").inverse()
			.expect("2 has inverse in Fq and is static; qed");

}

impl Fq {
    pub fn sqrt(&self) -> Option<Self> {
        let a1 = self.pow(*FQ_MINUS3_DIV4);
        let a1a = a1 * *self;
        let a0 = a1 * (a1a);

        let mut am1 = *FQ;
        am1.sub(&1.into(), &*FQ);

        if a0 == Fq::new(am1).unwrap() {
            None
        } else {
            Some(a1a)
        }
    }
}

#[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
#[inline]
pub fn const_fq(i: [u64; 4]) -> Fq {
    Fq(U256::from(i))
}

// TODO: Add a variant of this that reads raw non-Montgomery bytes
#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
#[inline]
pub fn const_fq(i: [u64; 4]) -> Fq {
    // The semantics assume the input is in Montgomery form, but we don't use Montgomery form internally
    Fq(U256::from(i)).from_montgomery()
}

// TODO: Skip: Passing
#[test]
fn test_rsquared() {
    let rng = &mut ::rand::thread_rng();

    for _ in 0..2 {  // TODO: From 1000
        let a = Fr::random(rng);
        let b: U256 = a.into();
        let c = Fr::new(b).unwrap();

        assert_eq!(a, c);
    }

    for _ in 0..2 {  // TODO: From 1000
        let a = Fq::random(rng);
        let b: U256 = a.into();
        let c = Fq::new(b).unwrap();

        assert_eq!(a, c);
    }
}

// TODO: Skip: Passing
#[test]
fn tnz_simple_square() {
    // TODO: Based on sqrt_fq
    let fq1 = Fq::from_str("5204065062716160319596273903996315000119019512886596366359652578430118331601").unwrap();
    let fq2 = Fq::from_str("348579348568").unwrap();

    assert_eq!(fq1 * fq1, fq2);
}

// TODO: Skip: Passing
#[test]
fn tnz_basic_mul() {
    // let two = Fq::from(U256::from(2u64));
    // let three = Fq::from(U256::from(3u64));
    let two = Fq::from_str("2").unwrap();
    let three = Fq::from_str("3").unwrap();

    assert_eq!(U256::from(two * three), U256::from(6u64));
}

#[test]
fn tnz_from_le_slice() {
    assert_eq!(Fq::one(), Fq::from_le_slice(&[
        1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]));
}

// TODO: Skip: Passing
#[test]
fn sqrt_fq() {
    // from zcash test_proof.cpp
    let fq1 = Fq::from_str("5204065062716160319596273903996315000119019512886596366359652578430118331601").unwrap();
    let fq2 = Fq::from_str("348579348568").unwrap();

    assert_eq!(fq1, fq2.sqrt().expect("348579348568 is quadratic residue"));
}
