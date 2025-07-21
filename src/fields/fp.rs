use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};
use bytemuck::{Zeroable, Pod};
use rand::Rng;
use crate::fields::FieldElement;
use crate::arith::{U256, U512};

macro_rules! field_impl {
    ($name:ident, $modulus:expr, $modulussquared:expr, $rsquared:expr, $rcubed:expr, $r:expr, $inv:expr, $rinv:expr) => {
        #[derive(Copy, Clone, PartialEq, Eq, Debug, Pod, Zeroable)]
        #[repr(C)]
        pub struct $name(U256);

        impl From<$name> for U256 {
            #[inline]
            fn from(a: $name) -> Self {
                // convert to canonical form
                a.reduce_mont().0
            }
        }

        impl $name {
            pub const MODULUS: U256 = U256($modulus);
            pub const MODULUS_SQUARED: U512 = U512($modulussquared);

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
            pub fn new(a: U256) -> Option<Self> {
                if a < U256($modulus) {
                    Some(Self::new_mul_factor(a))
                } else {
                    None
                }
            }

            /// Converts a U256 to an Fr regardless of modulus.
            pub fn new_mul_factor(mut a: U256) -> Self {
                #[cfg(not(any(target_r0vm, feature = "risc0")))]
                {
                    // Montgomery multiplication by R² converts an integer to Montgomery form
                    a.mul_mont(&U256($rsquared), &U256($modulus), $inv);
                }

                #[cfg(any(target_r0vm, feature = "risc0"))]
                {
                    // R0VM fast-path: plain modular multiply by R
                    a.mul(&U256($r), &U256($modulus));
                }

                Self(a)
            }

            #[inline]
            pub fn reduce_mont(mut self) -> Self {
                #[cfg(not(any(target_r0vm, feature = "risc0")))]
                {
                    self.0.mul_mont(&U256($r), &U256($modulus), $inv);
                }

                #[cfg(any(target_r0vm, feature = "risc0"))]
                {
                    self.0.mul(&U256($rinv), &U256($modulus));
                }

                self
            }

            pub fn interpret(buf: &[u8; 64]) -> Self {
                $name::new(U512::interpret(buf).divrem(&U256($modulus)).1).unwrap()
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
                $name(U256([0, 0]))
            }

            #[inline]
            fn one() -> Self {
                $name(U256($r))
            }

            fn random<R: Rng>(rng: &mut R) -> Self {
                $name(U256::random(rng, &U256($modulus)))
            }

            #[inline]
            fn is_zero(&self) -> bool {
                self.0.is_zero()
            }

            fn inverse(mut self) -> Option<Self> {
                if self.is_zero() {
                    return None;
                }

                // (xR)^-1 = x^-1 R^-1, thus we need to multiply by R² for correct Montgomery form
                self.0.invert(&U256($modulus));

                #[cfg(not(any(target_r0vm, feature = "risc0")))]
                {
                    // Montgomery multiplication by R³ effectively multiplies the integer by R²
                    self.0.mul_mont(&U256($rcubed), &U256($modulus), $inv);
                }

                #[cfg(any(target_r0vm, feature = "risc0"))]
                {
                    // R0VM fast-path: plain modular multiply by R²
                    self.0.mul(&U256($rsquared), &U256($modulus));
                }

                Some(self)
            }
        }

        impl Add for $name {
            type Output = $name;

            #[inline]
            fn add(mut self, other: $name) -> $name {
                self.0.add(&other.0, &U256($modulus));

                self
            }
        }

        impl Sub for $name {
            type Output = $name;

            #[inline]
            fn sub(mut self, other: $name) -> $name {
                self.0.sub(&other.0, &U256($modulus));

                self
            }
        }

        impl Mul for $name {
            type Output = $name;

            #[inline]
            fn mul(mut self, other: $name) -> $name {
                #[cfg(not(any(target_r0vm, feature = "risc0")))]
                {
                    self.0.mul_mont(&other.0, &U256($modulus), $inv);
                }

                #[cfg(any(target_r0vm, feature = "risc0"))]
                {
                    self.0.mul_mont(&other.0, &U256($modulus), &U256($rinv));
                }

                self
            }
        }

        impl Neg for $name {
            type Output = $name;

            #[inline]
            fn neg(mut self) -> $name {
                self.0.neg(&U256($modulus));

                self
            }
        }
    }
}

field_impl!(
    Fr,
    [
        0x2833e84879b9709143e1f593f0000001,
        0x30644e72e131a029b85045b68181585d,
    ],
        [
        0xC7F26223DCB3400008C3EB27E0000001,
        0xA6CE1975E821DDB0FFE9A62C68C9BB7F,
        0x85F73BB0D379D3DF2C77527B47B62FE7,
        0x0925C4B8763CBF9C599A6F7C0348D21C,
    ],
    [
        0x53fe3ab1e35c59e31bb8e645ae216da7,
        0x0216d0b17f4e44a58c49833d53bb8085,
    ],
    [
        0x2a489cbe1cfbb6b85e94d8e1b4bf0040,
        0x0cf8594b7fcc657c893cc664a19fcfed,
    ],
    [
        0x36fc76959f60cd29ac96341c4ffffffb,
        0xe0a77c19a07df2f666ea36f7879462e,
    ],
    0x6586864b4c6911b3c2e1f593efffffff,
    [
        0x90ef5a9e111ec87dc5ba0056db1194e,
        0x15ebf95182c5551cc8260de4aeb85d5d,
    ]
);

field_impl!(
    Fq,
    [
        0x97816a916871ca8d3c208c16d87cfd47,
        0x30644e72e131a029b85045b68181585d,
    ],
    [
        0xA602072D09EAC1013B5458A2275D69B1,
        0x04689E957A1242C84A50189C6D96CADC,
        0xB00B85511637560626EDFA5C34C6B38D,
        0x0925C4B8763CBF9C599A6F7C0348D21C,
    ],
    [
        0xb5e71911d44501fbf32cfc5b538afa89,
        0x06d89f71cab8351f47ab1eff0a417ff6,
    ],
    [
        0x62f210e6a7283db6b1cd6dafda1530df,
        0x20fd6e902d592544ef7f0b0c0ada0afb,
    ],
    [
        0xa78eb28f5c70b3dd35d438dc58f0d9d,
        0xe0a77c19a07df2f666ea36f7879462c,
    ],
    0x9ede7d651eca6ac987d20782e4866389,
    [
        0xeb2022850278edf8ed84884a014afa37,
        0x2e67157159e5c639cf63e9cfb74492d9,
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

#[inline]
pub fn const_fq(i: [u64; 4]) -> Fq {
    Fq(U256::from(i))
}

#[test]
fn test_rsquared() {
    let rng = &mut ::rand::thread_rng();

    for _ in 0..1000 {
        let a = Fr::random(rng);
        let b: U256 = a.into();
        let c = Fr::new(b).unwrap();

        assert_eq!(a, c);
    }

    for _ in 0..1000 {
        let a = Fq::random(rng);
        let b: U256 = a.into();
        let c = Fq::new(b).unwrap();

        assert_eq!(a, c);
    }
}


#[test]
fn sqrt_fq() {
    // from zcash test_proof.cpp
    let fq1 = Fq::from_str("5204065062716160319596273903996315000119019512886596366359652578430118331601").unwrap();
    let fq2 = Fq::from_str("348579348568").unwrap();

    assert_eq!(fq1, fq2.sqrt().expect("348579348568 is quadratic residue"));
}
