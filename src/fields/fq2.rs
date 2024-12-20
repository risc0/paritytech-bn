use core::ops::{Add, Mul, Neg, Sub};
use rand::Rng;
use crate::fields::{const_fq, FieldElement, Fq};
use crate::arith::{U256, U512};

#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use bytemuck;
#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use core::mem::MaybeUninit;
#[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
use risc0_bigint2::field;

// #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
#[inline]
fn fq_non_residue() -> Fq {
    // (q - 1) is a quadratic nonresidue in Fq
    // 21888242871839275222246405745257275088696311157297823662689037894645226208582
    const_fq([
        0x68c3488912edefaa,
        0x8d087f6872aabf4f,
        0x51e1a24709081231,
        0x2259d6b14729c0fa,
    ])
}

// #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
// #[inline]
// fn fq_non_residue() -> Fq {
//     // (q - 1) is a quadratic nonresidue in Fq
//     // 21888242871839275222246405745257275088696311157297823662689037894645226208582
//     const_fq([
//         0x3C208C16D87CFD46,
//         0x97816A916871CA8D,
//         0xB85045B68181585D,
//         0x30644E72E131A029,
//     ])
// }

#[inline]
pub fn fq2_nonresidue() -> Fq2 {
    Fq2::new(
        const_fq([
            0xf60647ce410d7ff7,
            0x2f3d6f4dd31bd011,
            0x2943337e3940c6d1,
            0x1d9598e8a7e39857,
        ]),
        const_fq([
            0xd35d438dc58f0d9d,
            0x0a78eb28f5c70b3d,
            0x666ea36f7879462c,
            0x0e0a77c19a07df2f,
        ]),
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fq2 {
    c0: Fq,
    c1: Fq,
}

impl Fq2 {
    pub fn new(c0: Fq, c1: Fq) -> Self {
        Fq2 { c0: c0, c1: c1 }
    }

    pub fn scale(&self, by: Fq) -> Self {
        Fq2 {
            c0: self.c0 * by,
            c1: self.c1 * by,
        }
    }

    pub fn mul_by_nonresidue(&self) -> Self {
        *self * fq2_nonresidue()
    }

    pub fn frobenius_map(&self, power: usize) -> Self {
        if power % 2 == 0 {
            *self
        } else {
            Fq2 {
                c0: self.c0,
                c1: self.c1 * fq_non_residue(),
            }
        }
    }

    pub fn real(&self) -> &Fq {
        &self.c0
    }

    pub fn imaginary(&self) -> &Fq {
        &self.c1
    }
}

impl FieldElement for Fq2 {
    fn zero() -> Self {
        Fq2 {
            c0: Fq::zero(),
            c1: Fq::zero(),
        }
    }

    fn one() -> Self {
        Fq2 {
            c0: Fq::one(),
            c1: Fq::zero(),
        }
    }

    fn random<R: Rng>(rng: &mut R) -> Self {
        Fq2 {
            c0: Fq::random(rng),
            c1: Fq::random(rng),
        }
    }

    fn is_zero(&self) -> bool {
        self.c0.is_zero() && self.c1.is_zero()
    }

    fn squared(&self) -> Self {
        // Devegili OhEig Scott Dahab
        //     Multiplication and Squaring on Pairing-Friendly Fields.pdf
        //     Section 3 (Complex squaring)

        let ab = self.c0 * self.c1;

        Fq2 {
            c0: (self.c1 * fq_non_residue() + self.c0) * (self.c0 + self.c1) - ab
                - ab * fq_non_residue(),
            c1: ab + ab,
        }
    }

    fn inverse(self) -> Option<Self> {
        // "High-Speed Software Implementation of the Optimal Ate Pairing
        // over Barreto–Naehrig Curves"; Algorithm 8

        match (self.c0.squared() - (self.c1.squared() * fq_non_residue())).inverse() {
            Some(t) => Some(Fq2 {
                c0: self.c0 * t,
                c1: -(self.c1 * t),
            }),
            None => None,
        }
    }
}

impl Mul for Fq2 {
    type Output = Fq2;

    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    fn mul(self, other: Fq2) -> Fq2 {
        let lhs0: [u32; 8] = bytemuck::cast(U256::from(self.c0).0);
        let lhs1: [u32; 8] = bytemuck::cast(U256::from(self.c1).0);
        let lhs = [lhs0, lhs1];

        let rhs0: [u32; 8] = bytemuck::cast(U256::from(other.c0).0);
        let rhs1: [u32; 8] = bytemuck::cast(U256::from(other.c1).0);
        let rhs = [rhs0, rhs1];

        let irred_poly0: [u32; 8] = bytemuck::cast(U256::from(fq_non_residue()).0);
        let irred_poly = [irred_poly0, [0; 8]];

        let prime: [u32; 8] = bytemuck::cast(Fq::modulus().0);

        // TODO: Review whether this is the architecture we want (incl. on the risc0 repo side)
        let mut result = MaybeUninit::<[[u128; 2]; 2]>::uninit();
        let result_mut: &mut [[u32; 8]; 2] = unsafe {
            bytemuck::cast_mut(result.assume_init_mut())
        };
        field::extfieldmul_256(&lhs, &rhs, &irred_poly, &prime, result_mut);
        let result = unsafe { result.assume_init() };
        Fq2 {
            c0: Fq::new(U256(result[0])).unwrap(),
            c1: Fq::new(U256(result[1])).unwrap(),
        }
    }

    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    fn mul(self, other: Fq2) -> Fq2 {
        // Devegili OhEig Scott Dahab
        //     Multiplication and Squaring on Pairing-Friendly Fields.pdf
        //     Section 3 (Karatsuba)

        let aa = self.c0 * other.c0;
        let bb = self.c1 * other.c1;

        Fq2 {
            c0: bb * fq_non_residue() + aa,
            c1: (self.c0 + self.c1) * (other.c0 + other.c1) - aa - bb,
        }
    }
}

impl Sub for Fq2 {
    type Output = Fq2;

    fn sub(self, other: Fq2) -> Fq2 {
        Fq2 {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
        }
    }
}

impl Add for Fq2 {
    type Output = Fq2;

    fn add(self, other: Fq2) -> Fq2 {
        Fq2 {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
        }
    }
}

impl Neg for Fq2 {
    type Output = Fq2;

    fn neg(self) -> Fq2 {
        Fq2 {
            c0: -self.c0,
            c1: -self.c1,
        }
    }
}

lazy_static::lazy_static! {
    static ref FQ: U256 = U256::from([
        0x3c208c16d87cfd47,
        0x97816a916871ca8d,
        0xb85045b68181585d,
        0x30644e72e131a029
    ]);

    static ref FQ_MINUS3_DIV4: Fq =
        Fq::new(3.into()).expect("3 is a valid field element and static; qed").neg() *
        Fq::new(4.into()).expect("4 is a valid field element and static; qed").inverse()
        .expect("4 has inverse in Fq and is static; qed");

    static ref FQ_MINUS1_DIV2: Fq =
        Fq::new(1.into()).expect("1 is a valid field element and static; qed").neg() *
        Fq::new(2.into()).expect("2 is a valid field element and static; qed").inverse()
            .expect("2 has inverse in Fq and is static; qed");
}

impl Fq2 {
    pub fn i() -> Fq2 {
        Fq2::new(Fq::zero(), Fq::one())
    }

    // TODO: What's going on here...?
    pub fn sqrt(&self) -> Option<Self> {
        let a1 = self.pow::<U256>((*FQ_MINUS3_DIV4).into());
        let a1a = a1 * *self;
        let alpha = a1 * a1a;
        let a0 = alpha.pow(*FQ) * alpha;

        if a0 == Fq2::one().neg() {
            return None;
        }

        if alpha == Fq2::one().neg() {
            Some(Self::i() * a1a)
        } else {
            let b = (alpha + Fq2::one()).pow::<U256>((*FQ_MINUS1_DIV2).into());
            Some(b * a1a)
        }
    }

    pub fn to_u512(&self) -> U512 {
        let c0: U256 = (*self.real()).into();
        let c1: U256 = (*self.imaginary()).into();

        U512::new(&c1, &c0, &FQ)
    }
}

// TODO: Skip: Passing
#[test]
fn tnz_simple_square() {
    // based on sqrt_fq2
    let x1 = Fq2::new(
        Fq::from_str("12844195307879678418043983815760255909500142247603239203345049921980497041944").unwrap(),
        Fq::from_str("7476417578426924565731404322659619974551724117137577781074613937423560117731").unwrap(),
    );
    let x2 = Fq2::new(
        Fq::from_str("3345897230485723946872934576923485762803457692345760237495682347502347589474").unwrap(),
        Fq::from_str("1234912378405347958234756902345768290345762348957605678245967234857634857676").unwrap(),
    );

    assert_eq!(x1 * x1, x2);
}

#[test]
fn tnz_simple_times() {
    assert_eq!(Fq2::one() * Fq2::one(), Fq2::one());
}

// TODO: Skip: Passing
#[test]
fn sqrt_fq2() {
    // from zcash test_proof.cpp
    let x1 = Fq2::new(
        Fq::from_str("12844195307879678418043983815760255909500142247603239203345049921980497041944").unwrap(),
        Fq::from_str("7476417578426924565731404322659619974551724117137577781074613937423560117731").unwrap(),
    );

    let x2 = Fq2::new(
        Fq::from_str("3345897230485723946872934576923485762803457692345760237495682347502347589474").unwrap(),
        Fq::from_str("1234912378405347958234756902345768290345762348957605678245967234857634857676").unwrap(),
    );

    assert_eq!(x2.sqrt().unwrap(), x1);

    // i is sqrt(-1)
    assert_eq!(
        Fq2::one().neg().sqrt().unwrap(),
        Fq2::i(),
    );

    // no sqrt for (1 + 2i)
    assert!(
        Fq2::new(Fq::from_str("1").unwrap(), Fq::from_str("2").unwrap()).sqrt().is_none()
    );
}
