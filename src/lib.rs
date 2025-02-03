#![no_std]

extern crate alloc;

pub mod arith;
mod fields;
mod groups;

use crate::fields::FieldElement;
use crate::groups::{GroupElement, G1Params, G2Params, GroupParams};

use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};
use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fr(fields::Fr);

impl Fr {
    pub fn zero() -> Self {
        Fr(fields::Fr::zero())
    }
    pub fn one() -> Self {
        Fr(fields::Fr::one())
    }
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Fr(fields::Fr::random(rng))
    }
    pub fn pow(&self, exp: Fr) -> Self {
        Fr(self.0.pow(exp.0))
    }
    pub fn from_str(s: &str) -> Option<Self> {
        fields::Fr::from_str(s).map(|e| Fr(e))
    }
    pub fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(|e| Fr(e))
    }
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    pub fn interpret(buf: &[u8; 64]) -> Fr {
        Fr(fields::Fr::interpret(buf))
    }
    pub fn from_slice(slice: &[u8]) -> Result<Self, FieldError> {
        arith::U256::from_slice(slice)
            .map_err(|_| FieldError::InvalidSliceLength) // todo: maybe more sensful error handling
            .map(|x| Fr::new_mul_factor(x))
    }
    pub const fn from_mont_le_slice(slice: &[u8]) -> Self {
        Fr(fields::Fr::from_mont_le_slice(slice))
    }
    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    pub fn to_big_endian(&self, slice: &mut [u8]) -> Result<(), FieldError> {
        self.0
            .raw()
            .to_big_endian(slice)
            .map_err(|_| FieldError::InvalidSliceLength)
    }
    /// Convert to a big-endian slice of `u8`s in Montgomery form
    ///
    /// Note that `Fr` and `Fq` have different behavior for `to_big_endian`,
    /// with `Fq` using standard form instead of Montgomery form.
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    pub fn to_big_endian(&self, slice: &mut [u8]) -> Result<(), FieldError> {
        let a = Fr(self.0.to_montgomery()).into_u256();

        a.to_big_endian(slice)
            .map_err(|_| FieldError::InvalidSliceLength)
    }
    /// Create a new Fr value
    ///
    /// Returns `None` if the input is greater than or equal to the field modulus
    pub fn new(val: arith::U256) -> Option<Self> {
        fields::Fr::new(val).map(|x| Fr(x))
    }
    /// Create a new Fr value
    ///
    /// If the input is greater than or equal to the field modulus, it will mod out by the modulus
    /// to create an element in reduced form.
    pub fn new_mul_factor(val: arith::U256) -> Self {
        Fr(fields::Fr::new_mul_factor(val))
    }
    pub fn into_u256(self) -> arith::U256 {
        (self.0).into()
    }
    pub fn set_bit(&mut self, bit: usize, to: bool) {
        self.0.set_bit(bit, to);
    }
}

impl Add<Fr> for Fr {
    type Output = Fr;

    fn add(self, other: Fr) -> Fr {
        Fr(self.0 + other.0)
    }
}

impl Sub<Fr> for Fr {
    type Output = Fr;

    fn sub(self, other: Fr) -> Fr {
        Fr(self.0 - other.0)
    }
}

impl Neg for Fr {
    type Output = Fr;

    fn neg(self) -> Fr {
        Fr(-self.0)
    }
}

impl Mul for Fr {
    type Output = Fr;

    fn mul(self, other: Fr) -> Fr {
        Fr(self.0 * other.0)
    }
}

#[derive(Debug)]
pub enum FieldError {
    InvalidSliceLength,
    InvalidU512Encoding,
    NotMember,
}

#[derive(Debug)]
pub enum CurveError {
    InvalidEncoding,
    NotMember,
    Field(FieldError),
    ToAffineConversion,
}

impl From<FieldError> for CurveError {
    fn from(fe: FieldError) -> Self {
        CurveError::Field(fe)
    }
}

pub use crate::groups::Error as GroupError;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fq(fields::Fq);

impl Fq {
    pub fn zero() -> Self {
        Fq(fields::Fq::zero())
    }
    pub fn one() -> Self {
        Fq(fields::Fq::one())
    }
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        Fq(fields::Fq::random(rng))
    }
    pub fn pow(&self, exp: Fq) -> Self {
        Fq(self.0.pow(exp.0))
    }
    pub fn from_str(s: &str) -> Option<Self> {
        fields::Fq::from_str(s).map(|e| Fq(e))
    }
    pub fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(|e| Fq(e))
    }
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    pub fn interpret(buf: &[u8; 64]) -> Fq {
        Fq(fields::Fq::interpret(buf))
    }
    pub fn from_slice(slice: &[u8]) -> Result<Self, FieldError> {
        arith::U256::from_slice(slice)
            .map_err(|_| FieldError::InvalidSliceLength) // todo: maybe more sensful error handling
            .and_then(|x| fields::Fq::new(x).ok_or(FieldError::NotMember))
            .map(|x| Fq(x))
    }
    pub const fn from_mont_le_slice(slice: &[u8]) -> Self {
        Fq(fields::Fq::from_mont_le_slice(slice))
    }
    #[cfg(not(all(target_os = "zkvm", target_arch = "riscv32")))]
    pub fn to_big_endian(&self, slice: &mut [u8]) -> Result<(), FieldError> {
        let mut a: arith::U256 = self.0.into();
        // convert from Montgomery representation
        a.mul(
            &fields::Fq::one().raw(),
            &fields::Fq::modulus(),
            self.0.inv(),
        );
        a.to_big_endian(slice)
            .map_err(|_| FieldError::InvalidSliceLength)
    }
    /// Convert to a big-endian slice of `u8`s in standard form
    ///
    /// Note that `Fr` and `Fq` have different behavior for `to_big_endian`,
    /// with `Fr` using Montgomery form instead of standard form.
    #[cfg(all(target_os = "zkvm", target_arch = "riscv32"))]
    pub fn to_big_endian(&self, slice: &mut [u8]) -> Result<(), FieldError> {
        let a: arith::U256 = self.0.into();
        a.to_big_endian(slice)
            .map_err(|_| FieldError::InvalidSliceLength)
    }
    pub fn from_u256(u256: arith::U256) -> Result<Self, FieldError> {
        Ok(Fq(fields::Fq::new(u256).ok_or(FieldError::NotMember)?))
    }
    pub fn into_u256(self) -> arith::U256 {
        (self.0).into()
    }
    pub fn modulus() -> arith::U256 {
        fields::Fq::modulus()
    }

    pub fn sqrt(&self) -> Option<Self> {
        self.0.sqrt().map(Fq)
    }
}

impl Add<Fq> for Fq {
    type Output = Fq;

    fn add(self, other: Fq) -> Fq {
        Fq(self.0 + other.0)
    }
}

impl Sub<Fq> for Fq {
    type Output = Fq;

    fn sub(self, other: Fq) -> Fq {
        Fq(self.0 - other.0)
    }
}

impl Neg for Fq {
    type Output = Fq;

    fn neg(self) -> Fq {
        Fq(-self.0)
    }
}

impl Mul for Fq {
    type Output = Fq;

    fn mul(self, other: Fq) -> Fq {
        Fq(self.0 * other.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Fq2(fields::Fq2);

impl Fq2 {
    pub fn one() -> Fq2 {
        Fq2(fields::Fq2::one())
    }

    pub fn i() -> Fq2 {
        Fq2(fields::Fq2::i())
    }

    pub fn zero() -> Fq2 {
        Fq2(fields::Fq2::zero())
    }

    /// Initalizes new F_q2(a + bi, a is real coeff, b is imaginary)
    pub fn new(a: Fq, b: Fq) -> Fq2 {
        Fq2(fields::Fq2::new(a.0, b.0))
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    pub fn pow(&self, exp: arith::U256) -> Self {
        Fq2(self.0.pow(exp))
    }

    pub fn real(&self) -> Fq {
        Fq(*self.0.real())
    }

    pub fn imaginary(&self) -> Fq {
        Fq(*self.0.imaginary())
    }

    pub fn sqrt(&self) -> Option<Self> {
        self.0.sqrt().map(Fq2)
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, FieldError> {
        let u512 = arith::U512::from_slice(bytes).map_err(|_| FieldError::InvalidU512Encoding)?;
        let (res, c0) = u512.divrem(&Fq::modulus());
        Ok(Fq2::new(
            Fq::from_u256(c0).map_err(|_| FieldError::NotMember)?,
            Fq::from_u256(res.ok_or(FieldError::NotMember)?).map_err(|_| FieldError::NotMember)?,
        ))
    }
}


impl Add<Fq2> for Fq2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Fq2(self.0 + other.0)
    }
}

impl Sub<Fq2> for Fq2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Fq2(self.0 - other.0)
    }
}

impl Neg for Fq2 {
    type Output = Self;

    fn neg(self) -> Self {
        Fq2(-self.0)
    }
}

impl Mul for Fq2 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Fq2(self.0 * other.0)
    }
}

pub trait Group
    : Send
    + Sync
    + Copy
    + Clone
    + PartialEq
    + Eq
    + Sized
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + Mul<Fr, Output = Self> {
    fn zero() -> Self;
    fn one() -> Self;
    fn random<R: Rng>(rng: &mut R) -> Self;
    fn is_zero(&self) -> bool;
    fn normalize(&mut self);
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct G1(groups::G1);

impl G1 {
    pub fn new(x: Fq, y: Fq, z: Fq) -> Self {
        G1(groups::G1::new(x.0, y.0, z.0))
    }

    pub fn x(&self) -> Fq {
        Fq(self.0.x().clone())
    }

    pub fn set_x(&mut self, x: Fq) {
        *self.0.x_mut() = x.0
    }

    pub fn y(&self) -> Fq {
        Fq(self.0.y().clone())
    }

    pub fn set_y(&mut self, y: Fq) {
        *self.0.y_mut() = y.0
    }

    pub fn z(&self) -> Fq {
        Fq(self.0.z().clone())
    }

    pub fn set_z(&mut self, z: Fq) {
        *self.0.z_mut() = z.0
    }

    pub fn b() -> Fq {
        Fq(G1Params::coeff_b())
    }

    pub fn from_compressed(bytes: &[u8]) -> Result<Self, CurveError> {
        if bytes.len() != 33 { return Err(CurveError::InvalidEncoding); }

        let sign = bytes[0];
        let fq = Fq::from_slice(&bytes[1..])?;
        let x = fq;
        let y_squared = (fq * fq * fq) + Self::b();

        let mut y = y_squared.sqrt().ok_or(CurveError::NotMember)?;

        if sign == 2 && y.into_u256().get_bit(0).expect("bit 0 always exist; qed") { y = y.neg(); }
        else if sign == 3 && !y.into_u256().get_bit(0).expect("bit 0 always exist; qed") { y = y.neg(); }
        else if sign != 3 && sign != 2 {
            return Err(CurveError::InvalidEncoding);
        }
        AffineG1::new(x, y).map_err(|_| CurveError::NotMember).map(Into::into)
    }
}

impl Group for G1 {
    fn zero() -> Self {
        G1(groups::G1::zero())
    }
    fn one() -> Self {
        G1(groups::G1::one())
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        G1(groups::G1::random(rng))
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn normalize(&mut self) {
        let new = match self.0.to_affine() {
            Some(a) => a,
            None => return,
        };

        self.0 = new.to_jacobian();
    }
}

impl Add<G1> for G1 {
    type Output = G1;

    fn add(self, other: G1) -> G1 {
        G1(self.0 + other.0)
    }
}

impl Sub<G1> for G1 {
    type Output = G1;

    fn sub(self, other: G1) -> G1 {
        G1(self.0 - other.0)
    }
}

impl Neg for G1 {
    type Output = G1;

    fn neg(self) -> G1 {
        G1(-self.0)
    }
}

impl Mul<Fr> for G1 {
    type Output = G1;

    fn mul(self, other: Fr) -> G1 {
        G1(self.0 * other.0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct AffineG1(groups::AffineG1);

impl AffineG1 {
    pub fn new(x: Fq, y: Fq) -> Result<Self, GroupError> {
        Ok(AffineG1(groups::AffineG1::new(x.0, y.0)?))
    }

    pub fn x(&self) -> Fq {
        Fq(self.0.x().clone())
    }

    pub fn set_x(&mut self, x: Fq) {
        *self.0.x_mut() = x.0
    }

    pub fn y(&self) -> Fq {
        Fq(self.0.y().clone())
    }

    pub fn set_y(&mut self, y: Fq) {
        *self.0.y_mut() = y.0
    }

    pub fn from_jacobian(g1: G1) -> Option<Self> {
        g1.0.to_affine().map(|x| AffineG1(x))
    }

    pub const fn from_mont_le_slice(bytes: &[u8]) -> Self {
        AffineG1(groups::AffineG1::from_mont_le_slice(bytes))
    }
}

impl From<AffineG1> for G1 {
    fn from(affine: AffineG1) -> Self {
        G1(affine.0.to_jacobian())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct G2(groups::G2);

impl G2 {
    pub fn new(x: Fq2, y: Fq2, z: Fq2) -> Self {
        G2(groups::G2::new(x.0, y.0, z.0))
    }

    pub fn x(&self) -> Fq2 {
        Fq2(self.0.x().clone())
    }

    pub fn set_x(&mut self, x: Fq2) {
        *self.0.x_mut() = x.0
    }

    pub fn y(&self) -> Fq2 {
        Fq2(self.0.y().clone())
    }

    pub fn set_y(&mut self, y: Fq2) {
        *self.0.y_mut() = y.0
    }

    pub fn z(&self) -> Fq2 {
        Fq2(self.0.z().clone())
    }

    pub fn set_z(&mut self, z: Fq2) {
        *self.0.z_mut() = z.0
    }

    pub fn b() -> Fq2 {
        Fq2(G2Params::coeff_b())
    }

    pub fn from_compressed(bytes: &[u8]) -> Result<Self, CurveError> {

        if bytes.len() != 65 { return Err(CurveError::InvalidEncoding); }

        let sign = bytes[0];
        let x = Fq2::from_slice(&bytes[1..])?;

        let y_squared = (x * x * x) + G2::b();
        let y = y_squared.sqrt().ok_or(CurveError::NotMember)?;
        let y_neg = -y;

        let y_gt = y.0.to_u512() > y_neg.0.to_u512();

        let e_y = if sign == 10 { if y_gt { y_neg } else { y } }
        else if sign == 11 { if y_gt { y } else { y_neg } }
        else {
            return Err(CurveError::InvalidEncoding);
        };

        AffineG2::new(x, e_y).map_err(|_| CurveError::NotMember).map(Into::into)
    }
}

impl Group for G2 {
    fn zero() -> Self {
        G2(groups::G2::zero())
    }
    fn one() -> Self {
        G2(groups::G2::one())
    }
    fn random<R: Rng>(rng: &mut R) -> Self {
        G2(groups::G2::random(rng))
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn normalize(&mut self) {
        let new = match self.0.to_affine() {
            Some(a) => a,
            None => return,
        };

        self.0 = new.to_jacobian();
    }
}

impl Add<G2> for G2 {
    type Output = G2;

    fn add(self, other: G2) -> G2 {
        G2(self.0 + other.0)
    }
}

impl Sub<G2> for G2 {
    type Output = G2;

    fn sub(self, other: G2) -> G2 {
        G2(self.0 - other.0)
    }
}

impl Neg for G2 {
    type Output = G2;

    fn neg(self) -> G2 {
        G2(-self.0)
    }
}

impl Mul<Fr> for G2 {
    type Output = G2;

    fn mul(self, other: Fr) -> G2 {
        G2(self.0 * other.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct Gt(fields::Fq12);

impl Gt {
    pub fn one() -> Self {
        Gt(fields::Fq12::one())
    }
    pub fn pow(&self, exp: Fr) -> Self {
        Gt(self.0.pow(exp.0))
    }
    pub fn inverse(&self) -> Option<Self> {
        self.0.inverse().map(Gt)
    }
    pub fn final_exponentiation(&self) -> Option<Self> {
        self.0.final_exponentiation().map(Gt)
    }
}

impl Mul<Gt> for Gt {
    type Output = Gt;

    fn mul(self, other: Gt) -> Gt {
        Gt(self.0 * other.0)
    }
}

pub fn pairing(p: G1, q: G2) -> Gt {
    Gt(groups::pairing(&p.0, &q.0))
}

pub fn pairing_batch(pairs: &[(G1, G2)]) -> Gt {
    let mut ps : Vec<groups::G1> = Vec::new();
    let mut qs : Vec<groups::G2> = Vec::new();
    for (p, q) in pairs {
        ps.push(p.0);
        qs.push(q.0);
    }
    Gt(groups::pairing_batch(&ps, &qs))
}

pub fn miller_loop_batch(pairs: &[(G2, G1)]) -> Result<Gt, CurveError> {
    let mut ps : Vec<groups::G2Precomp> = Vec::new();
    let mut qs : Vec<groups::AffineG<groups::G1Params>> = Vec::new();
    for (p, q) in pairs {
        ps.push(p.0.to_affine().ok_or(CurveError::ToAffineConversion)?.precompute());
        qs.push(q.0.to_affine().ok_or(CurveError::ToAffineConversion)?);
    }
    Ok(Gt(groups::miller_loop_batch(&ps, &qs)))
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub struct AffineG2(groups::AffineG2);

impl AffineG2 {
    pub fn new(x: Fq2, y: Fq2) -> Result<Self, GroupError> {
        Ok(AffineG2(groups::AffineG2::new(x.0, y.0)?))
    }

    pub fn x(&self) -> Fq2 {
        Fq2(self.0.x().clone())
    }

    pub fn set_x(&mut self, x: Fq2) {
        *self.0.x_mut() = x.0
    }

    pub fn y(&self) -> Fq2 {
        Fq2(self.0.y().clone())
    }

    pub fn set_y(&mut self, y: Fq2) {
        *self.0.y_mut() = y.0
    }

    pub fn from_jacobian(g2: G2) -> Option<Self> {
        g2.0.to_affine().map(|x| AffineG2(x))
    }

    pub const fn from_mont_le_slice(bytes: &[u8]) -> Self {
        AffineG2(groups::AffineG2::from_mont_le_slice(bytes))
    }
}

impl From<AffineG2> for G2 {
    fn from(affine: AffineG2) -> Self {
        G2(affine.0.to_jacobian())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use super::{G1, Fq, G2, Fq2, Fr};

    fn hex(s: &'static str) -> Vec<u8> {
        use rustc_hex::FromHex;
        s.from_hex().unwrap()
    }

    #[test]
    fn g1_from_compressed() {
        let g1 = G1::from_compressed(&hex("0230644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd46"))
            .expect("Invalid g1 decompress result");
        assert_eq!(g1.x(), Fq::from_str("21888242871839275222246405745257275088696311157297823662689037894645226208582").unwrap());
        assert_eq!(g1.y(), Fq::from_str("3969792565221544645472939191694882283483352126195956956354061729942568608776").unwrap());
        assert_eq!(g1.z(), Fq::one());
    }


    #[test]
    fn g2_from_compressed() {
        let g2 = G2::from_compressed(
            &hex("0a023aed31b5a9e486366ea9988b05dba469c6206e58361d9c065bbea7d928204a761efc6e4fa08ed227650134b52c7f7dd0463963e8a4bf21f4899fe5da7f984a")
        ).expect("Valid g2 point hex encoding");

        assert_eq!(g2.x(),
                   Fq2::new(
                       Fq::from_str("5923585509243758863255447226263146374209884951848029582715967108651637186684").unwrap(),
                       Fq::from_str("5336385337059958111259504403491065820971993066694750945459110579338490853570").unwrap(),
                   )
        );

        assert_eq!(g2.y(),
                   Fq2::new(
                       Fq::from_str("10374495865873200088116930399159835104695426846400310764827677226300185211748").unwrap(),
                       Fq::from_str("5256529835065685814318509161957442385362539991735248614869838648137856366932").unwrap(),
                   )
        );

        // 0b prefix is point reflection on the curve
        let g2 = -G2::from_compressed(
            &hex("0b023aed31b5a9e486366ea9988b05dba469c6206e58361d9c065bbea7d928204a761efc6e4fa08ed227650134b52c7f7dd0463963e8a4bf21f4899fe5da7f984a")
        ).expect("Valid g2 point hex encoding");

        assert_eq!(g2.x(),
                   Fq2::new(
                       Fq::from_str("5923585509243758863255447226263146374209884951848029582715967108651637186684").unwrap(),
                       Fq::from_str("5336385337059958111259504403491065820971993066694750945459110579338490853570").unwrap(),
                   )
        );

        assert_eq!(g2.y(),
                   Fq2::new(
                       Fq::from_str("10374495865873200088116930399159835104695426846400310764827677226300185211748").unwrap(),
                       Fq::from_str("5256529835065685814318509161957442385362539991735248614869838648137856366932").unwrap(),
                   )
        );

        // valid point but invalid sign prefix
        assert!(
            G2::from_compressed(
                &hex("0c023aed31b5a9e486366ea9988b05dba469c6206e58361d9c065bbea7d928204a761efc6e4fa08ed227650134b52c7f7dd0463963e8a4bf21f4899fe5da7f984a")
            ).is_err()
        );
    }

    #[test]
    fn r0_new() {
        assert!(Fr::new(crate::fields::Fr::modulus()).is_none());
        let mut modulus_plus_one = crate::fields::Fr::modulus();
        modulus_plus_one.0[0] += 1;
        assert!(Fr::new(modulus_plus_one).is_none());
        assert_eq!(Fr::new(0.into()).unwrap(), Fr::zero());
        assert_eq!(Fr::new(1.into()).unwrap(), Fr::one());
    }

    #[test]
    fn r0_new_mul_factor() {
        assert_eq!(Fr::new_mul_factor(crate::fields::Fr::modulus()), Fr::zero());
        let mut modulus_plus_one = crate::fields::Fr::modulus();
        modulus_plus_one.0[0] += 1;
        assert_eq!(Fr::new_mul_factor(modulus_plus_one), Fr::one());

        assert_eq!(Fr::new_mul_factor(0.into()), Fr::zero());
        assert_eq!(Fr::new_mul_factor(1.into()), Fr::one());
    }

    #[test]
    fn r0_from_str() {
        let q4 = Fq::from_str("4").unwrap();
        let q9 = Fq::from_str("9").unwrap();
        let q36 = Fq::from_str("36").unwrap();
        let r4 = Fr::from_str("4").unwrap();
        let r9 = Fr::from_str("9").unwrap();
        let r36 = Fr::from_str("36").unwrap();

        assert_eq!(q4 * q9, q36);
        assert_eq!(r4 * r9, r36);
    }

    #[test]
    fn r0_zero() {
        assert!(Fq::zero().is_zero());
        assert!(Fr::zero().is_zero());
    }

    #[test]
    fn r0_random() {
        let rng = &mut rand::thread_rng();
        let random_q = Fq::random(rng);
        let random_r = Fr::random(rng);

        assert!((random_q - random_q).is_zero());
        assert!((random_r - random_r).is_zero());
    }

    #[test]
    fn r0_interpret() {
        // Interpretting one should give one
        let one_bytes: [u8; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let interp_one = Fq::interpret(&one_bytes);
        assert_eq!(interp_one, Fq::one());
        let interp_one = Fr::interpret(&one_bytes);
        assert_eq!(interp_one, Fr::one());

        // Interpretting a multiple of the modulus should give zero
        // So we interpret modulus * (1 << 240)
        let mut modulus_bytes = [0u8; 64];
        let modulus = Fq::modulus();
        modulus.to_big_endian(&mut modulus_bytes[2..34]).unwrap();
        let interp_modulus = Fq::interpret(&modulus_bytes);
        assert_eq!(interp_modulus, Fq::zero());
        // Fr doesn't expose a `modulus` function, so no testing this for it
    }

    #[test]
    fn r0_from_slice() {
        let one_bytes: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
        ];
        let from_slice_one = Fq::from_slice(&one_bytes).unwrap();
        assert_eq!(from_slice_one, Fq::one());
        let from_slice_one = Fr::from_slice(&one_bytes).unwrap();
        assert_eq!(from_slice_one, Fr::one());
    }

    #[test]
    fn r0_from_mont_le_slice() {
        // This tests that reading in the Montgomery forms of 1 gives 1
        let montgomery_one_fq = crate::arith::U256::from([
            0xd35d438dc58f0d9d,
            0xa78eb28f5c70b3d,
            0x666ea36f7879462c,
            0xe0a77c19a07df2f,
        ]);
        let mut mont_one_bytes = [0u8; 32];
        montgomery_one_fq.to_big_endian(&mut mont_one_bytes).unwrap();
        mont_one_bytes.reverse();
        assert_eq!(Fq::one(), Fq::from_mont_le_slice(&mont_one_bytes));

        let montgomery_one_fr = crate::arith::U256::from([
            0xac96341c4ffffffb,
            0x36fc76959f60cd29,
            0x666ea36f7879462e,
            0xe0a77c19a07df2f,
        ]);
        let mut mont_one_bytes = [0u8; 32];
        montgomery_one_fr.to_big_endian(&mut mont_one_bytes).unwrap();
        mont_one_bytes.reverse();
        assert_eq!(Fr::one(), Fr::from_mont_le_slice(&mont_one_bytes));
    }

    #[test]
    fn r0_from_u256() {
        assert!(Fq::from_u256(Fq::modulus()).is_err());
        let mut modulus_plus_one = Fq::modulus();
        modulus_plus_one.0[0] += 1;
        assert!(Fq::from_u256(modulus_plus_one).is_err());
        assert_eq!(Fq::from_u256(0.into()).unwrap(), Fq::zero());
        assert_eq!(Fq::from_u256(1.into()).unwrap(), Fq::one());
    }

    #[test]
    fn r0_into_u256() {
        let q5 = Fq::from_str("5").unwrap();
        let r5 = Fr::from_str("5").unwrap();
        assert_eq!(q5.into_u256(), 5.into());
        assert_eq!(r5.into_u256(), 5.into());
    }

    #[test]
    fn r0_big_endian() {
        let mut computed_bytes = [0u8; 32];
        let one_bytes: [u8; 32] = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
        ];
        Fq::one().to_big_endian(&mut computed_bytes).unwrap();
        assert_eq!(computed_bytes, one_bytes);
        computed_bytes = [0u8; 32];
        // Note: For `Fr`, `to_big_endian` gives Montgomery form
        // Note also that this is not the behavior of `Fq`.
        let one_bytes_montgomery_r: [u8; 32] = [
            14,  10,  119, 193, 154, 7,   223, 47,
            102, 110, 163, 111, 120, 121, 70,  46,
            54,  252, 118, 149, 159, 96,  205, 41,
            172, 150, 52,  28,  79,  255, 255, 251,
        ];
        Fr::one().to_big_endian(&mut computed_bytes).unwrap();
        assert_eq!(computed_bytes, one_bytes_montgomery_r);
    }

    #[test]
    fn r0_modulus() {
        let mut modulus_minus_one = Fq::modulus();
        modulus_minus_one.0[0] -= 1;
        assert_eq!(Fq::from_u256(modulus_minus_one).unwrap(), Fq::zero() - Fq::one());
    }

    #[test]
    fn r0_arithmetic() {
        assert_eq!(Fq::one() + Fq::one(), Fq::from_str("2").unwrap());
        assert_eq!(Fq::one() - Fq::one(), Fq::zero());
        assert_eq!(-Fq::zero(), Fq::zero());
        assert_eq!(-Fq::one() + Fq::one(), Fq::zero());
        assert_eq!(Fq::one() * Fq::one(), Fq::one());
        assert_eq!(Fq::one() * Fq::zero(), Fq::zero());
        assert_eq!(Fq::from_str("8").unwrap() * Fq::from_str("9").unwrap(), Fq::from_str("72").unwrap());

        assert_eq!(Fr::one() + Fr::one(), Fr::from_str("2").unwrap());
        assert_eq!(Fr::one() - Fr::one(), Fr::zero());
        assert_eq!(-Fr::zero(), Fr::zero());
        assert_eq!(-Fr::one() + Fr::one(), Fr::zero());
        assert_eq!(Fr::one() * Fr::one(), Fr::one());
        assert_eq!(Fr::one() * Fr::zero(), Fr::zero());
        assert_eq!(Fr::from_str("8").unwrap() * Fr::from_str("9").unwrap(), Fr::from_str("72").unwrap());
    }

    #[test]
    fn r0_pow() {
        let rng = &mut rand::thread_rng();
        let random_q = Fq::random(rng);
        let random_r = Fr::random(rng);

        assert_eq!(random_q.pow(Fq::one()), random_q);
        assert_eq!(random_r.pow(Fr::one()), random_r);
        assert_eq!(Fq::one().pow(random_q), Fq::one());
        assert_eq!(Fr::one().pow(random_r), Fr::one());
        assert_eq!(random_q.pow(Fq::zero()), Fq::one());
        assert_eq!(random_r.pow(Fr::zero()), Fr::one());
    }

    #[test]
    fn r0_inverse() {
        assert!(Fq::zero().inverse().is_none());
        assert!(Fr::zero().inverse().is_none());

        let rng = &mut rand::thread_rng();
        let random_q = Fq::random(rng);
        let random_r = Fr::random(rng);

        let inv_q = random_q.inverse();
        let inv_r = random_r.inverse();

        if random_q.is_zero() {
            assert!(inv_q.is_none());
        } else {
            assert_eq!(inv_q.unwrap() * random_q, Fq::one());
        }
        if random_r.is_zero() {
            assert!(inv_r.is_none());
        } else {
            assert_eq!(inv_r.unwrap() * random_r, Fr::one());
        }
    }

    // Note: Not testing set_bit because it's not implemented for RISC Zero

    #[test]
    fn r0_sqrt() {
        assert_eq!(Fq::one(), Fq::sqrt(&Fq::one()).unwrap());
        assert_eq!(Fq::zero(), Fq::sqrt(&Fq::zero()).unwrap());
        assert_eq!(Fq::from_str("2").unwrap(), Fq::sqrt(&Fq::from_str("4").unwrap()).unwrap());
        assert_eq!(Fq::from_str("17918450306617730576773466553562392805212959031101866706334976164702657599807").unwrap(),
                Fq::sqrt(&Fq::from_str("2").unwrap()).unwrap());
        assert!(Fq::sqrt(&(Fq::zero() - Fq::one())).is_none());
    }

    #[test]
    fn r0_fq2_constants() {
        assert!(Fq2::zero().is_zero());
        assert_eq!(Fq2::one() + Fq2::i() * Fq2::i(), Fq2::zero());
        assert_eq!(Fq2::one() * Fq2::i(), Fq2::i());
        assert_eq!(Fq2::zero() * Fq2::i(), Fq2::zero());

        assert_eq!(Fq2::zero().real(), Fq::zero());
        assert_eq!(Fq2::zero().imaginary(), Fq::zero());
        assert_eq!(Fq2::one().real(), Fq::one());
        assert_eq!(Fq2::one().imaginary(), Fq::zero());
        assert_eq!(Fq2::i().real(), Fq::zero());
        assert_eq!(Fq2::i().imaginary(), Fq::one());
    }

    #[test]
    fn r0_fq2_new_and_parts() {
        let ext_elem = Fq2::new(Fq::from_str("4").unwrap(), Fq::from_str("3").unwrap());
        assert_eq!(ext_elem.real(), Fq::from_str("4").unwrap());
        assert_eq!(ext_elem.imaginary(), Fq::from_str("3").unwrap());
    }

    #[test]
    fn r0_fq2_pow() {
        let ext_elem = Fq2::new(Fq::from_str("5").unwrap(), Fq::from_str("2").unwrap());
        assert_eq!(ext_elem.pow(3.into()), Fq2::new(Fq::from_str("65").unwrap(), Fq::from_str("142").unwrap()));
    }

    #[test]
    fn r0_fq2_sqrt() {
        assert_eq!(Fq2::sqrt(&Fq2::zero()).unwrap(), Fq2::zero());
        assert_eq!(Fq2::sqrt(&Fq2::one()).unwrap(), Fq2::one());
        assert_eq!(Fq2::sqrt(&-Fq2::one()).unwrap(), Fq2::i());
        assert_eq!(Fq2::sqrt(&Fq2::new(Fq::from_str("4").unwrap(), Fq::zero())).unwrap(),
                Fq2::new(Fq::from_str("2").unwrap(), Fq::zero()));
    }

    #[test]
    fn r0_fq2_from_slice() {
        // This is 1 << 256; it represents ((1 << 256) % q) + ((1 << 256) / q) * i
        let example: [u8; 64] = [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let val = Fq2::from_slice(&example).unwrap();
        assert_eq!(val.real(), Fq::from_str("6350874878119819312338956282401532409788428879151445726012394534686998597021").unwrap());
        assert_eq!(val.imaginary(), Fq::from_str("5").unwrap());
    }
}

// TODO: Do we want to test the groups (i.e. G1, G2, Gt, AffineG1, and AffineG2)?
