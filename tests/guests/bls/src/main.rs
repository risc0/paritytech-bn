#![no_main]

use bn254::{PrivateKey, PublicKey, Signature, ECDSA as BLS};
use risc0_zkvm::guest::env;
use std::vec::Vec;

risc0_zkvm::guest::entry!(main);

const MSG: &[u8] = b"This message will be signed using BLS-BN254 within the RISC Zero ZKVM.";

pub fn main() {
    let sk_be_bytes_vec: Vec<Vec<u8>> = env::read();
    env::log("Input read");
    assert!(!sk_be_bytes_vec.is_empty());

    let mut pks = Vec::new();
    let mut sigs = Vec::new();

    for sk_be_bytes in &sk_be_bytes_vec {
        let sk = PrivateKey::try_from(sk_be_bytes.as_slice()).unwrap();
        let pk = PublicKey::from_private_key(&sk);
        let compressed = pk.to_compressed().unwrap();
        assert_eq!(pk.0, PublicKey::from_compressed(&compressed).unwrap().0);

        let sig = BLS::sign(&MSG, &sk).unwrap();
        assert_eq!(
            sig.0,
            Signature::from_compressed(sig.to_compressed().unwrap())
                .unwrap()
                .0
        );

        BLS::verify(&MSG, &sig, &pk).unwrap();
        env::log("verification passed");

        pks.push(pk);
        sigs.push(sig);
    }

    let agg_pk = pks.into_iter().reduce(|a, b| a + b).unwrap();
    let agg_sig = sigs.into_iter().reduce(|a, b| a + b).unwrap();

    BLS::verify(&MSG, &agg_sig, &agg_pk).unwrap();
    env::log("aggregated verification passed");

    env::commit(&agg_sig.to_compressed().unwrap());
}
