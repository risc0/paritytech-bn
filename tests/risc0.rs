use bn254::{PrivateKey, PublicKey, Signature, ECDSA};
use guests::{ECDSA_ELF, ECDSA_ID};
use rand::{rngs::StdRng, SeedableRng};
use risc0_zkvm::{default_prover, ExecutorEnv};
use test_log::test;
use tracing_subscriber::fmt::writer::TestWriter;

const MSG: &[u8] = b"This is the message to be signed by BN-254 within RISC Zero ZKVM";

fn bls(seed: u64, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let sks = (0..n)
        .map(|_| PrivateKey::random(&mut rng))
        .collect::<Vec<_>>();
    let sk_be_bytes_vec = sks
        .iter()
        .map(PrivateKey::to_bytes)
        .collect::<Result<Vec<_>, _>>()?;

    let env = ExecutorEnv::builder()
        .stdout(TestWriter::new())
        .write(&sk_be_bytes_vec)?
        .build()?;

    let prover = default_prover();

    println!("Generating proof ({})...", prover.get_name());
    let prove_info = prover.prove(env, ECDSA_ELF)?;

    println!("Verifying proof...");
    prove_info.receipt.verify(ECDSA_ID)?;

    println!("Validating Journal...");
    let journal = &prove_info.receipt.journal;

    let agg_sig = Signature::from_compressed(&journal.decode::<Vec<u8>>()?)?;
    let agg_pk = sks
        .iter()
        .map(PublicKey::from_private_key)
        .reduce(|a, b| a + b)
        .unwrap();

    ECDSA::verify(&MSG, &agg_sig, &agg_pk).unwrap();

    Ok(())
}

#[test]
#[cfg_attr(not(feature = "cuda"), ignore = "proving takes a long time")]
fn r0vm_prove_ecdsa_signatures() -> Result<(), Box<dyn std::error::Error>> {
    bls(42, 3)
}
