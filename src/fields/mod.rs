mod fp;
mod fq2;
mod fq6;
mod fq12;

use crate::arith::U256;
use rand::Rng;
use core::ops::{Add, Mul, Neg, Sub};
use alloc::fmt::Debug;

pub use self::fp::{const_fq, Fq, Fr};
pub use self::fq2::{Fq2, fq2_nonresidue};
pub use self::fq6::Fq6;
pub use self::fq12::Fq12;

pub trait FieldElement
    : Sized
    + Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + Eq
    + Debug {
    fn zero() -> Self;
    fn one() -> Self;
    fn random<R: Rng>(_: &mut R) -> Self;
    fn is_zero(&self) -> bool;
    fn squared(&self) -> Self {
        (*self) * (*self)
    }
    fn inverse(self) -> Option<Self>;
    fn pow<I: Into<U256>>(&self, by: I) -> Self {
        let mut res = Self::one();

        for i in by.into().bits() {
            res = res.squared();
            if i {
                res = *self * res;
            }
        }

        res
    }
}

#[cfg(test)]
mod tests;

#[test]
fn test_fr() {
    tests::field_trials::<Fr>();
}

#[test]
fn test_fq() {
    tests::field_trials::<Fq>();
}

#[test]
fn test_fq2() {
    tests::field_trials::<Fq2>();
}

#[test]
fn test_str() {
    assert_eq!(
        -Fr::one(),
        Fr::from_str(
            "21888242871839275222246405745257275088548364400416034343698204186575808495616"
        ).unwrap()
    );
    assert_eq!(
        -Fq::one(),
        Fq::from_str(
            "21888242871839275222246405745257275088696311157297823662689037894645226208582"
        ).unwrap()
    );
}

#[test]
fn test_fq6() {
    tests::field_trials::<Fq6>();
}

#[test]
fn test_fq12() {
    tests::field_trials::<Fq12>();
}

#[test]
fn fq12_test_vector() {
    let start = Fq12::new(
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "19797905000333868150253315089095386158892526856493194078073564469188852136946",
                ).unwrap(),
                Fq::from_str(
                    "10509658143212501778222314067134547632307419253211327938344904628569123178733",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "208316612133170645758860571704540129781090973693601051684061348604461399206",
                ).unwrap(),
                Fq::from_str(
                    "12617661120538088237397060591907161689901553895660355849494983891299803248390",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "2897490589776053688661991433341220818937967872052418196321943489809183508515",
                ).unwrap(),
                Fq::from_str(
                    "2730506433347642574983433139433778984782882168213690554721050571242082865799",
                ).unwrap(),
            ),
        ),
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "17870056122431653936196746815433147921488990391314067765563891966783088591110",
                ).unwrap(),
                Fq::from_str(
                    "14314041658607615069703576372547568077123863812415914883625850585470406221594",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "10123533891707846623287020000407963680629966110211808794181173248765209982878",
                ).unwrap(),
                Fq::from_str(
                    "5062091880848845693514855272640141851746424235009114332841857306926659567101",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "9839781502639936537333620974973645053542086898304697594692219798017709586567",
                ).unwrap(),
                Fq::from_str(
                    "1583892292110602864638265389721494775152090720173641072176370350017825640703",
                ).unwrap(),
            ),
        ),
    );

    // Do a bunch of arbitrary stuff to the element

    let mut next = start.clone();
    for _ in 0..100 {
        next = next * start;
    }

    let cpy = next.clone();

    for _ in 0..10 {
        next = next.squared();
    }

    for _ in 0..10 {
        next = next + start;
        next = next - cpy;
        next = -next;
    }

    next = next.squared();

    let finally = Fq12::new(
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "18388750939593263065521177085001223024106699964957029146547831509155008229833",
                ).unwrap(),
                Fq::from_str(
                    "18370529854582635460997127698388761779167953912610241447912705473964014492243",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "3691824277096717481466579496401243638295254271265821828017111951446539785268",
                ).unwrap(),
                Fq::from_str(
                    "20513494218085713799072115076991457239411567892860153903443302793553884247235",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "12214155472433286415803224222551966441740960297013786627326456052558698216399",
                ).unwrap(),
                Fq::from_str(
                    "10987494248070743195602580056085773610850106455323751205990078881956262496575",
                ).unwrap(),
            ),
        ),
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "5134522153456102954632718911439874984161223687865160221119284322136466794876",
                ).unwrap(),
                Fq::from_str(
                    "20119236909927036376726859192821071338930785378711977469360149362002019539920",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "8839766648621210419302228913265679710586991805716981851373026244791934012854",
                ).unwrap(),
                Fq::from_str(
                    "9103032146464138788288547957401673544458789595252696070370942789051858719203",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "10378379548636866240502412547812481928323945124508039853766409196375806029865",
                ).unwrap(),
                Fq::from_str(
                    "9021627154807648093720460686924074684389554332435186899318369174351765754041",
                ).unwrap(),
            ),
        ),
    );

    assert_eq!(finally, next);
}

#[test]
fn test_cyclotomic_exp() {
    let orig = Fq12::new(
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "2259924035228092997691937637688451143058635253053054071159756458902878894295",
                ).unwrap(),
                Fq::from_str(
                    "13145690032701362144460254305183927872683620413225364127064863863535255135244",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "9910063591662383599552477067956819406417086889312288278252482503717089428441",
                ).unwrap(),
                Fq::from_str(
                    "537414042055419261990282459138081732565514913399498746664966841152381183961",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "15311812409497308894370893420777496684951030254049554818293571309705780605004",
                ).unwrap(),
                Fq::from_str(
                    "13657107176064455789881282546557276003626320193974643644160350907227082365810",
                ).unwrap(),
            ),
        ),
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "4913017949003742946864670837361832856526234260447029873580022776602534856819",
                ).unwrap(),
                Fq::from_str(
                    "7834351480852267338070670220119081676575418514182895774094743209915633114041",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "12837298223308203788092748646758194441270207338661891973231184407371206766993",
                ).unwrap(),
                Fq::from_str(
                    "12756474445699147370503225379431475413909971718057034061593007812727141391799",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "9473802207170192255373153510655867502408045964296373712891954747252332944018",
                ).unwrap(),
                Fq::from_str(
                    "4583089109360519374075173304035813179013579459429335467869926761027310749713",
                ).unwrap(),
            ),
        ),
    );

    let expected = Fq12::new(
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "14722956046055152398903846391223329501345567382234608299399030576415080188350",
                ).unwrap(),
                Fq::from_str(
                    "14280703280777926697010730619606819467080027543707671882210769811674790473417",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "19969875076083990244184003223190771301761436396530543002586073549972410735411",
                ).unwrap(),
                Fq::from_str(
                    "10717335566913889643303549252432531178405520196706173198634734518494041323243",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "6063612626166484870786832843320782567259894784043383626084549455432890717937",
                ).unwrap(),
                Fq::from_str(
                    "17089783040131779205038789608891431427943860868115199598200376195935079808729",
                ).unwrap(),
            ),
        ),
        Fq6::new(
            Fq2::new(
                Fq::from_str(
                    "10029863438921507421569931792104023129735006154272482043027653425575205672906",
                ).unwrap(),
                Fq::from_str(
                    "6406252222753462799887280578845937185621081001436094637606245493619821542775",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "1048245462913506652602966692378792381004227332967846949234978073448561848050",
                ).unwrap(),
                Fq::from_str(
                    "1444281375189053827455518242624554285012408033699861764136810522738182087554",
                ).unwrap(),
            ),
            Fq2::new(
                Fq::from_str(
                    "8839610992666735109106629514135300820412539620261852250193684883379364789120",
                ).unwrap(),
                Fq::from_str(
                    "11347360242067273846784836674906058940820632082713814508736182487171407730718",
                ).unwrap(),
            ),
        ),
    );

    let e = orig.exp_by_neg_z();

    assert_eq!(e, expected);
}
