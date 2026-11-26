use numbam::{dot, exp, log};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("dot: {}", dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])?);
    println!("exp: {:?}", exp(&[-1.0, 0.0, 1.0]));
    println!("log: {}", log(2.0));
    Ok(())
}
