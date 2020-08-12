use std::ops::Add;
use std::cmp;

/// A float type with precision
///
///
#[derive(Debug, Clone)]
pub struct Float {
   pub number: f64,
}

impl Add<Float> for Float {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            number: self.number + other.number,
        }
    }
}

impl Add<f64> for Float {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self {
            number: self.number + other,
        }
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        let round_number = (self.number * 100.0).round() / 100.0;
        let round_other = (other.number * 100.0).round() / 100.0;
        round_number == round_other
    }
}

impl Eq for Float {}

impl Ord for Float {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
