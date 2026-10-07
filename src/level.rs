//! Level bands: the 1.0–6.0 float the dataset stores, and its A1–C2 label.

/// The CEFR label a fractional level rounds to (1=A1 … 6=C2), the upstream
/// notebook's `DIFFICULTY_MAPPING_REVERSE`.
pub fn level_to_cefr(level: f64) -> &'static str {
    match level.round() as i64 {
        1 => "A1",
        2 => "A2",
        3 => "B1",
        4 => "B2",
        5 => "C1",
        6 => "C2",
        _ => "?",
    }
}

/// The 1..=6 band a fractional level rounds into; 0 when the level is not
/// a finite number, so a corrupt row reads as "unknown", never as A1.
pub fn level_band(level: f64) -> u8 {
    if !level.is_finite() {
        return 0;
    }
    level.round().clamp(1.0, 6.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bands_round_to_their_nearest_label() {
        assert_eq!(level_to_cefr(1.0), "A1");
        assert_eq!(level_to_cefr(2.49), "A2");
        assert_eq!(level_to_cefr(2.5), "B1");
        assert_eq!(level_to_cefr(4.0), "B2");
        assert_eq!(level_to_cefr(5.6), "C2");
    }

    #[test]
    fn bands_clamp_into_one_through_six() {
        assert_eq!(level_band(0.2), 1);
        assert_eq!(level_band(3.4), 3);
        assert_eq!(level_band(9.0), 6);
    }

    #[test]
    fn a_non_finite_level_is_unknown_never_a1() {
        assert_eq!(level_band(f64::NAN), 0);
        assert_eq!(level_band(f64::INFINITY), 0);
    }
}
