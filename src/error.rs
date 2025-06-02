use hound::Error as HoundError;
use ndarray::ShapeError;
use ort::Error as OrtError;
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    io::Error as IoError,
    time::SystemTimeError,
};
#[derive(Debug)]
pub enum GptSoVitsError {
    Io(IoError),
    Ort(OrtError),
    Shape(ShapeError),
    SystemTime(SystemTimeError),
    Hound(HoundError),
}

impl Error for GptSoVitsError {}

impl Display for GptSoVitsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GptSoVitsError: ")?;
        match self {
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
            Self::SystemTime(e) => Display::fmt(e, f),
            Self::Hound(e) => Display::fmt(e, f),
        }
    }
}

impl From<OrtError> for GptSoVitsError {
    fn from(value: OrtError) -> Self {
        Self::Ort(value)
    }
}

impl From<IoError> for GptSoVitsError {
    fn from(value: IoError) -> Self {
        Self::Io(value)
    }
}

impl From<ShapeError> for GptSoVitsError {
    fn from(value: ShapeError) -> Self {
        Self::Shape(value)
    }
}

impl From<SystemTimeError> for GptSoVitsError {
    fn from(value: SystemTimeError) -> Self {
        Self::SystemTime(value)
    }
}

impl From<HoundError> for GptSoVitsError {
    fn from(value: HoundError) -> Self {
        Self::Hound(value)
    }
}
