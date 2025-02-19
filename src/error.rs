use ndarray::ShapeError;
use ort::Error as OrtError;
use rodio::decoder::DecoderError;
use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
    io::Error as IoError,
    time::SystemTimeError,
};

#[derive(Debug)]
pub enum GptSoVitsError {
    Decoder(DecoderError),
    Io(IoError),
    Ort(OrtError),
    Shape(ShapeError),
    SystemTime(SystemTimeError),
}

impl Error for GptSoVitsError {}

impl Display for GptSoVitsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "GptSoVitsError: ")?;
        match self {
            Self::Decoder(e) => Display::fmt(e, f),
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
            Self::SystemTime(e) => Display::fmt(e, f),
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

impl From<DecoderError> for GptSoVitsError {
    fn from(value: DecoderError) -> Self {
        Self::Decoder(value)
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
