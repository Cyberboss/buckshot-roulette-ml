#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RelativePlayer {
    Own,
    Other(OtherPlayer),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum OtherPlayer {
    Left,
    Opposite,
    Right,
}
