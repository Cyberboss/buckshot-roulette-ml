pub enum GameAction {
    Observe,
    Shoot(PlayerNumber),
    UnaryItem(UnaryItem),
    Jammer(PlayerNumber),
    Adreneline(AdrenelineTarget),
}

pub struct AdrenelineTarget {
    target_player: PlayerNumber,
    item: AdrenelineItem,
}

pub enum AdrenelineItem {
    Unary(UnaryItem),
    Jammer(PlayerNumber),
}

const ACTION_OBSERVE: usize = 0;
const ACTION_SHOOT_SELF: usize = 1;
const ACTION_INDEX_SHOOT_OTHER: usize = 2;

const ACTION_INDEX_UNARY_ITEM: usize = ACTION_INDEX_SHOOT_OTHER + OTHER_SEATS;

const ACTION_INDEX_JAMMER: usize = ACTION_INDEX_UNARY_ITEM + TOTAL_UNARY_ITEMS;

const ACTION_INDEX_ADRENELINE: usize = ACTION_INDEX_JAMMER + OTHER_SEATS;

pub const TOTAL_ACTIONS: usize =
    ACTION_INDEX_ADRENELINE + (OTHER_SEATS * (TOTAL_UNARY_ITEMS + OTHER_SEATS)) + 1;

impl GameAction {
    pub fn parse(action: usize) -> Self {
        todo!();
    }
}
