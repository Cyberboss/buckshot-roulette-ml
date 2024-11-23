use buckshot_roulette_gameplay_engine::item::{UnaryItem, TOTAL_UNARY_ITEMS};

use crate::{
    game_session_domain::OTHER_SEATS,
    relative_player::{OtherPlayer, RelativePlayer},
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum GameAction {
    Observe,
    Act(ActiveGameAction),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ActiveGameAction {
    Shoot(RelativePlayer),
    UnaryItem(UnaryItem),
    Jammer(OtherPlayer),
    Adreneline(AdrenelineTarget),
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct AdrenelineTarget {
    pub target_player: OtherPlayer,
    pub item: AdrenelineItem,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum AdrenelineItem {
    Unary(UnaryItem),
    Jammer(OtherPlayer),
}

const ACTION_OBSERVE: usize = 0;
const ACTION_SHOOT_SELF: usize = 1;
const ACTION_INDEX_SHOOT_OTHER: usize = 2;

const ACTION_INDEX_UNARY_ITEM: usize = ACTION_INDEX_SHOOT_OTHER + OTHER_SEATS;

const ACTION_INDEX_JAMMER: usize = ACTION_INDEX_UNARY_ITEM + TOTAL_UNARY_ITEMS;

const ACTION_INDEX_ADRENELINE: usize = ACTION_INDEX_JAMMER + OTHER_SEATS;

const NON_UNARY_ITEM_POSSIBILITIES: usize = TOTAL_UNARY_ITEMS + OTHER_SEATS;
const ADRENELINE_POSSIBILITIES: usize = OTHER_SEATS * NON_UNARY_ITEM_POSSIBILITIES;

pub const TOTAL_ACTIONS: usize = ACTION_INDEX_ADRENELINE + ADRENELINE_POSSIBILITIES;

impl GameAction {
    pub fn parse(action: usize) -> Self {
        assert!(action < TOTAL_ACTIONS);

        if action == ACTION_OBSERVE {
            GameAction::Observe
        } else {
            GameAction::Act(if action < ACTION_INDEX_UNARY_ITEM {
                ActiveGameAction::Shoot(calc_relative_player(action - ACTION_SHOOT_SELF))
            } else if action < ACTION_INDEX_JAMMER {
                ActiveGameAction::UnaryItem(calc_unary_item(action - ACTION_INDEX_UNARY_ITEM))
            } else if action < ACTION_INDEX_ADRENELINE {
                let relative_index = action - ACTION_INDEX_JAMMER;
                ActiveGameAction::Jammer(calc_other_player(relative_index))
            } else {
                let relative_index = action - ACTION_INDEX_ADRENELINE;
                let target_player_index = relative_index / NON_UNARY_ITEM_POSSIBILITIES;
                let target_item_index =
                    relative_index - (NON_UNARY_ITEM_POSSIBILITIES * target_player_index);
                ActiveGameAction::Adreneline(AdrenelineTarget {
                    target_player: calc_other_player(target_player_index),
                    item: if target_item_index < TOTAL_UNARY_ITEMS {
                        AdrenelineItem::Unary(calc_unary_item(target_item_index))
                    } else {
                        AdrenelineItem::Jammer(calc_other_player(
                            target_item_index - TOTAL_UNARY_ITEMS,
                        ))
                    },
                })
            })
        }
    }
}

#[test]
fn test_all_actions_achievable() {
    let mut set = std::collections::HashSet::with_capacity(TOTAL_ACTIONS);
    for i in 0..TOTAL_ACTIONS {
        println!("Parse: {}", i);
        if !set.insert(GameAction::parse(i)) {
            panic!("Failed to parse action: {}", i);
        }
    }
}

fn calc_relative_player(index: usize) -> RelativePlayer {
    match index {
        0 => RelativePlayer::Own,
        other_index => RelativePlayer::Other(calc_other_player(other_index - 1)),
    }
}

fn calc_other_player(index: usize) -> OtherPlayer {
    match index {
        0 => OtherPlayer::Left,
        1 => OtherPlayer::Opposite,
        2 => OtherPlayer::Right,
        _ => unreachable!("Invalid other player {}", index),
    }
}

fn calc_unary_item(index: usize) -> UnaryItem {
    assert_eq!(7, TOTAL_UNARY_ITEMS);
    match index {
        0 => UnaryItem::Beer,
        1 => UnaryItem::Cigarettes,
        2 => UnaryItem::Handsaw,
        3 => UnaryItem::Inverter,
        4 => UnaryItem::MagnifyingGlass,
        5 => UnaryItem::Phone,
        6 => UnaryItem::Remote,
        _ => unreachable!("Invalid unary item {}", index),
    }
}
