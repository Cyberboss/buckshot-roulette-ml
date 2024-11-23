use std::{
    cell::RefCell, collections::{HashMap, VecDeque}, fmt::format, os::linux::raw::stat
};

use buckshot_roulette_gameplay_engine::{
    game_session::{self, GameSession},
    item::{
        initialize_item_count_map, Item, NotAdreneline, UnaryItem, TOTAL_ITEMS, TOTAL_UNARY_ITEMS,
    },
    loadout::MAX_SHELLS,
    multiplayer_count::MultiplayerCount,
    player_number::{self, PlayerNumber},
    round::TurnSummary,
    round_number::RoundNumber,
    round_player::StunState,
    shell::ShellType,
};
use rand::Rng;
use rsrl::{
    domains::{self, Action, Domain, Observation, Reward, State},
    spaces::{
        discrete::{Interval, Ordinal},
        ProductSpace,
    },
};

use crate::{
    game_action::{GameAction, TOTAL_ACTIONS},
    seat_map::SeatMap,
};

const REWARD_INVALID_ACTION: f64 = -1000000.0;
const REWARD_ROUND_WIN: f64 = 4000.0;
const REWARD_ROUND_LOSS: f64 = -1000.0;

const REWARD_GAIN_HEALTH: f64 = 50.0;
const REWARD_USELESS_SMOKE: f64 = 0.0;

const REWARD_SHOOT_SELF: f64 = -100.0;
const REWARD_SHOOT_SELF_BLANK: f64 = 50.0;
const REWARD_SHOOT_SELF_SAWN: f64 = -200.0;

const REWARD_SHOOT_OTHER: f64 = 100.0;
const REWARD_SHOOT_OTHER_BLANK: f64 = -10.0;
const REWARD_SHOOT_OTHER_SAWN: f64 = 200.0;

const NUM_ITEM_SLOTS: i64 = 8;

const SHELL_GONE: i64 = 0;
const SHELL_UNKNOWN: i64 = 1;
const SHELL_BLANK: i64 = 2;
const SHELL_LIVE: i64 = 3;

const STUN_GONE: i64 = 0;
const STUN_HEALTHY: i64 = 1;
const STUN_RECOVERING: i64 = 2;
const STUN_STUNNED: i64 = 3;

const MAX_HEALTH: i64 = 6;

const TOTAL_SEATS: usize = 4;
const OTHER_SEATS: usize = TOTAL_SEATS - 1;

struct PlayerKnowledge {
    shells: VecDeque<i64>,
    remaining_live_rounds: i64,
    seat_map: SeatMap,
}

fn generate_master_item_list() -> Vec<Item> {
    initialize_item_count_map()
        .keys()
        .map(|item| *item)
        .collect()
}

struct GameSessionDomain<'session, TRng, F> {
    game_session: &'session RefCell<GameSession<TRng>>,
    player_knowledge: PlayerKnowledge,
    player_number: PlayerNumber,
    shell_inverted: bool,
    master_item_list: Vec<Item>,
    logger: F,
    logging_enabled: bool,
}

impl PlayerKnowledge {
    fn new(seat_map: SeatMap) -> Self {
        PlayerKnowledge {
            shells: VecDeque::with_capacity(MAX_SHELLS.try_into().unwrap()),
            remaining_live_rounds: 0,
            seat_map,
        }
    }
}

impl PlayerKnowledge {
    fn initialize(&mut self, total_shells: usize, live_shells: usize) {
        self.shells.clear();
        for _ in 0..total_shells {
            self.shells.push_back(SHELL_UNKNOWN);
        }

        let max_shells: usize = MAX_SHELLS.try_into().unwrap();
        for _ in 0..(max_shells - total_shells) {
            self.shells.push_back(SHELL_GONE);
        }

        self.remaining_live_rounds = live_shells.try_into().unwrap();
    }
}

pub enum ActionOrTurnSummary<'turn, TRng> {
    Action(String),
    TurnSummary(&'turn TurnSummary<TRng>),
}

impl<'session, TRng, F> GameSessionDomain<'session, TRng, F>
where
    TRng: Rng,
    F: FnMut(ActionOrTurnSummary<TRng>),
{
    pub fn new(
        game_session: &'session RefCell<GameSession<TRng>>,
        player_number: PlayerNumber,
        logger: F,
        logging_enabled: bool,
    ) -> Self {
        let session = game_session.borrow();
        let seats = session.round().unwrap().seats();
        let player_knowledge = PlayerKnowledge::new(SeatMap::new(player_number, seats));

        let mut domain = GameSessionDomain {
            game_session,
            player_knowledge,
            player_number,
            shell_inverted: false,
            master_item_list: generate_master_item_list(),
            logger,
            logging_enabled,
        };

        domain.initialize_knowledge();

        domain
    }

    fn initialize_knowledge(&mut self) {
        let session = self.game_session.borrow();
        let shells = session.round().unwrap().shells();
        let shell_count = shells.len();
        let live_shell_count = shells
            .iter()
            .filter(|shell| shell.shell_type() == ShellType::Live)
            .count();

        self.player_knowledge
            .initialize(shell_count, live_shell_count);
    }
}

/// Seats are always coded (self?)left/opposite/right
impl<'session, TRng, F> Domain for GameSessionDomain<'session, TRng, F>
where
    TRng: Rng,
{
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Ordinal;

    fn state_space(&self) -> Self::StateSpace {
        let mut space = ProductSpace::empty();

        // round number or 0 for complete
        space = space + Interval::bounded(0, 3);

        // turn calculator with 0 being current player
        space = space + Interval::bounded(0, 4);

        // knowledge of number of remaining live shells
        space = space + Interval::bounded(0, 9);

        // inversion bit
        space = space + Interval::bounded(0, 1);

        // turn order inversion bit
        space = space + Interval::bounded(0, 1);

        // sawn bit
        space = space + Interval::bounded(0, 1);

        // player to go in next round 4: undetermined, 5: N/A
        space = space + Interval::bounded(0, 6);

        // knowledge of all ten shells positions starting with what's chambered
        for _ in 0..10 {
            // all 4 states
            space = space + Interval::bounded(0, 3); // total shells remaining
        }

        // health of self
        space = space + Interval::bounded(1, 6);

        // health of other seats
        for _ in 0..OTHER_SEATS {
            space = space + Interval::bounded(0, MAX_HEALTH);
        }

        // stun state of other seats
        for _ in 0..TOTAL_SEATS {
            space = space + Interval::bounded(0, 3);
        }

        // count of each item for all 4 seats
        for _ in 0..TOTAL_ITEMS {
            for _ in 0..TOTAL_SEATS {
                space = space + Interval::bounded(0, NUM_ITEM_SLOTS);
            }
        }

        space
    }

    fn action_space(&self) -> Self::ActionSpace {
        Ordinal::new(TOTAL_ACTIONS)
    }

    fn emit(&self) -> Observation<State<Self>> {
        let expected_capacity = 6 + 10 + 4 + 4 + TOTAL_ITEMS;
        let session = self.game_session.borrow();
        match session.round() {
            Some(round) => {
                let mut state: Vec<i64> = Vec::with_capacity(expected_capacity);

                state.push(match round.number() {
                    RoundNumber::One => 1,
                    RoundNumber::Two => 2,
                    RoundNumber::Three => 3,
                });

                let current_player = round.next_player();

                state.push(turn_distance(current_player, self.player_number));

                let knowledge = &self.player_knowledge;
                state.push(knowledge.remaining_live_rounds);
                state.push(if self.shell_inverted { 1 } else { 0 });
                let modifiers = round.game_modifiers();
                state.push(if modifiers.turn_order_inverted { 1 } else { 0 });
                state.push(if modifiers.shotgun_sawn { 1 } else { 0 });
                state.push(match round.number() {
                    RoundNumber::One | RoundNumber::Two => match round.first_dead_player() {
                        Some(player_number) => player_as_index(player_number),
                        None => 4,
                    },
                    RoundNumber::Three => 5,
                });

                for i in 0..knowledge.shells.len() {
                    state.push(knowledge.shells[i]);
                }

                let seats = round.seats();
                let seat_map = &knowledge.seat_map;
                let mut add_seat_health = |seat_index: Option<usize>| {
                    state.push(match seat_index {
                        Some(seat_index) => match seats[seat_index].player() {
                            Some(player) => player.health().into(),
                            None => 0,
                        },
                        None => 0,
                    })
                };

                add_seat_health(Some(seat_map.own_seat_index));
                add_seat_health(seat_map.left_seat_index);
                add_seat_health(Some(seat_map.opposite_seat_index));
                add_seat_health(seat_map.right_seat_index);

                let mut add_seat_stun = |seat_index: Option<usize>| {
                    state.push(match seat_index {
                        Some(seat_index) => match seats[seat_index].player() {
                            Some(player) => match player.stun_state() {
                                StunState::Unstunned => STUN_HEALTHY,
                                StunState::Stunned => STUN_STUNNED,
                                StunState::Recovering => STUN_RECOVERING,
                            },
                            None => STUN_GONE,
                        },
                        None => STUN_GONE,
                    })
                };

                add_seat_stun(Some(seat_map.own_seat_index));
                add_seat_stun(seat_map.left_seat_index);
                add_seat_stun(Some(seat_map.opposite_seat_index));
                add_seat_stun(seat_map.right_seat_index);

                let mut add_seat_items = |seat_index: Option<usize>| {
                    let seat_base_index = state.len();
                    for _ in 0..TOTAL_ITEMS {
                        state.push(0);
                    }

                    if let Some(seat_index) = seat_index {
                        for item in seats[seat_index].items() {
                            let index = get_item_index(item);
                            state[seat_base_index + index] += 1;
                        }
                    }
                };

                add_seat_items(Some(seat_map.own_seat_index));
                add_seat_items(seat_map.left_seat_index);
                add_seat_items(Some(seat_map.opposite_seat_index));
                add_seat_items(seat_map.right_seat_index);

                assert!(state.len() == expected_capacity);
                Observation::Full(state)
            }
            None => {
                let mut state = Vec::with_capacity(expected_capacity);
                for _ in 0..expected_capacity {
                    state.push(0);
                }

                Observation::Terminal(state)
            }
        }
    }

    fn step(&mut self, a: &Action<Self>) -> (Observation<State<Self>>, Reward) {
        let mut session = self.game_session.borrow_mut();
        let round = session.round().unwrap();
        let current_player = round.next_player();
        let own_player = self.player_number;

        // first validate the action
        let action = GameAction::parse(a);

        let reward = match action {
            GameAction::Observe => if current_player == self.player_number {
                self.log(|| format!("Player {} invalidly tries to observe!", player_number));
                REWARD_INVALID_ACTION
            }else {},
            GameAction::Shoot(_) => todo!(),
            GameAction::UnaryItem(_) => todo!(),
            GameAction::Jammer(_) => todo!(),
            GameAction::Adreneline(adreneline_target) => todo!(),
        }

        session
            .with_turn(|turn| turn.shoot(current_player), |summary| {})
            .unwrap();
        todo!()
    }
}

fn get_item_index(item: &Item) -> usize {
    match item {
        Item::NotAdreneline(not_adreneline) => match not_adreneline {
            NotAdreneline::UnaryItem(unary_item) => match unary_item {
                UnaryItem::Remote => 0,
                UnaryItem::Phone => 1,
                UnaryItem::Inverter => 2,
                UnaryItem::MagnifyingGlass => 3,
                UnaryItem::Cigarettes => 4,
                UnaryItem::Handsaw => 5,
                UnaryItem::Beer => 6,
            },
            NotAdreneline::Jammer => 7,
        },
        Item::Adreneline => 8,
    }
}

fn turn_distance(current: PlayerNumber, target: PlayerNumber) -> i64 {
    let current_index = player_as_index(current);
    let target_index = player_as_index(target);

    let raw_distance = target_index - current_index;
    if raw_distance >= 0 {
        raw_distance
    } else {
        raw_distance + 4
    }
}

#[test]
fn test_turn_disance() {
    assert_eq!(0, turn_distance(PlayerNumber::One, PlayerNumber::One));
    assert_eq!(0, turn_distance(PlayerNumber::Three, PlayerNumber::Three));
    assert_eq!(2, turn_distance(PlayerNumber::One, PlayerNumber::Three));
    assert_eq!(3, turn_distance(PlayerNumber::One, PlayerNumber::Four));
    assert_eq!(1, turn_distance(PlayerNumber::Four, PlayerNumber::One));
    assert_eq!(3, turn_distance(PlayerNumber::Four, PlayerNumber::Three));
}

fn player_as_index(player: PlayerNumber) -> i64 {
    match player {
        PlayerNumber::One => 0,
        PlayerNumber::Two => 1,
        PlayerNumber::Three => 2,
        PlayerNumber::Four => 3,
    }
}
