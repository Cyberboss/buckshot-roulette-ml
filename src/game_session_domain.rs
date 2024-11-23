use std::collections::{HashMap, VecDeque};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession,
    item::{
        initialize_item_count_map, Item, NotAdreneline, UnaryItem, TOTAL_ITEMS, TOTAL_UNARY_ITEMS,
    },
    loadout::MAX_SHELLS,
    multiplayer_count::MultiplayerCount,
    player_number::PlayerNumber,
    round_number::RoundNumber,
    round_player::StunState,
    shell::ShellType,
};
use rsrl::{
    domains::{Action, Domain, Observation, Reward, State},
    spaces::{
        discrete::{Interval, Ordinal},
        ProductSpace,
    },
};

const ACTION_SHOOT_SELF: u32 = 0;
const ACTION_SHOOT_LEFT: u32 = 1;
const ACTION_SHOOT_ACROSS: u32 = 2;

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

use rand::Rng;

use crate::seat_map::SeatMap;

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

struct GameSessionDomain<TRng> {
    game_session: GameSession<TRng>,
    player_knowledge: HashMap<PlayerNumber, PlayerKnowledge>,
    shell_inverted: bool,
    master_item_list: Vec<Item>,
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

impl<TRng> GameSessionDomain<TRng>
where
    TRng: Rng,
{
    pub fn new(multiplayer_count: MultiplayerCount, rng: TRng) -> Self {
        let mut player_knowledge = HashMap::with_capacity(TOTAL_SEATS);

        let game_session = GameSession::new(multiplayer_count, rng);
        let seats = game_session.round().unwrap().seats();

        player_knowledge.insert(
            PlayerNumber::One,
            PlayerKnowledge::new(SeatMap::new(PlayerNumber::One, seats)),
        );
        player_knowledge.insert(
            PlayerNumber::Two,
            PlayerKnowledge::new(SeatMap::new(PlayerNumber::One, seats)),
        );

        match multiplayer_count {
            MultiplayerCount::Two => {}
            MultiplayerCount::Three | MultiplayerCount::Four => {
                player_knowledge.insert(
                    PlayerNumber::Three,
                    PlayerKnowledge::new(SeatMap::new(PlayerNumber::One, seats)),
                );

                if multiplayer_count == MultiplayerCount::Four {
                    player_knowledge.insert(
                        PlayerNumber::One,
                        PlayerKnowledge::new(SeatMap::new(PlayerNumber::One, seats)),
                    );
                }
            }
        }

        let mut domain = GameSessionDomain {
            game_session,
            player_knowledge,
            shell_inverted: false,
            master_item_list: generate_master_item_list(),
        };

        domain.initialize_knowledge();

        domain
    }

    fn initialize_knowledge(&mut self) {
        let shells = self.game_session.round().unwrap().shells();
        let shell_count = shells.len();
        let live_shell_count = shells
            .iter()
            .filter(|shell| shell.shell_type() == ShellType::Live)
            .count();

        self.player_knowledge
            .iter_mut()
            .for_each(|(_, knowledge)| knowledge.initialize(shell_count, live_shell_count));
    }
}

/// Seats are always coded (self?)left/opposite/right
impl<TRng> Domain for GameSessionDomain<TRng>
where
    TRng: Rng,
{
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Ordinal;

    fn state_space(&self) -> Self::StateSpace {
        let mut space = ProductSpace::empty();

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
        for _ in 0..OTHER_SEATS {
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
        let unary_item_use_possibilities = TOTAL_UNARY_ITEMS;

        let jammer_use_possibilities = OTHER_SEATS;

        Ordinal::new(
            1 // shoot self
            + OTHER_SEATS // shoot other seats
            + unary_item_use_possibilities // use unary items
            + jammer_use_possibilities // use jammer
            + OTHER_SEATS * (unary_item_use_possibilities + jammer_use_possibilities), // use adreneline
        )
    }

    fn emit(&self) -> Observation<State<Self>> {
        let expected_capacity = 5 + 10 + 4 + 3 + TOTAL_ITEMS;
        match self.game_session.round() {
            Some(round) => {
                let mut state: Vec<i64> = Vec::with_capacity(expected_capacity);

                let knowledge = self.player_knowledge.get(&round.next_player()).unwrap();
                state.push(knowledge.remaining_live_rounds);
                state.push(if self.shell_inverted { 1 } else { 0 });
                let modifiers = round.game_modifiers();
                state.push(if modifiers.turn_order_inverted { 1 } else { 0 });
                state.push(if modifiers.shotgun_sawn { 1 } else { 0 });
                state.push(match round.number() {
                    RoundNumber::One | RoundNumber::Two => match round.first_dead_player() {
                        Some(player_number) => match player_number {
                            PlayerNumber::One => 0,
                            PlayerNumber::Two => 1,
                            PlayerNumber::Three => 2,
                            PlayerNumber::Four => 3,
                        },
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
