use std::{cell::RefCell, ops::Index};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession,
    item::{initialize_item_count_map, Item, NotAdreneline, UnaryItem, TOTAL_ITEMS},
    player_number::PlayerNumber,
    round::{Round, RoundContinuation, TurnContinuation, TurnSummary},
    round_number::RoundNumber,
    round_player::StunState,
    seat::Seat,
    shell::{ShellType, ShotgunDamage},
    turn::{ItemUseResult, TakenAction, TerminalAction},
};
use rand::Rng;
use rsrl::{
    domains::{Action, Domain, Observation, Reward, State},
    spaces::{discrete::Ordinal, real::Interval, ProductSpace},
};

use crate::{
    game_action::{ActiveGameAction, AdrenelineItem, GameAction, TOTAL_ACTIONS},
    player_knowledge::{GlobalShellUpdate, PlayerKnowledge, ShellUpdate},
    relative_player::{OtherPlayer, RelativePlayer},
    seat_map::SeatMap,
};

const REWARD_INVALID_ACTION: f64 = -1000000.0;
const REWARD_ROUND_WIN: f64 = 4000.0;
const REWARD_ROUND_LOSS: f64 = -1000.0;

const REWARD_GAIN_HEALTH: f64 = 50.0;
const REWARD_USELESS_SMOKE: f64 = 0.0;
const REWARD_LOST_HEALTH: f64 = -25.0;

const REWARD_ITEM_GENERIC: f64 = 0.0;

const REWARD_SHOOT_SELF: f64 = -100.0;
const REWARD_KILL_SELF: f64 = -500.0 + REWARD_ROUND_LOSS;
const REWARD_SHOOT_SELF_BLANK: f64 = 50.0;
const REWARD_SHOOT_SELF_SAWN: f64 = -200.0;

const REWARD_SHOOT_OTHER: f64 = 100.0;
const REWARD_SHOOT_OTHER_BLANK: f64 = -10.0;
const REWARD_SHOOT_OTHER_SAWN: f64 = 200.0;

const NUM_ITEM_SLOTS: i32 = 8;

const STUN_GONE: i32 = 0;
const STUN_HEALTHY: i32 = 1;
const STUN_RECOVERING: i32 = 2;
const STUN_STUNNED: i32 = 3;

const MAX_HEALTH: i32 = 6;

pub const TOTAL_SEATS: usize = 4;
pub const OTHER_SEATS: usize = TOTAL_SEATS - 1;
fn generate_master_item_list() -> Vec<Item> {
    initialize_item_count_map()
        .keys()
        .map(|item| *item)
        .collect()
}

#[derive(Debug, Clone)]
struct PriorObservation {
    observation: Observation<Vec<f64>>,
    prior_health: i32,
}

#[derive(Debug, Clone)]
pub struct GameSessionDomain<'session, TRng> {
    prior_observation: Option<PriorObservation>,
    game_session: &'session RefCell<GameSession<TRng>>,
    knowledge: PlayerKnowledge,
    player_number: PlayerNumber,
    shell_inverted: bool,
    logging_enabled: bool,
    action_update: Option<ActionUpdate>,
}

#[derive(Debug, Clone, Default)]
pub struct ActionUpdate {
    pub global_shell_update: Option<GlobalShellUpdate>,
    pub new_loadout: bool,
}

pub enum ActionOrTurnSummary<'turn, TRng> {
    Action(String),
    TurnSummary(&'turn TurnSummary<TRng>),
}

impl<'session, TRng> GameSessionDomain<'session, TRng>
where
    TRng: Rng,
{
    pub fn new(
        game_session: &'session RefCell<GameSession<TRng>>,
        player_number: PlayerNumber,
        logging_enabled: bool,
    ) -> Self {
        let session = game_session.borrow();
        let seats = session.round().unwrap().seats();
        let seat_map = SeatMap::new(player_number, seats);
        let knowledge = PlayerKnowledge::new(seat_map);

        GameSessionDomain {
            game_session,
            knowledge,
            player_number,
            logging_enabled,
            prior_observation: None,
            shell_inverted: false,
            action_update: None,
        }
    }

    pub fn action_update(&mut self) -> Option<ActionUpdate> {
        self.action_update.take()
    }

    fn log_action<FAction>(&mut self, mut message_factory: FAction)
    where
        FAction: FnMut() -> String,
    {
        if self.logging_enabled {
            basic_log(ActionOrTurnSummary::<TRng>::Action(message_factory()));
        }
    }

    fn set_action_update<F>(&mut self, updater: F)
    where
        F: FnOnce(&mut ActionUpdate),
    {
        let update = self.action_update.get_or_insert(Default::default());
        updater(update);
    }

    fn log_summary(&mut self, summary: &TurnSummary<TRng>) {
        if let RoundContinuation::RoundContinues(continuation) = &summary.round_continuation {
            if let TurnContinuation::LoadoutEnds(_) = &continuation.turn_continuation {
                self.set_action_update(|update| update.new_loadout = true);
            }
        }

        if self.logging_enabled {
            basic_log(ActionOrTurnSummary::TurnSummary(summary));
        }
    }

    pub fn update_global_knowledge(&mut self, update: &GlobalShellUpdate) {
        self.knowledge.update(ShellUpdate::Global(*update));
    }

    pub fn pre_action_observe(&mut self) -> bool {
        assert!(self.prior_observation.is_none());

        let session = self.game_session.borrow();
        match session.round() {
            Some(round) => {
                let prior_health = get_player_health(round, self.player_number);
                self.prior_observation = Some(PriorObservation {
                    observation: self.emit(),
                    prior_health,
                });
                return round.next_player() == self.player_number;
            }
            None => panic!("Should not be observing a completed game!"),
        }
    }

    fn bad_active_game_action(&mut self) -> f64 {
        self.log_action(|| "Invalid action attempt!".to_string());
        let own_player = self.player_number;
        let mut session = self.game_session.borrow_mut();
        let ejected_shell = session
            .with_turn(
                |turn| turn.shoot(own_player),
                |summary| {
                    self.log_summary(summary);
                    match summary.shot_result.as_ref().unwrap().damage {
                        ShotgunDamage::Blank => GlobalShellUpdate::Ejected(ShellType::Blank),
                        ShotgunDamage::RegularShot(_) | ShotgunDamage::SawedShot(_) => {
                            GlobalShellUpdate::Ejected(ShellType::Live)
                        }
                    }
                },
            )
            .unwrap()
            .unwrap();

        self.set_action_update(|update| update.global_shell_update = Some(ejected_shell));

        REWARD_INVALID_ACTION
    }

    fn item_action_available_check(&mut self, item: Item) -> Option<f64> {
        let session = self.game_session.borrow();
        let round = session.round().unwrap();

        let seat = get_player_seat(round, self.player_number).unwrap();
        let valid_use = seat_has_item(seat, item);
        if valid_use {
            self.log_action(|| format!("Uses item: {}", item));
            None
        } else {
            Some(self.bad_active_game_action())
        }
    }
}
/// Seats are always coded (self?)left/opposite/right
impl<'session, TRng> Domain for GameSessionDomain<'session, TRng>
where
    TRng: Rng,
{
    type StateSpace = ProductSpace<Interval>;
    type ActionSpace = Ordinal;

    fn state_space(&self) -> Self::StateSpace {
        state_space_static()
    }

    fn action_space(&self) -> Self::ActionSpace {
        action_space_static()
    }

    fn emit(&self) -> Observation<State<Self>> {
        let expected_capacity = 6 + 10 + 4 + 4 + TOTAL_ITEMS;
        let session = self.game_session.borrow();
        match session.round() {
            Some(round) => {
                let mut state: Vec<f64> = Vec::with_capacity(expected_capacity);

                state.push(match round.number() {
                    RoundNumber::One => 1.0,
                    RoundNumber::Two => 2.0,
                    RoundNumber::Three => 3.0,
                });

                let current_player = round.next_player();

                state.push(turn_distance(current_player, self.player_number).into());

                let knowledge = &self.knowledge;
                state.push(knowledge.remaining_live_rounds.into());
                state.push(if self.shell_inverted { 1.0 } else { 0.0 });
                let modifiers = round.game_modifiers();
                state.push(if modifiers.turn_order_inverted {
                    1.0
                } else {
                    0.0
                });
                state.push(if modifiers.shotgun_sawn { 1.0 } else { 0.0 });
                state.push(match round.number() {
                    RoundNumber::One | RoundNumber::Two => match round.first_dead_player() {
                        Some(player_number) => player_as_index(player_number).into(),
                        None => 4.0,
                    },
                    RoundNumber::Three => 5.0,
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
                            None => 0.0,
                        },
                        None => 0.0,
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
                                StunState::Unstunned => STUN_HEALTHY.into(),
                                StunState::Stunned => STUN_STUNNED.into(),
                                StunState::Recovering => STUN_RECOVERING.into(),
                            },
                            None => STUN_GONE.into(),
                        },
                        None => STUN_GONE.into(),
                    })
                };

                add_seat_stun(Some(seat_map.own_seat_index));
                add_seat_stun(seat_map.left_seat_index);
                add_seat_stun(Some(seat_map.opposite_seat_index));
                add_seat_stun(seat_map.right_seat_index);

                let mut add_seat_items = |seat_index: Option<usize>| {
                    let seat_base_index = state.len();
                    for _ in 0..TOTAL_ITEMS {
                        state.push(0.0);
                    }

                    if let Some(seat_index) = seat_index {
                        for item in seats[seat_index].items() {
                            let index = get_item_index(item);
                            state[seat_base_index + index] += 1.0;
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
                    state.push(0.0);
                }

                Observation::Terminal(state)
            }
        }
    }

    fn step(&mut self, a: &Action<Self>) -> (Observation<State<Self>>, Reward) {
        assert!(self.action_update.is_none());
        let mut session = self.game_session.borrow_mut();
        let round = session.round().unwrap();
        let current_player = round.next_player();
        let own_player = self.player_number;

        // first validate the action
        let action = GameAction::parse(*a);

        let prior_observation = self.prior_observation.take().unwrap();

        let seat_map = self.knowledge.seat_map.clone();
        let reward = match action {
            GameAction::Observe => {
                if current_player == own_player {
                    self.log_action(|| "Invalid observation attempt!".to_string());
                    REWARD_INVALID_ACTION
                } else {
                    let current_player_health = get_player_health(round, own_player);
                    let delta = prior_observation.prior_health - current_player_health;
                    assert!(current_player_health <= prior_observation.prior_health);
                    REWARD_LOST_HEALTH * f64::from(delta)
                }
            }
            GameAction::Act(active_game_action) => {
                if current_player != own_player {
                    self.log_action(|| "Invalid action attempt!".to_string());
                    REWARD_INVALID_ACTION
                } else {
                    match active_game_action {
                        ActiveGameAction::Shoot(target_player) => match target_player {
                            RelativePlayer::Own => {
                                self.log_action(|| "Shoots self".to_string());
                                let (ejected_shell, reward) = session
                                    .with_turn(
                                        |turn| turn.shoot(own_player),
                                        |summary| {
                                            self.log_summary(summary);
                                            let shot_result = summary.shot_result.as_ref().unwrap();
                                            match shot_result.damage {
                                                ShotgunDamage::Blank => (
                                                    GlobalShellUpdate::Ejected(ShellType::Blank),
                                                    REWARD_SHOOT_SELF_BLANK,
                                                ),
                                                ShotgunDamage::RegularShot(killed) => (
                                                    GlobalShellUpdate::Ejected(ShellType::Live),
                                                    if killed {
                                                        REWARD_KILL_SELF
                                                    } else {
                                                        REWARD_SHOOT_SELF
                                                    },
                                                ),
                                                ShotgunDamage::SawedShot(killed) => (
                                                    GlobalShellUpdate::Ejected(ShellType::Live),
                                                    if killed {
                                                        REWARD_KILL_SELF
                                                    } else {
                                                        REWARD_SHOOT_SELF_SAWN
                                                    },
                                                ),
                                            }
                                        },
                                    )
                                    .unwrap()
                                    .unwrap();

                                self.set_action_update(|update| {
                                    update.global_shell_update = Some(ejected_shell)
                                });
                                reward
                            }
                            RelativePlayer::Other(other_player) => {
                                match get_other_player(round, &seat_map, other_player) {
                                    Some(target_seat) => match target_seat.player() {
                                        Some(target_player) => {
                                            let target_player_number =
                                                target_player.player_number();
                                            self.log_action(|| {
                                                format!("Shoots {}", target_player_number)
                                            });
                                            let (ejected_shell, reward) = session
                                                .with_turn(
                                                    |turn| turn.shoot(own_player),
                                                    |summary| {
                                                        self.log_summary(summary);
                                                        let shot_result =
                                                            summary.shot_result.as_ref().unwrap();
                                                        match shot_result.damage {
                                                            ShotgunDamage::Blank => (
                                                                GlobalShellUpdate::Ejected(
                                                                    ShellType::Blank,
                                                                ),
                                                                REWARD_SHOOT_OTHER_BLANK,
                                                            ),
                                                            ShotgunDamage::RegularShot(_)
                                                            | ShotgunDamage::SawedShot(_) => (
                                                                GlobalShellUpdate::Ejected(
                                                                    ShellType::Live,
                                                                ),
                                                                REWARD_SHOOT_OTHER,
                                                            ),
                                                        }
                                                    },
                                                )
                                                .unwrap()
                                                .unwrap();

                                            self.set_action_update(|update| {
                                                update.global_shell_update = Some(ejected_shell)
                                            });
                                            reward
                                        }
                                        None => self.bad_active_game_action(),
                                    },
                                    None => self.bad_active_game_action(),
                                }
                            }
                        },
                        ActiveGameAction::UnaryItem(unary_item) => {
                            let reward = self.item_action_available_check(Item::NotAdreneline(
                                NotAdreneline::UnaryItem(unary_item),
                            ));
                            match reward {
                                Some(reward) => reward,
                                None => {
                                    if unary_item == UnaryItem::Handsaw
                                        && round.game_modifiers().shotgun_sawn
                                    {
                                        self.bad_active_game_action()
                                    } else {
                                        let mut shell_update = None;
                                        session
                                        .with_turn(
                                            |turn| {
                                                let taken_action = turn.use_unary_item(unary_item);
                                                shell_update = match &taken_action {
                                                    TakenAction::Continued(continued_turn) => {
                                                        match continued_turn.item_result().as_ref().unwrap() {
                                                            ItemUseResult::Default => None,
                                                            ItemUseResult::ShotgunRacked(shotgun_rack_result) => Some(ShellUpdate::Global(GlobalShellUpdate::Ejected(shotgun_rack_result.ejected_shell_type))),
                                                            ItemUseResult::LearnedShell(learned_shell) => Some(ShellUpdate::Learned(learned_shell.clone())),
                                                            ItemUseResult::StunnedPlayer(_) => unreachable!(),
                                                        }
                                                    }
                                                    TakenAction::Terminal(taken_turn) => {
                                                        match &taken_turn.action {
                                                            TerminalAction::Item(
                                                                item_use_result,
                                                            ) => match item_use_result {
                                                                ItemUseResult::ShotgunRacked(shotgun_rack_result) => Some(ShellUpdate::Global(GlobalShellUpdate::Ejected(shotgun_rack_result.ejected_shell_type))),
                                                                ItemUseResult::Default | ItemUseResult::LearnedShell(_) | ItemUseResult::StunnedPlayer(_) => unreachable!(),
                                                            },
                                                            TerminalAction::Shot(_) => unreachable!()
                                                        }
                                                    }
                                                };

                                                taken_action
                                            },
                                            |summary| self.log_summary(summary),
                                        )
                                        .unwrap();

                                        if unary_item == UnaryItem::Inverter {
                                            shell_update = Some(ShellUpdate::Global(
                                                GlobalShellUpdate::Inverted,
                                            ));
                                        }

                                        if let Some(shell_update) = shell_update {
                                            match shell_update {
                                                ShellUpdate::Learned(_) => {
                                                    self.knowledge.update(shell_update)
                                                }
                                                ShellUpdate::Global(global_shell_update) => {
                                                    self.set_action_update(|update| {
                                                        update.global_shell_update =
                                                            Some(global_shell_update)
                                                    });
                                                }
                                            };
                                        }

                                        if unary_item == UnaryItem::Cigarettes {
                                            let current_player_health = get_player_health(
                                                session.round().unwrap(),
                                                own_player,
                                            );
                                            if prior_observation.prior_health
                                                == current_player_health
                                            {
                                                REWARD_USELESS_SMOKE
                                            } else {
                                                REWARD_GAIN_HEALTH
                                            }
                                        } else {
                                            REWARD_ITEM_GENERIC
                                        }
                                    }
                                }
                            }
                        }
                        ActiveGameAction::Jammer(other_player) => {
                            let reward = self.item_action_available_check(Item::NotAdreneline(
                                NotAdreneline::Jammer,
                            ));
                            match reward {
                                Some(reward) => reward,
                                None => match get_other_player(round, &seat_map, other_player) {
                                    Some(target_seat) => match target_seat.player() {
                                        Some(target_player) => match target_player.stun_state() {
                                            StunState::Unstunned => {
                                                let target_player_number =
                                                    target_player.player_number();

                                                self.log_action(|| {
                                                    format!(
                                                        "Targeting {} with Jammer",
                                                        target_player_number
                                                    )
                                                });
                                                session
                                                    .with_turn(
                                                        |turn| {
                                                            turn.use_jammer(target_player_number)
                                                        },
                                                        |_| unreachable!(),
                                                    )
                                                    .unwrap();
                                                REWARD_ITEM_GENERIC
                                            }
                                            StunState::Stunned | StunState::Recovering => {
                                                self.bad_active_game_action()
                                            }
                                        },
                                        None => self.bad_active_game_action(),
                                    },
                                    None => self.bad_active_game_action(),
                                },
                            }
                        }
                        ActiveGameAction::Adreneline(adreneline_target) => {
                            let reward = self.item_action_available_check(Item::Adreneline);
                            match reward {
                                Some(reward) => reward,
                                None => {
                                    match get_other_player(
                                        round,
                                        &seat_map,
                                        adreneline_target.target_player,
                                    ) {
                                        Some(seat) => {
                                            let target_item = match adreneline_target.item {
                                                AdrenelineItem::Unary(unary_item) => {
                                                    NotAdreneline::UnaryItem(unary_item)
                                                }
                                                AdrenelineItem::Jammer(_) => NotAdreneline::Jammer,
                                            };

                                            if seat_has_item(seat, Item::NotAdreneline(target_item))
                                            {
                                                let theive_from = seat.player_number();
                                                match adreneline_target.item {
                                                    AdrenelineItem::Unary(unary_item) => {
                                                        if unary_item == UnaryItem::Handsaw
                                                            && round.game_modifiers().shotgun_sawn
                                                        {
                                                            self.bad_active_game_action()
                                                        } else {
                                                            self.log_action(|| format!("Using {} stolen from player {}", Item::NotAdreneline(NotAdreneline::UnaryItem(unary_item)), theive_from));
                                                            session
                                                                .with_turn(
                                                                    |turn| {
                                                                        turn.use_adreneline(
                                                                            theive_from,
                                                                            unary_item,
                                                                        )
                                                                    },
                                                                    |_| unreachable!(),
                                                                )
                                                                .unwrap();
                                                            REWARD_ITEM_GENERIC
                                                        }
                                                    }
                                                    AdrenelineItem::Jammer(other_player) => {
                                                        match get_other_player(
                                                            round,
                                                            &seat_map,
                                                            other_player,
                                                        ) {
                                                            Some(target_seat) => {
                                                                match target_seat.player() {
                                                                    Some(target_player) =>  match target_player.stun_state() {
                                                                        StunState::Unstunned => {
                                                                            let target_player_number =
                                                                                target_player.player_number();
                                                                            self.log_action(|| format!("Targeting {} with jammer stolen from player {}", target_player_number, theive_from));
                                                                            session
                                                                                .with_turn(
                                                                                    |turn| {
                                                                                        turn.use_adreneline_then_jammer(theive_from, target_player_number)
                                                                                    },
                                                                                    |_| unreachable!(),
                                                                                )
                                                                                .unwrap();
                                                                            REWARD_ITEM_GENERIC
                                                                        }
                                                                        StunState::Stunned | StunState::Recovering => {
                                                                            self.bad_active_game_action()
                                                                        }
                                                                    },
                                                                    None => self
                                                                        .bad_active_game_action(),
                                                                }
                                                            }
                                                            None => self.bad_active_game_action(),
                                                        }
                                                    }
                                                }
                                            } else {
                                                self.bad_active_game_action()
                                            }
                                        }
                                        None => self.bad_active_game_action(),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };
        (self.emit(), reward)
    }
}

pub fn state_space_static() -> ProductSpace<Interval> {
    let mut space = ProductSpace::empty();

    // round number or 0 for complete
    space = space + Interval::bounded(0.0, 3.0);

    // turn calculator with 0 being current player
    space = space + Interval::bounded(0.0, 4.0);

    // knowledge of number of remaining live shells
    space = space + Interval::bounded(0.0, 9.0);

    // inversion bit
    space = space + Interval::bounded(0.0, 1.0);

    // turn order inversion bit
    space = space + Interval::bounded(0.0, 1.0);

    // sawn bit
    space = space + Interval::bounded(0.0, 1.0);

    // player to go in next round 4: undetermined, 5: N/A
    space = space + Interval::bounded(0.0, 6.0);

    // knowledge of all ten shells positions starting with what's chambered
    for _ in 0..10 {
        // all 4 states
        space = space + Interval::bounded(0.0, 3.0); // total shells remaining
    }

    // health of self
    space = space + Interval::bounded(1.0, 6.0);

    // health of other seats
    for _ in 0..OTHER_SEATS {
        space = space + Interval::bounded(0.0, MAX_HEALTH.into());
    }

    // stun state of other seats
    for _ in 0..TOTAL_SEATS {
        space = space + Interval::bounded(0.0, 3.0);
    }

    // count of each item for all 4 seats
    for _ in 0..TOTAL_ITEMS {
        for _ in 0..TOTAL_SEATS {
            space = space + Interval::bounded(0.0, NUM_ITEM_SLOTS.into());
        }
    }

    space
}

pub fn action_space_static() -> Ordinal {
    Ordinal::new(TOTAL_ACTIONS)
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

fn turn_distance(current: PlayerNumber, target: PlayerNumber) -> i32 {
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

fn player_as_index(player: PlayerNumber) -> i32 {
    match player {
        PlayerNumber::One => 0,
        PlayerNumber::Two => 1,
        PlayerNumber::Three => 2,
        PlayerNumber::Four => 3,
    }
}

fn get_other_player<'round, TRng>(
    round: &'round Round<TRng>,
    seat_map: &SeatMap,
    other_player: OtherPlayer,
) -> Option<&'round Seat>
where
    TRng: Rng,
{
    let seats = round.seats();

    let index = match other_player {
        OtherPlayer::Left => seat_map.left_seat_index,
        OtherPlayer::Opposite => Some(seat_map.opposite_seat_index),
        OtherPlayer::Right => seat_map.right_seat_index,
    };

    index.map(|index| seats.index(index))
}

fn get_player_seat<TRng>(round: &Round<TRng>, player_number: PlayerNumber) -> Option<&Seat>
where
    TRng: Rng,
{
    round
        .seats()
        .iter()
        .filter(|seat| seat.player_number() == player_number)
        .next()
}

fn get_player_health<TRng>(round: &Round<TRng>, player_number: PlayerNumber) -> i32
where
    TRng: Rng,
{
    round
        .seats()
        .iter()
        .filter_map(|seat| match seat.player() {
            Some(player) => Some(if player_number == player.player_number() {
                player.health()
            } else {
                0
            }),
            None => {
                if player_number == seat.player_number() {
                    Some(0)
                } else {
                    None
                }
            }
        })
        .next()
        .unwrap()
}

fn seat_has_item(seat: &Seat, item: Item) -> bool {
    seat.items().iter().any(|seat_item| *seat_item == item)
}

fn basic_log<'turn, TRng>(action_or_summary: ActionOrTurnSummary<'turn, TRng>) {
    match action_or_summary {
        ActionOrTurnSummary::Action(_) => todo!(),
        ActionOrTurnSummary::TurnSummary(_) => todo!(),
    }
}
