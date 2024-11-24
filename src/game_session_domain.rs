use std::{
    cell::{RefCell, RefMut},
    cmp::{max, min},
    ops::Index,
};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession,
    item::{
        global_item_limit, initialize_item_count_map, player_item_limit, Item, NotAdreneline,
        UnaryItem, ALL_ITEMS, TOTAL_ITEMS, TOTAL_UNARY_ITEMS,
    },
    loadout::MAX_SHELLS,
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
    player_knowledge::{GlobalShellUpdate, PlayerKnowledge, ShellUpdate, SHELL_STATE_MAX},
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

const NUM_ITEM_SLOTS: u32 = 8;

const STUN_HEALTHY_OR_GONE: i32 = 0;
const STUN_RECOVERING: i32 = 1;
const STUN_STUNNED: i32 = 2;
const STUN_MAX: i32 = STUN_STUNNED;

const MAX_HEALTH: u32 = 6;

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

        let mut domain = GameSessionDomain {
            game_session,
            knowledge,
            player_number,
            logging_enabled,
            prior_observation: None,
            shell_inverted: false,
            action_update: None,
        };

        domain.reset_knowledge();
        domain
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

    fn bad_active_game_action(&mut self, session: &mut RefMut<GameSession<TRng>>) -> f64 {
        self.log_action(|| "Invalid action attempt!".to_string());
        let own_player = self.player_number;
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

    pub fn reset_knowledge(&mut self) {
        let session = self.game_session.borrow();
        let loadout = session.round().unwrap().loadout();
        self.knowledge.initialize(
            loadout.initial_blank_rounds + loadout.initial_live_rounds,
            loadout.initial_live_rounds,
        );
    }

    fn item_action_available_check(
        &mut self,
        session: &mut RefMut<GameSession<TRng>>,
        item: Item,
    ) -> Option<f64> {
        let valid_use = match session.round() {
            Some(round) => {
                let seat = get_player_seat(round, self.player_number).unwrap();
                seat_has_item(seat, item)
            }
            None => false,
        };
        if valid_use {
            self.log_action(|| format!("Uses item: {}", item));
            None
        } else {
            Some(self.bad_active_game_action(session))
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
        let session = self.game_session.borrow();
        match session.round() {
            Some(round) => {
                let mut state: Vec<f64> = Vec::with_capacity(STATE_SIZE);

                let current_player = round.next_player();

                state.push(match current_player {
                    PlayerNumber::One => 0.0,
                    PlayerNumber::Two => 1.0,
                    PlayerNumber::Three => 2.0,
                    PlayerNumber::Four => 3.0,
                });

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

                for i in 0..knowledge.shells.len() {
                    state.push(knowledge.shells[i]);
                }

                let seats = round.seats();
                let seat_map = &knowledge.seat_map;
                let mut add_seat_health = |seat_index: Option<usize>| {
                    let health = match seat_index {
                        Some(seat_index) => {
                            let seat = &seats[seat_index];
                            match seat.player() {
                                Some(player) => player.health().into(),
                                None => {
                                    assert_ne!(seat.player_number(), current_player);
                                    0.0
                                }
                            }
                        }
                        None => 0.0,
                    };
                    state.push(health);
                };

                add_seat_health(Some(seat_map.own_seat_index));
                add_seat_health(seat_map.left_seat_index);
                add_seat_health(seat_map.opposite_seat_index);
                add_seat_health(seat_map.right_seat_index);

                let mut add_seat_stun = |seat_index: Option<usize>| {
                    state.push(match seat_index {
                        Some(seat_index) => match seats[seat_index].player() {
                            Some(player) => match player.stun_state() {
                                StunState::Unstunned => STUN_HEALTHY_OR_GONE.into(),
                                StunState::Stunned => STUN_STUNNED.into(),
                                StunState::Recovering => STUN_RECOVERING.into(),
                            },
                            None => STUN_HEALTHY_OR_GONE.into(),
                        },
                        None => STUN_HEALTHY_OR_GONE.into(),
                    })
                };

                add_seat_stun(Some(seat_map.own_seat_index));
                add_seat_stun(seat_map.left_seat_index);
                add_seat_stun(seat_map.opposite_seat_index);
                add_seat_stun(seat_map.right_seat_index);

                let get_item_index = |item| {
                    let result = match item {
                        Item::NotAdreneline(not_adreneline) => match not_adreneline {
                            NotAdreneline::UnaryItem(unary_item) => match unary_item {
                                UnaryItem::Remote => None,
                                UnaryItem::Phone => Some(0),
                                UnaryItem::Inverter => Some(1),
                                UnaryItem::MagnifyingGlass => Some(2),
                                UnaryItem::Cigarettes => Some(3),
                                UnaryItem::Handsaw => Some(4),
                                UnaryItem::Beer => Some(5),
                            },
                            NotAdreneline::Jammer => None,
                        },
                        Item::Adreneline => Some(6),
                    };

                    assert_eq!(result.is_some(), item_is_globally_limited(item).is_none());
                    result
                };
                let mut add_seat_non_global_items = |seat_index: Option<usize>| {
                    let seat_base_index = state.len();
                    for _ in 0..(TOTAL_ITEMS - GLOBALLY_LIMITED_ITEMS.len()) {
                        state.push(0.0);
                    }

                    if let Some(seat_index) = seat_index {
                        for item in seats[seat_index].items() {
                            if let Some(index) = get_item_index(*item) {
                                state[seat_base_index + index] += 1.0;
                            }
                        }
                    }
                };

                add_seat_non_global_items(Some(seat_map.own_seat_index));
                add_seat_non_global_items(seat_map.left_seat_index);
                add_seat_non_global_items(seat_map.opposite_seat_index);
                add_seat_non_global_items(seat_map.right_seat_index);

                let seat_has_item = |seat_index: Option<usize>, item| match seat_index {
                    Some(seat_index) => seats[seat_index]
                        .items()
                        .iter()
                        .any(|seat_item| *seat_item == item),
                    None => false,
                };

                let players_that_own_item = |item, limit| {
                    let mut result = Vec::with_capacity(limit);

                    if seat_has_item(Some(seat_map.own_seat_index), item) {
                        result.push(1.0);
                    }

                    if seat_has_item(seat_map.left_seat_index, item) {
                        result.push(2.0);
                    }

                    if seat_has_item(seat_map.opposite_seat_index, item) {
                        result.push(3.0);
                    }

                    if seat_has_item(seat_map.right_seat_index, item) {
                        result.push(4.0);
                    }

                    result
                };

                for item in GLOBALLY_LIMITED_ITEMS {
                    let global_item_limit = global_item_limit(item);
                    let mut found = 0;

                    // add player that owns globally limited item
                    for player in players_that_own_item(item, global_item_limit) {
                        state.push(player);
                        found += 1;
                    }

                    for _ in found..global_item_limit {
                        state.push(0.0);
                    }
                }

                assert_eq!(state.len(), STATE_SIZE);
                Observation::Full(state)
            }
            None => {
                let mut state = Vec::with_capacity(STATE_SIZE);
                for _ in 0..STATE_SIZE {
                    state.push(0.0);
                }

                Observation::Terminal(state)
            }
        }
    }

    fn step(&mut self, a: &Action<Self>) -> (Observation<State<Self>>, Reward) {
        assert!(self.action_update.is_none());
        let mut session = self.game_session.borrow_mut();
        let round = match session.round() {
            Some(round) => round,
            None => {
                drop(session);
                return (self.emit(), REWARD_ROUND_LOSS);
            }
        };
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
                    if current_player_health == 0 && prior_observation.prior_health > 0 {
                        REWARD_ROUND_LOSS
                    } else {
                        assert!(current_player_health <= prior_observation.prior_health);
                        REWARD_LOST_HEALTH * f64::from(delta)
                    }
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
                                                        let (ejected_shell, damage_reward) =
                                                            match shot_result.damage {
                                                                ShotgunDamage::Blank => (
                                                                    GlobalShellUpdate::Ejected(
                                                                        ShellType::Blank,
                                                                    ),
                                                                    REWARD_SHOOT_OTHER_BLANK,
                                                                ),
                                                                ShotgunDamage::RegularShot(_) => (
                                                                    GlobalShellUpdate::Ejected(
                                                                        ShellType::Live,
                                                                    ),
                                                                    REWARD_SHOOT_OTHER,
                                                                ),
                                                                ShotgunDamage::SawedShot(
                                                                    killed,
                                                                ) => (
                                                                    GlobalShellUpdate::Ejected(
                                                                        ShellType::Live,
                                                                    ),
                                                                    // don't overeward overkill
                                                                    if killed {
                                                                        REWARD_SHOOT_OTHER
                                                                    } else {
                                                                        REWARD_SHOOT_OTHER_SAWN
                                                                    },
                                                                ),
                                                            };

                                                        if let RoundContinuation::RoundEnds(_) =
                                                            summary.round_continuation
                                                        {
                                                            (ejected_shell, REWARD_ROUND_WIN)
                                                        } else {
                                                            (ejected_shell, damage_reward)
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
                                        None => self.bad_active_game_action(&mut session),
                                    },
                                    None => self.bad_active_game_action(&mut session),
                                }
                            }
                        },
                        ActiveGameAction::UnaryItem(unary_item) => {
                            let reward = self.item_action_available_check(
                                &mut session,
                                Item::NotAdreneline(NotAdreneline::UnaryItem(unary_item)),
                            );
                            let round = session.round().unwrap();
                            match reward {
                                Some(reward) => reward,
                                None => {
                                    if unary_item == UnaryItem::Handsaw
                                        && round.game_modifiers().shotgun_sawn
                                    {
                                        self.bad_active_game_action(&mut session)
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
                                                            TerminalAction::Shot(_) => unreachable!("Bad terminal action!")
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
                            let reward = self.item_action_available_check(
                                &mut session,
                                Item::NotAdreneline(NotAdreneline::Jammer),
                            );
                            let round = session.round().unwrap();
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
                                                self.bad_active_game_action(&mut session)
                                            }
                                        },
                                        None => self.bad_active_game_action(&mut session),
                                    },
                                    None => self.bad_active_game_action(&mut session),
                                },
                            }
                        }
                        ActiveGameAction::Adreneline(adreneline_target) => {
                            let reward =
                                self.item_action_available_check(&mut session, Item::Adreneline);
                            let round = session.round().unwrap();
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
                                                            self.bad_active_game_action(
                                                                &mut session,
                                                            )
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
                                                                            self.bad_active_game_action(
                                                                                &mut session,)
                                                                        }
                                                                    },
                                                                    None => self
                                                                        .bad_active_game_action(
                                                                            &mut session,),
                                                                }
                                                            }
                                                            None => self.bad_active_game_action(
                                                                &mut session,
                                                            ),
                                                        }
                                                    }
                                                }
                                            } else {
                                                self.bad_active_game_action(&mut session)
                                            }
                                        }
                                        None => self.bad_active_game_action(&mut session),
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        drop(session);
        (self.emit(), reward)
    }
}

struct SpaceBuilder {
    space: ProductSpace<Interval>,
    total_variables: usize,
}

impl SpaceBuilder {
    fn add_range(&mut self, limit_inclusive: u32) {
        self.add_range_with_lower_bound(0, limit_inclusive);
    }

    fn add_range_with_lower_bound(&mut self, lower_bound: u32, upper_bound: u32) {
        self.space = self.space.clone() + Interval::bounded(lower_bound.into(), upper_bound.into());
        self.total_variables += 1;
    }
}

const STATE_SIZE: usize = 52;

#[test]
fn test_state_size() {
    state_space_static();
}

pub fn state_space_static() -> ProductSpace<Interval> {
    let mut builder = SpaceBuilder {
        space: ProductSpace::empty(),
        total_variables: 0,
    };

    // turn calculator with 0 being current player
    builder.add_range(3);

    // knowledge of number of remaining live shells
    builder.add_range(7);

    // inversion bit
    builder.add_range(1);

    // turn order inversion bit
    builder.add_range(1);

    // sawn bit
    builder.add_range(1);

    // knowledge of all shell position starting with what's chambered
    for _ in 0..MAX_SHELLS {
        // all 4 states
        builder.add_range(SHELL_STATE_MAX); // total shells remaining
    }

    // health of self
    builder.add_range_with_lower_bound(1, MAX_HEALTH);

    // health/presence of other seats
    for _ in 0..OTHER_SEATS {
        builder.add_range(MAX_HEALTH);
    }

    // stun state of all seats
    for _ in 0..TOTAL_SEATS {
        builder.add_range(STUN_MAX.try_into().unwrap());
    }

    // count of each non-globally limited item for all 4 seats
    for _ in 0..TOTAL_SEATS {
        for item in ALL_ITEMS {
            if item_is_globally_limited(item).is_some() {
                continue;
            }

            builder.add_range(player_max_of_item(item));
        }
    }

    // for each globally limited item
    for item in ALL_ITEMS {
        if let Some(global_limit) = item_is_globally_limited(item) {
            // indicate the player that has it, 0 being none
            for _ in 0..global_limit {
                builder.add_range(4);
            }
        }
    }

    assert_eq!(STATE_SIZE, builder.total_variables);

    builder.space
}

const GLOBALLY_LIMITED_ITEMS: [Item; 2] = [
    Item::NotAdreneline(NotAdreneline::UnaryItem(UnaryItem::Remote)),
    Item::NotAdreneline(NotAdreneline::Jammer),
];

fn item_is_globally_limited(item: Item) -> Option<usize> {
    let limit = global_item_limit(item);
    if limit < max_items_on_table() {
        Some(limit)
    } else {
        None
    }
}

#[test]
fn test_globally_limited_items() {
    let mut limited = 0;
    for item in ALL_ITEMS {
        if let Some(_) = item_is_globally_limited(item) {
            limited += 1;
        }
    }

    assert_eq!(limited, GLOBALLY_LIMITED_ITEMS.len());
}

fn max_items_on_table() -> usize {
    let max_items_per_player: usize = NUM_ITEM_SLOTS.try_into().unwrap();
    TOTAL_SEATS * max_items_per_player
}

fn player_max_of_item(item: Item) -> u32 {
    min(player_item_limit(item).try_into().unwrap(), NUM_ITEM_SLOTS)
}

pub fn action_space_static() -> Ordinal {
    Ordinal::new(TOTAL_ACTIONS)
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
        OtherPlayer::Opposite => seat_map.opposite_seat_index,
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
        ActionOrTurnSummary::Action(action) => println!("{}", action),
        ActionOrTurnSummary::TurnSummary(summary) => {
            if let Some(shot_result) = &summary.shot_result {
                let killed;
                let damage = match shot_result.damage {
                    ShotgunDamage::Blank => {
                        killed = false;
                        "0"
                    }
                    ShotgunDamage::RegularShot(inner_killed) => {
                        killed = inner_killed;
                        "1"
                    }
                    ShotgunDamage::SawedShot(inner_killed) => {
                        killed = inner_killed;
                        "2"
                    }
                };

                println!(
                    "Player {} was shot taking {} damage. They were{} killed",
                    shot_result.target_player,
                    damage,
                    if killed { "" } else { " NOT" }
                );
            }

            match &summary.round_continuation {
                RoundContinuation::RoundContinues(_) => {}
                RoundContinuation::RoundEnds(finished_round) => {
                    println!("Player {} wins the round!", finished_round.winner())
                }
            }
        }
    }
}
