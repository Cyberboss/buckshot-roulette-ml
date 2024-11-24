use std::{cell::RefCell, collections::HashMap};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession,
    player_number::PlayerNumber,
    round::TurnSummary,
    turn::{TakenAction, Turn},
};
use rand::Rng;
use rsrl::domains::{Domain, Transition};

use crate::{game_session_domain::GameSessionDomain, player_knowledge::GlobalShellUpdate};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct GameController<'session, TRng, TAgent, FSample, FHandle> {
    session: &'session RefCell<GameSession<TRng>>,
    agent: Option<TAgent>,
    sample: FSample,
    handle: FHandle,
    enable_updates: bool,
    enable_logging: bool,
    domains: HashMap<PlayerNumber, GameSessionDomain<'session, TRng>>,
}

impl<'session, TRng, TAgent, FSample, FHandle>
    GameController<'session, TRng, TAgent, FSample, FHandle>
where
    TRng: Rng,
    FSample: Fn(TAgent, &GameSessionDomain<'session, TRng>) -> (usize, TAgent),
    FHandle: Fn(TAgent, &Transition<Vec<f64>, usize>) -> TAgent,
{
    pub fn new(
        session: &'session RefCell<GameSession<TRng>>,
        agent: TAgent,
        sample: FSample,
        handle: FHandle,
        enable_updates: bool,
        enable_logging: bool,
    ) -> Self {
        let controller = GameController {
            session,
            domains: HashMap::with_capacity(4),
            agent: Some(agent),
            sample,
            handle,
            enable_updates,
            enable_logging,
        };
        controller.log_loadout(true);
        controller
    }

    pub fn next_action_is_domain(&self) -> Option<bool> {
        match self.session.borrow().round() {
            Some(round) => {
                let next_player = round.next_player();
                Some(self.domains.contains_key(&next_player))
            }
            None => None,
        }
    }

    pub fn take_domain_action(&mut self) -> bool {
        let starting_round = self.session.borrow().round().unwrap().number();

        let mut current_players_turn = None;
        let domains = &mut self.domains;

        domains.iter_mut().for_each(|(player_number, domain)| {
            if domain.pre_action_observe() {
                current_players_turn = Some(*player_number)
            }
        });

        let acting_player = current_players_turn.unwrap();
        let acting_domain = domains.get_mut(&acting_player).unwrap();

        if self.enable_logging {
            println!("Processing action for player: {}", acting_player);
        }

        let (sample, mut agent) = (self.sample)(self.agent.take().unwrap(), &acting_domain);
        let transition = acting_domain.transition(sample);
        if self.enable_updates {
            agent = (self.handle)(agent, &transition);
        }

        self.agent = Some(agent);

        let action_update = acting_domain.action_update();

        if self.enable_logging {
            println!("Processing observations");
        }

        // update observing domains
        domains
            .iter_mut()
            .filter(|(player_number, _)| **player_number != acting_player)
            .for_each(|(_, domain)| {
                let (sample, mut agent) = (self.sample)(self.agent.take().unwrap(), &domain);
                let transition = domain.transition(sample);
                if self.enable_updates {
                    agent = (self.handle)(agent, &transition);
                }
                self.agent = Some(agent);

                if let Some(action_update) = &action_update {
                    if let Some(shell_update) = &action_update.global_shell_update {
                        domain.update_global_knowledge(shell_update);
                    }
                }
            });

        let mut new_loadout = false;
        if let Some(action_update) = &action_update {
            new_loadout = action_update.new_loadout
        }

        let new_round = match self.session.borrow().round() {
            Some(round) => round.number() != starting_round,
            None => true,
        };

        if new_loadout || new_round {
            self.log_loadout(new_round);
        }

        new_round && self.session.borrow().round().is_none()
    }

    fn log_loadout(&self, new_round: bool) {
        if !self.enable_logging {
            return;
        }
        let session = self.session.borrow();
        let round = session.round();
        if new_round {
            println!(
                "Starting round: {}",
                match round {
                    Some(round) => format!("{}", round.number()),
                    None => "End of game!".to_string(),
                }
            );

            if round.is_none() {
                return;
            }
        }

        let round = round.unwrap();
        let loadout = round.loadout();
        println!(
            "New loadout: {} Items, {} Lives, {} Blanks",
            loadout.new_items, loadout.initial_live_rounds, loadout.initial_blank_rounds
        );

        for i in 0..loadout.new_items {
            for player in round.living_players() {
                let items = player.items();
                let item_to_log = &items[(items.len() - (loadout.new_items - i)) - 1];
                println!("Player {} grabs {}", player.player_number(), item_to_log);
            }
        }
    }

    pub fn extract_agent(mut self) -> TAgent {
        self.agent.take().unwrap()
    }

    pub fn take_human_action<TurnF, SummaryF, TRet>(
        &mut self,
        turn_func: TurnF,
        summary_func: SummaryF,
    ) -> Result<Option<TRet>>
    where
        TurnF: FnMut(Turn<TRng>) -> TakenAction<TRng>,
        SummaryF: FnMut(&TurnSummary<TRng>) -> TRet,
    {
        todo!()
    }

    pub fn register_domain(&mut self, player_number: PlayerNumber, logging_enabled: bool) {
        let domain = GameSessionDomain::new(self.session, player_number, logging_enabled);

        assert!(self.domains.insert(player_number, domain).is_none());
    }

    fn update_knowledge(&mut self, update: GlobalShellUpdate) {
        self.domains
            .iter_mut()
            .for_each(|(_, domain)| domain.update_global_knowledge(&update));
    }
}
