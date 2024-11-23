use std::{cell::RefCell, collections::HashMap, rc::Rc};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession, multiplayer_count::MultiplayerCount, player_number::PlayerNumber,
};
use rand::Rng;

use crate::game_session_domain::{
    ActionOrTurnSummary, GameSessionDomain, PlayerKnowledge, ShellUpdate,
};

#[derive(Debug, Clone)]
pub struct GameController<TRng> {
    pub session: RefCell<GameSession<TRng>>,
    knowledge: HashMap<PlayerNumber, Rc<RefCell<PlayerKnowledge>>>,
}

impl<TRng> GameController<TRng>
where
    TRng: Rng,
{
    pub fn new(multiplayer_count: MultiplayerCount, rng: TRng) -> Self {
        GameController {
            session: RefCell::new(GameSession::new(multiplayer_count, rng)),
            knowledge: HashMap::with_capacity(4),
        }
    }

    pub fn register_knowledge(
        &mut self,
        player_number: PlayerNumber,
        knowledge: PlayerKnowledge,
    ) -> Rc<RefCell<PlayerKnowledge>> {
        let pre_existing = self
            .knowledge
            .insert(player_number, Rc::new(RefCell::new(knowledge)));

        assert!(pre_existing.is_none());

        Rc::clone(&self.knowledge.get(&player_number).unwrap())
    }

    pub fn update_knowledge(&mut self, update: ShellUpdate) {
        self.knowledge
            .iter_mut()
            .for_each(|(_, rc)| rc.borrow_mut().update(&update));
    }
}

pub fn register_player<'session, TRng, F>(
    controller: &'session mut Rc<RefCell<GameController<TRng>>>,
    player_number: PlayerNumber,
    logger: F,
    logging_enabled: bool,
) -> GameSessionDomain<'session, TRng, F>
where
    TRng: Rng,
    F: FnMut(ActionOrTurnSummary<TRng>),
{
    let borrowed_controller = controller.borrow();
    GameSessionDomain::new(
        Rc::clone(controller),
        &borrowed_controller.session,
        player_number,
        logger,
        logging_enabled,
    )
}
