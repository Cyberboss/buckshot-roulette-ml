extern crate blas_src;

mod game_action;
mod game_controller;
mod game_session_domain;
mod player_knowledge;
mod relative_player;
mod seat_map;

use std::{
    cell::RefCell,
    hash::{BuildHasher, DefaultHasher, RandomState},
};

use buckshot_roulette_gameplay_engine::{
    game_session::GameSession, multiplayer_count::MultiplayerCount, player_number::PlayerNumber,
};
use game_controller::GameController;
use game_session_domain::{action_space_static, state_space_static};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rsrl::{
    control::td::SARSALambda,
    domains::Domain,
    fa::linear::{
        basis::{Combinators, Fourier, TileCoding},
        optim::SGD,
        LFA,
    },
    make_shared,
    params::Parameterised,
    policies::{EpsilonGreedy, Greedy, Policy, Random},
    spaces::Space,
    traces::Trace,
    Handler,
};

const ALPHA: f64 = 0.01;
const GAMMA: f64 = 0.99;
const LAMBDA: f64 = 0.7;

struct DeterministicHasher {}

impl BuildHasher for DeterministicHasher {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        DefaultHasher::new()
    }
}

fn main() {
    let mut rng = StdRng::seed_from_u64(0);
    let mut agent = {
        let n_actions = action_space_static().card().into();

        let memory_size = 1024 * 1024;

        let s = DeterministicHasher {};
        let basis = TileCoding::new(s, 1000, memory_size).with_bias();

        // let basis = Fourier::from_space(5, state_space_static()).with_bias();
        let fa_theta = make_shared(LFA::vector(basis, SGD(1.0), n_actions));

        let policy = EpsilonGreedy::new(Greedy::new(fa_theta.clone()), Random::new(n_actions), 0.2);
        let wdim = fa_theta.weights_dim();

        let trace = Trace::replacing(wdim, GAMMA, LAMBDA);

        SARSALambda {
            fa_theta,
            policy,
            trace,
            alpha: ALPHA,
            gamma: GAMMA,
        }
    };

    for e in 0..1000 {
        // Episode loop:
        let mut j = 0;
        let session = RefCell::new(GameSession::new(
            MultiplayerCount::Two,
            StdRng::seed_from_u64(rng.next_u64()),
        ));

        let mut controller = GameController::new(
            &session,
            (agent, rng),
            |(agent, mut rng), domain| {
                (
                    agent.policy.sample(&mut rng, domain.emit().state()),
                    (agent, rng),
                )
            },
            |(mut agent, rng), transition| {
                agent.handle(transition).unwrap();
                (agent, rng)
            },
            true,
            true,
        );

        controller.register_domain(PlayerNumber::One, true);
        controller.register_domain(PlayerNumber::Two, true);

        for i in 0.. {
            j = i;
            if controller.take_domain_action() {
                break;
            }
        }

        (agent, rng) = controller.extract_agent();
        agent.policy.epsilon *= 0.995;

        println!("Batch {}: {} steps...", e + 1, j + 1);
    }

    let session = RefCell::new(GameSession::new(
        MultiplayerCount::Two,
        StdRng::seed_from_u64(rng.next_u64()),
    ));

    let mut controller = GameController::new(
        &session,
        (agent, rng),
        |(agent, mut rng), domain| {
            (
                agent.policy.sample(&mut rng, domain.emit().state()),
                (agent, rng),
            )
        },
        |(mut agent, rng), transition| {
            agent.handle(transition).unwrap();
            (agent, rng)
        },
        true,
        true,
    );

    controller.register_domain(PlayerNumber::One, true);
    controller.register_domain(PlayerNumber::Two, true);

    loop {
        if controller.take_domain_action() {
            break;
        }
    }
}
